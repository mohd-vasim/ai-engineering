"""RAG Utils"""

# pylint: disable=W1203

import os
import logging
import numpy as np
from pymilvus import MilvusClient, DataType
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.novita import NovitaAI
from llama_index.core.base.llms.types import ChatMessage, MessageRole

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def batch_iterate(lst, batch_size):
    """Yield successive n-sized chunks from list."""
    for i in range(0, len(lst), batch_size):
        yield lst[i : i + batch_size]


class EmbedData:
    """Embed data"""

    def __init__(self, embed_model_name="BAAI/bge-large-en-v1.5", batch_size=512):
        self.embed_model_name = embed_model_name
        self.embed_model = self._load_embed_model()
        self.batch_size = batch_size
        self.contexts = None
        self.embeddings = []
        self.binary_embeddings = []  # Store binary quantized embeddings

    def _load_embed_model(self):
        embed_model = HuggingFaceEmbedding(
            model_name=self.embed_model_name,
            trust_remote_code=True,
            cache_folder="./hf_cache",
        )
        return embed_model

    def generate_embedding(self, context):
        """Generate embeddings"""
        return self.embed_model.get_text_embedding_batch(context)

    def _binary_quantize(self, embeddings):
        """Convert float32 embeddings to binary vectors"""
        embeddings_array = np.array(embeddings)
        binary_embeddings = np.where(embeddings_array > 0, 1, 0).astype(np.uint8)
        # Pack bits into bytes (8 dimensions per byte)
        packed_embeddings = np.packbits(binary_embeddings, axis=1)
        return [vec.tobytes() for vec in packed_embeddings]

    def embed(self, contexts):
        """Embed contexts"""
        self.contexts = contexts

        logger.info(f"Generating embeddings for {len(contexts)} contexts...")

        for batch_context in batch_iterate(contexts, self.batch_size):
            # Generate float32 embeddings
            batch_embeddings = self.generate_embedding(batch_context)
            self.embeddings.extend(batch_embeddings)

            # Convert to binary and store
            binary_batch = self._binary_quantize(batch_embeddings)
            self.binary_embeddings.extend(binary_batch)

        logger.info(
            f"Generated {len(self.embeddings)} embeddings with binary quantization"
        )


class MilvusVDB_BQ:
    """Milvus Vector DB"""

    def __init__(
        self,
        collection_name,
        vector_dim=1024,
        batch_size=512,
        db_file="milvus_binary_quantized.db",
    ):
        self.collection_name = collection_name
        self.batch_size = batch_size
        self.vector_dim = vector_dim
        self.db_file = db_file
        self.client = None

    def define_client(self):
        """Define client"""
        try:
            self.client = MilvusClient(self.db_file)
            logger.info(f"Initialized Milvus Lite client with database: {self.db_file}")
        except Exception as e:
            logger.error(f"Failed to initialize Milvus client: {e}")
            raise e

    def create_collection(self):
        """Create collection"""
        # Drop existing collection if it exists
        if self.client.has_collection(collection_name=self.collection_name):
            self.client.drop_collection(collection_name=self.collection_name)
            logger.info(f"Dropped existing collection: {self.collection_name}")

        # Create schema for binary vectors
        schema = self.client.create_schema(
            auto_id=True,
            enable_dynamic_fields=True,
        )

        # Add fields to schema
        schema.add_field(
            field_name="id", datatype=DataType.INT64, is_primary=True, auto_id=True
        )
        schema.add_field(
            field_name="context", datatype=DataType.VARCHAR, max_length=65535
        )
        schema.add_field(
            field_name="binary_vector",
            datatype=DataType.BINARY_VECTOR,
            dim=self.vector_dim,
        )

        # Create index parameters for binary vectors
        index_params = self.client.prepare_index_params()
        index_params.add_index(
            field_name="binary_vector",
            index_name="binary_vector_index",
            index_type="BIN_FLAT",  # Exact search for binary vectors
            metric_type="HAMMING",  # Hamming distance for binary vectors
        )

        # Create collection with schema and index
        self.client.create_collection(
            collection_name=self.collection_name,
            schema=schema,
            index_params=index_params,
        )

        logger.info(
            f"Created collection '{self.collection_name}' with binary vectors (dim={self.vector_dim})"
        )

    def ingest_data(self, embeddata):
        """Ingest data"""
        logger.info(f"Ingesting {len(embeddata.contexts)} documents...")

        total_inserted = 0
        for batch_context, batch_binary_embeddings in zip(
            batch_iterate(embeddata.contexts, self.batch_size),
            batch_iterate(embeddata.binary_embeddings, self.batch_size),
        ):
            # Prepare data for insertion
            data_batch = []
            for context, binary_embedding in zip(
                batch_context, batch_binary_embeddings
            ):
                data_batch.append(
                    {"context": context, "binary_vector": binary_embedding}
                )

            # Insert batch
            self.client.insert(collection_name=self.collection_name, data=data_batch)

            total_inserted += len(batch_context)
            logger.info(f"Inserted batch: {len(batch_context)} documents")

        logger.info(
            f"Successfully ingested {total_inserted} documents with binary quantization"
        )


class Retriever:
    """Retriever"""

    def __init__(self, vector_db, embeddata, top_k=5):
        self.vector_db = vector_db
        self.embeddata = embeddata
        self.top_k = top_k

    def _binary_quantize_query(self, query_embedding):
        """Binary qant for query"""
        embedding_array = np.array([query_embedding])
        binary_embedding = np.where(embedding_array > 0, 1, 0).astype(np.uint8)
        # Pack bits into bytes (8 dimensions per byte)
        packed_embedding = np.packbits(binary_embedding, axis=1)
        return packed_embedding[0].tobytes()

    def search(self, query, top_k=None):
        """Search query"""
        if top_k is None:
            top_k = self.top_k

        # Generate query embedding (float32)
        query_embedding = self.embeddata.embed_model.get_query_embedding(query)
        # Convert to binary vectors
        binary_query = self._binary_quantize_query(query_embedding)

        # Perform search using MilvusClient
        search_results = self.vector_db.client.search(
            collection_name=self.vector_db.collection_name,
            data=[binary_query],
            anns_field="binary_vector",
            search_params={"metric_type": "HAMMING", "params": {}},
            limit=top_k,
            output_fields=["context"],
        )

        # Format results
        formatted_results = []
        for result in search_results[0]:
            formatted_results.append(
                {
                    "id": result["id"],
                    "score": 1.0
                    / (
                        1.0 + result["distance"]
                    ),  # Convert Hamming distance to similarity score
                    "payload": {"context": result["entity"]["context"]},
                }
            )

        return formatted_results


class RAG:
    """RAG Pipeline"""

    def __init__(self, retriever, llm_model="openai/gpt-oss-120b", api_key=None):
        system_msg = ChatMessage(
            role=MessageRole.SYSTEM,
            content="You are a helpful assistant that answers questions about the user's document.",
        )
        self.messages = [system_msg]
        self.llm_model = llm_model
        self.api_key = api_key or os.getenv("NOVITA_API_KEY")
        self.llm = self._setup_llm()
        self.retriever = retriever
        self.prompt_template = (
            "CONTEXT: {context}\n"
            "---------------------\n"
            "Given the context information above I want you to think step by step to answer the user's query in a crisp and concise manner. "
            "In case you don't know the answer simply say 'I don't know!'. Don't try to make up an answer. Only answer based on facts and contextual information.\n"
            "QUERY: {query}\n"
            "ANSWER: "
        )

    def _setup_llm(self):
        """Setup llm"""
        if not self.api_key:
            raise ValueError(
                "API key is required. Set api_key environment variable or pass api_key parameter."
            )

        return NovitaAI(
            model=self.llm_model,
            api_key=self.api_key,
            temperature=0.4,
            max_tokens=1000,
        )

    def generate_context(self, query, top_k=5):
        """Generate context"""
        results = self.retriever.search(query, top_k=top_k)

        combined_context = []
        for entry in results:
            context = entry["payload"]["context"]
            combined_context.append(context)

        return "\n\n---\n\n".join(combined_context)

    def query(self, query, stream=True):
        """Query"""
        # Generate context from retrieval
        context = self.generate_context(query=query)
        # Create prompt from prompt template
        prompt = self.prompt_template.format(context=context, query=query)

        if stream:
            # Stream response
            streaming_response = self.llm.stream_complete(prompt)
            return streaming_response
        else:
            # Complete response
            response = self.llm.complete(prompt)
            return response.text

    def chat_query(self, query, stream=True):
        """Chat query"""
        context = self.generate_context(query=query)
        prompt = self.prompt_template.format(context=context, query=query)
        user_msg = ChatMessage(role=MessageRole.USER, content=prompt)

        if stream:
            # Stream chat response
            streaming_response = self.llm.stream_chat([user_msg])
            return streaming_response
        else:
            # Complete chat response
            chat_response = self.llm.chat([user_msg])
            return chat_response.message.content

# -------- CONFIG --------
REMOTE=origin

MM_RAG_FOLDER=06-projects/04-multimodal-rag
MM_RAG_BRANCH=multimodal-rag


# -------- MultimodalRAG CHATBOT --------

deploy-mm-rag:
	git subtree push --prefix=$(MM_RAG_FOLDER) $(REMOTE) $(MM_RAG_BRANCH)

pull-mm-rag:
	git subtree pull --prefix=$(MM_RAG_FOLDER) $(REMOTE) $(MM_RAG_BRANCH) --squash


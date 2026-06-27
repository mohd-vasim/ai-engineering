"""Load emb model"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    AutoModel,
    AutoTokenizer,
    SiglipModel,
    SiglipProcessor,
    WhisperModel,
    WhisperFeatureExtractor,
)
from huggingface_hub import hf_hub_download


class MultiModalEmbedder(nn.Module):
    """Standalone multimodal embedder - no external dependencies."""

    def __init__(self):
        super().__init__()
        # Text encoder (384d, no projection needed)
        self.text_tokenizer = AutoTokenizer.from_pretrained(
            "sentence-transformers/all-MiniLM-L6-v2"
        )
        self.text_encoder = AutoModel.from_pretrained(
            "sentence-transformers/all-MiniLM-L6-v2"
        )

        # Image encoder (768d -> 384d projection)
        self.image_processor = SiglipProcessor.from_pretrained(
            "google/siglip-base-patch16-512"
        )
        self.image_encoder = SiglipModel.from_pretrained(
            "google/siglip-base-patch16-512"
        ).vision_model
        self.image_proj = nn.Linear(768, 384)

        # Audio encoder (384d, no projection needed)
        self.audio_processor = WhisperFeatureExtractor.from_pretrained(
            "openai/whisper-tiny"
        )
        self.audio_encoder = WhisperModel.from_pretrained("openai/whisper-tiny").encoder

    def encode_text(self, texts):
        if isinstance(texts, str):
            texts = [texts]
        inputs = self.text_tokenizer(
            texts, padding=True, truncation=True, return_tensors="pt"
        )
        inputs = {k: v.to(next(self.parameters()).device) for k, v in inputs.items()}
        outputs = self.text_encoder(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1)  # Mean pooling
        return F.normalize(embeddings, p=2, dim=-1)

    def encode_image(self, images):
        inputs = self.image_processor(images=images, return_tensors="pt")
        inputs = {k: v.to(next(self.parameters()).device) for k, v in inputs.items()}
        outputs = self.image_encoder(**inputs)
        embeddings = outputs.pooler_output
        embeddings = self.image_proj(embeddings)  # 768 -> 384
        return F.normalize(embeddings, p=2, dim=-1)

    def encode_audio(self, waveform):
        # waveform: numpy array or tensor at 16kHz
        if isinstance(waveform, torch.Tensor):
            waveform = waveform.squeeze().numpy()
        inputs = self.audio_processor(
            waveform, sampling_rate=16000, return_tensors="pt"
        )
        inputs = {k: v.to(next(self.parameters()).device) for k, v in inputs.items()}
        outputs = self.audio_encoder(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1)  # Mean pooling
        return F.normalize(embeddings, p=2, dim=-1)


def load_model():
    """Load model"""
    # Load model
    model = MultiModalEmbedder()

    # Download and load trained weights
    checkpoint_path = hf_hub_download(
        repo_id="llm-semantic-router/multi-modal-embed-small", filename="model.pt"
    )
    state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    all_keys = list(state_dict.keys())
    print(f"Checkpoint has {len(all_keys)} keys. First 10:")
    for k in all_keys[:10]:
        print(f"  {k}")
    prefixes = sorted({k.split(".")[0] + "." + k.split(".")[1] for k in all_keys if "." in k})
    print(f"Unique top-2 prefixes: {prefixes}")

    # Load text encoder weights
    text_keys = {k.replace("text_encoder.encoder.", ""): v for k, v in state_dict.items() if k.startswith("text_encoder.encoder.")}
    print(f"Text encoder keys matched: {len(text_keys)}")
    model.text_encoder.load_state_dict(text_keys)

    # Load image encoder and projection weights
    img_keys = {k.replace("image_encoder.vision_encoder.", ""): v for k, v in state_dict.items() if k.startswith("image_encoder.vision_encoder.")}
    proj_keys = {k.replace("image_encoder.projection.", ""): v for k, v in state_dict.items() if k.startswith("image_encoder.projection.")}
    print(f"Image encoder keys matched: {len(img_keys)}, projection keys: {len(proj_keys)}")
    model.image_encoder.load_state_dict(img_keys)
    model.image_proj.load_state_dict(proj_keys)

    # Load audio encoder weights
    audio_keys = {k.replace("audio_encoder.encoder.", ""): v for k, v in state_dict.items() if k.startswith("audio_encoder.encoder.")}
    print(f"Audio encoder keys matched: {len(audio_keys)}")
    model.audio_encoder.load_state_dict(audio_keys)

    model.eval()
    print("Model loaded successfully!")

    return model


if __name__ == "__main__":
    print("Testing load emb model script")
    load_model()

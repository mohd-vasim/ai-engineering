"""Load multimodal embedding model using from_config to skip downloading base model weights."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    AutoConfig,
    AutoModel,
    AutoTokenizer,
    SiglipConfig,
    SiglipModel,
    SiglipProcessor,
    WhisperConfig,
    WhisperModel,
    WhisperFeatureExtractor,
)
from huggingface_hub import hf_hub_download


class MultiModalEmbedder(nn.Module):

    def __init__(self):
        super().__init__()
        # Text encoder — download config + tokenizer only, no model weights
        text_config = AutoConfig.from_pretrained("sentence-transformers/all-MiniLM-L6-v2", token=False)
        self.text_tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2", token=False)
        self.text_encoder = AutoModel.from_config(text_config)

        # Image encoder — download config + processor only, no model weights
        image_config = SiglipConfig.from_pretrained("google/siglip-base-patch16-512", token=False)
        self.image_processor = SiglipProcessor.from_pretrained("google/siglip-base-patch16-512", token=False)
        self.image_encoder = SiglipModel(image_config).vision_model
        self.image_proj = nn.Linear(768, 384)

        # Audio encoder — download config + feature extractor only, no model weights
        audio_config = WhisperConfig.from_pretrained("openai/whisper-tiny", token=False)
        self.audio_processor = WhisperFeatureExtractor.from_pretrained("openai/whisper-tiny", token=False)
        self.audio_encoder = WhisperModel(audio_config).encoder

    def encode_text(self, texts):
        if isinstance(texts, str):
            texts = [texts]
        inputs = self.text_tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
        inputs = {k: v.to(next(self.parameters()).device) for k, v in inputs.items()}
        outputs = self.text_encoder(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1)
        return F.normalize(embeddings, p=2, dim=-1)

    def encode_image(self, images):
        inputs = self.image_processor(images=images, return_tensors="pt")
        inputs = {k: v.to(next(self.parameters()).device) for k, v in inputs.items()}
        outputs = self.image_encoder(**inputs)
        embeddings = outputs.pooler_output
        embeddings = self.image_proj(embeddings)
        return F.normalize(embeddings, p=2, dim=-1)

    def encode_audio(self, waveform):
        if isinstance(waveform, torch.Tensor):
            waveform = waveform.squeeze().numpy()
        inputs = self.audio_processor(waveform, sampling_rate=16000, return_tensors="pt")
        inputs = {k: v.to(next(self.parameters()).device) for k, v in inputs.items()}
        outputs = self.audio_encoder(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1)
        return F.normalize(embeddings, p=2, dim=-1)


def load_model():
    print("Initializing model architecture (no base model weights downloaded)...")
    model = MultiModalEmbedder()

    print("Downloading custom checkpoint...")
    checkpoint_path = hf_hub_download(
        repo_id="llm-semantic-router/multi-modal-embed-small", filename="model.pt", token=False
    )
    state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    all_keys = list(state_dict.keys())
    print(f"Checkpoint has {len(all_keys)} keys. Sample:")
    for k in all_keys[:10]:
        print(f"  {k}")

    text_keys = {k.replace("text_encoder.encoder.", ""): v for k, v in state_dict.items() if k.startswith("text_encoder.encoder.")}
    img_keys = {k.replace("image_encoder.vision_encoder.", ""): v for k, v in state_dict.items() if k.startswith("image_encoder.vision_encoder.")}
    proj_keys = {k.replace("image_encoder.projection.", ""): v for k, v in state_dict.items() if k.startswith("image_encoder.projection.")}
    audio_keys = {k.replace("audio_encoder.encoder.", ""): v for k, v in state_dict.items() if k.startswith("audio_encoder.encoder.")}

    print(f"Keys matched — text: {len(text_keys)}, image: {len(img_keys)}, proj: {len(proj_keys)}, audio: {len(audio_keys)}")

    model.text_encoder.load_state_dict(text_keys)
    model.image_encoder.load_state_dict(img_keys)
    model.image_proj.load_state_dict(proj_keys)
    model.audio_encoder.load_state_dict(audio_keys)

    model.eval()
    print("Model loaded successfully!")
    return model


if __name__ == "__main__":
    print("Testing load_emb_model_v2")
    load_model()

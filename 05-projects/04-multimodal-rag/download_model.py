from huggingface_hub import snapshot_download

print("Downloading jinaai/jina-clip-v2 ...")
snapshot_download("jinaai/jina-clip-v2", allow_patterns=["*.safetensors", "*.bin", "model.safetensors", "pytorch_model.bin", "config.json", "*.json"])
print("Model downloaded successfully!")

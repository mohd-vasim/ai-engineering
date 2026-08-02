# Shoplifting Detection & Video Captioning

This project extracts frames from surveillance video feeds and leverages a local vision-language model via Ollama to generate chronological captions describing activities and detecting suspicious behaviors (e.g., shoplifting).

---

## 1. Setup with `uv`

[uv](https://github.com/astral-sh/uv) is an extremely fast Python package installer and resolver.

### Install `uv`
If you don't have `uv` installed, run one of the following commands:
* **macOS/Linux:**
  ```bash
  curl -LsSf https://astral.sh/uv/install.sh | sh
  ```
* **Homebrew (macOS):**
  ```bash
  brew install uv
  ```

### Initialize Virtual Environment & Install Dependencies
Navigate to the project root directory and run:
```bash
# Create a virtual environment using uv
uv venv

# Activate the virtual environment
source .venv/bin/activate

# Install required dependencies
uv pip install opencv-python ollama
```

---

## 2. Setup Ollama and Download the Model

### Install Ollama
1. Download and install Ollama from the official website: [ollama.com](https://ollama.com).
2. Launch the Ollama application to start the local server.

### Pull the Vision Model
We use a Qwen vision model (which supports text and image inputs). Run the following command in your terminal to download it:
```bash
# Pull the standard Qwen 3.5 2B model
ollama pull qwen3.5:2b

# Alternatively, pull the MLX-optimized version (highly recommended for Apple Silicon Macs)
ollama pull qwen3.5:2b-mlx
```

---

## 3. Running the Analysis

To extract frames from the sample video at `0.5s` intervals, save them into a folder, and run a batched sequence analysis through Ollama, execute:

```bash
# Make sure your virtual environment is active
python pose_est_vlm/test_ollama_caption_v2.py
```

* The frame JPEGs will be saved to `sample_videos/gettyimages-1995820194-640_adpp/`.
* The script sends frames in batches of `10` to avoid local memory limits and provides chronological descriptions of the actions.

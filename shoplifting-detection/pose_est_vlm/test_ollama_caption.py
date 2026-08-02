import base64
import os
import cv2
import ollama

def extract_and_analyze(video_path, model_name="qwen3.5:2b-mlx", interval_seconds=5.0):
    if not os.path.exists(video_path):
        print(f"Error: Sample video not found at {video_path}", flush=True)
        return

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}", flush=True)
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0

    print(f"Video Info: {duration:.2f}s duration | {fps:.2f} FPS | {total_frames} frames", flush=True)

    step = max(1, int(fps * interval_seconds))

    prompt = (
        "Analyze this frame from a surveillance camera. "
        "Describe what the person is doing in 1-2 concise sentences. "
        "Pay close attention to suspicious activities such as shoplifting, "
        "taking items from shelves, concealing items, or looking around nervously."
    )

    current_frame = 0
    
    try:
        while current_frame < total_frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, current_frame)
            ret, frame = cap.read()
            if not ret:
                break

            timestamp = current_frame / fps
            print("-" * 60, flush=True)
            print(f"Processing frame at {timestamp:.2f}s (Frame #{current_frame})", flush=True)

            # 1. Encode frame to JPG buffer
            success, buffer = cv2.imencode('.jpg', frame)
            if not success:
                print(f"Failed to encode frame at {timestamp:.2f}s", flush=True)
                current_frame += step
                continue

            # 2. Convert JPG buffer to Base64 String
            base64_image = base64.b64encode(buffer).decode('utf-8')

            try:
                # Pass explicit Base64 string to Ollama API
                response = ollama.chat(
                    model=model_name,
                    think=False,
                    messages=[
                        {
                            'role': 'user',
                            'content': prompt,
                            'images': [base64_image]  # Explicit Base64 string
                        }
                    ]
                )
                
                # Retrieve content safely from response object or dict
                if hasattr(response, 'message'):
                    content = response.message.content.strip()
                else:
                    content = response['message']['content'].strip()

                print(f"Model Response:\n{content}", flush=True)

            except Exception as e:
                print(f"Error during Ollama API call at {timestamp:.2f}s: {e}", flush=True)

            current_frame += step

    finally:
        cap.release()
        print("-" * 60, flush=True)
        print("Analysis complete.", flush=True)


if __name__ == "__main__":
    video_path = "sample_videos/gettyimages-1995820194-640_adpp.mp4"
    model_name = "qwen3.5:2b-mlx"
    
    print(f"Analyzing {video_path} with {model_name}...", flush=True)
    extract_and_analyze(video_path, model_name=model_name, interval_seconds=0.5)
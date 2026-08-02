import os
import cv2
import ollama

def extract_and_analyze_v2(video_path, model_name="qwen3.5:2b-mlx", interval_seconds=0.5, batch_size=10):
    if not os.path.exists(video_path):
        print(f"Error: Video file not found at {video_path}", flush=True)
        return

    # 1. Create output directory for frames
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    output_dir = os.path.join(os.path.dirname(video_path), video_name)
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory for frames: {output_dir}", flush=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error: Could not open video file {video_path}", flush=True)
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0

    print(f"Video Info: {duration:.2f}s duration | {fps:.2f} FPS | {total_frames} frames", flush=True)

    step = max(1, int(fps * interval_seconds))
    print(f"Sampling every {step} frames (approx {interval_seconds}s intervals)", flush=True)

    frame_idx = 0
    saved_paths = []

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % step == 0:
                timestamp = frame_idx / fps
                frame_filename = f"frame_{timestamp:.2f}s.jpg"
                frame_path = os.path.abspath(os.path.join(output_dir, frame_filename))
                
                # Save frame as JPEG
                cv2.imwrite(frame_path, frame)
                saved_paths.append(frame_path)
                print(f"Saved: {frame_filename}", flush=True)

            frame_idx += 1
    finally:
        cap.release()

    total_extracted = len(saved_paths)
    print(f"\nExtraction complete. Saved {total_extracted} frames.", flush=True)

    if total_extracted == 0:
        print("No frames were extracted. Exiting.", flush=True)
        return

    # 2. Analyze the saved frames with Ollama
    print(f"\nStarting analysis with Ollama model '{model_name}'...", flush=True)

    prompt = (
        "Analyze this sequence of surveillance camera frames in chronological order. "
        "Describe what the person is doing in 2-3 concise sentences. "
        "Pay close attention to suspicious activities such as shoplifting, "
        "taking items from shelves, concealing items, or looking around nervously."
    )

    # Process in batches to prevent hitting context limits or OOM errors
    # If batch_size is None or <= 0, process all in one go
    if not batch_size or batch_size <= 0:
        batches = [saved_paths]
    else:
        batches = [saved_paths[i:i + batch_size] for i in range(0, total_extracted, batch_size)]

    for idx, batch in enumerate(batches):
        print("-" * 60, flush=True)
        if len(batches) > 1:
            print(f"Processing Batch {idx+1}/{len(batches)} ({len(batch)} frames)...", flush=True)
        else:
            print(f"Processing all {len(batch)} frames in one go...", flush=True)
        
        # Print the frames in the current batch
        for p in batch:
            print(f"  - {os.path.basename(p)}", flush=True)

        try:
            response = ollama.chat(
                model=model_name,
                think=False,
                messages=[
                    {
                        'role': 'user',
                        'content': prompt,
                        'images': batch
                    }
                ]
            )

            if hasattr(response, 'message'):
                content = response.message.content.strip()
            else:
                content = response['message']['content'].strip()

            print(f"\nModel Response for Batch {idx+1}:\n{content}\n", flush=True)

        except Exception as e:
            print(f"Error during Ollama API call: {e}", flush=True)

    print("-" * 60, flush=True)
    print("Analysis complete.", flush=True)


if __name__ == "__main__":
    video_path = "sample_videos/video_1.mp4"
    model_name = "qwen3.5:2b-mlx"
    
    # You can set batch_size=0 to pass all frames in a single call,
    # or set a positive integer (like 10) to process them in groups.
    extract_and_analyze_v2(
        video_path=video_path,
        model_name=model_name,
        interval_seconds=1,
        batch_size=3
    )

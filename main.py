#!/usr/bin/env python3
import argparse
import cv2
import whisper
import warnings
from transformers import AutoModelForCausalLM, AutoTokenizer
from PIL import Image
import torch
import os
from tqdm import tqdm
import numpy as np
import json
from datetime import datetime
import os
import transformers
from openai import OpenAI
from dotenv import load_dotenv
import time
from frame_selection import process_video
import gradio as gr

# Function to transcribe audio from a video file using Whisper model
def transcribe_video(video_path):
    print(f"Loading audio model")

    # Suppress the FutureWarning from torch.load
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=FutureWarning)
        model = whisper.load_model("large")

    print(f"Audio model loaded")

    print(f"Transcribing video")

    # Transcribe the audio from the video file
    result = model.transcribe(video_path)

    # Process the result to include timestamps
    timestamped_transcript = []
    for segment in result["segments"]:
        timestamped_transcript.append({
            "start": segment["start"],
            "end": segment["end"],
            "text": segment["text"]
        })

    # Unload the model
    del model
    torch.cuda.empty_cache()  # If using CUDA
    print("Audio model unloaded")

    return timestamped_transcript

# Function to describe frames using a Vision-Language Model
def describe_frames(video_path, frame_numbers):
    print("Loading Vision-Language model...")
    model_id = "vikhyatk/moondream2"
    revision = "2024-07-23"
    model = AutoModelForCausalLM.from_pretrained(
        model_id, trust_remote_code=True, revision=revision, torch_dtype=torch.bfloat16
    ).to(device=torch.device("cuda"))
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
    print("Vision-Language model loaded")

    # Load prompt from file
    with open('prompts/moondream_prompt.md', 'r') as file:
        prompt = file.read().strip()

    print("Extracting frames...")
    video = cv2.VideoCapture(video_path)
    fps = video.get(cv2.CAP_PROP_FPS)
    frames = []

    for frame_number in frame_numbers:
        video.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        success, frame = video.read()
        if success:
            timestamp = frame_number / fps
            frames.append((frame_number, frame, timestamp))

    video.release()
    print(f"{len(frames)} frames extracted")

    print("Describing frames...")
    batch_size = 8  # Adjust this based on your GPU memory
    results = []

    for i in tqdm(range(0, len(frames), batch_size), desc="Processing batches"):
        batch_frames = frames[i:i+batch_size]
        batch_images = [Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)) for _, frame, _ in batch_frames]
        batch_prompts = [prompt] * len(batch_images)

        batch_answers = model.batch_answer(
            images=batch_images,
            prompts=batch_prompts,
            tokenizer=tokenizer,
        )

        for (frame_number, _, timestamp), answer in zip(batch_frames, batch_answers):
            results.append((frame_number, timestamp, answer))

    # Unload the model
    del model
    torch.cuda.empty_cache()  # If using CUDA
    print("Vision-Language model unloaded")

    return results

def display_described_frames(video_path, descriptions):
    video = cv2.VideoCapture(video_path)
    current_index = 0

    def add_caption_to_frame(frame, caption):
        height, width = frame.shape[:2]
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        font_thickness = 1
        line_spacing = 10
        margin = 10

        # Split caption into lines
        words = caption.split()
        lines = []
        current_line = []

        for word in words:
            test_line = ' '.join(current_line + [word])
            (text_width, _), _ = cv2.getTextSize(test_line, font, font_scale, font_thickness)

            if text_width <= width - 2 * margin:
                current_line.append(word)
            else:
                lines.append(' '.join(current_line))
                current_line = [word]

        if current_line:
            lines.append(' '.join(current_line))

        # Calculate caption height
        (_, text_height), _ = cv2.getTextSize('Tg', font, font_scale, font_thickness)
        total_height = (text_height + line_spacing) * len(lines) + 2 * margin

        # Create caption image
        caption_image = np.zeros((total_height, width, 3), dtype=np.uint8)

        # Add text to the caption image
        y = margin + text_height
        for line in lines:
            cv2.putText(caption_image, line, (margin, y), font, font_scale, (255, 255, 255), font_thickness)
            y += text_height + line_spacing

        return np.vstack((caption_image, frame))

    print("\nKey controls:")
    print("  Right arrow or Space: Next frame")
    print("  Left arrow: Previous frame")
    print("  'q': Quit")

    try:
        while True:
            frame_number, timestamp, description = descriptions[current_index]
            video.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            success, frame = video.read()
            if not success:
                print(f"Failed to read frame {frame_number}")
                break

            # Resize the frame if it's too large
            max_height = 800
            height, width = frame.shape[:2]
            if height > max_height:
                ratio = max_height / height
                frame = cv2.resize(frame, (int(width * ratio), max_height))

            # Add caption to the frame
            frame_with_caption = add_caption_to_frame(frame, f"[{timestamp:.2f}s] Frame {frame_number}: {description}")

            # Display the frame with the caption
            cv2.imshow('Video Frames', frame_with_caption)
            cv2.setWindowTitle('Video Frames', f"Frame {frame_number}")

            # Wait for key press with a timeout
            key = cv2.waitKey(100) & 0xFF  # Wait for 100ms
            if key == ord('q'):  # Press 'q' to quit
                break
            elif key in (83, 32):  # Right arrow or space bar for next frame
                current_index = min(current_index + 1, len(descriptions) - 1)
            elif key == 81:  # Left arrow for previous frame
                current_index = max(current_index - 1, 0)
    except KeyboardInterrupt:
        print("\nDisplay interrupted by user.")
    finally:
        video.release()
        cv2.destroyAllWindows()

def get_frame_count(video_path):
    video = cv2.VideoCapture(video_path)
    if not video.isOpened():
        print(f"Error: Could not open video {video_path}")
        return 0
    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    video.release()
    print(f"Total frames: {frame_count}")
    return frame_count

def save_output(video_path, frame_count, transcript, descriptions, summary, total_run_time):
    os.makedirs('logs', exist_ok=True)
    timestamp = datetime.now().isoformat().replace(':', '-')
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    filename = f"logs/{timestamp}_{video_name}.json"

    # Read the current prompt from moondream_prompt.md
    with open('prompts/moondream_prompt.md', 'r') as prompt_file:
        current_prompt = prompt_file.read().strip()

    output = {
        "video_path": video_path,
        "frame_count": frame_count,
        "transcript": transcript,
        "moondream_prompt": current_prompt,
        "frame_descriptions": [
            {
                "frame_number": frame,
                "timestamp": timestamp,
                "description": desc
            } for frame, timestamp, desc in descriptions
        ],
        "summary": summary,
        "total_run_time": total_run_time
    }

    with open(filename, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"Output saved to {filename}")

def summarize_with_llama(transcript, descriptions):
    # Load the synthesis prompt
    with open('prompts/llama_synthesis_prompt.md', 'r') as file:
        synthesis_prompt = file.read().strip()

    # Prepare the input for the model
    timestamped_transcript = "\n".join([f"[{segment['start']:.2f}s-{segment['end']:.2f}s] {segment['text']}" for segment in transcript])
    frame_descriptions = "\n".join([f"[{timestamp:.2f}s] Frame {frame}: {desc}" for frame, timestamp, desc in descriptions])
    user_content = f"<transcript>\n{timestamped_transcript}\n</transcript>\n\n<frame_descriptions>\n{frame_descriptions}\n</frame_descriptions>"

    messages = [
        {"role": "system", "content": synthesis_prompt},
        {"role": "user", "content": user_content}
    ]

    # Construct the prompt
    prompt = messages[0]["content"] + "\n" + "\n".join([f"{m['role']}: {m['content']}" for m in messages[1:]])

    # Initialize the pipeline
    pipeline = transformers.pipeline(
        "text-generation",
        model="meta-llama/Meta-Llama-3.1-8B-Instruct",
        model_kwargs={"torch_dtype": torch.bfloat16},
        device_map="auto",
    )

    # Generate the summary
    outputs = pipeline(
        prompt,
        max_new_tokens=512,
        return_full_text=False
    )

    return outputs[0]['generated_text'].strip()

def summarize_with_hosted_llm(transcript, descriptions):
    # Load environment variables from .env file
    load_dotenv()

    # Load the synthesis prompt from file
    with open('prompts/synthesis_prompt.md', 'r') as file:
        synthesis_prompt = file.read().strip()

    # Prepare the input for the model
    timestamped_transcript = "\n".join([f"[{segment['start']:.2f}s-{segment['end']:.2f}s] {segment['text']}" for segment in transcript])
    frame_descriptions = "\n".join([f"[{timestamp:.2f}s] Frame {frame}: {desc}" for frame, timestamp, desc in descriptions])
    user_content = f"<transcript>\n{timestamped_transcript}\n</transcript>\n\n<frame_descriptions>\n{frame_descriptions}\n</frame_descriptions>"

    # Check if OPENAI_API_KEY is set
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable is not set. Please set it in your .env file or environment.")

    # Initialize the OpenAI client
    client = OpenAI(api_key=api_key)

    # Create the chat completion
    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": synthesis_prompt},
            {"role": "user", "content": user_content}
        ],
        max_tokens=1024
    )

    # Extract and return the generated summary
    return completion.choices[0].message.content.strip()

def process_video_web(video_file):
    # This function will be called by Gradio when a video is uploaded
    video_path = video_file.name
    print(f"Processing video: {video_path}")

    start_time = time.time()

    # Get frame count
    frame_count = get_frame_count(video_path)
    if frame_count == 0:
        return "Error: Could not process video."

    # Transcribe the video
    transcript = transcribe_video(video_path)
    print(f"Transcript generated.")

    # Select frames (using default method)
    print("Sampling every 50 frames")
    frame_numbers = list(range(0, frame_count, 50))

    # Describe frames
    descriptions = describe_frames(video_path, frame_numbers)
    print("Frame descriptions generated.")

    # Generate summary (using hosted LLM by default)
    print("Generating video summary...")
    summary = summarize_with_hosted_llm(transcript, descriptions)
    print("Summary generation complete.")

    total_run_time = time.time() - start_time
    print(f"Time taken: {total_run_time:.2f} seconds")

    # Save output to JSON file
    print("Saving output to JSON file...")
    save_output(video_path, frame_count, transcript, descriptions, summary, total_run_time)

    return f"Video Summary:\n{summary}\n\nTime taken: {total_run_time:.2f} seconds"

def main():
    parser = argparse.ArgumentParser(description="Process a video file.")
    parser.add_argument('video', type=str, nargs='?', help='Path to the video file')
    parser.add_argument('--debug', action='store_true', help='Display frames with descriptions')
    parser.add_argument('--save', action='store_true', help='Save output to a JSON file')
    parser.add_argument('--local', action='store_true', help='Use local Llama model for summarization')
    parser.add_argument('--frame-selection', action='store_true', help='Use CLIP-based key frame selection algorithm')
    parser.add_argument('--web', action='store_true', help='Start Gradio web interface')
    args = parser.parse_args()

    if args.web:
        # Create Gradio interface
        iface = gr.Interface(
            fn=process_video_web,
            inputs=gr.File(label="Upload Video"),
            outputs="text",
            title="Video Summarizer",
            description="Upload a video to get a summary."
        )
        iface.launch()
    elif args.video:
        # Existing command-line processing logic
        print(f"Processing video: {args.video}")

        start_time = time.time()

        # Get frame count
        frame_count = get_frame_count(args.video)
        if frame_count == 0:
            return

        # Transcribe the video to get the audio captions
        transcript = transcribe_video(args.video)
        print(f"Transcript: {transcript}")

        # Select frames based on the chosen method
        if args.frame_selection:
            print("Using frame selection algorithm")
            frame_numbers = process_video(args.video)
        else:
            print("Sampling every 50 frames")
            # Calculate frame numbers to describe every 50 frames
            frame_numbers = list(range(0, frame_count, 50))

        # Describe frames
        descriptions = describe_frames(args.video, frame_numbers)

        print("Generating video summary...")
        if args.local:
            print("Using local Llama model for summarization...")
            summary = summarize_with_llama(transcript, descriptions)
        else:
            print("Using hosted LLM for summarization...")
            summary = summarize_with_hosted_llm(transcript, descriptions)
        print("Summary generation complete.")
        print(f"Video Summary:\n{summary}")

        total_run_time = time.time() - start_time
        print(f"Time taken: {total_run_time:.2f} seconds")

        if args.save:
            print("Saving output to JSON file...")
            save_output(args.video, frame_count, transcript, descriptions, summary, total_run_time)

        if args.debug:
            display_described_frames(args.video, descriptions)
    else:
        print("Please provide a video file path or use the --web flag to start the web interface.")

if __name__ == "__main__":
    main()
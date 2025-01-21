#!/usr/bin/env python3
import argparse
import cv2
import whisper
import warnings
from transformers import AutoModelForCausalLM, AutoTokenizer
from PIL import Image, ImageDraw, ImageFont
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
import subprocess
import re

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
    model = AutoModelForCausalLM.from_pretrained(
        "vikhyatk/moondream2",
        revision="2025-01-09",
        trust_remote_code=True,
        device_map={"": "cuda"}
    )
    tokenizer = AutoTokenizer.from_pretrained("vikhyatk/moondream2", revision="2025-01-09")
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

def get_synthesis_prompt(num_keyframes: int) -> str:
    """Generate a dynamic synthesis prompt based on number of keyframes."""
    num_captions = max(4, num_keyframes // 3)  # At least 4 captions, or 1/3 of keyframes
    
    return f"""You are tasked with summarizing and captioning a video based on its transcript and frame descriptions. You MUST follow the exact format specified below.

Output Format:
1. A summary section wrapped in <summary></summary> tags
2. A captions section wrapped in <captions></captions> tags
3. Exactly {num_captions} individual captions, each wrapped in <caption></caption> tags
4. Each caption MUST start with a timestamp in seconds in square brackets, e.g. [2.5]

Example format:
<summary>
The video shows a classroom scene where...
</summary>

<captions>
<caption>[0.0] A teacher stands at the front of the classroom...</caption>
<caption>[5.2] Students raise their hands to answer...</caption>
...additional captions...
</captions>

Requirements for Summary:
1. Provide a clear overview of the video's main content and purpose
2. Include key events, characters, and settings
3. Integrate both visual and audio information
4. Keep it concise (3-5 sentences)

Requirements for Captions:
1. Generate exactly {num_captions} captions
2. Each caption MUST:
   - Start with a timestamp in [X.X] format
   - Be 20-50 words long
   - Focus on key events and context
   - Integrate both visual and audio information when relevant
3. Timestamps should be reasonably spaced throughout the video
4. Focus on narrative flow and context, not just describing what's visible

Input sections:
<transcript>
Timestamped transcript of the video
</transcript>

<frame_descriptions>
Timestamped descriptions of key frames
</frame_descriptions>"""

def summarize_with_hosted_llm(transcript, descriptions):
    # Load environment variables from .env file
    load_dotenv()

    # Generate dynamic synthesis prompt based on number of keyframes
    num_keyframes = len(descriptions)
    synthesis_prompt = get_synthesis_prompt(num_keyframes)

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

def parse_synthesis_output(output: str) -> tuple:
    """Parse the synthesis output to extract summary and captions."""
    try:
        # Extract summary (required)
        summary_match = re.search(r'<summary>(.*?)</summary>', output, re.DOTALL)
        if not summary_match:
            raise ValueError("No summary found in synthesis output")
        summary = summary_match.group(1).strip()
        
        # Extract captions (required)
        captions_match = re.search(r'<captions>(.*?)</captions>', output, re.DOTALL)
        if not captions_match:
            raise ValueError("No captions found in synthesis output")
        
        captions_text = captions_match.group(1).strip()
        caption_matches = re.finditer(r'<caption>\s*\[(\d+\.?\d*)\](.*?)</caption>', captions_text, re.DOTALL)
        
        caption_list = []
        for match in caption_matches:
            timestamp = float(match.group(1))
            text = match.group(2).strip()
            if text:  # Only add if we have actual text
                caption_list.append((timestamp, text))
        
        if not caption_list:
            raise ValueError("No valid captions found in synthesis output")
        
        return summary, caption_list
    except (ValueError, AttributeError) as e:
        print(f"\nError parsing synthesis output: {str(e)}")
        print("Raw output:", output)
        return None, None

def create_summary_clip(summary: str, width: int, height: int, fps: int) -> str:
    """Create a video clip with centered summary text."""
    # Calculate duration based on reading speed (400 words/min) + 20% buffer
    word_count = len(summary.split())
    base_duration = (word_count / 400) * 60  # Convert to seconds
    duration = base_duration * 1.2  # Add 20% buffer
    total_frames = int(duration * fps)
    
    # Create output path
    os.makedirs('outputs', exist_ok=True)
    temp_summary_path = os.path.join('outputs', f'temp_summary_{int(time.time())}.mp4')
    
    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(temp_summary_path, fourcc, fps, (width, height))
    
    # Create black frame
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Prepare text
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = min(height * 0.04 / 20, 1.0)  # Slightly larger font for summary
    margin = 40
    max_width = int(width * 0.8)  # Use 80% of frame width
    
    # Split text into lines
    words = summary.split()
    lines = []
    current_line = []
    
    for word in words:
        test_line = ' '.join(current_line + [word])
        (text_width, _), _ = cv2.getTextSize(test_line, font, font_scale, 1)
        
        if text_width <= max_width:
            current_line.append(word)
        else:
            if current_line:
                lines.append(' '.join(current_line))
            current_line = [word]
    
    if current_line:
        lines.append(' '.join(current_line))
    
    # Calculate text block height
    line_height = max(25, int(height * 0.04))
    total_height = len(lines) * line_height
    start_y = (height - total_height) // 2  # Center vertically
    
    # Write frames
    for _ in range(total_frames):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        y = start_y
        
        for line in lines:
            # Get text size for centering
            (text_width, _), _ = cv2.getTextSize(line, font, font_scale, 1)
            x = (width - text_width) // 2  # Center horizontally
            
            cv2.putText(frame,
                       line,
                       (x, y),
                       font,
                       font_scale,
                       (255, 255, 255),
                       1,
                       cv2.LINE_AA)
            y += line_height
        
        out.write(frame)
    
    out.release()
    return temp_summary_path, duration

def create_captioned_video(video_path: str, descriptions: list, summary: str, transcript: list, synthesis_captions: list = None, use_synthesis_captions: bool = False, output_path: str = None) -> str:
    """Create a video with persistent captions from keyframe descriptions and transcriptions."""
    print("\nCreating captioned video...")
    
    # Validate synthesis captions if requested
    if use_synthesis_captions:
        if not synthesis_captions:
            print("\nWarning: Synthesis captions were requested but none were generated.")
            print("This might indicate an issue with the synthesis process.")
            print("Falling back to frame descriptions.")
            use_synthesis_captions = False
        else:
            print(f"\nUsing {len(synthesis_captions)} synthesis captions for video output...")
    
    # Create output directory if needed
    if output_path is None:
        os.makedirs('outputs', exist_ok=True)
        base_name = os.path.splitext(os.path.basename(video_path))[0]
        temp_output = os.path.join('outputs', f'temp_{base_name}.mp4')
        final_output = os.path.join('outputs', f'captioned_{base_name}.mp4')
    else:
        temp_output = output_path + '.temp'
        final_output = output_path

    video = cv2.VideoCapture(video_path)
    fps = video.get(cv2.CAP_PROP_FPS)
    frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))

    # Choose which captions to use
    if use_synthesis_captions and synthesis_captions:
        print("\nUsing synthesis captions for video output...")
        # Convert synthesis captions to frame info format
        frame_info = []
        min_time_gap = 1.2  # Minimum time gap between captions in seconds
        
        for i, (timestamp, text) in enumerate(synthesis_captions):
            # Skip if this caption is too close to the previous one
            if i > 0:
                prev_timestamp = synthesis_captions[i-1][0]
                if timestamp - prev_timestamp < min_time_gap:
                    # Remove the previous caption if it exists in frame_info
                    if frame_info and frame_info[-1][1] == prev_timestamp:
                        frame_info.pop()
            
            # Find nearest frame number
            frame_number = int(timestamp * fps)
            frame_info.append((frame_number, timestamp, text))
            
        print(f"After filtering close captions: {len(frame_info)} captions remaining")
    else:
        print("\nUsing frame descriptions for video output...")
        # Use all frame descriptions
        frame_info = [(frame_num, timestamp, desc) for frame_num, timestamp, desc in descriptions]
    
    # Sort by timestamp
    frame_info.sort(key=lambda x: x[1])

    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(temp_output, fourcc, fps, (frame_width, frame_height))

    # Process the main video
    print("\nProcessing video...")
    frame_count = 0
    current_desc_idx = 0
    current_transcript_idx = 0

    with tqdm(total=total_frames, desc="Creating video") as pbar:
        while True:
            ret, frame = video.read()
            if not ret:
                break

            current_time = frame_count / fps

            # Find appropriate description for current timestamp
            while (current_desc_idx < len(frame_info) - 1 and 
                   current_time >= frame_info[current_desc_idx + 1][1]):
                current_desc_idx += 1

            # Find appropriate transcript segment
            while (current_transcript_idx < len(transcript) - 1 and 
                   current_time >= transcript[current_transcript_idx + 1]["start"]):
                current_transcript_idx += 1

            # Get current description
            frame_number, timestamp, description = frame_info[current_desc_idx]

            # Get current transcript segment
            transcript_segment = transcript[current_transcript_idx]
            transcript_text = transcript_segment["text"]
            segment_duration = transcript_segment["end"] - transcript_segment["start"]

            # Check for potential hallucination in transcript
            words_per_second = len(transcript_text.split()) / segment_duration if segment_duration > 0 else 0
            is_likely_hallucination = words_per_second < 0.5

            # Add caption using our existing logic
            height, width = frame.shape[:2]
            margin = 8
            padding = 10
            min_line_height = 20
            font = cv2.FONT_HERSHEY_SIMPLEX
            max_width = int(width * 0.9)
            
            # Start with timestamp and description (different format for synthesis captions)
            if use_synthesis_captions:
                full_text = description  # Synthesis captions already include context
            else:
                full_text = f"[{timestamp:.2f}s] {description}"
                
            # Normalize smart quotes and apostrophes to ASCII
            full_text = full_text.replace('"', '"').replace('"', '"')
            full_text = full_text.replace("'", "'").replace("'", "'")
            words = full_text.split()
            top_lines = []
            current_line = []
            
            font_scale = min(height * 0.03 / min_line_height, 0.7)
            
            # Split text into lines that fit the width
            for word in words:
                test_line = ' '.join(current_line + [word])
                (text_width, _), _ = cv2.getTextSize(test_line, font, font_scale, 1)
                
                if text_width <= max_width - 2 * margin:
                    current_line.append(word)
                else:
                    if current_line:
                        top_lines.append(' '.join(current_line))
                    current_line = [word]
            
            if current_line:
                top_lines.append(' '.join(current_line))

            # Add transcript as subtitle if not likely hallucination
            bottom_lines = []
            if not is_likely_hallucination and transcript_text.strip():
                # Normalize smart quotes and apostrophes in transcript
                transcript_text = transcript_text.replace('"', '"').replace('"', '"')
                transcript_text = transcript_text.replace("'", "'").replace("'", "'")
                # Split transcript into lines
                subtitle_words = transcript_text.split()
                current_line = []
                
                # Larger font scale for transcripts (4 sizes bigger)
                transcript_font_scale = min(height * 0.03 / min_line_height, 0.7) + 0.4
                
                for word in subtitle_words:
                    test_line = ' '.join(current_line + [word])
                    (text_width, _), _ = cv2.getTextSize(test_line, font, transcript_font_scale, 1)
                    
                    if text_width <= max_width - 2 * margin:
                        current_line.append(word)
                    else:
                        if current_line:
                            bottom_lines.append(' '.join(current_line))
                        current_line = [word]
                
                if current_line:
                    bottom_lines.append(' '.join(current_line))
            
            # Calculate box dimensions for top overlay (frame descriptions)
            top_line_count = len(top_lines)
            line_height = max(min_line_height, int(height * 0.03))
            top_box_height = top_line_count * line_height + 2 * padding
            
            # Create top overlay for frame descriptions
            overlay = frame.copy()
            cv2.rectangle(overlay, 
                         (margin, margin),
                         (width - margin, margin + top_box_height),
                         (0, 0, 0),
                         -1)
            
            # Add frame description text
            y = margin + padding + line_height
            for line in top_lines:
                cv2.putText(overlay,
                           line,
                           (margin + padding, y),
                           font,
                           font_scale,
                           (255, 255, 255),
                           1,
                           cv2.LINE_AA)
                y += line_height
            
            # If we have speech transcription, add bottom overlay
            if bottom_lines:
                bottom_line_count = len(bottom_lines)
                bottom_box_height = bottom_line_count * line_height + 2 * padding
                bottom_margin = int(height * 0.15)  # Position overlay 15% from bottom
                
                # Add speech transcription text (centered with tight background)
                y = height - bottom_box_height - bottom_margin + padding + line_height
                for line in bottom_lines:
                    # Calculate text width for centering and background
                    (text_width, text_height), _ = cv2.getTextSize(line, font, transcript_font_scale, 1)
                    x = (width - text_width) // 2  # Center the text
                    
                    # Create tight background just for this line
                    bg_padding = 10  # Padding around text
                    bg_x1 = x - bg_padding
                    bg_x2 = x + text_width + bg_padding
                    bg_y1 = y - text_height - bg_padding
                    bg_y2 = y + bg_padding
                    
                    # Draw background rectangle
                    cv2.rectangle(overlay,
                                (bg_x1, bg_y1),
                                (bg_x2, bg_y2),
                                (0, 0, 0),
                                -1)
                    
                    # Draw text
                    cv2.putText(overlay,
                               line,
                               (x, y),
                               font,
                               transcript_font_scale,
                               (255, 255, 255),
                               1,
                               cv2.LINE_AA)
                    y += line_height * 1.5  # Increase spacing between lines
            
            # Blend overlay with original frame
            alpha = 0.7
            frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
            
            out.write(frame)
            frame_count += 1
            pbar.update(1)

    video.release()
    out.release()

    # Create summary intro clip
    print("\nCreating summary intro...")
    summary_path, summary_duration = create_summary_clip(summary, frame_width, frame_height, fps)
    
    # Create a file list for concatenation
    concat_file = os.path.join('outputs', 'concat_list.txt')
    with open(concat_file, 'w') as f:
        f.write(f"file '{os.path.abspath(summary_path)}'\n")
        f.write(f"file '{os.path.abspath(temp_output)}'\n")
    
    # Concatenate videos
    print("\nCombining summary and main video...")
    concat_output = os.path.join('outputs', f'concat_{os.path.basename(temp_output)}')
    concat_cmd = [
        'ffmpeg', '-y',
        '-f', 'concat',
        '-safe', '0',
        '-i', concat_file,
        '-c', 'copy',
        concat_output
    ]
    subprocess.run(concat_cmd, check=True)
    
    # Now merge with audio, adjusting for summary duration
    print("\nMerging with audio...")
    cmd = [
        'ffmpeg', '-y',
        '-i', concat_output,      # Input concatenated video
        '-i', video_path,         # Input original video (for audio)
        '-filter_complex', f'[1:a]adelay={int(summary_duration*1000)}|{int(summary_duration*1000)}[delayed_audio]',  # Delay audio
        '-c:v', 'copy',           # Copy video stream as is
        '-map', '0:v',            # Use video from first input
        '-map', '[delayed_audio]', # Use delayed audio
        final_output
    ]
    
    subprocess.run(cmd, check=True)
    
    # Clean up temporary files
    os.remove(temp_output)
    os.remove(summary_path)
    os.remove(concat_file)
    os.remove(concat_output)
    
    print(f"\nCaptioned video saved to: {final_output}")
    return final_output

def process_video_web(video_file, use_frame_selection=False, use_synthesis_captions=False):
    """Process video through web interface."""
    video_path = video_file.name
    print(f"Processing video: {video_path}")
    print(f"Using synthesis captions: {use_synthesis_captions}")  # Debug print

    start_time = time.time()

    # Get frame count
    frame_count = get_frame_count(video_path)
    if frame_count == 0:
        return "Error: Could not process video.", None, None

    # Transcribe the video
    transcript = transcribe_video(video_path)
    print(f"Transcript generated.")

    # Select frames based on the chosen method
    if use_frame_selection:
        print("Using frame selection algorithm")
        frame_numbers = process_video(video_path)
    else:
        print("Sampling every 50 frames")
        frame_numbers = list(range(0, frame_count, 50))

    # Describe frames
    descriptions = describe_frames(video_path, frame_numbers)
    print("Frame descriptions generated.")

    # Generate summary and captions
    print("Generating video summary...")
    synthesis_output = summarize_with_hosted_llm(transcript, descriptions)
    summary, synthesis_captions = parse_synthesis_output(synthesis_output)
    
    if use_synthesis_captions and (synthesis_captions is None or len(synthesis_captions) == 0):
        print("\nError: Failed to generate synthesis captions. Please try again or use frame descriptions.")
        return "Error: Failed to generate synthesis captions. Please try again or use frame descriptions.", None, None
    
    print(f"Summary generation complete. Generated {len(synthesis_captions) if synthesis_captions else 0} synthesis captions.")

    # Create captioned video with summary intro
    output_video_path = create_captioned_video(
        video_path, 
        descriptions, 
        summary,
        transcript,
        synthesis_captions,
        use_synthesis_captions  # Make sure this flag is passed through
    )

    total_run_time = time.time() - start_time
    print(f"Time taken: {total_run_time:.2f} seconds")

    # Save output to JSON file
    print("Saving output to JSON file...")
    save_output(video_path, frame_count, transcript, descriptions, summary, total_run_time)

    # Prepare frames with captions for gallery display
    gallery_images = []
    video = cv2.VideoCapture(video_path)
    
    # For gallery display, always use frame descriptions (not synthesis captions)
    # This helps debug the frame selection and description process
    for frame_number, timestamp, description in descriptions:
        video.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        success, frame = video.read()
        if success:
            # Convert BGR to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Get frame dimensions
            height, width = frame.shape[:2]
            
            # Calculate text parameters
            margin = 8  # pixels from edge
            padding = 10  # pixels inside box
            min_line_height = 20  # minimum pixels per line
            
            # Split description into lines if too long
            font = cv2.FONT_HERSHEY_SIMPLEX
            max_width = int(width * 0.9)  # Use up to 90% of frame width
            
            # Add timestamp line and description
            full_text = f"Frame {frame_number} [{timestamp:.2f}s]\n{description}"
            words = full_text.split()
            lines = []
            current_line = []
            
            # Calculate appropriate font scale
            font_scale = min(height * 0.03 / min_line_height, 0.7)  # Scale with frame height but cap it
            
            for word in words:
                test_line = ' '.join(current_line + [word])
                (text_width, _), _ = cv2.getTextSize(test_line, font, font_scale, 1)
                
                if text_width <= max_width - 2 * margin:
                    current_line.append(word)
                else:
                    if current_line:
                        lines.append(' '.join(current_line))
                    current_line = [word]
            
            if current_line:
                lines.append(' '.join(current_line))
            
            # Calculate box dimensions
            line_count = len(lines)
            line_height = max(min_line_height, int(height * 0.03))  # Scale with frame height
            box_height = line_count * line_height + 2 * padding
            
            # Create overlay for text background
            overlay = frame.copy()
            cv2.rectangle(overlay, 
                         (margin, margin),
                         (width - margin, margin + box_height),
                         (0, 0, 0),
                         -1)
            
            # Add text
            y = margin + padding + line_height
            for line in lines:
                cv2.putText(overlay,
                           line,
                           (margin + padding, y),
                           font,
                           font_scale,
                           (255, 255, 255),
                           1,
                           cv2.LINE_AA)
                y += line_height
            
            # Blend overlay with original frame
            alpha = 0.7
            frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
            
            # Convert back to PIL for gallery display
            frame_pil = Image.fromarray(frame)
            gallery_images.append(frame_pil)
    
    video.release()

    return (
        f"Video Summary:\n{summary}\n\nTime taken: {total_run_time:.2f} seconds", 
        gallery_images,
        output_video_path
    )

def main():
    parser = argparse.ArgumentParser(description="Process a video file.")
    parser.add_argument('video', type=str, nargs='?', help='Path to the video file')
    parser.add_argument('--save', action='store_true', help='Save output to a JSON file')
    parser.add_argument('--local', action='store_true', help='Use local Llama model for summarization')
    parser.add_argument('--frame-selection', action='store_true', help='Use CLIP-based key frame selection algorithm')
    parser.add_argument('--web', action='store_true', help='Start Gradio web interface')
    args = parser.parse_args()

    if args.web:
        # Create Gradio interface with gallery and video output
        iface = gr.Interface(
            fn=process_video_web,
            inputs=[
                gr.File(label="Upload Video"),
                gr.Checkbox(label="Use Frame Selection", value=False),
                gr.Checkbox(label="Use Synthesis Captions (fewer, more contextual)", value=False)
            ],
            outputs=[
                gr.Textbox(label="Summary"),
                gr.Gallery(label="Analyzed Frames"),
                gr.Video(label="Captioned Video")
            ],
            title="Video Summarizer",
            description="Upload a video to get a summary and view analyzed frames.",
            allow_flagging="never"
        )
        iface.launch()
    elif args.video:
        # Process video and prepare gallery output
        print(f"Processing video: {args.video}")

        start_time = time.time()

        # Get frame count
        frame_count = get_frame_count(args.video)
        if frame_count == 0:
            return

        # Process video using the web function for consistency
        summary, gallery_images, video_path = process_video_web(
            type('VideoFile', (), {'name': args.video})(),
            use_frame_selection=args.frame_selection,
            use_synthesis_captions=False  # Default to all captions in CLI mode
        )
        
        # Print the summary and processing time
        print("\nVideo Summary:")
        print(summary)

        total_run_time = time.time() - start_time
        print(f"\nTotal processing time: {total_run_time:.2f} seconds")
        print(f"\nCaptioned video saved to: {video_path}")
    else:
        print("Please provide a video file path or use the --web flag to start the web interface.")

if __name__ == "__main__":
    main()
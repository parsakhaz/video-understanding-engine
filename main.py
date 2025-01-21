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

    # Check if video has an audio stream using ffprobe
    has_audio = False
    try:
        cmd = [
            'ffprobe', 
            '-i', video_path,
            '-show_streams', 
            '-select_streams', 'a', 
            '-loglevel', 'error'
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        has_audio = len(result.stdout.strip()) > 0
    except subprocess.CalledProcessError:
        print("Error checking audio stream")
    if not has_audio:
        print("No audio track detected. Processing video with empty transcript.")
        # Return empty transcript with same structure
        return [{
            "start": 0,
            "end": 0,
            "text": ""
        }]
    # Suppress the FutureWarning from torch.load
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=FutureWarning)
        model = whisper.load_model("large")

    print(f"Audio model loaded")
    print(f"Transcribing video")

    try:
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
    except RuntimeError as e:
        print(f"Error transcribing video: {str(e)}")
        # Return empty transcript if transcription fails
        timestamped_transcript = [{
            "start": 0,
            "end": 0,
            "text": ""
        }]
    finally:
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

def get_synthesis_prompt(num_keyframes: int, is_long_video: bool = False) -> str:
    """Generate a dynamic synthesis prompt based on number of keyframes."""
    # For longer videos, we want fewer captions per minute to avoid overwhelming
    if is_long_video:
        num_captions = max(4, num_keyframes // 4)  # 1/4 of keyframes for long videos
    else:
        num_captions = min(100, max(12, num_keyframes // 1.5))  # Keep original ratio for short videos but cap at 100
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
   - Use the EARLIEST timestamp where a scene or action begins
   - Be 20-50 words long
   - Focus on key events and context
   - Integrate both visual and audio information when relevant
3. Timestamps should be reasonably spaced throughout the video
4. Focus on narrative flow and context, not just describing what's visible
5. IMPORTANT: When multiple frames describe the same scene or action, use the EARLIEST timestamp

Input sections:
<transcript>
Timestamped transcript of the video
</transcript>

<frame_descriptions>
Timestamped descriptions of key frames
</frame_descriptions>"""

def chunk_video_data(transcript: list, descriptions: list, chunk_duration: int = 60) -> list:
    """Split video data into chunks for processing longer videos."""
    chunks = []
    current_chunk_start = 0
    min_descriptions_per_chunk = 4  # Ensure at least 4 descriptions per chunk
    
    while current_chunk_start < descriptions[-1][1]:  # Until we reach the end timestamp
        current_chunk_end = current_chunk_start + chunk_duration
        
        # Get descriptions in this time window
        chunk_descriptions = [
            desc for desc in descriptions 
            if current_chunk_start <= desc[1] < current_chunk_end
        ]
        
        # If chunk has too few descriptions, extend the window
        while len(chunk_descriptions) < min_descriptions_per_chunk and current_chunk_end < descriptions[-1][1]:
            current_chunk_end += 15  # Extend by 15 seconds (reduced from 30)
            chunk_descriptions = [
                desc for desc in descriptions 
                if current_chunk_start <= desc[1] < current_chunk_end
            ]
        
        # Get transcript segments in this time window
        chunk_transcript = [
            seg for seg in transcript
            if (seg["start"] >= current_chunk_start and seg["start"] < current_chunk_end) or
               (seg["end"] > current_chunk_start and seg["end"] <= current_chunk_end)
        ]
        
        if chunk_descriptions:  # Only add chunk if it has descriptions
            chunks.append((chunk_transcript, chunk_descriptions))
        
        current_chunk_start = current_chunk_end
    
    return chunks
def summarize_with_hosted_llm(transcript, descriptions):
    # Load environment variables from .env file
    load_dotenv()

    # Check if this is a long video (over 2 minutes)
    video_duration = descriptions[-1][1]  # Last frame timestamp
    is_long_video = video_duration > 150  # 2.5 minutes threshold
    
    if is_long_video:
        print("\nLong video detected. Processing in chunks...")
        chunks = chunk_video_data(transcript, descriptions)
        all_summaries = []
        all_captions = []
        
        for i, (chunk_transcript, chunk_descriptions) in enumerate(chunks, 1):
            print(f"\nProcessing chunk {i}/{len(chunks)}...")
            
            # Generate synthesis prompt for this chunk
            synthesis_prompt = get_synthesis_prompt(len(chunk_descriptions), is_long_video=True)

            # Prepare the input for the model
            timestamped_transcript = "\n".join([
                f"[{segment['start']:.2f}s-{segment['end']:.2f}s] {segment['text']}"
                for segment in chunk_transcript
            ])
            frame_descriptions = "\n".join([
                f"[{timestamp:.2f}s] Frame {frame}: {desc}"
                for frame, timestamp, desc in chunk_descriptions
            ])
            
            user_content = f"<transcript>\n{timestamped_transcript}\n</transcript>\n\n<frame_descriptions>\n{frame_descriptions}\n</frame_descriptions>"
            
            # Get completion for this chunk
            completion = get_llm_completion(synthesis_prompt, user_content)
            chunk_summary, chunk_captions = parse_synthesis_output(completion)
            
            if chunk_summary and chunk_captions:
                all_summaries.append(chunk_summary)
                all_captions.extend(chunk_captions)

        # Make a final pass to synthesize all summaries into one coherent summary
        print("\nSynthesizing final summary...")
        final_summary_prompt = """You are tasked with synthesizing multiple summaries from different parts of a video into one coherent, comprehensive summary. 
        The summaries are presented in chronological order. Create a single summary that:
        1. Captures the main narrative arc of the entire video
        2. Highlights key themes and important elements
        3. Maintains a clear and concise flow
        4. Is roughly the same length as one of the input summaries
        
        Input Format:
        <chunk_summaries>
        [Chronological summaries of video segments]
        </chunk_summaries>"""
        
        chunk_summaries_content = "\n\n".join([f"Chunk {i+1}:\n{summary}" for i, summary in enumerate(all_summaries)])
        final_summary_content = f"<chunk_summaries>\n{chunk_summaries_content}\n</chunk_summaries>"
        
        final_summary = get_llm_completion(final_summary_prompt, final_summary_content)
        
        # Create final synthesis format with the new summary
        final_output = f"<summary>\n{final_summary}\n</summary>\n\n<captions>\n"
        for timestamp, text in sorted(all_captions, key=lambda x: x[0]):
            final_output += f"<caption>[{timestamp:.1f}] {text}</caption>\n"
        final_output += "</captions>"
        
        return final_output
    else:
        # Original logic for short videos
        synthesis_prompt = get_synthesis_prompt(len(descriptions), is_long_video=False)
        timestamped_transcript = "\n".join([f"[{segment['start']:.2f}s-{segment['end']:.2f}s] {segment['text']}" for segment in transcript])
        frame_descriptions = "\n".join([f"[{timestamp:.2f}s] Frame {frame}: {desc}" for frame, timestamp, desc in descriptions])
        user_content = f"<transcript>\n{timestamped_transcript}\n</transcript>\n\n<frame_descriptions>\n{frame_descriptions}\n</frame_descriptions>"
        return get_llm_completion(synthesis_prompt, user_content)

def get_llm_completion(prompt: str, content: str) -> str:
    """Helper function to get LLM completion with error handling."""
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable is not set. Please set it in your .env file or environment.")

    max_retries = 3
    retry_delay = 1
    
    for attempt in range(max_retries):
        try:
            client = OpenAI(api_key=api_key)
            completion = client.chat.completions.create(
                model="gpt-4o",  # Fixed model name
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": content}
                ],
                max_tokens=1024,
                temperature=0.3
            )
            return completion.choices[0].message.content.strip()
        except Exception as e:
            if attempt == max_retries - 1:  # Last attempt
                print(f"\nError getting LLM completion after {max_retries} attempts: {str(e)}")
                return f"<summary>\nError generating summary: API error after {max_retries} attempts.\n</summary>\n\n<captions>\n</captions>"
            print(f"\nRetrying LLM completion ({attempt + 1}/{max_retries})...")
            time.sleep(retry_delay * (attempt + 1))  # Exponential backoff

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
    
    # Check if video has an audio stream
    has_audio = False
    try:
        cmd = [
            'ffprobe', 
            '-i', video_path,
            '-show_streams', 
            '-select_streams', 'a', 
            '-loglevel', 'error'
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        has_audio = len(result.stdout.strip()) > 0
    except subprocess.CalledProcessError:
        print("Error checking audio stream")

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
            # Adjust timestamp to be 0.5s earlier, but not before 0
            adjusted_timestamp = max(0.0, timestamp - 0.5)
            
            # Skip if this caption is too close to the previous one
            if i > 0:
                prev_timestamp = max(0.0, synthesis_captions[i-1][0] - 0.5)  # Use adjusted previous timestamp
                if adjusted_timestamp - prev_timestamp < min_time_gap:
                    # Remove the previous caption if it exists in frame_info
                    if frame_info and frame_info[-1][1] == prev_timestamp:
                        frame_info.pop()
            
            # Find nearest frame number using adjusted timestamp
            frame_number = int(adjusted_timestamp * fps)
            frame_info.append((frame_number, adjusted_timestamp, text))
            
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

            # Find appropriate transcript segment and check if we're within its time period
            current_transcript_text = ""
            while (current_transcript_idx < len(transcript) - 1 and 
                   current_time >= transcript[current_transcript_idx + 1]["start"]):
                current_transcript_idx += 1
            
            # Only use transcript text if we're within its time period
            transcript_segment = transcript[current_transcript_idx]
            if (current_time >= transcript_segment["start"] and 
                current_time <= transcript_segment["end"]):
                transcript_text = transcript_segment["text"]
                segment_duration = transcript_segment["end"] - transcript_segment["start"]
                
                # Check for potential hallucination in transcript
                words_per_second = len(transcript_text.split()) / segment_duration if segment_duration > 0 else 0
                is_likely_hallucination = words_per_second < 0.5
                
                if not is_likely_hallucination:
                    current_transcript_text = transcript_text

            # Get current description
            frame_number, timestamp, description = frame_info[current_desc_idx]

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

            # If we have speech transcription, add bottom overlay
            if current_transcript_text:  # Changed from bottom_lines check to current_transcript_text
                # Normalize smart quotes and apostrophes in transcript
                current_transcript_text = current_transcript_text.replace('"', '"').replace('"', '"')
                current_transcript_text = current_transcript_text.replace("'", "'").replace("'", "'")
                # Split transcript into lines
                subtitle_words = current_transcript_text.split()
                current_line = []
                bottom_lines = []
                
                # Adjust font and padding based on resolution tiers
                if height >= 1440:  # 2K and above
                    base_increase = 0.6
                    bg_padding = 20
                    bottom_margin = int(height * 0.2)
                elif height >= 720:  # HD
                    base_increase = 0.4
                    bg_padding = 10
                    bottom_margin = int(height * 0.15)
                else:  # SD
                    base_increase = 0.1
                    bg_padding = 5
                    bottom_margin = int(height * 0.1)
                
                transcript_font_scale = min(height * 0.03 / min_line_height, 0.7) + base_increase
                
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
            # Adjust line height based on resolution
            if height >= 1440:  # 2K and above
                line_height = max(min_line_height, int(height * 0.04))
                padding = 15
            elif height >= 720:  # HD
                line_height = max(min_line_height, int(height * 0.03))
                padding = 10
            else:  # SD
                line_height = max(min_line_height, int(height * 0.02))
                padding = 8
                
            top_box_height = top_line_count * line_height + 2 * padding
            
            # Create overlay for text background
            overlay = frame.copy()
            cv2.rectangle(overlay, 
                         (int(margin), int(margin)),
                         (int(width - margin), int(margin + top_box_height)),
                         (0, 0, 0),
                         -1)
            
            # Blend background overlay with original frame
            alpha = 0.7
            frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
            
            # Draw text directly on frame (not on overlay) to keep it fully opaque
            y = margin + padding + line_height
            for line in top_lines:
                cv2.putText(frame,
                           line,
                           (int(margin + padding), int(y)),
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
                
                # Add speech transcription text (centered with tight background)
                y = height - bottom_box_height - bottom_margin + padding + line_height
                for line in bottom_lines:
                    # Calculate text width for centering and background
                    (text_width, text_height), _ = cv2.getTextSize(line, font, transcript_font_scale, 1)
                    x = (width - text_width) // 2  # Center the text
                    
                    # Create tight background just for this line
                    bg_x1 = int(x - bg_padding)
                    bg_x2 = int(x + text_width + bg_padding)
                    bg_y1 = int(y - text_height - bg_padding)
                    bg_y2 = int(y + bg_padding)
                    
                    # Draw background rectangle with transparency
                    overlay = frame.copy()
                    cv2.rectangle(overlay,
                                (bg_x1, bg_y1),
                                (bg_x2, bg_y2),
                                (0, 0, 0),
                                -1)
                    frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
                    
                    # Draw text directly on frame to keep it fully opaque
                    cv2.putText(frame,
                               line,
                               (int(x), int(y)),
                               font,
                               transcript_font_scale,
                               (255, 255, 255),
                               1,
                               cv2.LINE_AA)
                    y += line_height * 1.5  # Increase spacing between lines
            
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
    
    # Modify ffmpeg command based on audio presence
    if has_audio:
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
    else:
        # If no audio, just copy the video stream
        cmd = [
            'ffmpeg', '-y',
            '-i', concat_output,
            '-c:v', 'copy',
            final_output
        ]
    
    subprocess.run(cmd, check=True)
    
    # Clean up temporary files
    os.remove(temp_output)
    os.remove(summary_path)
    os.remove(concat_file)
    os.remove(concat_output)
    
    # Convert to web-compatible format (h264)
    web_output = final_output.replace('.mp4', '_web.mp4')
    convert_cmd = [
        'ffmpeg', '-y',
        '-i', final_output,
        '-c:v', 'libx264',
        '-preset', 'ultrafast',  # Use ultrafast preset for speed
        '-crf', '23',           # Reasonable quality
        '-c:a', 'aac',          # AAC audio for web compatibility
        '-movflags', '+faststart',  # Enable fast start for web playback
        web_output
    ]
    subprocess.run(convert_cmd, check=True)
    
    # Remove the non-web version
    os.remove(final_output)
    
    print(f"\nCaptioned video saved to: {web_output}")
    return web_output

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
                         (int(margin), int(margin)),
                         (int(width - margin), int(margin + box_height)),
                         (0, 0, 0),
                         -1)
            
            # Blend overlay with original frame
            alpha = 0.7
            frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
            
            # Add text directly on frame to keep it fully opaque
            y = margin + padding + line_height
            for line in lines:
                cv2.putText(frame,
                           line,
                           (int(margin + padding), int(y)),
                           font,
                           font_scale,
                           (255, 255, 255),
                           1,
                           cv2.LINE_AA)
                y += line_height
            
            # Convert back to PIL for gallery display
            frame_pil = Image.fromarray(frame)
            gallery_images.append(frame_pil)
    
    video.release()

    return (
        output_video_path,
        f"Video Summary:\n{summary}\n\nTime taken: {total_run_time:.2f} seconds", 
        gallery_images
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
                gr.Checkbox(label="Use Frame Selection", value=True, info="Recommended: Intelligently selects key frames"),
                gr.Checkbox(label="Use Synthesis Captions (fewer, more contextual)", value=True, info="Recommended: Creates a more pleasant viewing experience")
            ],
            outputs=[
                gr.Video(label="Captioned Video"),
                gr.Textbox(label="Summary"),
                gr.Gallery(label="Analyzed Frames")
            ],
            title="Video Summarizer",
            description="Upload a video to get a summary and view analyzed frames. Both Frame Selection and Synthesis Captions are recommended for the best results.",
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
        output_video_path, summary, gallery_images = process_video_web(
            type('VideoFile', (), {'name': args.video})(),
            use_frame_selection=args.frame_selection,
            use_synthesis_captions=False  # Default to all captions in CLI mode
        )
        
        # Print the summary and processing time
        print("\nVideo Summary:")
        print(summary)

        total_run_time = time.time() - start_time
        print(f"\nTotal processing time: {total_run_time:.2f} seconds")
        print(f"\nCaptioned video saved to: {output_video_path}")
    else:
        print("Please provide a video file path or use the --web flag to start the web interface.")

if __name__ == "__main__":
    main()
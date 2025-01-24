#!/usr/bin/env python3
import argparse
import cv2
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
from difflib import SequenceMatcher
import ast
from transformers import pipeline
import whisper
import traceback

def similar(a, b, threshold=0.90):
    """Return True if strings are similar above threshold."""
    return SequenceMatcher(None, a, b).ratio() > threshold

def clean_transcript(segments):
    """Clean up transcript by removing duplicates and background noise."""
    # Common background noise phrases to filter
    noise_phrases = [
        "ambient music",
        "music playing",
        "background music",
        "music",
        "captions",
        "[Music]",
        "♪",
        "♫"
    ]
    
    # Common boilerplate/disclaimer phrases to detect
    boilerplate_phrases = [
        "work of fiction",
        "any resemblance",
        "coincidental",
        "unintentional",
        "all rights reserved",
        "copyright",
        "trademark"
    ]
    
    # First pass: Remove segments that are just noise or boilerplate
    filtered_segments = []
    for segment in segments:
        text = segment["text"].strip().lower()
        
        # Skip if just noise
        if any(phrase in text.lower() for phrase in noise_phrases):
            continue
            
        # Skip if contains multiple boilerplate phrases
        boilerplate_count = sum(1 for phrase in boilerplate_phrases if phrase in text.lower())
        if boilerplate_count >= 2:
            continue
            
        filtered_segments.append(segment)
    
    # Second pass: Remove duplicates and near-duplicates
    cleaned_segments = []
    for i, current in enumerate(filtered_segments):
        # Skip if too similar to previous segment
        if i > 0 and similar(current["text"], filtered_segments[i-1]["text"]):
            continue
            
        # Skip if too similar to any of the next 3 segments (for repeated content)
        next_segments = filtered_segments[i+1:i+4]
        if any(similar(current["text"], next_seg["text"]) for next_seg in next_segments):
            continue
            
        cleaned_segments.append(current)
    
    return cleaned_segments if cleaned_segments else [{
        "start": 0,
        "end": 0,
        "text": ""
    }]

def get_video_duration(video_path):
    """Get video duration using ffprobe."""
    try:
        cmd = [
            'ffprobe', 
            '-v', 'error',
            '-show_entries', 'format=duration',
            '-of', 'default=noprint_wrappers=1:nokey=1',
            video_path
        ]
        duration = float(subprocess.check_output(cmd).decode().strip())
        print(f"Video duration: {duration:.2f} seconds")
        return duration
    except Exception as e:
        print(f"Warning: Could not get video duration: {str(e)}")
        return None

def extract_audio(video_path):
    """Extract audio from video to MP3."""
    audio_path = os.path.join('outputs', f'temp_audio_{int(time.time())}.mp3')
    os.makedirs('outputs', exist_ok=True)
    
    try:
        cmd = [
            'ffmpeg', '-y',
            '-i', video_path,
            '-vn',  # No video
            audio_path
        ]
        subprocess.run(cmd, check=True)
        return audio_path
    except Exception as e:
        print(f"Error extracting audio: {str(e)}")
        return None

def process_transcript(segments):
    """Convert Whisper segments to our transcript format."""
    processed_transcript = []
    
    for segment in segments:
        start = segment.get("start")
        end = segment.get("end")
        text = segment.get("text", "").strip()
        
        if start is None or end is None or not text:
            continue
            
        processed_transcript.append({
            "start": start,
            "end": end,
            "text": text
        })
    
    return processed_transcript

# Function to transcribe audio from a video file using Whisper model
def transcribe_video(video_path):
    print(f"Loading audio model")

    # Get video duration using ffprobe
    duration = get_video_duration(video_path)
    if duration is None:
        return [{
            "start": 0,
            "end": 0,
            "text": ""
        }]

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
        return [{
            "start": 0,
            "end": 0,
            "text": ""
        }]

    # Extract audio from video
    audio_path = extract_audio(video_path)
    if not audio_path:
        print("Failed to extract audio")
        return [{
            "start": 0,
            "end": 0,
            "text": ""
        }]
    
    try:
        # Load Whisper model
        print("Loading Whisper model...")
        model = whisper.load_model("large")
        
        print("Transcribing audio...")
        
        # Run transcription
        result = model.transcribe(audio_path)
        
        # Save raw output for debugging
        timestamp = int(time.time())
        raw_output_path = os.path.join('outputs', f'whisper_raw_{timestamp}.txt')
        
        # # Save raw output with detailed information
        # with open(raw_output_path, 'w', encoding='utf-8') as f:
        #     f.write("=== RAW WHISPER OUTPUT ===\n\n")
        #     f.write(f"Video path: {video_path}\n")
        #     f.write(f"Video duration: {duration:.2f}s\n")
        #     f.write(f"Timestamp: {timestamp}\n")
        #     f.write("\n=== FULL RESULT ===\n")
        #     f.write(json.dumps(result, indent=2, ensure_ascii=False))
        
        # print(f"\nRaw output saved to: {raw_output_path}")
        
        # Process segments
        processed_transcript = process_transcript(result["segments"])
        
        # Clean up the transcript
        return clean_transcript(processed_transcript)
            
    except Exception as e:
        print(f"Error during transcription: {str(e)}")
        return [{
            "start": 0,
            "end": 0,
            "text": ""
        }]
    finally:
        # Clean up temporary files
        try:
            if os.path.exists(audio_path):
                os.remove(audio_path)
        except Exception as e:
            print(f"Error cleaning up temporary files: {str(e)}")

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

def save_output(video_path, frame_count, transcript, descriptions, summary, total_run_time, synthesis_output=None, synthesis_captions=None):
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
        "total_run_time": total_run_time,
        "synthesis": {
            "raw_output": synthesis_output,
            "captions": [
                {
                    "timestamp": timestamp,
                    "text": text
                } for timestamp, text in (synthesis_captions or [])
            ]
        }
    }

    with open(filename, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"Output saved to {filename}")

def get_synthesis_prompt(num_keyframes: int, video_duration: float, metadata_context: str = "") -> str:
    """Generate a dynamic synthesis prompt based on number of keyframes, video duration, and metadata."""
    print("\n=== Caption Calculation Logic ===")
    print(f"Input: {num_keyframes} keyframes, {video_duration:.1f}s duration")
    print(f"Has metadata context: {bool(metadata_context)}")
    
    # Calculate number of captions (existing logic)
    if video_duration > 150:  # > 150s
        print("Long video detected (>150s)")
        num_captions = max(15, num_keyframes // 2)
        print(f"Long video calculation: max(12, {num_keyframes} // 2) = {num_captions}")
    elif video_duration > 90:  # 90-150s
        print("Medium-long video detected (90-150s)")
        base_captions = int(video_duration / 6)
        num_captions = min(int(base_captions * 1.25), num_keyframes // 2)
        num_captions = max(9, num_captions)
        print(f"Medium-long video calculation:")
        print(f"- Base captions (1/6s): {base_captions}")
        print(f"- After 25% increase: {int(base_captions * 1.25)}")
        print(f"- Final (min 15, max {num_keyframes//2}): {num_captions}")
    else:
        if video_duration < 30:  # Short videos < 30s
            print("Short video detected (<30s)")
            target_captions = int(video_duration / 3.5)
            num_captions = min(100, max(4, target_captions))
            print(f"Short video calculation:")
            print(f"- Target: {video_duration:.1f}s / 3.5s = {target_captions}")
            print(f"- Final: min(100, max(4, {target_captions})) = {num_captions}")
        else:  # Medium videos 30-90s
            print("Medium video detected (30-90s)")
            caption_interval = 5.0 + (video_duration - 30) * (7.0 - 5.0) / (90 - 30)
            target_captions = int(video_duration / caption_interval)
            max_captions = min(int(video_duration / 4), num_keyframes // 3)
            num_captions = min(target_captions, max_captions)
            num_captions = max(8, num_captions)
            print(f"Medium video calculation:")
            print(f"- Caption interval: {caption_interval:.1f}s")
            print(f"- Initial target: {video_duration:.1f}s / {caption_interval:.1f}s = {target_captions}")
            print(f"- Max captions: min({video_duration:.1f}/4, {num_keyframes}/3) = {max_captions}")
            print(f"- After max cap: min({target_captions}, {max_captions}) = {min(target_captions, max_captions)}")
            print(f"- Final (after min 8): {num_captions}")
    
    print(f"Final decision: {num_captions} captions for {video_duration:.1f}s video ({num_captions/video_duration:.1f} captions/second)")
    print("================================\n")

    # Build the prompt with optional metadata context
    metadata_section = ""
    if metadata_context:
        metadata_section = f"""You are tasked with summarizing and captioning a video based on its transcript and frame descriptions, with the following important context about the video's origin and purpose:

{metadata_context}

This metadata should inform your understanding of:
- The video's intended purpose and audience
- Appropriate style and tone for descriptions
- Professional vs. amateur context
- Genre-specific considerations
"""
    else:
        metadata_section = "You are tasked with summarizing and captioning a video based on its transcript and frame descriptions."
    
    return f"""{metadata_section}

You MUST follow the exact format specified below.

Output Format:
1. A summary section wrapped in <summary></summary> tags. The summary MUST start with "This video presents" to maintain consistent style. 
  - This video summary never refers to frames as "stills" or "images" - they are frames from a continuous sequence.
2. A captions section wrapped in <captions></captions> tags
3. Exactly {num_captions} individual captions, each wrapped in <caption></caption> tags
4. Each caption MUST start with a timestamp in square brackets, e.g. [2.5]

Example format:
<summary>
This video presents a [the high level action or narrative that takes place over the frames] in a [location], where [action/assumptions]...
</summary>

<captions>
<caption>[0.0] A teacher stands at the front of the classroom...</caption>
<caption>[3.2] Students raise their hands to answer...</caption>
...additional captions...
</captions>

IMPORTANT NOTES ON HANDLING TRANSCRIPT AND DESCRIPTIONS:
1. The transcript provides helpful context but should not be directly quoted in captions
2. Focus captions on describing visual elements and actions:
   - What is happening in the scene
   - Physical movements and gestures
   - Changes in environment or setting
   - Observable facial expressions and body language

Frame Description Guidelines:
1. Frame descriptions may vary in detail and consistency between frames
2. Be cautious with frame descriptions that make assumptions about:
   - Who is speaking or performing actions
   - Relationships between people
   - Roles or identities of individuals
   - Continuity between frames
3. When synthesizing descriptions:
   - Default to neutral terms like "a person", "someone", or "the speaker"
   - Only attribute actions/speech if explicitly clear from transcript
   - Focus on clearly visible elements rather than interpretations
   - Use passive voice when action source is ambiguous
   - Cross-reference transcript before assigning names/identities
4. If frame descriptions conflict or make assumptions:
   - Favor the most conservative interpretation
   - Omit speculative details
   - Focus on concrete visual elements
   - Maintain consistent neutral language
5. Remember that gaps may exist between frames, so avoid assuming continuity.

Requirements for Summary:
1. IMPORTANT: Always start with "This video presents" to maintain consistent style
2. Provide a clear overview of this video's main content and purpose
3. Include key events and settings without attributing actions to specific people
4. Integrate both visual and audio information while avoiding assumptions
5. Keep it concise (3-5 sentences)
6. Use transcript information to establish this video's context and purpose
7. Cross-reference transcript for useful context and understanding of emotions associated with this video
8. Focus on clearly visible elements rather than interpretations

Requirements for Captions:
1. Generate exactly {num_captions} captions
2. Each caption MUST:
   - Start with a timestamp in [X.X] format
   - Use the EARLIEST timestamp where a scene or action begins
   - Be 20-50 words long
   - Focus on observable events and context
   - Avoid attributing speech or actions unless explicitly clear
3. Timestamps should be reasonably spaced throughout this video
4. Focus on what is definitively shown or heard, not assumptions
5. IMPORTANT: When multiple frames describe the same scene or action, use the EARLIEST timestamp
6. Default to neutral terms like "a person" or "someone" when identities are unclear
7. Use passive voice when action source is ambiguous
8. Describe only what is visually observable and keep descriptions objective

Input sections:
<transcript>
Timestamped transcript of this video
</transcript>

<frame_descriptions>
Timestamped descriptions of key frames
</frame_descriptions>"""

def chunk_video_data(transcript: list, descriptions: list, chunk_duration: int = 60) -> list:
    """Split video data into chunks for processing longer videos."""
    print("\n=== Video Chunking Logic ===")
    print(f"Total video length: {descriptions[-1][1]:.1f}s")
    print(f"Base chunk duration: {chunk_duration}s")
    print(f"Total descriptions: {len(descriptions)}")
    print(f"Total transcript segments: {len(transcript)}")
    
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
        
        print(f"\nChunk {len(chunks) + 1}:")
        print(f"- Time window: {current_chunk_start:.1f}s - {current_chunk_end:.1f}s")
        print(f"- Initial descriptions: {len(chunk_descriptions)}")
        
        # If chunk has too few descriptions, extend the window
        original_end = current_chunk_end
        while len(chunk_descriptions) < min_descriptions_per_chunk and current_chunk_end < descriptions[-1][1]:
            current_chunk_end += 15  # Extend by 15 seconds
            chunk_descriptions = [
                desc for desc in descriptions 
                if current_chunk_start <= desc[1] < current_chunk_end
            ]
            if current_chunk_end > original_end:
                print(f"- Extended window to {current_chunk_end:.1f}s to get more descriptions: now have {len(chunk_descriptions)}")
        
        # Get transcript segments in this time window
        chunk_transcript = [
            seg for seg in transcript
            if (seg["start"] >= current_chunk_start and seg["start"] < current_chunk_end) or
               (seg["end"] > current_chunk_start and seg["end"] <= current_chunk_end)
        ]
        
        if chunk_descriptions:  # Only add chunk if it has descriptions
            chunks.append((chunk_transcript, chunk_descriptions))
            print(f"- Final chunk size: {current_chunk_end - current_chunk_start:.1f}s")
            print(f"- Final descriptions: {len(chunk_descriptions)}")
            print(f"- Transcript segments: {len(chunk_transcript)}")
        else:
            print("- Skipping chunk: no descriptions found")
        
        current_chunk_start = current_chunk_end
    
    print(f"\nFinal chunks: {len(chunks)}")
    print("=========================\n")
    return chunks

def summarize_with_hosted_llm(transcript, descriptions, video_duration: float, use_local_llm=False, video_path: str = None):
    # Load environment variables from .env file
    load_dotenv()

    # Extract metadata at start
    metadata_context = ""
    if video_path:
        max_metadata_retries = 3
        for attempt in range(max_metadata_retries):
            try:
                print(f"\nExtracting video metadata (attempt {attempt + 1}/{max_metadata_retries})...")
                cmd = [
                    'ffprobe',
                    '-v', 'quiet',
                    '-print_format', 'json',
                    '-show_format',
                    '-show_streams',
                    video_path
                ]
                metadata_json = subprocess.check_output(cmd).decode('utf-8')
                metadata = json.loads(metadata_json)
                
                # Extract relevant metadata fields
                format_metadata = metadata.get('format', {}).get('tags', {})
                title = format_metadata.get('title', '')
                artist = format_metadata.get('artist', '')
                duration = float(metadata.get('format', {}).get('duration', 0))
                
                print(f"Found metadata - Title: {title}, Artist: {artist}, Duration: {duration:.2f}s")
                
                # Format metadata context
                metadata_context = "Video Metadata:\n"
                if title:
                    metadata_context += f"- Title: {title}\n"
                if artist:
                    metadata_context += f"- Creator: {artist}\n"
                metadata_context += f"- Duration: {duration:.2f} seconds\n"
                
                # Add any other relevant metadata fields
                for key, value in format_metadata.items():
                    if key not in ['title', 'artist'] and value:
                        metadata_context += f"- {key}: {value}\n"
                break
            except Exception as e:
                print(f"Warning: Metadata extraction attempt {attempt + 1} failed: {str(e)}")
                if attempt < max_metadata_retries - 1:
                    time.sleep(2 * (attempt + 1))
                else:
                    print("Failed to extract metadata after all retries")
                    metadata_context = ""

    # Check if this is a long video (over 2 minutes)
    is_long_video = video_duration > 150  # 2.5 minutes threshold
    
    initial_synthesis = None
    initial_captions = None
    
    if is_long_video:
        print("\nLong video detected. Processing in chunks...")
        chunks = chunk_video_data(transcript, descriptions)
        all_summaries = []
        all_captions = []
        
        for i, (chunk_transcript, chunk_descriptions) in enumerate(chunks, 1):
            print(f"\nProcessing chunk {i}/{len(chunks)}...")
            
            # Generate synthesis prompt for this chunk with metadata context
            synthesis_prompt = get_synthesis_prompt(
                len(chunk_descriptions), 
                video_duration,
                metadata_context  # Pass metadata to each chunk
            )

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
            
            # Get completion for this chunk with retries
            max_retries = 5  # Increased from 3 to 5
            for attempt in range(max_retries):
                print(f"\nAttempt {attempt + 1}/{max_retries} to get completion...")
                completion = get_llm_completion(synthesis_prompt, user_content, use_local_llm=use_local_llm)
                
                if completion is None:
                    print(f"Got None completion on attempt {attempt + 1}")
                    if attempt < max_retries - 1:
                        print("Retrying...")
                        time.sleep(2 * (attempt + 1))
                        continue
                    else:
                        print("Failed all attempts to get completion")
                        return None, None
                
                print("\nGot completion, attempting to parse...")
                chunk_summary, chunk_captions = parse_synthesis_output(completion)
                
                if chunk_summary and chunk_captions:
                    all_summaries.append(chunk_summary)
                    all_captions.extend(chunk_captions)
                    break
                else:
                    if attempt < max_retries - 1:
                        print(f"\nRetrying chunk {i} synthesis (attempt {attempt + 2}/{max_retries})...")
                        time.sleep(2 * (attempt + 1))  # Exponential backoff
                    else:
                        print(f"\nFailed to generate synthesis for chunk {i} after {max_retries} attempts.")
                        return None, None

        # Make a final pass to synthesize all summaries into one coherent summary
        print("\nSynthesizing final summary...")
        final_summary_prompt = """You are tasked with synthesizing multiple summaries and transcript segments from different parts of a single video into one coherent, comprehensive summary of the entire video.
        The frame and scene summaries, along with transcript segments, are presented in chronological order. This video does not contain photos, any mentions of photos are incorrect and hallucinations.
        
        Output a single overall summary of all the frames, scenes, and dialogue that:
        1. Always starts with "This video presents" to maintain consistent style.
        2. Captures the main narrative arc, inferring the most likely high level overview and theme with the given information, along with emotions and feelings.
        3. Maintains a clear and concise flow
        4. Is roughly the same length as one of the input summaries
        5. Ends immediately after the summary (no extra text)
        6. IMPORTANT: Never refers to frames as "stills" or "images" - they are frames from a continuous sequence
        7. You must focus on observable events and context without making assumptions about who is doing what
        8. You must use neutral language and avoids attributing actions unless explicitly clear
        
        Input Format:
        <chunk_summaries>
        [Chronological summaries of segments]
        </chunk_summaries>

        <transcript>
        [Chronological transcript segments]
        </transcript>"""
        
        chunk_summaries_content = "\n\n".join([f"Chunk {i+1}:\n{summary}" for i, summary in enumerate(all_summaries)])
        # Format transcript for final summary
        timestamped_transcript = "\n".join([
            f"[{segment['start']:.2f}s-{segment['end']:.2f}s] {segment['text']}"
            for segment in transcript
        ])
        
        final_summary_content = f"<chunk_summaries>\n{chunk_summaries_content}\n</chunk_summaries>\n\n<transcript>\n{timestamped_transcript}\n</transcript>"

        # Get final summary with retries
        max_retries = 5
        final_summary = None
        for attempt in range(max_retries):
            print(f"\nAttempt {attempt + 1}/{max_retries} to get final summary...")
            final_summary = get_llm_completion(final_summary_prompt, final_summary_content, use_local_llm=use_local_llm)
            if final_summary:
                break
            elif attempt < max_retries - 1:
                print(f"\nRetrying final summary generation (attempt {attempt + 2}/{max_retries})...")
                time.sleep(2 * (attempt + 1))
            else:
                print("\nFailed to generate final summary after multiple attempts.")
                return None, None

        if not final_summary:
            return None, None
            
        # For long videos, validate and retry the final recontextualized synthesis
        if is_long_video:
            max_recontextualize_retries = 3
            for attempt in range(max_recontextualize_retries):
                print(f"\nValidating final synthesis (attempt {attempt + 1}/{max_recontextualize_retries})...")
                
                # Recontextualize the final summary with metadata
                if metadata_context:
                    final_summary = recontextualize_summary(final_summary, metadata_context, use_local_llm)
                
                # Create final synthesis format with the new summary
                initial_synthesis = f"<summary>\n{final_summary}\n</summary>\n\n<captions>\n"
                for timestamp, text in sorted(all_captions, key=lambda x: x[0]):
                    initial_synthesis += f"<caption>[{timestamp:.1f}] {text}</caption>\n"
                initial_synthesis += "</captions>"

                # Parse and validate the final output
                initial_summary, initial_captions = parse_synthesis_output(initial_synthesis)
                if initial_summary and initial_captions:
                    break
                
                if attempt < max_recontextualize_retries - 1:
                    print(f"\nRetrying final synthesis validation (attempt {attempt + 2}/{max_recontextualize_retries})...")
                    time.sleep(2 * (attempt + 1))
                else:
                    print("\nFailed to validate final synthesis after all retries")
                    return None, None
        
        # Recontextualize the final summary with metadata
        if metadata_context:
            final_summary = recontextualize_summary(final_summary, metadata_context, use_local_llm)
        
        # Create final synthesis format with the new summary
        initial_synthesis = f"<summary>\n{final_summary}\n</summary>\n\n<captions>\n"
        for timestamp, text in sorted(all_captions, key=lambda x: x[0]):
            initial_synthesis += f"<caption>[{timestamp:.1f}] {text}</caption>\n"
        initial_synthesis += "</captions>"
    else:
        # Original logic for short videos with retries
        synthesis_prompt = get_synthesis_prompt(len(descriptions), video_duration)
        timestamped_transcript = "\n".join([f"[{segment['start']:.2f}s-{segment['end']:.2f}s] {segment['text']}" for segment in transcript])
        frame_descriptions = "\n".join([f"[{timestamp:.2f}s] Frame {frame}: {desc}" for frame, timestamp, desc in descriptions])
        user_content = f"<transcript>\n{timestamped_transcript}\n</transcript>\n\n<frame_descriptions>\n{frame_descriptions}\n</frame_descriptions>"
        
        max_retries = 5  # Increased from 3 to 5
        for attempt in range(max_retries):
            print(f"\nAttempt {attempt + 1}/{max_retries} to get completion...")
            completion = get_llm_completion(synthesis_prompt, user_content, use_local_llm=use_local_llm)
            
            if completion is None:
                print(f"Got None completion on attempt {attempt + 1}")
                if attempt < max_retries - 1:
                    print("Retrying...")
                    time.sleep(2 * (attempt + 1))
                    continue
                else:
                    print("Failed all attempts to get completion")
                    print("Falling back to frame descriptions.")
                    return None, None
            
            print("\nGot completion, attempting to parse...")
            initial_summary, initial_captions = parse_synthesis_output(completion)
            
            if initial_summary and initial_captions:
                initial_synthesis = completion
                break
            else:
                if attempt < max_retries - 1:
                    print(f"\nRetrying synthesis generation (attempt {attempt + 2}/{max_retries})...")
                    time.sleep(2 * (attempt + 1))  # Exponential backoff
                else:
                    print(f"\nFailed to generate synthesis after {max_retries} attempts.")
                    print("Falling back to frame descriptions.")
                    return None, None
    
    # At this point we have initial_synthesis and initial_captions
    if initial_synthesis and initial_captions:
        # Recontextualize the final summary with metadata
        if metadata_context:
            initial_summary = recontextualize_summary(initial_summary, metadata_context, use_local_llm)
            # Update the synthesis with the recontextualized summary
            initial_synthesis = f"<summary>\n{initial_summary}\n</summary>\n\n<captions>\n"
            for timestamp, text in initial_captions:
                initial_synthesis += f"<caption>[{timestamp:.1f}] {text}</caption>\n"
            initial_synthesis += "</captions>"
        return initial_synthesis, initial_captions

    return None, None

def get_llm_completion(prompt: str, content: str, use_local_llm: bool = False) -> str:
    """Helper function to get LLM completion with error handling."""
    print("\n=== Starting LLM Completion ===")
    print(f"Using local LLM: {use_local_llm}")
    
    try:
        if use_local_llm:
            pipeline = None
            try:
                print("\nInitializing local Llama model...")
                pipeline = transformers.pipeline(
                    "text-generation",
                    model="meta-llama/Meta-Llama-3.1-8B-Instruct",
                    model_kwargs={"torch_dtype": torch.bfloat16},
                    device_map="auto",
                )
                
                # Format messages like in the docs
                messages = [
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": content},
                ]
                
                print("\nGenerating response with local model...")
                outputs = pipeline(
                    messages,
                    max_new_tokens=8000,
                    temperature=0.3,
                )
                raw_response = outputs[0]["generated_text"]
                
                print("\nRaw local LLM response:")
                print("=" * 80)
                print(raw_response)
                print("=" * 80)
                
                # Extract just the assistant's response
                try:
                    # Convert the list object to a string first
                    raw_response_str = str(raw_response)
                    
                    # Try different quote styles and formats
                    patterns = [
                        "'role': 'assistant', 'content': '",
                        '"role": "assistant", "content": "',
                        "'role': 'assistant', 'content': \"",
                        "role': 'assistant', 'content': '",
                        "role\": \"assistant\", \"content\": \""
                    ]
                    
                    content = None
                    for pattern in patterns:
                        print(f"\nTrying pattern: {pattern}")
                        assistant_start = raw_response_str.find(pattern)
                        if assistant_start != -1:
                            content_start = assistant_start + len(pattern)
                            # Try different ending patterns
                            for end_pattern in ["'}", '"}', "\"}", "'}]", '"}]']:
                                content_end = raw_response_str.find(end_pattern, content_start)
                                if content_end != -1:
                                    content = raw_response_str[content_start:content_end]
                                    # Unescape the content
                                    content = content.encode().decode('unicode_escape')
                                    print("\nExtracted and unescaped content:")
                                    print("-" * 80)
                                    print(content)
                                    print("-" * 80)
                                    return content
                    
                    # If we get here, try to find XML tags directly
                    if "<summary>" in raw_response_str and "</summary>" in raw_response_str:
                        print("\nFalling back to direct XML parsing...")
                        return raw_response_str
                    
                    print("\nNo valid assistant response found in messages")
                    print("Raw response:")
                    print(raw_response_str)
                    return None
                
                except Exception as e:
                    print(f"\nError extracting content from local LLM response: {str(e)}")
                    print("Raw response that failed:", raw_response)
                    return None
                    
            except Exception as e:
                print(f"\nError with local LLM: {str(e)}")
                print("Falling back to OpenAI API...")
                return None
                    
            finally:
                # Clean up resources
                if pipeline is not None:
                    del pipeline
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        print("\nCleared CUDA cache and released model resources")
        
        # OpenAI API fallback
        print("\nAttempting OpenAI API...")
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            print("\nNo OpenAI API key found in environment")
            return None

        max_retries = 5
        retry_delay = 1
        
        for attempt in range(max_retries):
            try:
                client = OpenAI(api_key=api_key)
                completion = client.chat.completions.create(
                    model="gpt-4o",
                    messages=[
                        {"role": "system", "content": prompt},
                        {"role": "user", "content": content}
                    ],
                    temperature=0.3
                )
                response = completion.choices[0].message.content.strip()
                print("\nRaw OpenAI response:")
                print("-" * 80)
                print(response)
                print("-" * 80)
                return response
            except Exception as e:
                if attempt == max_retries - 1:  # Last attempt
                    print(f"\nError getting OpenAI completion after {max_retries} attempts: {str(e)}")
                    return None
                print(f"\nRetrying OpenAI completion ({attempt + 1}/{max_retries})...")
                time.sleep(retry_delay * (attempt + 1))
    except Exception as e:
        print(f"\nUnexpected error in get_llm_completion: {str(e)}")
        print("Stack trace:", traceback.format_exc())
        return None

def parse_synthesis_output(output: str) -> tuple:
    """Parse the synthesis output to extract summary and captions."""
    try:
        print("\nParsing synthesis output...")
        
        # Extract summary (required)
        summary_match = re.search(r'<summary>(.*?)</summary>', output, re.DOTALL)
        if not summary_match:
            print("\nFailed to find <summary> tags in output")
            raise ValueError("No summary found in synthesis output")
        summary = summary_match.group(1).strip()
        print("\nFound summary:", summary[:100] + "..." if len(summary) > 100 else summary)
        
        # Extract captions (required)
        captions_match = re.search(r'<captions>(.*?)</captions>', output, re.DOTALL)
        if not captions_match:
            print("\nFailed to find <captions> tags in output")
            raise ValueError("No captions found in synthesis output")
        
        captions_text = captions_match.group(1).strip()
        # Extract all caption tags from within the captions section
        caption_matches = re.findall(r'<caption>\[([\d.]+)s?\](.*?)</caption>', captions_text, re.DOTALL)
        
        caption_list = []
        for timestamp_str, text in caption_matches:
            try:
                timestamp = float(timestamp_str)
                text = text.strip()
                if text:  # Only add if we have actual text
                    caption_list.append((timestamp, text))
            except ValueError as e:
                print(f"Warning: Could not parse timestamp {timestamp_str}: {e}")
                continue
        
        if not caption_list:
            print("\nNo valid captions found in parsed output")
            raise ValueError("No valid captions found in synthesis output")
            
        print(f"\nSuccessfully parsed {len(caption_list)} captions")
        print("First caption:", caption_list[0] if caption_list else "None")
        print("Last caption:", caption_list[-1] if caption_list else "None")
        
        return summary, caption_list
    except (ValueError, AttributeError) as e:
        print(f"\nError parsing synthesis output: {str(e)}")
        print("\nRaw output:")
        print("-" * 80)
        print(output)
        print("-" * 80)
        return None, None

def create_summary_clip(summary: str, width: int, height: int, fps: int, debug: bool = False, metadata: dict = None) -> str:
    """Create a video clip with centered summary text."""
    # Calculate duration based on reading speed (400 words/min) * 0% buffer
    word_count = len(summary.split())
    base_duration = (word_count / 400) * 60  # Convert to seconds
    duration = base_duration * 1
    total_frames = int(duration * fps)
    
    # Create output path
    os.makedirs('outputs', exist_ok=True)
    temp_summary_path = os.path.join('outputs', f'temp_summary_{int(time.time())}.mp4')
    
    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(temp_summary_path, fourcc, fps, (width, height))
    
    # Create black frame
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Add attribution text - now at bottom middle with adjusted font size
    attribution = "Local Video Understanding Engine - powered by Moondream 2B, CLIP, LLama 3.1 8b Instruct, and Whisper Large"
    if debug:
        attribution = "DEBUG " + attribution
    font = cv2.FONT_HERSHEY_SIMPLEX
    # Increased font scale for attribution
    attr_font_scale = min(height * 0.02 / 20, 0.6)  # Increased from 0.015 to 0.02
    
    # Calculate text size and position for centering
    (text_width, text_height), _ = cv2.getTextSize(attribution, font, attr_font_scale, 1)
    attr_x = (width - text_width) // 2  # Center horizontally
    attr_y = height - 20  # 20 pixels from bottom
    
    # Draw attribution text
    if debug:
        # Draw DEBUG in red with monospace font first
        debug_font = cv2.FONT_HERSHEY_COMPLEX_SMALL  # Using complex small for monospace-like appearance
        debug_text = "DEBUG"
        (debug_width, _), _ = cv2.getTextSize(debug_text, debug_font, attr_font_scale, 1)
        # Draw DEBUG text in red
        cv2.putText(frame, debug_text, (attr_x, attr_y), debug_font, attr_font_scale, (0, 0, 255), 1, cv2.LINE_AA)
        # Draw the rest of the attribution after DEBUG
        rest_text = " " + attribution[6:]  # Skip the "DEBUG " we added earlier
        cv2.putText(frame, rest_text, (attr_x + debug_width, attr_y), font, attr_font_scale, (255, 255, 255), 1, cv2.LINE_AA)
    else:
        cv2.putText(frame, attribution, (attr_x, attr_y), font, attr_font_scale, (255, 255, 255), 1, cv2.LINE_AA)
    
    # Add debug metadata if enabled
    if debug and metadata:
        # Increased font scale for debug metadata
        debug_font_scale = min(height * 0.018 / 20, 0.5)  # Increased from 0.015 to 0.018
        y_pos = 30  # Start position from top
        x_pos = 20  # Left margin
        
        # Format and display metadata
        for key, value in metadata.items():
            if isinstance(value, float):
                debug_text = f"{key}: {value:.2f}"
            else:
                debug_text = f"{key}: {value}"
            cv2.putText(frame, debug_text, (x_pos, y_pos), font, debug_font_scale, (200, 200, 200), 1, cv2.LINE_AA)
            y_pos += 25  # Space between lines
    
    # Prepare text
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
        
        # Add attribution text to each frame - at bottom middle
        (text_width, text_height), _ = cv2.getTextSize(attribution, font, attr_font_scale, 1)
        attr_x = (width - text_width) // 2  # Center horizontally
        attr_y = height - 20  # 20 pixels from bottom
        
        # Draw attribution text
        if debug:
            # Draw DEBUG in red with monospace font first
            debug_font = cv2.FONT_HERSHEY_COMPLEX_SMALL  # Using complex small for monospace-like appearance
            debug_text = "DEBUG"
            (debug_width, _), _ = cv2.getTextSize(debug_text, debug_font, attr_font_scale, 1)
            # Draw DEBUG text in red
            cv2.putText(frame, debug_text, (attr_x, attr_y), debug_font, attr_font_scale, (0, 0, 255), 1, cv2.LINE_AA)
            # Draw the rest of the attribution after DEBUG
            rest_text = " " + attribution[6:]  # Skip the "DEBUG " we added earlier
            cv2.putText(frame, rest_text, (attr_x + debug_width, attr_y), font, attr_font_scale, (255, 255, 255), 1, cv2.LINE_AA)
        else:
            cv2.putText(frame, attribution, (attr_x, attr_y), font, attr_font_scale, (255, 255, 255), 1, cv2.LINE_AA)
        
        # Add debug metadata if enabled
        if debug and metadata:
            y_pos = 30  # Start position from top
            x_pos = 20  # Left margin
            for key, value in metadata.items():
                if isinstance(value, float):
                    debug_text = f"{key}: {value:.2f}"
                else:
                    debug_text = f"{key}: {value}"
                cv2.putText(frame, debug_text, (x_pos, y_pos), font, debug_font_scale, (200, 200, 200), 1, cv2.LINE_AA)
                y_pos += 25  # Space between lines
        
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

def create_captioned_video(video_path: str, descriptions: list, summary: str, transcript: list, synthesis_captions: list = None, use_synthesis_captions: bool = False, show_transcriptions: bool = False, output_path: str = None, debug: bool = False, debug_metadata: dict = None) -> str:
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
        # Calculate video duration
        video_duration = total_frames / fps
        print(f"\nVideo duration: {video_duration:.2f}s")
        
        # Convert synthesis captions to frame info format
        frame_info = []
        
        if len(synthesis_captions) > 0:
            print(f"\nInitial synthesis captions count: {len(synthesis_captions)}")
            print(f"First caption: {synthesis_captions[0][1][:50]}...")
            print(f"Last caption: {synthesis_captions[-1][1][:50]}...")
            
            # First pass: Adjust timestamps slightly earlier for better timing
            adjusted_captions = []
            for timestamp, text in synthesis_captions:
                # Adjust timestamp to be 0.1s earlier, but not before 0
                adjusted_timestamp = max(0.0, timestamp - 0.1)
                adjusted_captions.append((adjusted_timestamp, text))
            
            print(f"\nAfter timestamp adjustment: {len(adjusted_captions)} captions")
            
            # Convert to frame info format with adjusted timestamps
            frame_info = [(int(timestamp * fps), timestamp, text) 
                         for timestamp, text in adjusted_captions]
            
            frame_info.sort(key=lambda x: x[1])  # Sort by timestamp
            print(f"\nFinal output: {len(frame_info)} captions")
            if frame_info:
                print(f"First caption at {frame_info[0][1]:.2f}s: {frame_info[0][2][:50]}...")
                print(f"Last caption at {frame_info[-1][1]:.2f}s: {frame_info[-1][2][:50]}...")
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

            # Handle transcriptions (bottom overlay)
            bottom_lines = []
            if show_transcriptions:
                # Find appropriate transcript segment and check if we're within its time period
                while (current_transcript_idx < len(transcript) - 1 and 
                       current_time >= transcript[current_transcript_idx + 1]["start"]):
                    current_transcript_idx += 1
                
                # Only use transcript text if we're within its time period
                transcript_segment = transcript[current_transcript_idx]
                if (current_time >= transcript_segment["start"] and 
                    current_time <= transcript_segment["end"]):
                    current_transcript_text = transcript_segment["text"]
                    
                    # Normalize smart quotes and apostrophes in transcript
                    current_transcript_text = current_transcript_text.replace('"', '"').replace('"', '"')
                    current_transcript_text = current_transcript_text.replace("'", "'").replace("'", "'")
                    
                    # Adjust font and padding based on resolution tiers
                    if height >= 1440:  # 2K and above
                        base_increase = 0.6
                        bg_padding = 20
                        bottom_margin = int(height * 0.25)
                    elif height >= 720:  # HD
                        base_increase = 0.2
                        bg_padding = 8
                        bottom_margin = int(height * 0.15)
                    else:  # SD
                        base_increase = 0.1
                        bg_padding = 5
                        bottom_margin = int(height * 0.18)
                    
                    transcript_font_scale = min(height * 0.025 / min_line_height, 0.6) + base_increase
                    
                    # Split transcript into lines
                    current_line = []
                    for word in current_transcript_text.split():
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

    # Create summary intro clip with debug metadata
    print("\nCreating summary intro...")
    summary_path, summary_duration = create_summary_clip(
        summary, 
        frame_width, 
        frame_height, 
        fps,
        debug=debug,
        metadata=debug_metadata
    )
    
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

def process_video_web(video_file, use_frame_selection=False, use_synthesis_captions=False, use_local_llm=False, show_transcriptions=False, debug=False):
    """Process video through web interface."""
    video_path = video_file.name
    print(f"Processing video: {video_path}")
    print(f"Using synthesis captions: {use_synthesis_captions}")
    print(f"Using local LLM: {use_local_llm}")
    print(f"Showing transcriptions: {show_transcriptions}")
    print(f"Debug mode: {debug}")

    # Skip if this is an output file
    if os.path.basename(video_path).startswith('captioned_') or video_path.endswith('_web.mp4'):
        print(f"Skipping output file: {video_path}")
        return "Skipped output file", None, None

    start_time = time.time()

    # Get frame count and duration
    frame_count = get_frame_count(video_path)
    if frame_count == 0:
        return "Error: Could not process video.", None, None
        
    # Get video duration
    video_duration = get_video_duration(video_path)
    if video_duration is None:
        return "Error: Could not get video duration.", None, None

    # Extract metadata if in debug mode
    debug_metadata = None
    if debug:
        try:
            cmd = [
                'ffprobe',
                '-v', 'quiet',
                '-print_format', 'json',
                '-show_format',
                '-show_streams',
                video_path
            ]
            metadata_json = subprocess.check_output(cmd).decode('utf-8')
            metadata = json.loads(metadata_json)
            
            # Extract relevant metadata
            debug_metadata = {
                'Duration': float(metadata.get('format', {}).get('duration', 0)),
                'Size': f"{int(int(metadata.get('format', {}).get('size', 0)) / 1024 / 1024)}MB",
                'Format': metadata.get('format', {}).get('format_name', ''),
                'Bitrate': f"{int(int(metadata.get('format', {}).get('bit_rate', 0)) / 1024)}Kbps",
                'Start Time': metadata.get('format', {}).get('start_time', '0'),
            }
            
            # Add video stream info if available
            video_stream = next((s for s in metadata.get('streams', []) if s.get('codec_type') == 'video'), None)
            if video_stream:
                debug_metadata.update({
                    'Resolution': f"{video_stream.get('width', '')}x{video_stream.get('height', '')}",
                    'Codec': video_stream.get('codec_name', ''),
                    'FPS': eval(video_stream.get('r_frame_rate', '0/1')),  # Safely evaluate frame rate fraction
                })
        except Exception as e:
            print(f"Error extracting debug metadata: {str(e)}")
            debug_metadata = {'Error': 'Failed to extract metadata'}

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
    synthesis_output, synthesis_captions = summarize_with_hosted_llm(
        transcript, 
        descriptions, 
        video_duration,
        use_local_llm=use_local_llm,
        video_path=video_path
    )
    
    if synthesis_output is None or (use_synthesis_captions and synthesis_captions is None):
        print("\nError: Failed to generate synthesis. Please try again.")
        return "Error: Failed to generate synthesis. Please try again.", None, None
    
    # Extract summary from synthesis output
    summary_match = re.search(r'<summary>(.*?)</summary>', synthesis_output, re.DOTALL)
    if not summary_match:
        print("\nError: Failed to extract summary from synthesis output")
        return "Error: Failed to extract summary from synthesis output.", None, None
    
    summary = summary_match.group(1).strip()
    print(f"Summary generation complete. Generated {len(synthesis_captions) if synthesis_captions else 0} synthesis captions.")

    # Create captioned video with summary intro
    output_video_path = create_captioned_video(
        video_path, 
        descriptions, 
        summary,
        transcript,
        synthesis_captions,
        use_synthesis_captions,
        show_transcriptions,
        debug=debug,
        debug_metadata=debug_metadata
    )

    total_run_time = time.time() - start_time
    print(f"Time taken: {total_run_time:.2f} seconds")

    # Save output to JSON file
    print("Saving output to JSON file...")
    save_output(
        video_path, 
        frame_count, 
        transcript, 
        descriptions, 
        summary, 
        total_run_time,
        synthesis_output,
        synthesis_captions
    )

    # Prepare frames with captions for gallery display
    gallery_images = []
    video = cv2.VideoCapture(video_path)
    
    for frame_number, timestamp, description in descriptions:
        video.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        success, frame = video.read()
        if success:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            height, width = frame.shape[:2]
            margin = 8
            padding = 10
            min_line_height = 20
            font = cv2.FONT_HERSHEY_SIMPLEX
            max_width = int(width * 0.9)
            
            full_text = f"Frame {frame_number} [{timestamp:.2f}s]\n{description}"
            words = full_text.split()
            lines = []
            current_line = []
            
            font_scale = min(height * 0.03 / min_line_height, 0.7)
            
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
            
            line_count = len(lines)
            line_height = max(min_line_height, int(height * 0.03))
            box_height = line_count * line_height + 2 * padding
            
            overlay = frame.copy()
            cv2.rectangle(overlay, 
                         (int(margin), int(margin)),
                         (int(width - margin), int(margin + box_height)),
                         (0, 0, 0),
                         -1)
            
            alpha = 0.7
            frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
            
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
            
            frame_pil = Image.fromarray(frame)
            gallery_images.append(frame_pil)
    
    video.release()

    return (
        output_video_path,
        f"Video Summary:\n{summary}\n\nTime taken: {total_run_time:.2f} seconds", 
        gallery_images
    )

def process_folder(folder_path, args):
    """Process all videos in a folder with retries."""
    # Get list of video files
    video_extensions = ('.mp4', '.avi', '.mov', '.mkv', '.webm')
    
    # Get all video files
    all_files = [f for f in os.listdir(folder_path) 
                if os.path.isfile(os.path.join(folder_path, f)) 
                and f.lower().endswith(video_extensions)]
    
    # Filter out output files and temporary files
    video_files = []
    for f in all_files:
        # Skip if it's an output file or temporary file
        if (f.startswith('captioned_') or 
            f.startswith('temp_') or 
            f.startswith('concat_') or 
            f.endswith('_web.mp4')):
            print(f"Skipping output/temp file: {f}")
            continue
        video_files.append(f)
    
    if not video_files:
        print(f"No video files found in {folder_path}")
        return

    print(f"Found {len(video_files)} video files to process")
    print("Files to process:", video_files)
    
    # Keep track of successfully processed videos
    processed_videos = set()
    
    # Process each video with retries
    for i, video_file in enumerate(video_files, 1):
        if video_file in processed_videos:
            print(f"Skipping already processed video: {video_file}")
            continue
            
        video_path = os.path.join(folder_path, video_file)
        print(f"\nProcessing video {i}/{len(video_files)}: {video_file}")
        
        # Try processing with up to 10 retries
        max_retries = 10
        success_marker = f"Captioned video saved to: outputs/captioned_{os.path.splitext(video_file)[0]}_web.mp4"
        
        for attempt in range(max_retries):
            try:
                if attempt > 0:
                    print(f"Retry attempt {attempt + 1}/{max_retries}")
                    # Force garbage collection and clear CUDA cache before retry
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                
                # Process video using the web function for consistency
                output_video_path, summary, gallery_images = process_video_web(
                    type('VideoFile', (), {'name': video_path})(),
                    use_frame_selection=args.frame_selection,
                    use_synthesis_captions=args.synthesis_captions,
                    use_local_llm=args.local,
                    show_transcriptions=args.transcribe,
                    debug=args.debug
                )
                
                # Check if the expected output file exists
                expected_output = os.path.join('outputs', f'captioned_{os.path.splitext(video_file)[0]}_web.mp4')
                if os.path.exists(expected_output):
                    processed_videos.add(video_file)
                    print(f"Successfully processed {video_file}")
                    print("\nVideo Summary:")
                    print(summary)
                    print(success_marker)  # Print the specific success marker
                    break  # Success, exit retry loop
                else:
                    print(f"Output file not found: {expected_output}")
                    if attempt == max_retries - 1:
                        print(f"Failed to process {video_file} after {max_retries} attempts")
                    else:
                        print("Retrying...")
                
            except Exception as e:
                print(f"Error processing {video_file} (attempt {attempt + 1}/{max_retries}): {str(e)}")
                # Check if the success marker is in the error message
                if str(e).find(success_marker) != -1:
                    print("Found success marker in error message - video was actually processed successfully")
                    processed_videos.add(video_file)
                    break
                elif attempt == max_retries - 1:  # Last attempt
                    print(f"Failed to process {video_file} after {max_retries} attempts")
                else:
                    print("Retrying immediately...")
    
    print(f"\nProcessing complete. Successfully processed {len(processed_videos)} out of {len(video_files)} videos.")
    print("Processed videos:", sorted(list(processed_videos)))

def recontextualize_summary(summary: str, metadata_context: str, use_local_llm: bool = False) -> str:
    """Recontextualize just the summary using metadata context."""
    print("\n=== Starting Recontextualization ===")
    print("Original Summary:", summary)
    print("\nMetadata Context:", metadata_context)
    print(f"Using local LLM: {use_local_llm}")
    
    recontextualize_prompt = """You are analyzing a video summary that needs to be enriched with metadata context.

Given the video's metadata and initial summary, provide a more complete understanding that incorporates the video's origin and purpose.

Guidelines for Recontextualization:
1. Consider the metadata FIRST - it provides crucial context about the video's origin and purpose
2. Use the video's title, creator, and other metadata to properly frame the context, but IGNORE:
   - Software/tool attributions (e.g. "Created with Clipchamp", "Made in Adobe", etc.)
   - Watermarks or branding from editing tools
   - Generic platform metadata (e.g. "Uploaded to YouTube")
3. Pay special attention to:
   - The video's intended purpose (based on meaningful metadata)
   - Professional vs. amateur content
   - Genre and style implications
4. Maintain objectivity while acknowledging the full context
5. Don't speculate beyond what's supported by meaningful metadata
6. Keep roughly the same length and style
7. Your output must include <recontextualized_summary> open tags and </recontextualized_summary> close tags, with the summary content between them. 
8. Don't be too specific or verbose in your summary, keep it general and concise.

Output Format:
<recontextualized_summary>
A well-written summary that integrates metadata context. Must start with "This video presents"
</recontextualized_summary>

Input:
<metadata>
{metadata}
</metadata>

<initial_summary>
{summary}
</initial_summary>"""

    # Create the prompt with context
    prompt_content = recontextualize_prompt.format(
        metadata=metadata_context,
        summary=summary
    )
    
    print("\nFull Prompt for Recontextualization:")
    print("=" * 80)
    print(prompt_content)
    print("=" * 80)

    max_retries = 3
    for attempt in range(max_retries):
        print(f"\nAttempt {attempt + 1}/{max_retries} to get recontextualized summary...")
        completion = get_llm_completion(prompt_content, "", use_local_llm=use_local_llm)
        
        if completion:
            print("\nGot completion response:")
            print("-" * 80)
            print(completion)
            print("-" * 80)
            
            # Extract summary from completion using same regex pattern style
            summary_match = re.search(r'<recontextualized_summary>(.*?)</recontextualized_summary>', completion, re.DOTALL)
            if summary_match:
                new_summary = summary_match.group(1).strip()
                print("\nSuccessfully extracted new summary:")
                print(new_summary)
                return new_summary
            else:
                print("\nFailed to extract summary from completion - no <recontextualized_summary> tags found")
        else:
            print("\nNo completion returned from LLM")
        
        if attempt < max_retries - 1:
            print(f"\nRetrying recontextualization (attempt {attempt + 2}/{max_retries})...")
            time.sleep(2 * (attempt + 1))
    
    print("\n=== Recontextualization Failed ===")
    print("Returning original summary:", summary)
    return summary

def main():
    parser = argparse.ArgumentParser(description="Process a video file or folder of videos.")
    parser.add_argument('path', type=str, nargs='?', help='Path to the video file or folder')
    parser.add_argument('--save', action='store_true', help='Save output to a JSON file')
    parser.add_argument('--local', action='store_true', help='Use local Llama model for summarization')
    parser.add_argument('--frame-selection', action='store_true', help='Use CLIP-based key frame selection algorithm')
    parser.add_argument('--web', action='store_true', help='Start Gradio web interface')
    parser.add_argument('--synthesis-captions', action='store_true', help='Use synthesized narrative captions (recommended)')
    parser.add_argument('--transcribe', action='store_true', help='Show speech transcriptions in the output video')
    parser.add_argument('--debug', action='store_true', help='Enable debug mode to show metadata in summary screen')
    args = parser.parse_args()

    if args.web:
        # Create Gradio interface with gallery and video output
        iface = gr.Interface(
            fn=process_video_web,
            inputs=[
                gr.File(label="Upload Video"),
                gr.Checkbox(label="Use Frame Selection", value=True, info="Recommended: Intelligently selects key frames"),
                gr.Checkbox(label="Use Synthesis Captions", value=True, info="Recommended: Creates a more pleasant viewing experience"),
                gr.Checkbox(label="Use Local LLM", value=True, info="Use local Llama model instead of OpenAI API (requires model weights)"),
                gr.Checkbox(label="Show Transcriptions", value=False, info="Show speech transcriptions in the output video"),
                gr.Checkbox(label="Debug Mode", value=False, info="Show metadata in summary screen")
            ],
            outputs=[
                gr.Video(label="Captioned Video"),
                gr.Textbox(label="Summary"),
                gr.Gallery(label="Analyzed Frames")
            ],
            title="Video Summarizer",
            description="Upload a video to get a summary and view analyzed frames.",
            allow_flagging="never"
        )
        iface.launch()
    elif args.path:
        start_time = time.time()

        if os.path.isdir(args.path):
            # Process folder
            print(f"Processing folder: {args.path}")
            process_folder(args.path, args)
        else:
            # Process single video
            print(f"Processing video: {args.path}")
            
            # Get frame count
            frame_count = get_frame_count(args.path)
            if frame_count == 0:
                return

            # Process video using the web function for consistency
            output_video_path, summary, gallery_images = process_video_web(
                type('VideoFile', (), {'name': args.path})(),
                use_frame_selection=args.frame_selection,
                use_synthesis_captions=args.synthesis_captions,
                use_local_llm=args.local,
                show_transcriptions=args.transcribe,
                debug=args.debug
            )
            
            # Print the summary and processing time
            print("\nVideo Summary:")
            print(summary)

            total_run_time = time.time() - start_time
            print(f"\nTotal processing time: {total_run_time:.2f} seconds")
            print(f"\nCaptioned video saved to: {output_video_path}")
    else:
        print("Please provide a video file/folder path or use the --web flag to start the web interface.")

if __name__ == "__main__":
    main()
#!/usr/bin/env python3
import torch
from transformers import pipeline
import subprocess
import json
import os
import time
from pathlib import Path
import cv2
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from tqdm import tqdm

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

def correct_timestamps(chunks):
    """
    Correct timestamps that reset every ~30 seconds.
    Algorithm:
    1. Track the current 30s segment we're in
    2. When we see a timestamp near 0 after being in the 20-30s range,
       we know we've wrapped to the next 30s segment
    3. Apply the current offset to all subsequent timestamps until next reset
    """
    corrected_transcript = []
    current_offset = 0
    last_end = 0
    in_high_range = False  # Flag to track if we were in 20-30s range
    
    print("\nTimestamp Correction Analysis:")
    print("------------------------------")
    
    for i, chunk in enumerate(chunks):
        start = chunk.get("timestamp", [None, None])[0]
        end = chunk.get("timestamp", [None, None])[1]
        text = chunk.get("text", "").strip()
        
        if start is None or start == 0 or end is None or end == 0 or not text:
            continue
            
        # Check if we were in high range (20-30s)
        if last_end > 20:
            in_high_range = True
            
        # Detect wrap-around to next 30s segment
        # If we were in high range and now we're back to low numbers (0-5s)
        if in_high_range and 0 <= start <= 5:
            current_offset += 30
            in_high_range = False
            print(f"\nRESET DETECTED at chunk {i}:")
            print(f"  Previous end time: {last_end:.2f}s")
            print(f"  Current start time: {start:.2f}s")
            print(f"  Adding 30s - New offset: {current_offset}s")
        
        # Always apply the current offset to both start and end times
        corrected_start = start + current_offset
        corrected_end = end + current_offset
        
        # Store corrected chunk
        if text:  # Only store non-empty chunks
            if current_offset > 0:
                print(f"\nChunk {i}:")
                print(f"  Text: {text}")
                print(f"  Original: {start:.2f}s -> {end:.2f}s")
                print(f"  Corrected: {corrected_start:.2f}s -> {corrected_end:.2f}s")
                print(f"  Current offset: +{current_offset}s")
            
            corrected_transcript.append({
                "start": corrected_start,
                "end": corrected_end,
                "text": text
            })
        
        # Update last_end using the original (non-offset) time for reset detection
        last_end = end
    
    print(f"\nFinal offset: {current_offset}s")
    return corrected_transcript

def add_caption_to_frame(frame, text, font_path=None):
    """Add caption text to a video frame."""
    height, width = frame.shape[:2]
    font_size = int(height / 20)  # Adjust font size based on video height
    
    # Convert frame to PIL Image for text rendering
    frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(frame_pil)
    
    try:
        if font_path and os.path.exists(font_path):
            font = ImageFont.truetype(font_path, font_size)
        else:
            # Fallback to default font
            font = ImageFont.load_default()
    except Exception as e:
        print(f"Error loading font: {e}")
        font = ImageFont.load_default()
    
    # Calculate text size and position
    text_width = draw.textlength(text, font=font)
    x = (width - text_width) / 2
    y = height - int(height / 6)  # Position near bottom
    
    # Add black outline/shadow for better visibility
    outline_color = "black"
    outline_width = 2
    for dx, dy in [(-1,-1), (-1,1), (1,-1), (1,1)]:
        draw.text((x + dx * outline_width, y + dy * outline_width), text, outline_color, font=font)
    
    # Draw main text in white
    draw.text((x, y), text, "white", font=font)
    
    # Convert back to OpenCV format
    return cv2.cvtColor(np.array(frame_pil), cv2.COLOR_RGB2BGR)

def create_captioned_video(video_path, transcript, output_path=None):
    """Create a new video with captions from the transcript using OpenCV."""
    if not output_path:
        timestamp = int(time.time())
        output_path = os.path.join('outputs', f'captioned_{timestamp}_temp.mp4')
        final_output = os.path.join('outputs', f'captioned_{timestamp}.mp4')
    
    print(f"\nCreating captioned video: {output_path}")
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error: Could not open video file")
        return
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    print(f"Video properties:")
    print(f"- FPS: {fps}")
    print(f"- Resolution: {width}x{height}")
    print(f"- Total frames: {total_frames}")
    print(f"- Duration: {total_frames/fps:.2f}s")
    
    # Try different codecs in order of preference
    codecs = ['avc1', 'mp4v']
    out = None
    for codec in codecs:
        try:
            fourcc = cv2.VideoWriter_fourcc(*codec)
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
            if out.isOpened():
                print(f"Using codec: {codec}")
                break
        except Exception as e:
            print(f"Failed to initialize codec {codec}: {e}")
            if out is not None:
                out.release()
                out = None
    
    if out is None:
        print("Error: Could not initialize video writer with any codec")
        return
    
    frame_count = 0
    current_text = ""
    font = cv2.FONT_HERSHEY_DUPLEX
    font_scale = height / 1000  # Scale font based on video height
    
    try:
        with tqdm(total=total_frames, desc="Processing frames") as pbar:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Calculate current time in video
                current_time = frame_count / fps
                
                # Find appropriate caption for current time
                for segment in transcript:
                    if segment["start"] <= current_time <= segment["end"]:
                        current_text = segment["text"].strip()
                        break
                    else:
                        current_text = ""
                
                # Add caption to frame if we have text
                if current_text:
                    # Calculate text size and position
                    thickness = max(2, int(height / 500))  # Scale thickness with height
                    (text_width, text_height), baseline = cv2.getTextSize(current_text, font, font_scale, thickness)
                    
                    # Center text horizontally, position near bottom
                    x = (width - text_width) // 2
                    y = height - int(height / 8)  # Position text higher from bottom
                    
                    # Draw black outline (thicker)
                    cv2.putText(frame, current_text, (x, y), font, font_scale, (0, 0, 0), thickness * 3)
                    # Draw white text
                    cv2.putText(frame, current_text, (x, y), font, font_scale, (255, 255, 255), thickness)
                
                # Write frame
                out.write(frame)
                frame_count += 1
                pbar.update(1)
    
    finally:
        cap.release()
        out.release()
    
    # Copy audio from original to final video using ffmpeg
    try:
        cmd = [
            'ffmpeg', '-y',
            '-i', output_path,  # Video with captions
            '-i', video_path,   # Original video (for audio)
            '-c:v', 'copy',     # Copy video stream
            '-c:a', 'aac',      # Re-encode audio as AAC
            final_output
        ]
        subprocess.run(cmd, check=True)
        
        # Remove temporary file
        if os.path.exists(output_path):
            os.remove(output_path)
            
        print(f"\nCaptioned video with audio saved to: {final_output}")
        return final_output
        
    except Exception as e:
        print(f"Error adding audio to video: {str(e)}")
        return output_path

def test_whisper_transcription(video_path):
    """Test Whisper transcription with detailed logging."""
    print("\nStarting Whisper transcription test...")
    
    # Get video duration
    duration = get_video_duration(video_path)
    
    # Extract audio
    audio_path = extract_audio(video_path)
    if not audio_path:
        print("Failed to extract audio")
        return
        
    try:
        # Setup device
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        print(f"Using device: {device}")
        
        # Create pipeline (no chunking)
        pipe = pipeline(
            task="automatic-speech-recognition",
            model="openai/whisper-large-v3-turbo",
            device=device,
        )
        
        print("Transcribing audio...")
        
        # Run transcription
        result = pipe(
            audio_path,
            batch_size=8,
            return_timestamps=True,
            generate_kwargs={"task": "transcribe"}
        )
        
        # Save raw output
        timestamp = int(time.time())
        raw_output_path = os.path.join('outputs', f'whisper_raw_{timestamp}.txt')
        json_output_path = os.path.join('outputs', f'whisper_processed_{timestamp}.json')
        
        # Save raw output with detailed information
        with open(raw_output_path, 'w', encoding='utf-8') as f:
            f.write("=== RAW WHISPER OUTPUT ===\n\n")
            f.write(f"Video path: {video_path}\n")
            f.write(f"Video duration: {duration:.2f}s\n")
            f.write(f"Timestamp: {timestamp}\n")
            f.write("\n=== FULL RESULT ===\n")
            f.write(json.dumps(result, indent=2, ensure_ascii=False))
            f.write("\n\n=== CHUNKS WITH ANALYSIS ===\n")
            
            # Analyze chunks
            prev_end = 0
            for i, chunk in enumerate(result.get("chunks", [])):
                f.write(f"\nChunk {i+1}:\n")
                f.write(json.dumps(chunk, indent=2, ensure_ascii=False))
                
                # Analyze timestamps
                timestamp = chunk.get("timestamp")
                if timestamp:
                    start, end = timestamp
                    if start is not None and end is not None:
                        gap = start - prev_end if i > 0 else 0
                        f.write(f"\nAnalysis:")
                        f.write(f"\n- Duration: {end - start:.2f}s")
                        f.write(f"\n- Gap from previous: {gap:.2f}s")
                        if gap < -10:  # Highlight potential resets
                            f.write(f"\n- WARNING: Timestamp reset detected!")
                        prev_end = end
                
                f.write("\n" + "-"*50)
        
        print(f"\nRaw output saved to: {raw_output_path}")
        
        # Process and correct timestamps
        corrected_transcript = correct_timestamps(result.get("chunks", []))
        
        # Handle final chunk's end time
        if corrected_transcript and corrected_transcript[-1]["end"] is None and duration:
            corrected_transcript[-1]["end"] = duration
        
        # Save processed output
        with open(json_output_path, 'w', encoding='utf-8') as f:
            json.dump({
                "video_path": video_path,
                "duration": duration,
                "transcript": corrected_transcript
            }, f, indent=2, ensure_ascii=False)
            
        print(f"Processed output saved to: {json_output_path}")
        
        # After saving the processed output, create captioned video
        create_captioned_video(video_path, corrected_transcript)
        
    except Exception as e:
        print(f"Error during transcription: {str(e)}")
    finally:
        # Cleanup
        if audio_path and os.path.exists(audio_path):
            os.remove(audio_path)

def main():
    # Check inputs directory for video files
    input_dir = Path('inputs')
    if not input_dir.exists():
        print("Creating inputs directory...")
        input_dir.mkdir(exist_ok=True)
        print("Please place video files in the 'inputs' directory")
        return
        
    video_files = list(input_dir.glob('*.mp4'))
    if not video_files:
        print("No .mp4 files found in inputs directory")
        return
        
    print(f"Found {len(video_files)} video files:")
    for i, video_path in enumerate(video_files, 1):
        print(f"{i}. {video_path.name}")
    
    # Process each video
    for video_path in video_files:
        print(f"\nProcessing: {video_path.name}")
        test_whisper_transcription(str(video_path))

if __name__ == "__main__":
    main() 
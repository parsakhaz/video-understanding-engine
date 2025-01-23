#!/usr/bin/env python3
import torch
from transformers import pipeline
import subprocess
import json
import os
import time
from pathlib import Path

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
        
        # Test both with and without chunking
        for is_chunked in [True, False]:
            print(f"\nTesting {'chunked' if is_chunked else 'non-chunked'} processing...")
            
            # Create pipeline
            pipe = pipeline(
                task="automatic-speech-recognition",
                model="openai/whisper-large-v3-turbo",
                device=device,
                chunk_length_s=30 if is_chunked else None,  # Only set chunk_length_s if chunking
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
            mode = "chunked" if is_chunked else "full"
            raw_output_path = os.path.join('outputs', f'whisper_raw_{mode}_{timestamp}.txt')
            json_output_path = os.path.join('outputs', f'whisper_processed_{mode}_{timestamp}.json')
            
            # Save raw output with detailed information
            with open(raw_output_path, 'w', encoding='utf-8') as f:
                f.write("=== RAW WHISPER OUTPUT ===\n\n")
                f.write(f"Video path: {video_path}\n")
                f.write(f"Video duration: {duration:.2f}s\n")
                f.write(f"Mode: {'Chunked (30s)' if is_chunked else 'Full audio'}\n")
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
                            if is_chunked:
                                f.write(f"\n- Expected chunk boundary: {(i * 30):.2f}s")
                                f.write(f"\n- Actual chunk boundary: {start:.2f}s")
                                f.write(f"\n- Offset from expected: {(start - (i * 30)):.2f}s")
                            prev_end = end
                    
                    f.write("\n" + "-"*50)
            
            print(f"\nRaw output saved to: {raw_output_path}")
            
            # Process and save timestamped transcript
            timestamped_transcript = []
            for chunk in result.get("chunks", []):
                timestamp = chunk.get("timestamp")
                if timestamp and len(timestamp) == 2:
                    start, end = timestamp
                    if start is not None:
                        if end is None and duration:
                            end = duration
                        
                        if end is not None:
                            text = chunk.get("text", "").strip()
                            if text:
                                timestamped_transcript.append({
                                    "start": float(start),
                                    "end": float(end),
                                    "text": text
                                })
            
            # Save processed output
            with open(json_output_path, 'w', encoding='utf-8') as f:
                json.dump({
                    "video_path": video_path,
                    "duration": duration,
                    "mode": "chunked" if is_chunked else "full",
                    "transcript": timestamped_transcript
                }, f, indent=2, ensure_ascii=False)
                
            print(f"Processed output saved to: {json_output_path}")
        
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
#!/usr/bin/env python3
import argparse
import cv2
import warnings
from PIL import Image
import torch
import os
from tqdm import tqdm
import numpy as np
import json
from datetime import datetime
from openai import OpenAI
from dotenv import load_dotenv
import time
from frame_selection import process_video
import gradio as gr
import subprocess
import re
from difflib import SequenceMatcher
from prompts import get_frame_description_prompt, get_recontextualization_prompt, get_final_summary_prompt
import video_utils
from config import (
    VIDEO_SETTINGS,
    FRAME_SELECTION,
    MODEL_SETTINGS,
    DISPLAY,
    RETRY,
    CAPTION,
    SIMILARITY
)
from model_loader import model_context

def similar(a, b, threshold=SIMILARITY['THRESHOLD']):
    """Return True if strings are similar above threshold."""
    return SequenceMatcher(None, a, b).ratio() > threshold

def clean_transcript(segments):
    """Clean up transcript by removing duplicates and background noise."""
    noise_phrases = SIMILARITY['NOISE_PHRASES']
    boilerplate_phrases = SIMILARITY['BOILERPLATE_PHRASES']
    
    filtered_segments = []
    for segment in segments:
        text = segment["text"].strip().lower()
        
        if any(phrase in text.lower() for phrase in noise_phrases):
            continue
            
        boilerplate_count = sum(1 for phrase in boilerplate_phrases if phrase in text.lower())
        if boilerplate_count >= 2:
            continue
            
        filtered_segments.append(segment)
    
    cleaned_segments = []
    for i, current in enumerate(filtered_segments):
        if i > 0 and similar(current["text"], filtered_segments[i-1]["text"]):
            continue
            
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
    return video_utils.get_video_duration(video_path)

def extract_audio(video_path):
    """Extract audio from video to MP3."""
    return video_utils.extract_audio(video_path)

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
    duration = get_video_duration(video_path)
    if duration is None:
        return [{
            "start": 0,
            "end": 0,
            "text": ""
        }]

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
        pass
    if not has_audio:
        return [{
            "start": 0,
            "end": 0,
            "text": ""
        }]

    audio_path = extract_audio(video_path)
    if not audio_path:
        return [{
            "start": 0,
            "end": 0,
            "text": ""
        }]
    
    try:
        with model_context("whisper") as model:
            if model is None:
                return [{
                    "start": 0,
                    "end": 0,
                    "text": ""
                }]
                
            result = model.transcribe(audio_path)
            timestamp = int(time.time())
            raw_output_path = os.path.join('outputs', f'whisper_raw_{timestamp}.txt')
            
            processed_transcript = process_transcript(result["segments"])
            
            return clean_transcript(processed_transcript)
            
    except Exception as e:
        return [{
            "start": 0,
            "end": 0,
            "text": ""
        }]
    finally:
        try:
            if os.path.exists(audio_path):
                os.remove(audio_path)
        except Exception:
            pass

# Function to describe frames using a Vision-Language Model
def describe_frames(video_path, frame_numbers):
    with model_context("moondream") as model_tuple:
        if model_tuple is None:
            return []
            
        model, tokenizer = model_tuple
        prompt = get_frame_description_prompt()

        props = video_utils.get_video_properties(video_path)
        fps = props['fps']
        frames = []

        video = cv2.VideoCapture(video_path)
        for frame_number in frame_numbers:
            video.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            success, frame = video.read()
            if success:
                timestamp = frame_number / fps
                frames.append((frame_number, frame, timestamp))

        video.release()

        batch_size = FRAME_SELECTION['BATCH_SIZE']
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

        (_, text_height), _ = cv2.getTextSize('Tg', font, font_scale, font_thickness)
        total_height = (text_height + line_spacing) * len(lines) + 2 * margin

        caption_image = np.zeros((total_height, width, 3), dtype=np.uint8)

        y = margin + text_height
        for line in lines:
            cv2.putText(caption_image, line, (margin, y), font, font_scale, (255, 255, 255), font_thickness)
            y += text_height + line_spacing

        return np.vstack((caption_image, frame))

    try:
        while True:
            frame_number, timestamp, description = descriptions[current_index]
            video.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
            success, frame = video.read()
            if not success:
                break

            max_height = 800
            height, width = frame.shape[:2]
            if height > max_height:
                ratio = max_height / height
                frame = cv2.resize(frame, (int(width * ratio), max_height))

            frame_with_caption = add_caption_to_frame(frame, f"[{timestamp:.2f}s] Frame {frame_number}: {description}")

            cv2.imshow('Video Frames', frame_with_caption)
            cv2.setWindowTitle('Video Frames', f"Frame {frame_number}")

            key = cv2.waitKey(100) & 0xFF
            if key == ord('q'):
                break
            elif key in (83, 32):
                current_index = min(current_index + 1, len(descriptions) - 1)
            elif key == 81:
                current_index = max(current_index - 1, 0)
    except KeyboardInterrupt:
        pass
    finally:
        video.release()
        cv2.destroyAllWindows()

def get_frame_count(video_path):
    video = cv2.VideoCapture(video_path)
    if not video.isOpened():
        return 0
    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    video.release()
    return frame_count

def save_output(video_path, frame_count, transcript, descriptions, summary, total_run_time, synthesis_output=None, synthesis_captions=None, use_local_llm=False):
    os.makedirs('logs', exist_ok=True)
    timestamp = datetime.now().isoformat().replace(':', '-')
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    filename = f"logs/{timestamp}_{video_name}.json"

    metadata, metadata_context = video_utils.get_video_metadata(video_path)

    props = video_utils.get_video_properties(video_path)
    fps = props['fps']
    width = props['frame_width']
    height = props['frame_height']
    video_duration = frame_count / fps if fps else None

    if video_duration < VIDEO_SETTINGS['MIN_VIDEO_DURATION']:
        video_type = "Short (<30s)"
    elif video_duration < VIDEO_SETTINGS['MEDIUM_VIDEO_DURATION']:
        video_type = "Medium (30-90s)"
    elif video_duration < VIDEO_SETTINGS['LONG_VIDEO_DURATION']:
        video_type = "Medium-long (90-150s)"
    else:
        video_type = "Long (>150s)"

    if video_duration > VIDEO_SETTINGS['LONG_VIDEO_DURATION']:
        num_captions = max(CAPTION['LONG_VIDEO']['MIN_CAPTIONS'], frame_count // 2)
        caption_calc = f"max({CAPTION['LONG_VIDEO']['MIN_CAPTIONS']}, {frame_count} // 2)"
    elif video_duration > VIDEO_SETTINGS['MEDIUM_VIDEO_DURATION']:
        base_captions = int(video_duration / CAPTION['LONG_VIDEO']['INTERVAL_RATIO'])
        num_captions = min(int(base_captions * 1.25), frame_count // 2)
        num_captions = max(CAPTION['MEDIUM_VIDEO']['MIN_CAPTIONS'], num_captions)
        caption_calc = f"min(max({CAPTION['MEDIUM_VIDEO']['MIN_CAPTIONS']}, {base_captions} * 1.25), {frame_count} // 2)"
    else:
        if video_duration < VIDEO_SETTINGS['MIN_VIDEO_DURATION']:
            target_captions = int(video_duration / CAPTION['SHORT_VIDEO']['INTERVAL'])
            num_captions = min(CAPTION['SHORT_VIDEO']['MAX_CAPTIONS'], 
                             max(CAPTION['SHORT_VIDEO']['MIN_CAPTIONS'], target_captions))
            caption_calc = f"min({CAPTION['SHORT_VIDEO']['MAX_CAPTIONS']}, max({CAPTION['SHORT_VIDEO']['MIN_CAPTIONS']}, {target_captions}))"
        else:
            caption_interval = (CAPTION['MEDIUM_VIDEO']['BASE_INTERVAL'] + 
                              (video_duration - VIDEO_SETTINGS['MIN_VIDEO_DURATION']) * 
                              (CAPTION['MEDIUM_VIDEO']['MAX_INTERVAL'] - CAPTION['MEDIUM_VIDEO']['BASE_INTERVAL']) / 
                              (VIDEO_SETTINGS['MEDIUM_VIDEO_DURATION'] - VIDEO_SETTINGS['MIN_VIDEO_DURATION']))
            target_captions = int(video_duration / caption_interval)
            max_captions = min(int(video_duration / 4), frame_count // 3)
            num_captions = min(target_captions, max_captions)
            num_captions = max(CAPTION['MEDIUM_VIDEO']['MIN_CAPTIONS'], num_captions)
            caption_calc = f"min(max({CAPTION['MEDIUM_VIDEO']['MIN_CAPTIONS']}, {target_captions}), {max_captions})"

    from prompts import get_frame_description_prompt
    frame_description_prompt = get_frame_description_prompt()

    gpu_memory = {}
    if torch.cuda.is_available():
        try:
            for i in range(torch.cuda.device_count()):
                gpu_memory[f"gpu_{i}"] = {
                    "total": torch.cuda.get_device_properties(i).total_memory,
                    "allocated": torch.cuda.memory_allocated(i),
                    "cached": torch.cuda.memory_reserved(i)
                }
        except Exception:
            pass

    cache_path = os.path.join('embedding_cache', f'{video_name}.npy')
    embedding_cache_exists = os.path.exists(cache_path)

    output = {
        "processing_info": {
            "timestamp": timestamp,
            "total_run_time": total_run_time,
            "gpu_memory": gpu_memory,
            "embedding_cache_status": {
                "exists": embedding_cache_exists,
                "path": cache_path if embedding_cache_exists else None
            }
        },
        "configuration": {
            "video_settings": VIDEO_SETTINGS,
            "frame_selection": FRAME_SELECTION,
            "model_settings": MODEL_SETTINGS,
            "display": DISPLAY,
            "retry": RETRY,
            "caption": CAPTION,
            "similarity": SIMILARITY
        },
        "prompts": {
            "frame_description": frame_description_prompt,
            "recontextualization": get_recontextualization_prompt(),
            "final_summary": get_final_summary_prompt(),
            "synthesis": get_synthesis_prompt(
                len(descriptions), 
                video_duration,
                metadata_context=metadata_context
            ),
            "actual_prompts_used": {
                "frame_description": frame_description_prompt,
                "synthesis": synthesis_output,
                "recontextualization": metadata_context if metadata else None,
                "final_summary": get_final_summary_prompt() if video_duration > VIDEO_SETTINGS['LONG_VIDEO_DURATION'] else None
            }
        },
        "video_metadata": {
            "path": video_path,
            "frame_count": frame_count,
            "duration": video_duration,
            "fps": fps,
            "resolution": {
                "width": width,
                "height": height
            },
            "ffprobe_data": metadata,
            "video_type": video_type
        },
        "caption_analysis": {
            "video_type": video_type,
            "target_captions": num_captions,
            "captions_per_second": num_captions/video_duration if video_duration else None,
            "calculation_formula": caption_calc,
            "actual_captions_generated": len(synthesis_captions) if synthesis_captions else 0
        },
        "model_info": {
            "frame_description_prompt": frame_description_prompt,
            "whisper_model": MODEL_SETTINGS['WHISPER_MODEL'],
            "clip_model": "ViT-SO400M-14-SigLIP-384",
            "synthesis_model": MODEL_SETTINGS['LOCAL_LLM_MODEL'] if use_local_llm else MODEL_SETTINGS['GPT_MODEL']
        },
        "processing_stages": {
            "frame_selection": {
                "method": "clip" if os.environ.get("USE_FRAME_SELECTION") else "regular_sampling",
                "parameters": {
                    "novelty_threshold": FRAME_SELECTION['NOVELTY_THRESHOLD'],
                    "min_skip": FRAME_SELECTION['MIN_SKIP_FRAMES'],
                    "n_clusters": FRAME_SELECTION['NUM_CLUSTERS']
                } if os.environ.get("USE_FRAME_SELECTION") else {
                    "sampling_interval": VIDEO_SETTINGS['FRAME_SAMPLING_INTERVAL']
                }
            },
            "transcription": {
                "segments": transcript,
                "total_segments": len(transcript)
            },
            "frame_descriptions": [
                {
                    "frame_number": frame,
                    "timestamp": timestamp,
                    "description": desc
                } for frame, timestamp, desc in descriptions
            ],
            "synthesis": {
                "raw_output": synthesis_output,
                "captions": [
                    {
                        "timestamp": timestamp,
                        "text": text
                    } for timestamp, text in (synthesis_captions or [])
                ],
                "summary": summary
            }
        },
        "error_log": {
            "warnings": [],
            "errors": [],
            "retries": {}
        }
    }

    with open(filename, 'w') as f:
        json.dump(output, f, indent=2)

def get_synthesis_prompt(num_keyframes: int, video_duration: float, metadata_context: str = "") -> str:
    if video_duration > VIDEO_SETTINGS['LONG_VIDEO_DURATION']:
        num_captions = max(CAPTION['LONG_VIDEO']['MIN_CAPTIONS'], num_keyframes // 2)
    elif video_duration > VIDEO_SETTINGS['MEDIUM_VIDEO_DURATION']:
        base_captions = int(video_duration / CAPTION['LONG_VIDEO']['INTERVAL_RATIO'])
        num_captions = min(int(base_captions * 1.25), num_keyframes // 2)
        num_captions = max(CAPTION['MEDIUM_VIDEO']['MIN_CAPTIONS'], num_captions)
    else:
        if video_duration < VIDEO_SETTINGS['MIN_VIDEO_DURATION']:
            target_captions = int(video_duration / CAPTION['SHORT_VIDEO']['INTERVAL'])
            num_captions = min(CAPTION['SHORT_VIDEO']['MAX_CAPTIONS'], 
                             max(CAPTION['SHORT_VIDEO']['MIN_CAPTIONS'], target_captions))
        else:
            caption_interval = (CAPTION['MEDIUM_VIDEO']['BASE_INTERVAL'] + 
                              (video_duration - VIDEO_SETTINGS['MIN_VIDEO_DURATION']) * 
                              (CAPTION['MEDIUM_VIDEO']['MAX_INTERVAL'] - CAPTION['MEDIUM_VIDEO']['BASE_INTERVAL']) / 
                              (VIDEO_SETTINGS['MEDIUM_VIDEO_DURATION'] - VIDEO_SETTINGS['MIN_VIDEO_DURATION']))
            target_captions = int(video_duration / caption_interval)
            max_captions = min(int(video_duration / 4), num_keyframes // 3)
            num_captions = min(target_captions, max_captions)
            num_captions = max(CAPTION['MEDIUM_VIDEO']['MIN_CAPTIONS'], num_captions)

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
    chunks = []
    current_chunk_start = 0
    min_descriptions_per_chunk = 4
    
    while current_chunk_start < descriptions[-1][1]:
        current_chunk_end = current_chunk_start + chunk_duration
        
        chunk_descriptions = [
            desc for desc in descriptions 
            if current_chunk_start <= desc[1] < current_chunk_end
        ]
        
        original_end = current_chunk_end
        while len(chunk_descriptions) < min_descriptions_per_chunk and current_chunk_end < descriptions[-1][1]:
            current_chunk_end += 15
            chunk_descriptions = [
                desc for desc in descriptions 
                if current_chunk_start <= desc[1] < current_chunk_end
            ]
        
        chunk_transcript = [
            seg for seg in transcript
            if (seg["start"] >= current_chunk_start and seg["start"] < current_chunk_end) or
               (seg["end"] > current_chunk_start and seg["end"] <= current_chunk_end)
        ]
        
        if chunk_descriptions:
            chunks.append((chunk_transcript, chunk_descriptions))
        
        current_chunk_start = current_chunk_end
    
    return chunks

def summarize_with_hosted_llm(transcript, descriptions, video_duration: float, use_local_llm=False, video_path: str = None):
    load_dotenv()

    metadata_context = ""
    if video_path:
        max_metadata_retries = 3
        for attempt in range(max_metadata_retries):
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
                
                format_metadata = metadata.get('format', {}).get('tags', {})
                title = format_metadata.get('title', '')
                artist = format_metadata.get('artist', '')
                duration = float(metadata.get('format', {}).get('duration', 0))
                
                metadata_context = "Video Metadata:\n"
                if title:
                    metadata_context += f"- Title: {title}\n"
                if artist:
                    metadata_context += f"- Creator: {artist}\n"
                metadata_context += f"- Duration: {duration:.2f} seconds\n"
                
                for key, value in format_metadata.items():
                    if key not in ['title', 'artist'] and value:
                        metadata_context += f"- {key}: {value}\n"
                break
            except Exception as e:
                if attempt < max_metadata_retries - 1:
                    time.sleep(2 * (attempt + 1))
                else:
                    metadata_context = ""

    is_long_video = video_duration > 150  # 2.5 minutes threshold
    
    initial_synthesis = None
    initial_captions = None
    
    if is_long_video:
        chunks = chunk_video_data(transcript, descriptions)
        all_summaries = []
        all_captions = []
        
        for i, (chunk_transcript, chunk_descriptions) in enumerate(chunks, 1):
            synthesis_prompt = get_synthesis_prompt(
                len(chunk_descriptions), 
                video_duration,
                metadata_context  # Pass metadata to each chunk
            )

            timestamped_transcript = "\n".join([
                f"[{segment['start']:.2f}s-{segment['end']:.2f}s] {segment['text']}"
                for segment in chunk_transcript
            ])
            frame_descriptions = "\n".join([
                f"[{timestamp:.2f}s] Frame {frame}: {desc}"
                for frame, timestamp, desc in chunk_descriptions
            ])
            
            user_content = f"<transcript>\n{timestamped_transcript}\n</transcript>\n\n<frame_descriptions>\n{frame_descriptions}\n</frame_descriptions>"
            
            max_retries = 5
            for attempt in range(max_retries):
                completion = get_llm_completion(synthesis_prompt, user_content, use_local_llm=use_local_llm)
                
                if completion is None:
                    if attempt < max_retries - 1:
                        time.sleep(2 * (attempt + 1))
                        continue
                    else:
                        return None, None
                
                chunk_summary, chunk_captions = parse_synthesis_output(completion)
                
                if chunk_summary and chunk_captions:
                    all_summaries.append(chunk_summary)
                    all_captions.extend(chunk_captions)
                    break
                else:
                    if attempt < max_retries - 1:
                        time.sleep(2 * (attempt + 1))  # Exponential backoff
                    else:
                        return None, None

        # Make a final pass to synthesize all summaries into one coherent summary
        from prompts import get_final_summary_prompt
        final_summary_prompt = get_final_summary_prompt()

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
            final_summary = get_llm_completion(final_summary_prompt, final_summary_content, use_local_llm=use_local_llm)
            if final_summary:
                break
            elif attempt < max_retries - 1:
                time.sleep(2 * (attempt + 1))
        
        if not final_summary:
            return None, None
            
        # For long videos, validate and retry the final recontextualized synthesis
        if is_long_video:
            max_recontextualize_retries = 3
            for attempt in range(max_recontextualize_retries):
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
                    time.sleep(2 * (attempt + 1))
        
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
            completion = get_llm_completion(synthesis_prompt, user_content, use_local_llm=use_local_llm)
            
            if completion is None:
                if attempt < max_retries - 1:
                    time.sleep(2 * (attempt + 1))
                    continue
                else:
                    return None, None
            
            initial_summary, initial_captions = parse_synthesis_output(completion)
            
            if initial_summary and initial_captions:
                initial_synthesis = completion
                break
            else:
                if attempt < max_retries - 1:
                    time.sleep(2 * (attempt + 1))  # Exponential backoff
                else:
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
    try:
        if use_local_llm:
            with model_context("llama") as pipeline:
                if pipeline is None:
                    return None
                    
                try:
                    # Format messages like in the docs
                    messages = [
                        {"role": "system", "content": prompt},
                        {"role": "user", "content": content},
                    ]
                    
                    outputs = pipeline(
                        messages,
                        max_new_tokens=8000,
                        temperature=MODEL_SETTINGS['TEMPERATURE'],
                    )
                    raw_response = outputs[0]["generated_text"]
                    
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
                                        return content
                        
                        # If we get here, try to find XML tags directly
                        if "<summary>" in raw_response_str and "</summary>" in raw_response_str:
                            return raw_response_str
                        
                        return None
                    
                    except Exception as e:
                        return None
                        
                except Exception as e:
                    return None
        
        # OpenAI API fallback
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            return None

        for attempt in range(RETRY['MAX_RETRIES']):
            try:
                client = OpenAI(api_key=api_key)
                completion = client.chat.completions.create(
                    model=MODEL_SETTINGS['GPT_MODEL'],
                    messages=[
                        {"role": "system", "content": prompt},
                        {"role": "user", "content": content}
                    ],
                    temperature=MODEL_SETTINGS['TEMPERATURE']
                )
                response = completion.choices[0].message.content.strip()
                return response
            except Exception as e:
                if attempt == RETRY['MAX_RETRIES'] - 1:  # Last attempt
                    return None
                time.sleep(RETRY['INITIAL_DELAY'] * (attempt + 1))
    except Exception as e:
        return None

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
                continue
        
        if not caption_list:
            raise ValueError("No valid captions found in synthesis output")
            
        return summary, caption_list
    except (ValueError, AttributeError) as e:
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
        line_spacing = 25  # Space between lines
        section_spacing = 35  # Space between sections
        
        # Format and display metadata by sections
        for section_name, section_data in metadata.items():
            # Draw section header
            cv2.putText(frame, f"{section_name}:", (x_pos, y_pos), font, debug_font_scale * 1.1, (255, 255, 255), 1, cv2.LINE_AA)
            y_pos += line_spacing
            
            # Draw section items
            for key, value in section_data.items():
                if isinstance(value, float):
                    debug_text = f"  {key}: {value:.2f}"
                else:
                    debug_text = f"  {key}: {value}"
                cv2.putText(frame, debug_text, (x_pos, y_pos), font, debug_font_scale, (200, 200, 200), 1, cv2.LINE_AA)
                y_pos += line_spacing
            
            # Add extra space between sections
            y_pos += section_spacing - line_spacing  # Subtract line_spacing to not add too much space
    
    # Prepare text
    font_scale = min(height * 0.03 / 20, 0.8)  # Reduced from 0.04 to 0.03 and max from 1.0 to 0.8
    margin = 40
    max_width = int(width * 0.75)  # Use 80% of frame width
    
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
    
    # Check if video has an audio stream using video_utils
    has_audio = video_utils.has_audio_stream(video_path)

    # Validate synthesis captions if requested
    if use_synthesis_captions:
        if not synthesis_captions:
            use_synthesis_captions = False
    
    # Create output directory if needed
    if output_path is None:
        os.makedirs(VIDEO_SETTINGS['OUTPUT_DIR'], exist_ok=True)
        base_name = os.path.splitext(os.path.basename(video_path))[0]
        temp_output = os.path.join(VIDEO_SETTINGS['OUTPUT_DIR'], f'temp_{base_name}.mp4')
        final_output = os.path.join(VIDEO_SETTINGS['OUTPUT_DIR'], f'captioned_{base_name}.mp4')
    else:
        temp_output = output_path + '.temp'
        final_output = output_path

    # Get video properties using video_utils
    props = video_utils.get_video_properties(video_path)
    fps = props['fps']
    frame_width = props['frame_width']
    frame_height = props['frame_height']
    total_frames = props['total_frames']

    # Initialize video capture
    video = cv2.VideoCapture(video_path)
    if not video.isOpened():
        return None

    # Choose which captions to use
    if use_synthesis_captions and synthesis_captions:
        # Calculate video duration
        video_duration = total_frames / fps
        
        # Convert synthesis captions to frame info format
        frame_info = []
        
        if len(synthesis_captions) > 0:
            # First pass: Adjust timestamps slightly earlier for better timing
            adjusted_captions = []
            for timestamp, text in synthesis_captions:
                # Adjust timestamp to be slightly earlier, but not before 0
                adjusted_timestamp = max(0.0, timestamp - CAPTION['TIMESTAMP_OFFSET'])
                adjusted_captions.append((adjusted_timestamp, text))
            
            # Convert to frame info format with adjusted timestamps
            frame_info = [(int(timestamp * fps), timestamp, text) 
                         for timestamp, text in adjusted_captions]
            
            frame_info.sort(key=lambda x: x[1])  # Sort by timestamp
    else:
        # Use all frame descriptions
        frame_info = [(frame_num, timestamp, desc) for frame_num, timestamp, desc in descriptions]
    
    # Sort by timestamp
    frame_info.sort(key=lambda x: x[1])

    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(temp_output, fourcc, fps, (frame_width, frame_height))

    # Process the main video
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

            # Add caption using display settings from config
            height, width = frame.shape[:2]
            margin = DISPLAY['MARGIN']['CAPTION']
            padding = DISPLAY['PADDING']
            min_line_height = DISPLAY['MIN_LINE_HEIGHT']
            font = DISPLAY['FONT']
            max_width = int(width * DISPLAY['MAX_WIDTH_RATIO'])
            
            # Start with timestamp and description
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
            
            font_scale = min(height * 0.03 / min_line_height, DISPLAY['FONT_SCALE']['CAPTION'])
            
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
                # Find appropriate transcript segment
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
                    
                    # Split transcript into lines
                    current_line = []
                    for word in current_transcript_text.split():
                        test_line = ' '.join(current_line + [word])
                        (text_width, _), _ = cv2.getTextSize(test_line, font, font_scale, 1)
                        
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
            
            # Create overlay for text background
            overlay = frame.copy()
            cv2.rectangle(overlay, 
                         (margin, margin),
                         (width - margin, margin + top_box_height),
                         (0, 0, 0),
                         -1)
            
            # Blend background overlay with original frame
            alpha = DISPLAY['OVERLAY_ALPHA']
            frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
            
            # Draw text directly on frame
            y = margin + padding + line_height
            for line in top_lines:
                cv2.putText(frame,
                           line,
                           (margin + padding, y),
                           font,
                           font_scale,
                           DISPLAY['TEXT_COLOR']['WHITE'],
                           1,
                           cv2.LINE_AA)
                y += line_height
            
            # If we have speech transcription, add bottom overlay
            if bottom_lines:
                bottom_line_count = len(bottom_lines)
                bottom_box_height = bottom_line_count * line_height + 2 * padding
                
                # Add speech transcription text (centered with tight background)
                y = height - bottom_box_height - margin + padding + line_height
                for line in bottom_lines:
                    # Calculate text width for centering and background
                    (text_width, text_height), _ = cv2.getTextSize(line, font, font_scale, 1)
                    x = (width - text_width) // 2  # Center the text
                    
                    # Create tight background just for this line
                    bg_x1 = x - padding
                    bg_x2 = x + text_width + padding
                    bg_y1 = y - text_height - padding
                    bg_y2 = y + padding
                    
                    # Draw background rectangle with transparency
                    overlay = frame.copy()
                    cv2.rectangle(overlay,
                                (bg_x1, bg_y1),
                                (bg_x2, bg_y2),
                                (0, 0, 0),
                                -1)
                    frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
                    
                    # Draw text
                    cv2.putText(frame,
                               line,
                               (x, y),
                               font,
                               font_scale,
                               DISPLAY['TEXT_COLOR']['WHITE'],
                               1,
                               cv2.LINE_AA)
                    y += line_height * 1.5  # Increase spacing between lines
            
            out.write(frame)
            frame_count += 1
            pbar.update(1)

    video.release()
    out.release()
    # Create summary intro clip with debug metadata
    summary_path, summary_duration = create_summary_clip(
        summary, 
        frame_width, 
        frame_height, 
        fps,
        debug=debug,
        metadata=debug_metadata
    )
    
    # Concatenate videos and add audio if present
    final_output = video_utils.concatenate_videos(
        video_paths=[summary_path, temp_output],
        output_path=final_output,
        has_audio=has_audio,
        audio_source=video_path if has_audio else None,
        audio_delay=summary_duration
    )
    
    # Clean up temporary files
    os.remove(temp_output)
    os.remove(summary_path)
    
    # Convert to web-compatible format
    web_output = video_utils.convert_to_web_format(final_output)
    
    # Remove the non-web version
    os.remove(final_output)
    
    return web_output

def process_video_web(video_file, use_frame_selection=False, use_synthesis_captions=False, use_local_llm=False, show_transcriptions=False, debug=False):
    """Process video through web interface."""
    video_path = video_file.name
    
    # Skip if this is an output file
    if os.path.basename(video_path).startswith('captioned_') or video_path.endswith('_web.mp4'):
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
            
            # Determine video type based on duration
            video_type = "Short (<30s)" if video_duration < 30 else "Medium (30-90s)" if video_duration < 90 else "Medium-long (90-150s)" if video_duration < 150 else "Long (>150s)"
            
            # Calculate target captions
            if video_duration > 150:
                num_captions = max(15, frame_count // 2)
                caption_calc = f"max(15, {frame_count} // 2)"
            elif video_duration > 90:
                base_captions = int(video_duration / 6)
                num_captions = min(int(base_captions * 1.25), frame_count // 2)
                num_captions = max(9, num_captions)
                caption_calc = f"min(max(9, {base_captions} * 1.25), {frame_count} // 2)"
            else:
                if video_duration < 30:
                    target_captions = int(video_duration / 3.5)
                    num_captions = min(100, max(4, target_captions))
                    caption_calc = f"min(100, max(4, {video_duration} / 3.5))"
                else:
                    caption_interval = 5.0 + (video_duration - 30) * (7.0 - 5.0) / (90 - 30)
                    target_captions = int(video_duration / caption_interval)
                    max_captions = min(int(video_duration / 4), frame_count // 3)
                    num_captions = min(target_captions, max_captions)
                    num_captions = max(8, num_captions)
                    caption_calc = f"min(max(8, {target_captions}), {max_captions})"
            
            # Extract relevant metadata
            debug_metadata = {
                'Video Info': {
                    'Duration': f"{video_duration:.1f}s",
                    'Resolution': f"{metadata.get('streams', [{}])[0].get('width', '')}x{metadata.get('streams', [{}])[0].get('height', '')}",
                    'FPS': f"{eval(metadata.get('streams', [{}])[0].get('r_frame_rate', '0/1')):.1f}",
                    'Size': f"{int(int(metadata.get('format', {}).get('size', 0)) / 1024 / 1024)}MB"
                },
                'Caption Info': {
                    'Video Type': video_type,
                    'Target Captions': str(num_captions),
                    'Captions/Second': f"{num_captions/video_duration:.2f}",
                    'Calculation': caption_calc.replace(' ', '')  # Remove spaces to make it more compact
                }
            }
            
        except Exception as e:
            debug_metadata = {'Error': 'Failed to extract metadata'}

    # Transcribe the video
    transcript = transcribe_video(video_path)

    # Select frames based on the chosen method
    if use_frame_selection:
        frame_numbers = process_video(video_path)
    else:
        frame_numbers = list(range(0, frame_count, 50))

    # Describe frames
    descriptions = describe_frames(video_path, frame_numbers)

    # Generate summary and captions
    synthesis_output, synthesis_captions = summarize_with_hosted_llm(
        transcript, 
        descriptions, 
        video_duration,
        use_local_llm=use_local_llm,
        video_path=video_path
    )
    
    if synthesis_output is None or (use_synthesis_captions and synthesis_captions is None):
        return "Error: Failed to generate synthesis. Please try again.", None, None
    
    # Extract summary from synthesis output
    summary_match = re.search(r'<summary>(.*?)</summary>', synthesis_output, re.DOTALL)
    if not summary_match:
        return "Error: Failed to extract summary from synthesis output.", None, None
    
    summary = summary_match.group(1).strip()

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

    # Save output to JSON file
    save_output(
        video_path, 
        frame_count, 
        transcript, 
        descriptions, 
        summary, 
        total_run_time,
        synthesis_output,
        synthesis_captions,
        use_local_llm=use_local_llm
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
            continue
        video_files.append(f)
    
    if not video_files:
        return

    # Keep track of successfully processed videos
    processed_videos = set()
    
    # Process each video with retries
    for i, video_file in enumerate(video_files, 1):
        if video_file in processed_videos:
            continue
            
        video_path = os.path.join(folder_path, video_file)
        
        # Try processing with up to 10 retries
        max_retries = 10
        success_marker = f"Captioned video saved to: outputs/captioned_{os.path.splitext(video_file)[0]}_web.mp4"
        
        for attempt in range(max_retries):
            try:
                if attempt > 0:
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
                    break  # Success, exit retry loop
                else:
                    if attempt == max_retries - 1:
                        pass
                
            except Exception as e:
                # Check if the success marker is in the error message
                if str(e).find(success_marker) != -1:
                    processed_videos.add(video_file)
                    break
                elif attempt == max_retries - 1:  # Last attempt
                    pass

def recontextualize_summary(summary: str, metadata_context: str, use_local_llm: bool = False) -> str:
    """Recontextualize just the summary using metadata context."""
    # Get the recontextualization prompt
    prompt = get_recontextualization_prompt()
    
    # Create the prompt with context
    prompt_content = prompt.format(
        metadata=metadata_context,
        summary=summary
    )

    max_retries = 3
    for attempt in range(max_retries):
        completion = get_llm_completion(prompt_content, "", use_local_llm=use_local_llm)
        
        if completion:
            # Extract summary from completion using same regex pattern style
            summary_match = re.search(r'<recontextualized_summary>(.*?)</recontextualized_summary>', completion, re.DOTALL)
            if summary_match:
                new_summary = summary_match.group(1).strip()
                return new_summary
        
        if attempt < max_retries - 1:
            time.sleep(2 * (attempt + 1))
    
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
            process_folder(args.path, args)
        else:
            # Process single video
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
    else:
        pass

if __name__ == "__main__":
    main()
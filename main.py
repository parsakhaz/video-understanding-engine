#!/usr/bin/env python3
import argparse, cv2, warnings, os, time, json, re, subprocess, traceback
from PIL import Image
import torch
from tqdm import tqdm
import numpy as np
from datetime import datetime
from openai import OpenAI
from dotenv import load_dotenv
from frame_selection import process_video
import gradio as gr
from difflib import SequenceMatcher
from prompts import get_frame_description_prompt, get_recontextualization_prompt, get_final_summary_prompt
import video_utils
from config import VIDEO_SETTINGS, FRAME_SELECTION, MODEL_SETTINGS, DISPLAY, RETRY, CAPTION, SIMILARITY, CAPTION_DENSITY
from model_loader import model_context

class ErrorCollector:
    def __init__(self):
        self.warnings = []
        self.errors = []
        self.retries = {}
        
    def add_warning(self, message, source=None):
        warning = {"message": str(message), "source": source, "timestamp": datetime.now().isoformat()}
        self.warnings.append(warning)
        
    def add_error(self, error, source=None):
        error_info = {
            "message": str(error),
            "traceback": traceback.format_exc(),
            "source": source,
            "timestamp": datetime.now().isoformat()
        }
        self.errors.append(error_info)
        
    def add_retry(self, operation, attempt, error=None):
        if operation not in self.retries:
            self.retries[operation] = []
        retry_info = {
            "attempt": attempt,
            "error": str(error) if error else None,
            "timestamp": datetime.now().isoformat()
        }
        self.retries[operation].append(retry_info)
        
    def get_logs(self):
        return {
            "warnings": self.warnings,
            "errors": self.errors,
            "retries": self.retries
        }

error_collector = ErrorCollector()

def similar(a, b, threshold=SIMILARITY['THRESHOLD']): return SequenceMatcher(None, a, b).ratio() > threshold

def clean_transcript(segments):
    noise_phrases, boilerplate_phrases = SIMILARITY['NOISE_PHRASES'], SIMILARITY['BOILERPLATE_PHRASES']
    filtered_segments = [s for s in segments if not any(p in s["text"].strip().lower() for p in noise_phrases) and sum(1 for p in boilerplate_phrases if p in s["text"].strip().lower()) < 2]
    cleaned_segments = []
    for i, current in enumerate(filtered_segments):
        if i > 0 and similar(current["text"], filtered_segments[i-1]["text"]): continue
        if any(similar(current["text"], next_seg["text"]) for next_seg in filtered_segments[i+1:i+4]): continue
        cleaned_segments.append(current)
    return cleaned_segments if cleaned_segments else [{"start": 0, "end": 0, "text": ""}]

def get_video_duration(video_path): return video_utils.get_video_duration(video_path)
def extract_audio(video_path): return video_utils.extract_audio(video_path)

def process_transcript(segments):
    return [{"start": s.get("start"), "end": s.get("end"), "text": s.get("text", "").strip()} 
            for s in segments if s.get("start") is not None and s.get("end") is not None and s.get("text", "").strip()]

def transcribe_video(video_path):
    duration = get_video_duration(video_path)
    if duration is None:
        error_collector.add_error("Could not get video duration", source="transcribe_video")
        return [{"start": 0, "end": 0, "text": ""}]
    
    has_audio = len(subprocess.run(['ffprobe', '-i', video_path, '-show_streams', '-select_streams', 'a', '-loglevel', 'error'], capture_output=True, text=True).stdout.strip()) > 0
    if not has_audio:
        error_collector.add_warning("No audio stream found in video", source="transcribe_video")
        return [{"start": 0, "end": 0, "text": ""}]
    
    audio_path = extract_audio(video_path)
    if not audio_path:
        error_collector.add_error("Failed to extract audio from video", source="transcribe_video")
        return [{"start": 0, "end": 0, "text": ""}]
    
    try:
        with model_context("whisper") as model:
            if model is None:
                error_collector.add_error("Failed to load Whisper model", source="transcribe_video")
                return [{"start": 0, "end": 0, "text": ""}]
            result = model.transcribe(audio_path)
            processed_transcript = process_transcript(result["segments"])
            return clean_transcript(processed_transcript)
    except Exception as e:
        error_collector.add_error(e, source="transcribe_video")
        return [{"start": 0, "end": 0, "text": ""}]
    finally:
        if os.path.exists(audio_path): os.remove(audio_path)

def describe_frames(video_path, frame_numbers):
    with model_context("moondream") as model_tuple:
        if model_tuple is None:
            error_collector.add_error("Failed to load Moondream model", source="describe_frames")
            return []
        model, tokenizer = model_tuple
        prompt = get_frame_description_prompt()
        try:
            props = video_utils.get_video_properties(video_path)
            fps = props['fps']
            frames = []
            video = cv2.VideoCapture(video_path)
            for frame_number in frame_numbers:
                video.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
                success, frame = video.read()
                if success:
                    frames.append((frame_number, frame, frame_number / fps))
                else:
                    error_collector.add_warning(f"Failed to read frame {frame_number}", source="describe_frames")
            video.release()
            
            if not frames:
                error_collector.add_error("No frames could be read from video", source="describe_frames")
                return []
                
            results = []
            for i in tqdm(range(0, len(frames), FRAME_SELECTION['BATCH_SIZE']), desc="Processing batches"):
                try:
                    batch_frames = frames[i:i+FRAME_SELECTION['BATCH_SIZE']]
                    batch_images = [Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)) for _, frame, _ in batch_frames]
                    batch_answers = model.batch_answer(images=batch_images, prompts=[prompt] * len(batch_images), tokenizer=tokenizer)
                    results.extend((frame_number, timestamp, answer) for (frame_number, _, timestamp), answer in zip(batch_frames, batch_answers))
                except Exception as e:
                    error_collector.add_error(e, source=f"describe_frames_batch_{i}")
            return results
        except Exception as e:
            error_collector.add_error(e, source="describe_frames")
            return []

def display_described_frames(video_path, descriptions):
    video = cv2.VideoCapture(video_path)
    current_index = 0
    def add_caption_to_frame(frame, caption):
        height, width = frame.shape[:2]
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale, font_thickness, line_spacing, margin = 0.7, 1, 10, 10
        lines, current_line = [], []
        for word in caption.split():
            test_line = ' '.join(current_line + [word])
            if cv2.getTextSize(test_line, font, font_scale, font_thickness)[0][0] <= width - 2 * margin: 
                current_line.append(word)
            else:
                lines.append(' '.join(current_line))
                current_line = [word]
        if current_line: lines.append(' '.join(current_line))
        text_height = cv2.getTextSize('Tg', font, font_scale, font_thickness)[0][1]
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
            if not success: break
            if frame.shape[0] > 800: frame = cv2.resize(frame, (int(frame.shape[1] * 800/frame.shape[0]), 800))
            frame_with_caption = add_caption_to_frame(frame, f"[{timestamp:.2f}s] Frame {frame_number}: {description}")
            cv2.imshow('Video Frames', frame_with_caption)
            cv2.setWindowTitle('Video Frames', f"Frame {frame_number}")
            key = cv2.waitKey(100) & 0xFF
            if key == ord('q'): break
            elif key in (83, 32): current_index = min(current_index + 1, len(descriptions) - 1)
            elif key == 81: current_index = max(current_index - 1, 0)
    except KeyboardInterrupt: pass
    finally:
        video.release()
        cv2.destroyAllWindows()

def get_frame_count(video_path):
    video = cv2.VideoCapture(video_path)
    if not video.isOpened(): return 0
    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    video.release()
    return frame_count

def save_output(video_path, frame_count, transcript=None, descriptions=None, summary=None, total_run_time=None, synthesis_output=None, synthesis_captions=None, use_local_llm=False, success=False, caption_density=CAPTION_DENSITY['DEFAULT']):
    """Save output to a log file. Creates a new file at start of run, updates it at the end."""
    os.makedirs('logs', exist_ok=True)
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    
    # For initial log creation (no transcript/descriptions/summary)
    if transcript is None and descriptions is None and summary is None:
        timestamp = datetime.now().isoformat().replace(':', '-')
        filename = f"logs/{timestamp}_{video_name}.json"
        output = {
            "processing_info": {
                "timestamp": datetime.now().isoformat(),
                "success": success
            },
            "video_metadata": {
                "path": video_path,
                "frame_count": frame_count
            }
        }
        with open(filename, 'w') as f:
            json.dump(output, f, indent=2)
        return

    # For updating existing log
    existing_logs = sorted([f for f in os.listdir('logs') if f.endswith(f'_{video_name}.json')], reverse=True)
    if existing_logs and time.time() - os.path.getctime(os.path.join('logs', existing_logs[0])) < 3600:  # Within last hour
        filename = os.path.join('logs', existing_logs[0])
        try:
            with open(filename, 'r') as f:
                output = json.load(f)
        except Exception:
            # If reading fails, start fresh
            timestamp = datetime.now().isoformat().replace(':', '-')
            filename = f"logs/{timestamp}_{video_name}.json"
            output = {}
    else:
        # Shouldn't happen as we should have created a log at start
        timestamp = datetime.now().isoformat().replace(':', '-')
        filename = f"logs/{timestamp}_{video_name}.json"
        output = {}

    metadata, metadata_context = video_utils.get_video_metadata(video_path)
    props = video_utils.get_video_properties(video_path)
    fps, width, height = props['fps'], props['frame_width'], props['frame_height']
    video_duration = frame_count / fps if fps else None
    video_type = "Short (<30s)" if video_duration < VIDEO_SETTINGS['MIN_VIDEO_DURATION'] else "Medium (30-90s)" if video_duration < VIDEO_SETTINGS['MEDIUM_VIDEO_DURATION'] else "Medium-long (90-150s)" if video_duration < VIDEO_SETTINGS['LONG_VIDEO_DURATION'] else "Long (>150s)"
    
    if video_duration > VIDEO_SETTINGS['LONG_VIDEO_DURATION']:
        # For long videos, base it on duration not frame count
        base_captions = int(video_duration / 8)  # One caption every 8 seconds
        num_captions = min(max(15, base_captions), 120)  # Cap at 120 captions
        caption_calc = f"min(max(15, {video_duration} / 8), 120)"
    elif video_duration > VIDEO_SETTINGS['MEDIUM_VIDEO_DURATION']:
        base_captions = int(video_duration / CAPTION['LONG_VIDEO']['INTERVAL_RATIO'])
        num_captions = min(int(base_captions * 1.25), frame_count // 2)
        num_captions = max(CAPTION['MEDIUM_VIDEO']['MIN_CAPTIONS'], num_captions)
        caption_calc = f"min(max({CAPTION['MEDIUM_VIDEO']['MIN_CAPTIONS']}, {base_captions} * 1.25), {frame_count} // 2)"
    else:
        if video_duration < VIDEO_SETTINGS['MIN_VIDEO_DURATION']:
            target_captions = int(video_duration / CAPTION['SHORT_VIDEO']['INTERVAL'])
            num_captions = min(CAPTION['SHORT_VIDEO']['MAX_CAPTIONS'], max(CAPTION['SHORT_VIDEO']['MIN_CAPTIONS'], target_captions))
            caption_calc = f"min({CAPTION['SHORT_VIDEO']['MAX_CAPTIONS']}, max({CAPTION['SHORT_VIDEO']['MIN_CAPTIONS']}, {target_captions}))"
        else:
            caption_interval = (CAPTION['MEDIUM_VIDEO']['BASE_INTERVAL'] + (video_duration - VIDEO_SETTINGS['MIN_VIDEO_DURATION']) * (CAPTION['MEDIUM_VIDEO']['MAX_INTERVAL'] - CAPTION['MEDIUM_VIDEO']['BASE_INTERVAL']) / (VIDEO_SETTINGS['MEDIUM_VIDEO_DURATION'] - VIDEO_SETTINGS['MIN_VIDEO_DURATION']))
            target_captions = int(video_duration / caption_interval)
            max_captions = min(int(video_duration / 4), frame_count // 3)
            num_captions = min(target_captions, max_captions)
            num_captions = max(CAPTION['MEDIUM_VIDEO']['MIN_CAPTIONS'], num_captions)
            caption_calc = f"min(max({CAPTION['MEDIUM_VIDEO']['MIN_CAPTIONS']}, {target_captions}), {max_captions})"

    frame_description_prompt = get_frame_description_prompt() if descriptions else None
    gpu_memory = {}
    if torch.cuda.is_available():
        try:
            gpu_memory = {f"gpu_{i}": {"total": torch.cuda.get_device_properties(i).total_memory, "allocated": torch.cuda.memory_allocated(i), "cached": torch.cuda.memory_reserved(i)} for i in range(torch.cuda.device_count())}
        except Exception: pass

    cache_path = os.path.join('embedding_cache', f'{video_name}.npy')
    embedding_cache_exists = os.path.exists(cache_path)
    
    # Update the output dictionary with new information
    output.update({
        "processing_info": {
            "timestamp": datetime.now().isoformat(),
            "total_run_time": total_run_time,
            "gpu_memory": gpu_memory,
            "embedding_cache_status": {"exists": embedding_cache_exists, "path": cache_path if embedding_cache_exists else None},
            "success": success
        },
        "configuration": {"video_settings": VIDEO_SETTINGS, "frame_selection": FRAME_SELECTION, "model_settings": MODEL_SETTINGS, "display": DISPLAY, "retry": RETRY, "caption": CAPTION, "similarity": SIMILARITY},
        "prompts": {
            "frame_description": get_frame_description_prompt(),
            "recontextualization_template": get_recontextualization_prompt(),
            "final_summary": get_final_summary_prompt(),
            "synthesis": get_synthesis_prompt(len(descriptions) if descriptions else 0, video_duration, metadata_context=metadata_context, caption_density=caption_density),
            "actual_prompts_used": {
                "frame_description": get_frame_description_prompt(),
                "synthesis": synthesis_output,
                "metadata_context": metadata_context if metadata else None,
                "final_summary": get_final_summary_prompt() if video_duration > VIDEO_SETTINGS['LONG_VIDEO_DURATION'] else None
            }
        },
        "video_metadata": {"path": video_path, "frame_count": frame_count, "duration": video_duration, "fps": fps, "resolution": {"width": width, "height": height}, "ffprobe_data": metadata, "video_type": video_type},
        "caption_analysis": {"video_type": video_type, "density_multiplier": caption_density, "target_captions": num_captions, "captions_per_second": num_captions/video_duration if video_duration else None, "calculation_formula": caption_calc, "actual_captions_generated": len(synthesis_captions) if synthesis_captions else 0},
        "model_info": {
            "frame_description_prompt": frame_description_prompt,
            "whisper_model": MODEL_SETTINGS['WHISPER_MODEL'],
            "clip_model": "ViT-SO400M-14-SigLIP-384",
            "synthesis_model": MODEL_SETTINGS['LOCAL_LLM_MODEL'] if use_local_llm else MODEL_SETTINGS['GPT_MODEL'],
            "using_local_llm": use_local_llm
        },
        "processing_stages": {"frame_selection": {"method": "clip" if os.environ.get("USE_FRAME_SELECTION") else "regular_sampling", "parameters": {"novelty_threshold": FRAME_SELECTION['NOVELTY_THRESHOLD'], "min_skip": FRAME_SELECTION['MIN_SKIP_FRAMES'], "n_clusters": FRAME_SELECTION['NUM_CLUSTERS']} if os.environ.get("USE_FRAME_SELECTION") else {"sampling_interval": VIDEO_SETTINGS['FRAME_SAMPLING_INTERVAL']}}, "transcription": {"segments": transcript, "total_segments": len(transcript) if transcript else 0}, "frame_descriptions": [{"frame_number": frame, "timestamp": timestamp, "description": desc} for frame, timestamp, desc in (descriptions or [])], "synthesis": {"raw_output": synthesis_output, "captions": [{"timestamp": timestamp, "text": text} for timestamp, text in (synthesis_captions or [])], "summary": summary}},
        "error_log": error_collector.get_logs()
    })

    with open(filename, 'w') as f:
        json.dump(output, f, indent=2)

def get_synthesis_prompt(num_keyframes: int, video_duration: float, metadata_context: str = "", caption_density: float = CAPTION_DENSITY['DEFAULT']) -> str:
    if video_duration > VIDEO_SETTINGS['LONG_VIDEO_DURATION']:
        num_captions = max(CAPTION['LONG_VIDEO']['MIN_CAPTIONS'], num_keyframes // 2)
    elif video_duration > VIDEO_SETTINGS['MEDIUM_VIDEO_DURATION']:
        base_captions = int(video_duration / CAPTION['LONG_VIDEO']['INTERVAL_RATIO'])
        num_captions = min(int(base_captions * 1.25), num_keyframes // 2)
        num_captions = max(CAPTION['MEDIUM_VIDEO']['MIN_CAPTIONS'], num_captions)
    else:
        if video_duration < VIDEO_SETTINGS['MIN_VIDEO_DURATION']:
            target_captions = int(video_duration / CAPTION['SHORT_VIDEO']['INTERVAL'])
            num_captions = min(CAPTION['SHORT_VIDEO']['MAX_CAPTIONS'], max(CAPTION['SHORT_VIDEO']['MIN_CAPTIONS'], target_captions))
        else:
            caption_interval = (CAPTION['MEDIUM_VIDEO']['BASE_INTERVAL'] + (video_duration - VIDEO_SETTINGS['MIN_VIDEO_DURATION']) * (CAPTION['MEDIUM_VIDEO']['MAX_INTERVAL'] - CAPTION['MEDIUM_VIDEO']['BASE_INTERVAL']) / (VIDEO_SETTINGS['MEDIUM_VIDEO_DURATION'] - VIDEO_SETTINGS['MIN_VIDEO_DURATION']))
            target_captions = int(video_duration / caption_interval)
            max_captions = min(int(video_duration / 4), num_keyframes // 3)
            num_captions = min(target_captions, max_captions)
            num_captions = max(CAPTION['MEDIUM_VIDEO']['MIN_CAPTIONS'], num_captions)

    # Apply density multiplier to final caption count
    num_captions = int(num_captions * caption_density)

    metadata_section = f"""You are tasked with summarizing and captioning a video based on its transcript and frame descriptions, with the following metadata context about the video's origin and purpose:\n\n{metadata_context}\n\nThis metadata (any resembling a title or description especially) might inform your understanding of the video, but don't lean on it too heavily, or mention it explicitly in your output.""" if metadata_context else "You are tasked with summarizing and captioning a video based on its transcript and frame descriptions."
    
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
   - Focus on clearly visible elements and their narrative significance
   - Use passive voice when action source is ambiguous
   - Cross-reference transcript before assigning names/identities
4. If frame descriptions conflict or make assumptions:
   - Favor the most conservative interpretation
   - Omit speculative details
   - Focus on concrete visual elements and their meaning
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
1. Generate exactly {num_captions} captions, with a minimum of 2 seconds between each caption.
2. Each caption MUST:
   - Start with a timestamp in [X.X] format
   - Use the EARLIEST timestamp where a scene or action that is clearly visible begins
   - Be 20-50 words long
   - Focus on observable events and context
   - Avoid attributing speech or actions unless explicitly clear
3. Timestamps should timed accurately throughout this video
4. Focus on what is definitively shown or heard, not assumptions. Overarching themes and emotions are more important than specific details.
5. IMPORTANT: When multiple frames describe the same scene or action, use the EARLIEST timestamp
6. Default to neutral terms like "a person" or "someone" when identities are unclear.
7. Use passive voice when action source is ambiguous.
8. Describe only what is visually observable and keep descriptions objective

Input sections:
<transcript>
Timestamped transcript of this video
</transcript>

<frame_descriptions>
Timestamped descriptions of key frames
</frame_descriptions>"""

def chunk_video_data(transcript: list, descriptions: list, chunk_duration: int = 60) -> list:
    chunks = []
    current_chunk_start = 0
    min_descriptions_per_chunk = 4
    
    while current_chunk_start < descriptions[-1][1]:
        current_chunk_end = current_chunk_start + chunk_duration
        chunk_descriptions = [desc for desc in descriptions if current_chunk_start <= desc[1] < current_chunk_end]
        
        while len(chunk_descriptions) < min_descriptions_per_chunk and current_chunk_end < descriptions[-1][1]:
            current_chunk_end += 15
            chunk_descriptions = [desc for desc in descriptions if current_chunk_start <= desc[1] < current_chunk_end]
        
        chunk_transcript = [seg for seg in transcript if (seg["start"] >= current_chunk_start and seg["start"] < current_chunk_end) or (seg["end"] > current_chunk_start and seg["end"] <= current_chunk_end)]
        
        if chunk_descriptions: chunks.append((chunk_transcript, chunk_descriptions))
        current_chunk_start = current_chunk_end
    
    return chunks

def summarize_with_hosted_llm(transcript: list, descriptions: list, video_duration: float, use_local_llm: bool = False, video_path: str = None, caption_density: float = CAPTION_DENSITY['DEFAULT']):
    # Clear CUDA cache at start of summarization
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        
    load_dotenv()
    metadata_context = ""
    if video_path:
        for attempt in range(3):
            try:
                metadata_json = subprocess.check_output(['ffprobe', '-v', 'quiet', '-print_format', 'json', '-show_format', '-show_streams', video_path]).decode('utf-8')
                metadata = json.loads(metadata_json)
                format_metadata = metadata.get('format', {}).get('tags', {})
                metadata_context = "Video Metadata:\n"
                if format_metadata.get('title'): metadata_context += f"- Title: {format_metadata['title']}\n"
                if format_metadata.get('artist'): metadata_context += f"- Creator: {format_metadata['artist']}\n"
                metadata_context += f"- Duration: {float(metadata.get('format', {}).get('duration', 0)):.2f} seconds\n"
                metadata_context += '\n'.join(f"- {k}: {v}" for k, v in format_metadata.items() if k not in ['title', 'artist'] and v)
                break
            except Exception as e:
                error_collector.add_retry("metadata_extraction", attempt, error=e)
                if attempt < 2:
                    time.sleep(2 * (attempt + 1))
                else:
                    error_collector.add_warning("Failed to extract video metadata", source="summarize_with_hosted_llm")
                    metadata_context = ""

    is_long_video = video_duration > 150
    initial_synthesis = initial_captions = None
    
    if is_long_video:
        chunks = chunk_video_data(transcript, descriptions)
        all_summaries, all_captions = [], []
        
        for chunk_idx, (chunk_transcript, chunk_descriptions) in enumerate(chunks):
            # Clear CUDA cache before each chunk
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            synthesis_prompt = get_synthesis_prompt(len(chunk_descriptions), video_duration, metadata_context, caption_density)
            timestamped_transcript = '\n'.join(f"[{s['start']:.2f}s-{s['end']:.2f}s] {s['text']}" for s in chunk_transcript)
            frame_descriptions = '\n'.join(f"[{t:.2f}s] Frame {f}: {d}" for f, t, d in chunk_descriptions)
            user_content = f"<transcript>\n{timestamped_transcript}\n</transcript>\n\n<frame_descriptions>\n{frame_descriptions}\n</frame_descriptions>"
            
            for attempt in range(5):
                completion = get_llm_completion(synthesis_prompt, user_content, use_local_llm)
                if completion is None:
                    error_collector.add_retry(f"chunk_synthesis_{chunk_idx}", attempt)
                    if attempt < 4:
                        time.sleep(2 * (attempt + 1))
                    continue
                
                chunk_summary, chunk_captions = parse_synthesis_output(completion)
                if chunk_summary and chunk_captions:
                    all_summaries.append(chunk_summary)
                    all_captions.extend(chunk_captions)
                    break
                elif attempt < 4:
                    error_collector.add_retry(f"chunk_synthesis_parsing_{chunk_idx}", attempt)
                    time.sleep(2 * (attempt + 1))
                else:
                    error_collector.add_error(f"Failed to parse chunk {chunk_idx} synthesis output", source="summarize_with_hosted_llm")

        if not all_summaries or not all_captions:
            error_collector.add_error("Failed to generate summaries or captions for all chunks", source="summarize_with_hosted_llm")
            return None, None

        final_summary_prompt = get_final_summary_prompt()
        chunk_summaries_content = '\n\n'.join(f"Chunk {i+1}:\n{s}" for i, s in enumerate(all_summaries))
        timestamped_transcript = '\n'.join(f"[{s['start']:.2f}s-{s['end']:.2f}s] {s['text']}" for s in transcript)
        final_summary_content = f"<chunk_summaries>\n{chunk_summaries_content}\n</chunk_summaries>\n\n<transcript>\n{timestamped_transcript}\n</transcript>"

        final_summary = None
        for attempt in range(5):
            final_summary = get_llm_completion(final_summary_prompt, final_summary_content, use_local_llm)
            if final_summary:
                break
            error_collector.add_retry("final_summary", attempt)
            if attempt < 4:
                time.sleep(2 * (attempt + 1))
        
        if not final_summary:
            error_collector.add_error("Failed to generate final summary", source="summarize_with_hosted_llm")
            return None, None

        if metadata_context:
            final_summary = recontextualize_summary(final_summary, metadata_context, use_local_llm)

        initial_synthesis = f"<summary>\n{final_summary}\n</summary>\n\n<captions>\n" + '\n'.join(f"<caption>[{t:.1f}] {txt}</caption>" for t, txt in sorted(all_captions, key=lambda x: x[0])) + "\n</captions>"
        initial_summary, initial_captions = parse_synthesis_output(initial_synthesis)
        
        if not (initial_summary and initial_captions):
            for attempt in range(3):
                if metadata_context:
                    final_summary = recontextualize_summary(final_summary, metadata_context, use_local_llm)
                initial_synthesis = f"<summary>\n{final_summary}\n</summary>\n\n<captions>\n" + '\n'.join(f"<caption>[{t:.1f}] {txt}</caption>" for t, txt in sorted(all_captions, key=lambda x: x[0])) + "\n</captions>"
                initial_summary, initial_captions = parse_synthesis_output(initial_synthesis)
                if initial_summary and initial_captions:
                    break
                error_collector.add_retry("final_synthesis_parsing", attempt)
                if attempt < 2:
                    time.sleep(2 * (attempt + 1))
    else:
        synthesis_prompt = get_synthesis_prompt(len(descriptions), video_duration, metadata_context, caption_density)
        timestamped_transcript = '\n'.join(f"[{s['start']:.2f}s-{s['end']:.2f}s] {s['text']}" for s in transcript)
        frame_descriptions = '\n'.join(f"[{t:.2f}s] Frame {f}: {d}" for f, t, d in descriptions)
        user_content = f"<transcript>\n{timestamped_transcript}\n</transcript>\n\n<frame_descriptions>\n{frame_descriptions}\n</frame_descriptions>"
        
        for attempt in range(5):
            completion = get_llm_completion(synthesis_prompt, user_content, use_local_llm)
            if completion is None:
                error_collector.add_retry("synthesis", attempt)
                if attempt < 4:
                    time.sleep(2 * (attempt + 1))
                continue
            
            initial_summary, initial_captions = parse_synthesis_output(completion)
            if initial_summary and initial_captions:
                initial_synthesis = completion
                break
            error_collector.add_retry("synthesis_parsing", attempt)
            if attempt < 4:
                time.sleep(2 * (attempt + 1))
    
    if initial_synthesis and initial_captions:
        if metadata_context:
            initial_summary = recontextualize_summary(initial_summary, metadata_context, use_local_llm)
            initial_synthesis = f"<summary>\n{initial_summary}\n</summary>\n\n<captions>\n" + '\n'.join(f"<caption>[{t:.1f}] {txt}</caption>" for t, txt in initial_captions) + "\n</captions>"
        return initial_synthesis, initial_captions

    error_collector.add_error("Failed to generate synthesis output", source="summarize_with_hosted_llm")
    return None, None

def get_llm_completion(prompt: str, content: str, use_local_llm: bool = False) -> str:
    try:
        if use_local_llm:
            # Clear CUDA cache before loading model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            with model_context("llama") as pipeline:
                if pipeline is None:
                    error_collector.add_error("Failed to load local LLM model", source="get_llm_completion")
                    return None
                try:
                    outputs = pipeline([{"role": "system", "content": prompt}, {"role": "user", "content": content}], max_new_tokens=8000, temperature=MODEL_SETTINGS['TEMPERATURE'])
                    # Clear CUDA cache after inference
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        
                    raw_response = str(outputs[0]["generated_text"])
                    for pattern in ["'role': 'assistant', 'content': '", '"role": "assistant", "content": "', "'role': 'assistant', 'content': \"", "role': 'assistant', 'content': '", "role\": \"assistant\", \"content\": \""]:
                        assistant_start = raw_response.find(pattern)
                        if assistant_start != -1:
                            content_start = assistant_start + len(pattern)
                            for end_pattern in ["'}", '"}', "\"}", "'}]", '"}]']:
                                content_end = raw_response.find(end_pattern, content_start)
                                if content_end != -1:
                                    content = raw_response[content_start:content_end].encode().decode('unicode_escape')
                                    return content
                    if "<summary>" in raw_response and "</summary>" in raw_response: return raw_response
                    error_collector.add_error("Failed to parse local LLM response", source="get_llm_completion_local")
                    return None
                except Exception as e:
                    error_collector.add_error(e, source="get_llm_completion_local")
                    return None
        
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            error_collector.add_error("OpenAI API key not found", source="get_llm_completion")
            return None

        for attempt in range(RETRY['MAX_RETRIES']):
            try:
                client = OpenAI(api_key=api_key)
                completion = client.chat.completions.create(model=MODEL_SETTINGS['GPT_MODEL'], messages=[{"role": "system", "content": prompt}, {"role": "user", "content": content}], temperature=MODEL_SETTINGS['TEMPERATURE'])
                return completion.choices[0].message.content.strip()
            except Exception as e:
                error_collector.add_retry("openai_completion", attempt, error=e)
                if attempt < RETRY['MAX_RETRIES'] - 1:
                    time.sleep(RETRY['INITIAL_DELAY'] * (attempt + 1))
                else:
                    error_collector.add_error(e, source="get_llm_completion_openai")
                    return None
    except Exception as e:
        error_collector.add_error(e, source="get_llm_completion")
        return None

def parse_synthesis_output(output: str) -> tuple:
    """Parse the synthesis output to extract summary and captions."""
    try:
        error_collector.add_warning(f"Raw LLM output to parse: {output[:200]}...", source="parse_synthesis_output")
        
        # Extract summary and captions (both required)
        summary_match = re.search(r'<summary>(.*?)</summary>', output, re.DOTALL)
        if not summary_match:
            error_collector.add_error("No summary tags found in output", source="parse_synthesis_output")
            error_collector.add_warning(f"Failed to match summary in: {output[:500]}...", source="parse_synthesis_output")
            return None, None
            
        summary = summary_match.group(1).strip()
        error_collector.add_warning(f"Successfully extracted summary: {summary[:100]}...", source="parse_synthesis_output")
        
        captions_match = re.search(r'<captions>(.*?)</captions>', output, re.DOTALL)
        if not captions_match:
            error_collector.add_error("No captions tags found in output", source="parse_synthesis_output")
            error_collector.add_warning(f"Failed to match captions in: {output[:500]}...", source="parse_synthesis_output")
            return None, None
            
        captions_text = captions_match.group(1).strip()
        error_collector.add_warning(f"Successfully extracted captions text: {captions_text[:100]}...", source="parse_synthesis_output")
        
        # Extract and validate caption timestamps/text
        caption_list = []
        caption_matches = re.findall(r'<caption>\[([\d.]+)s?\](.*?)</caption>', captions_text, re.DOTALL)
        error_collector.add_warning(f"Found {len(caption_matches)} caption matches", source="parse_synthesis_output")
        
        for timestamp_str, text in caption_matches:
            try:
                text = text.strip()
                if text:
                    timestamp = float(timestamp_str)
                    caption_list.append((timestamp, text))
                    error_collector.add_warning(f"Successfully parsed caption: [{timestamp}] {text[:50]}...", source="parse_synthesis_output")
            except ValueError as e:
                error_collector.add_error(f"Failed to parse timestamp '{timestamp_str}': {str(e)}", source="parse_synthesis_output")
                continue
                
        if not caption_list:
            error_collector.add_error("No valid captions found in synthesis output", source="parse_synthesis_output")
            return None, None
            
        error_collector.add_warning(f"Successfully parsed {len(caption_list)} captions", source="parse_synthesis_output")
        return summary, caption_list
    except Exception as e:
        error_collector.add_error(f"Exception in parse_synthesis_output: {str(e)}", source="parse_synthesis_output")
        error_collector.add_warning(f"Failed to parse output: {output[:500]}...", source="parse_synthesis_output")
        return None, None

def create_summary_clip(summary: str, width: int, height: int, fps: int, debug: bool = False, metadata: dict = None) -> str:
    """Create a video clip with centered summary text."""
    # Setup video writer
    duration = (len(summary.split()) / 400) * 60
    total_frames = int(duration * fps)
    temp_summary_path = os.path.join('outputs', f'temp_summary_{int(time.time())}.mp4')
    os.makedirs('outputs', exist_ok=True)
    out = cv2.VideoWriter(temp_summary_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
    
    # Configure fonts and text settings
    font = cv2.FONT_HERSHEY_SIMPLEX
    attr_font_scale = min(height * 0.014 / 20, 0.42)
    debug_font_scale = min(height * 0.0126 / 20, 0.35) if debug else None
    font_scale = min(height * 0.03 / 20, 0.8)
    line_height = max(25, int(height * 0.04))
    
    # Prepare attribution text
    attribution = "Local Video Understanding Engine - powered by Moondream 2B, CLIP, LLama 3.1 8b Instruct, and Whisper Large"
    if debug:
        attribution = "DEBUG " + attribution
    
    # Split summary into lines that fit width
    lines = []
    current_line = []
    max_width = int(width * 0.75)
    
    for word in summary.split():
        test_line = ' '.join(current_line + [word])
        if cv2.getTextSize(test_line, font, font_scale, 1)[0][0] <= max_width:
            current_line.append(word)
        else:
            if current_line:
                lines.append(' '.join(current_line))
            current_line = [word]
    if current_line:
        lines.append(' '.join(current_line))
        
    # Calculate vertical positioning
    total_height = len(lines) * line_height
    start_y = (height - total_height) // 2
    
    # Generate frames
    for _ in range(total_frames):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Draw attribution
        (text_width, _), _ = cv2.getTextSize(attribution, font, attr_font_scale, 1)
        attr_x = (width - text_width) // 2
        attr_y = height - 20
        
        if debug:
            debug_font = cv2.FONT_HERSHEY_COMPLEX_SMALL
            debug_text = "DEBUG"
            (debug_width, _), _ = cv2.getTextSize(debug_text, debug_font, attr_font_scale, 1)
            cv2.putText(frame, debug_text, (attr_x, attr_y), debug_font, attr_font_scale, (0, 0, 255), 1, cv2.LINE_AA)
            cv2.putText(frame, " " + attribution[6:], (attr_x + debug_width, attr_y), font, attr_font_scale, (255, 255, 255), 1, cv2.LINE_AA)
        else:
            cv2.putText(frame, attribution, (attr_x, attr_y), font, attr_font_scale, (255, 255, 255), 1, cv2.LINE_AA)
            
        # Draw debug metadata
        if debug and metadata:
            y_pos = 30
            for section_name, section_data in metadata.items():
                cv2.putText(frame, f"{section_name}:", (20, y_pos), font, debug_font_scale * 1.1, (255, 255, 255), 1, cv2.LINE_AA)
                y_pos += 25
                for key, value in section_data.items():
                    debug_text = f"  {key}: {value:.2f}" if isinstance(value, float) else f"  {key}: {value}"
                    cv2.putText(frame, debug_text, (20, y_pos), font, debug_font_scale, (200, 200, 200), 1, cv2.LINE_AA)
                    y_pos += 25
                y_pos += 10
                
        # Draw summary text
        y = start_y
        for line in lines:
            x = (width - cv2.getTextSize(line, font, font_scale, 1)[0][0]) // 2
            cv2.putText(frame, line, (x, y), font, font_scale, (255, 255, 255), 1, cv2.LINE_AA)
            y += line_height
            
        out.write(frame)
        
    out.release()
    return temp_summary_path, duration

def create_captioned_video(video_path: str, descriptions: list, summary: str, transcript: list, synthesis_captions: list = None, use_synthesis_captions: bool = False, show_transcriptions: bool = False, output_path: str = None, debug: bool = False, debug_metadata: dict = None) -> str:
    """Create a video with persistent captions from keyframe descriptions and transcriptions."""
    for attempt in range(3):  # Add retry at this level
        try:
            has_audio = video_utils.has_audio_stream(video_path)
            use_synthesis_captions = use_synthesis_captions and synthesis_captions
            
            if output_path is None:
                os.makedirs(VIDEO_SETTINGS['OUTPUT_DIR'], exist_ok=True)
                base_name = os.path.splitext(os.path.basename(video_path))[0]
                temp_output = os.path.join(VIDEO_SETTINGS['OUTPUT_DIR'], f'temp_{base_name}.mp4')
                final_output = os.path.join(VIDEO_SETTINGS['OUTPUT_DIR'], f'captioned_{base_name}.mp4')
            else:
                temp_output, final_output = output_path + '.temp', output_path

            props = video_utils.get_video_properties(video_path)
            fps, frame_width, frame_height, total_frames = props['fps'], props['frame_width'], props['frame_height'], props['total_frames']

            video = cv2.VideoCapture(video_path)
            if not video.isOpened():
                error_collector.add_error("Could not open video file", source="create_captioned_video")
                return None

            if use_synthesis_captions and synthesis_captions:
                # Filter out captions containing "the same"
                filtered_captions = [(timestamp, text) for timestamp, text in synthesis_captions if "the same" not in text.lower()]
                adjusted_captions = [(max(0.0, timestamp - CAPTION['TIMESTAMP_OFFSET']), text) for timestamp, text in filtered_captions]
                frame_info = [(int(timestamp * fps), timestamp, text) for timestamp, text in adjusted_captions]
            else:
                # Filter out descriptions containing "the same"
                frame_info = [(frame_num, timestamp, desc) for frame_num, timestamp, desc in descriptions if "the same" not in desc.lower()]
            
            frame_info.sort(key=lambda x: x[1])

            out = cv2.VideoWriter(temp_output, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))
            frame_count = current_desc_idx = current_transcript_idx = 0

            with tqdm(total=total_frames, desc="Creating video") as pbar:
                while True:
                    ret, frame = video.read()
                    if not ret:
                        break

                    current_time = frame_count / fps
                    while current_desc_idx < len(frame_info) - 1 and current_time >= frame_info[current_desc_idx + 1][1]:
                        current_desc_idx += 1

                    frame_number, timestamp, description = frame_info[current_desc_idx]
                    height, width = frame.shape[:2]
                    margin, padding = DISPLAY['MARGIN']['CAPTION'], DISPLAY['PADDING']
                    min_line_height, font = DISPLAY['MIN_LINE_HEIGHT'], DISPLAY['FONT']
                    max_width = int(width * DISPLAY['MAX_WIDTH_RATIO'])
                    
                    full_text = description if use_synthesis_captions else f"[{timestamp:.2f}s] {description}"
                    full_text = full_text.replace('"', '"').replace('"', '"').replace("'", "'").replace("'", "'")
                    
                    font_scale = min(height * 0.03 / min_line_height, DISPLAY['FONT_SCALE']['CAPTION'])
                    
                    # Process text into lines
                    top_lines = []
                    current_line = []
                    for word in full_text.split():
                        test_line = ' '.join(current_line + [word])
                        if cv2.getTextSize(test_line, font, font_scale, 1)[0][0] <= max_width - 2 * margin:
                            current_line.append(word)
                        else:
                            if current_line:
                                top_lines.append(' '.join(current_line))
                            current_line = [word]
                    if current_line:
                        top_lines.append(' '.join(current_line))

                    bottom_lines = []
                    if show_transcriptions:
                        while current_transcript_idx < len(transcript) - 1 and current_time >= transcript[current_transcript_idx + 1]["start"]:
                            current_transcript_idx += 1
                        
                        segment = transcript[current_transcript_idx]
                        if current_time >= segment["start"] and current_time <= segment["end"]:
                            text = segment["text"].replace('"', '"').replace('"', '"').replace("'", "'").replace("'", "'")
                            current_line = []
                            for word in text.split():
                                test_line = ' '.join(current_line + [word])
                                if cv2.getTextSize(test_line, font, font_scale, 1)[0][0] <= max_width - 2 * margin:
                                    current_line.append(word)
                                else:
                                    if current_line:
                                        bottom_lines.append(' '.join(current_line))
                                    current_line = [word]
                            if current_line:
                                bottom_lines.append(' '.join(current_line))

                    line_height = max(min_line_height, int(height * 0.03))
                    top_box_height = len(top_lines) * line_height + 2 * padding
                    
                    # Draw top caption box
                    overlay = frame.copy()
                    cv2.rectangle(overlay, 
                                (int(margin), int(margin)), 
                                (int(width - margin), int(margin + top_box_height)), 
                                (0, 0, 0), 
                                -1)
                    alpha = DISPLAY['OVERLAY_ALPHA']
                    frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
                    
                    # Draw top captions
                    y = int(margin + padding + line_height)
                    for line in top_lines:
                        cv2.putText(frame, line, (int(margin + padding), y), font, font_scale, DISPLAY['TEXT_COLOR']['WHITE'], 1, cv2.LINE_AA)
                        y += line_height
                    
                    # Draw bottom transcriptions if any
                    if bottom_lines:
                        bottom_box_height = len(bottom_lines) * line_height + 2 * padding
                        
                        # Calculate max text width and box width
                        max_text_width = max(cv2.getTextSize(line, font, font_scale, 1)[0][0] for line in bottom_lines)
                        box_width = max_text_width + 2 * padding
                        box_x_start = (width - box_width) // 2
                        
                        # Draw bottom caption box first
                        overlay = frame.copy()
                        cv2.rectangle(overlay, 
                                    (box_x_start, int(height - margin - bottom_box_height)),
                                    (box_x_start + box_width, int(height - margin)),
                                    (0, 0, 0), 
                                    -1)
                        frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
                        
                        # Draw text
                        y = int(height - margin - bottom_box_height + padding + line_height)
                        for line in bottom_lines:
                            text_width = cv2.getTextSize(line, font, font_scale, 1)[0][0]
                            x = (width - text_width) // 2
                            cv2.putText(frame, line, (x, y), font, font_scale, DISPLAY['TEXT_COLOR']['WHITE'], 1, cv2.LINE_AA)
                            y += line_height

                    out.write(frame)
                    frame_count += 1
                    pbar.update(1)

            video.release()
            out.release()

            summary_path, summary_duration = create_summary_clip(summary, frame_width, frame_height, fps, debug=debug, metadata=debug_metadata)
            final_output = video_utils.concatenate_videos([summary_path, temp_output], final_output, has_audio, video_path if has_audio else None, summary_duration)
            
            os.remove(temp_output)
            os.remove(summary_path)
            web_output = video_utils.convert_to_web_format(final_output)
            os.remove(final_output)
            
            return web_output
            
        except Exception as e:
            error_collector.add_retry("create_captioned_video", attempt, error=e)
            if attempt < 2:  # Only retry twice
                error_collector.add_warning(f"Retrying video creation, attempt {attempt + 2}/3", source="create_captioned_video")
                continue
            error_collector.add_error(e, source="create_captioned_video")
            return None

def calculate_target_captions(video_duration: float, frame_count: int, density: float = CAPTION_DENSITY['DEFAULT']) -> tuple:
    """Calculate the target number of captions based on video duration and density preference."""
    # Clamp density between MIN and MAX values
    density = max(CAPTION_DENSITY['MIN'], min(CAPTION_DENSITY['MAX'], float(density)))
    
    if video_duration > VIDEO_SETTINGS['LONG_VIDEO_DURATION']:
        base_captions = int(video_duration / 8)  # One caption every 8 seconds
        num_captions = min(max(15, base_captions), 120)  # Cap at 120 captions
        caption_calc = f"min(max(15, {video_duration} / 8), 120) * {density:.2f}"
    elif video_duration > VIDEO_SETTINGS['MEDIUM_VIDEO_DURATION']:
        base_captions = int(video_duration / CAPTION['LONG_VIDEO']['INTERVAL_RATIO'])
        num_captions = min(int(base_captions * 1.25), frame_count // 2)
        num_captions = max(CAPTION['MEDIUM_VIDEO']['MIN_CAPTIONS'], num_captions)
        caption_calc = f"min(max({CAPTION['MEDIUM_VIDEO']['MIN_CAPTIONS']}, {base_captions} * 1.25), {frame_count} // 2) * {density:.2f}"
    else:
        if video_duration < VIDEO_SETTINGS['MIN_VIDEO_DURATION']:
            target_captions = int(video_duration / CAPTION['SHORT_VIDEO']['INTERVAL'])
            num_captions = min(CAPTION['SHORT_VIDEO']['MAX_CAPTIONS'], max(CAPTION['SHORT_VIDEO']['MIN_CAPTIONS'], target_captions))
            caption_calc = f"min({CAPTION['SHORT_VIDEO']['MAX_CAPTIONS']}, max({CAPTION['SHORT_VIDEO']['MIN_CAPTIONS']}, {target_captions})) * {density:.2f}"
        else:
            caption_interval = (CAPTION['MEDIUM_VIDEO']['BASE_INTERVAL'] + 
                              (video_duration - VIDEO_SETTINGS['MIN_VIDEO_DURATION']) * 
                              (CAPTION['MEDIUM_VIDEO']['MAX_INTERVAL'] - CAPTION['MEDIUM_VIDEO']['BASE_INTERVAL']) / 
                              (VIDEO_SETTINGS['MEDIUM_VIDEO_DURATION'] - VIDEO_SETTINGS['MIN_VIDEO_DURATION']))
            target_captions = int(video_duration / caption_interval)
            max_captions = min(int(video_duration / 4), frame_count // 3)
            num_captions = min(target_captions, max_captions)
            num_captions = max(CAPTION['MEDIUM_VIDEO']['MIN_CAPTIONS'], num_captions)
            caption_calc = f"min(max({CAPTION['MEDIUM_VIDEO']['MIN_CAPTIONS']}, {target_captions}), {max_captions}) * {density:.2f}"
    
    num_captions = int(num_captions * density)
    return num_captions, caption_calc

def update_caption_estimate(video_file, density=CAPTION_DENSITY['DEFAULT']):
    """Calculate and return estimated caption count for a video."""
    if not video_file:
        return "Upload a video to see caption estimate"
    
    try:
        density = float(density)
    except (TypeError, ValueError):
        density = CAPTION_DENSITY['DEFAULT']
        
    video = cv2.VideoCapture(video_file.name)
    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = video.get(cv2.CAP_PROP_FPS)
    video_duration = frame_count / fps if fps > 0 else None
    video.release()
    
    if not video_duration:
        return "Could not calculate video duration"
        
    num_captions, _ = calculate_target_captions(video_duration, frame_count, density)
    return f"Estimated {num_captions} captions ({density:.2f}x density) for {video_duration:.1f}s video"

def process_video_web(video_file, use_frame_selection=False, use_synthesis_captions=False, 
                     use_local_llm=False, show_transcriptions=False, debug=False,
                     caption_density=CAPTION_DENSITY['DEFAULT']):
    """Process video through web interface."""
    video_path = video_file.name
    if os.path.basename(video_path).startswith('captioned_') or video_path.endswith('_web.mp4'):
        error_collector.add_warning("Skipped output file", source="process_video_web")
        return "Skipped output file", None, None

    start_time = time.time()
    
    # Get basic video info from OpenCV first
    video = cv2.VideoCapture(video_path)
    if not video.isOpened():
        error_collector.add_error("Could not open video file", source="process_video_web")
        save_output(video_path, 0, success=False, caption_density=caption_density)
        return "Error: Could not open video file.", None, None
        
    frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = video.get(cv2.CAP_PROP_FPS)
    width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_duration = frame_count / fps if fps > 0 else None
    video.release()

    error_collector.add_warning(f"Video properties - frame_count: {frame_count}, fps: {fps}, duration: {video_duration}", source="process_video_web")

    # Initialize log at start with basic info
    save_output(video_path, frame_count, success=False, caption_density=caption_density)

    if frame_count == 0:
        error_collector.add_error("Could not process video - zero frames detected", source="process_video_web")
        return "Error: Could not process video.", None, None
        
    if video_duration is None or fps <= 0:
        error_collector.add_error("Could not determine video duration or invalid FPS", source="process_video_web")
        return "Error: Could not get video duration.", None, None

    # Calculate target captions using new function
    try:
        error_collector.add_warning(f"Calculating captions for duration: {video_duration}", source="process_video_web")
        base_captions, caption_calc = calculate_target_captions(video_duration, frame_count, 1.0)  # Get base count with no multiplier
        final_captions, _ = calculate_target_captions(video_duration, frame_count, caption_density)  # Get final count with multiplier
        error_collector.add_warning(f"Caption calculation result - base_captions: {base_captions}, final_captions: {final_captions}, calc: {caption_calc}", source="process_video_web")
    except Exception as e:
        error_collector.add_error(f"Caption calculation failed - duration: {video_duration}, frame_count: {frame_count}, error: {str(e)}", source="process_video_web_caption_calc")
        # Fallback to a simple calculation
        base_captions = max(8, min(frame_count // 50, 100))
        final_captions = int(base_captions * caption_density)
        caption_calc = f"fallback: {base_captions} * {caption_density:.2f}"

    debug_metadata = None
    if debug:
        try:
            cmd = ['ffprobe', '-v', 'quiet', '-print_format', 'json', '-show_format', '-show_streams', video_path]
            metadata = json.loads(subprocess.check_output(cmd).decode('utf-8'))
            video_type = "Short (<30s)" if video_duration < 30 else "Medium (30-90s)" if video_duration < 90 else "Medium-long (90-150s)" if video_duration < 150 else "Long (>150s)"
            
            debug_metadata = {}
            
            # Add ffprobe metadata first if available
            if metadata.get('format', {}).get('tags', {}):
                debug_metadata['FFprobe Metadata'] = metadata.get('format', {}).get('tags', {})

            # Add caption info second
            debug_metadata['Caption Info'] = {
                'Video Type': video_type,
                'Density Multiplier': f"{caption_density:.2f}x",
                'Base Caption Count': str(base_captions),
                'Final Caption Count': str(final_captions),
                'Captions/Second': f"{final_captions/video_duration:.2f}",
                'Raw Calculation': caption_calc.replace(' ', '')
            }

            # Add video info last
            debug_metadata['Video Info'] = {
                'Duration': f"{video_duration:.1f}s",
                'Resolution': f"{width}x{height}",
                'FPS': f"{fps:.1f}",
                'Frame Count': str(frame_count),
                'File Size': f"{os.path.getsize(video_path) / (1024*1024):.1f}MB"
            }
                
        except Exception as e:
            error_collector.add_warning(f"Could not get detailed video metadata: {str(e)}", source="process_video_web")
            # Still provide basic metadata from OpenCV, maintaining the same order
            debug_metadata = {
                'Caption Info': {
                    'Target Captions': str(final_captions),
                    'Captions/Second': f"{final_captions/video_duration:.2f}",
                    'Calculation': caption_calc.replace(' ', '')
                },
                'Video Info': {
                    'Duration': f"{video_duration:.1f}s",
                    'Resolution': f"{width}x{height}",
                    'FPS': f"{fps:.1f}",
                    'Frame Count': str(frame_count),
                    'File Size': f"{os.path.getsize(video_path) / (1024*1024):.1f}MB"
                }
            }

    try:
        transcript = transcribe_video(video_path)
        frame_numbers = process_video(video_path) if use_frame_selection else list(range(0, frame_count, 50))
        descriptions = describe_frames(video_path, frame_numbers)
        synthesis_output, synthesis_captions = summarize_with_hosted_llm(
            transcript, 
            descriptions, 
            video_duration, 
            use_local_llm=use_local_llm, 
            video_path=video_path,
            caption_density=caption_density
        )
        
        if synthesis_output is None or (use_synthesis_captions and synthesis_captions is None):
            error_collector.add_error("Failed to generate synthesis", source="process_video_web")
            save_output(video_path, frame_count, transcript, descriptions, success=False, caption_density=caption_density)
            return "Error: Failed to generate synthesis. Please try again.", None, None
        
        summary_match = re.search(r'<summary>(.*?)</summary>', synthesis_output, re.DOTALL)
        if not summary_match:
            error_collector.add_error("Failed to extract summary from synthesis output", source="process_video_web")
            save_output(video_path, frame_count, transcript, descriptions, success=False, caption_density=caption_density)
            return "Error: Failed to extract summary from synthesis output.", None, None
        
        summary = summary_match.group(1).strip()
        output_video_path = create_captioned_video(video_path, descriptions, summary, transcript, synthesis_captions, use_synthesis_captions, show_transcriptions, debug=debug, debug_metadata=debug_metadata)
        total_run_time = time.time() - start_time

        # Final save with success flag
        save_output(video_path, frame_count, transcript, descriptions, summary, total_run_time, synthesis_output, synthesis_captions, use_local_llm=use_local_llm, success=True, caption_density=caption_density)

        gallery_images = []
        video = cv2.VideoCapture(video_path)
        for frame_number, timestamp, description in descriptions:
            try:
                video.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
                success, frame = video.read()
                if success:
                    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    height, width = frame.shape[:2]
                    margin, padding, min_line_height = 8, 10, 20
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    max_width = int(width * 0.9)
                    full_text = f"Frame {frame_number} [{timestamp:.2f}s]\n{description}"
                    words, lines, current_line = full_text.split(), [], []
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

                    line_height = max(min_line_height, int(height * 0.03))
                    box_height = len(lines) * line_height + 2 * padding
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
                    
                    gallery_images.append(Image.fromarray(frame))
                else:
                    error_collector.add_warning(f"Failed to read frame {frame_number} for gallery", source="process_video_web")
            except Exception as e:
                error_collector.add_error(e, source=f"process_video_web_gallery_frame_{frame_number}")
        
        video.release()
        return output_video_path, f"Video Summary:\n{summary}\n\nTime taken: {total_run_time:.2f} seconds", gallery_images
    except Exception as e:
        error_collector.add_error(e, source="process_video_web")
        save_output(video_path, frame_count, transcript, descriptions, success=False, caption_density=caption_density)
        return "Error: An unexpected error occurred while processing the video.", None, None

def process_folder(folder_path, args):
    """Process all videos in a folder with retries."""
    video_extensions = ('.mp4', '.avi', '.mov', '.mkv', '.webm')
    video_files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f)) and f.lower().endswith(video_extensions) and not (f.startswith(('captioned_', 'temp_', 'concat_')) or f.endswith('_web.mp4'))]
    
    if not video_files:
        error_collector.add_warning("No video files found to process", source="process_folder")
        print("No video files found to process.")
        return

    processed_videos, failed_videos = set(), set()
    print(f"\nFound {len(video_files)} videos to process in {folder_path}")
    
    for i, video_file in enumerate(video_files, 1):
        if video_file in processed_videos or video_file in failed_videos:
            continue
            
        video_path = os.path.join(folder_path, video_file)
        print(f"\nProcessing video {i}/{len(video_files)}: {video_file}")
        
        success = False
        for attempt in range(10):
            if attempt > 0:
                print(f"Retry attempt {attempt + 1}/10 for {video_file}")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            
            try:
                output_video_path, summary, gallery_images = process_video_web(
                    type('VideoFile', (), {'name': video_path})(),
                    use_frame_selection=args.frame_selection,
                    use_synthesis_captions=args.synthesis_captions,
                    use_local_llm=args.local,
                    show_transcriptions=args.transcribe,
                    debug=args.debug,
                    caption_density=args.density
                )
                expected_output = os.path.join('outputs', f'captioned_{os.path.splitext(video_file)[0]}_web.mp4')
                if os.path.exists(expected_output):
                    processed_videos.add(video_file)
                    print(f"Successfully processed {video_file}")
                    success = True
                    break
            except Exception as e:
                error_collector.add_retry(f"process_folder_{video_file}", attempt, error=e)
                if attempt == 9:
                    error_collector.add_error(e, source=f"process_folder_final_{video_file}")
                    print(f"Failed to process {video_file} after 10 attempts: {str(e)}")
                    failed_videos.add(video_file)
                else:
                    print(f"Error processing {video_file} (attempt {attempt + 1}): {str(e)}")
                    if not args.local:
                        time.sleep(2 * (attempt + 1))
        
        if not success and video_file not in failed_videos:
            failed_videos.add(video_file)
            error_collector.add_error(f"Failed to process {video_file} after all attempts", source="process_folder")
            print(f"Failed to process {video_file} after all attempts")
    
    print("\nProcessing complete!")
    print(f"Successfully processed: {len(processed_videos)}/{len(video_files)} videos")
    if failed_videos:
        error_collector.add_warning(f"Failed to process {len(failed_videos)} videos", source="process_folder")
        print(f"Failed to process {len(failed_videos)} videos:")
        for video in failed_videos:
            print(f"- {video}")

def recontextualize_summary(summary: str, metadata_context: str, use_local_llm: bool = False) -> str:
    """Recontextualize just the summary using metadata context."""
    prompt = get_recontextualization_prompt()
    prompt_content = prompt.format(metadata=metadata_context, summary=summary)

    for attempt in range(3):
        completion = get_llm_completion(prompt_content, "", use_local_llm=use_local_llm)
        if completion:
            summary_match = re.search(r'<recontextualized_summary>(.*?)</recontextualized_summary>', completion, re.DOTALL)
            if summary_match:
                return summary_match.group(1).strip()
        if attempt < 2:
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
    parser.add_argument('--density', type=float, default=CAPTION_DENSITY['DEFAULT'], 
                       help=f'Caption density multiplier (between {CAPTION_DENSITY["MIN"]} and {CAPTION_DENSITY["MAX"]}, default: {CAPTION_DENSITY["DEFAULT"]})')
    args = parser.parse_args()

    if args.web:
        with gr.Blocks(title="Video Summarizer") as iface:
            gr.Markdown("# Video Summarizer")
            gr.Markdown("Upload a video to get a summary and view analyzed frames.")
            
            with gr.Row():
                with gr.Column():
                    video_input = gr.File(label="Upload Video")
                    density_input = gr.Number(
                        value=CAPTION_DENSITY['DEFAULT'],
                        label="Caption Density",
                        info=f"Multiplier for number of captions (between {CAPTION_DENSITY['MIN']} and {CAPTION_DENSITY['MAX']})",
                        minimum=CAPTION_DENSITY['MIN'],
                        maximum=CAPTION_DENSITY['MAX'],
                        step=0.1
                    )
                    estimate_text = gr.Textbox(label="Caption Estimate", interactive=False)
                    
                    frame_selection = gr.Checkbox(
                        label="Use Frame Selection",
                        value=True,
                        info="Recommended: Intelligently selects key frames"
                    )
                    synthesis_captions = gr.Checkbox(
                        label="Use Synthesis Captions",
                        value=True,
                        info="Recommended: Creates a more pleasant viewing experience"
                    )
                    use_local_llm = gr.Checkbox(
                        label="Use Local LLM",
                        value=True,
                        info="Use local Llama model instead of OpenAI API (requires model weights)"
                    )
                    show_transcriptions = gr.Checkbox(
                        label="Show Transcriptions",
                        value=True,
                        info="Show speech transcriptions in the output video"
                    )
                    debug_mode = gr.Checkbox(
                        label="Debug Mode",
                        value=False,
                        info="Show metadata in summary screen"
                    )
                    
                with gr.Column():
                    video_output = gr.Video(label="Captioned Video")
                    summary_output = gr.Textbox(label="Summary")
                    gallery_output = gr.Gallery(label="Analyzed Frames")
            
            # Update estimate when video or density changes
            video_input.change(
                fn=update_caption_estimate,
                inputs=[video_input, density_input],
                outputs=estimate_text
            )
            density_input.change(
                fn=update_caption_estimate,
                inputs=[video_input, density_input],
                outputs=estimate_text
            )
            
            # Process video button
            process_btn = gr.Button("Process Video")
            process_btn.click(
                fn=process_video_web,
                inputs=[
                    video_input,
                    frame_selection,
                    synthesis_captions,
                    use_local_llm,
                    show_transcriptions,
                    debug_mode,
                    density_input
                ],
                outputs=[video_output, summary_output, gallery_output]
            )
            
        iface.launch()
    elif args.path:
        start_time = time.time()
        if os.path.isdir(args.path):
            process_folder(args.path, args)
        else:
            frame_count = get_frame_count(args.path)
            if frame_count > 0:
                output_video_path, summary, gallery_images = process_video_web(
                    type('VideoFile', (), {'name': args.path})(),
                    use_frame_selection=args.frame_selection,
                    use_synthesis_captions=args.synthesis_captions,
                    use_local_llm=args.local,
                    show_transcriptions=args.transcribe,
                    debug=args.debug,
                    caption_density=args.density
                )

if __name__ == "__main__":
    main()
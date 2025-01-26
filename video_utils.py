import os
import subprocess
import json
import cv2
import time

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
        return duration
    except Exception as e:
        print(f"Warning: Could not get video duration: {str(e)}")
        return None

def get_video_metadata(video_path):
    """Get detailed video metadata using ffprobe."""
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
        metadata_context = json.dumps(metadata.get('format', {}).get('tags', {}), indent=2)
        return metadata, metadata_context
    except Exception as e:
        print(f"Warning: Could not get detailed video metadata: {str(e)}")
        return {}, ""

def has_audio_stream(video_path):
    """Check if video has an audio stream."""
    try:
        cmd = [
            'ffprobe', 
            '-i', video_path,
            '-show_streams', 
            '-select_streams', 'a', 
            '-loglevel', 'error'
        ]
        result = subprocess.run(cmd, capture_output=True, text=True)
        return len(result.stdout.strip()) > 0
    except subprocess.CalledProcessError:
        print("Error checking audio stream")
        return False

def get_video_properties(video_path):
    """Get basic video properties using OpenCV."""
    video = cv2.VideoCapture(video_path)
    properties = {
        'fps': video.get(cv2.CAP_PROP_FPS),
        'frame_width': int(video.get(cv2.CAP_PROP_FRAME_WIDTH)),
        'frame_height': int(video.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        'total_frames': int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    }
    video.release()
    return properties

def convert_to_web_format(input_path, output_path=None):
    """Convert video to web-compatible format (h264)."""
    if output_path is None:
        output_path = input_path.replace('.mp4', '_web.mp4')
    
    convert_cmd = [
        'ffmpeg', '-y',
        '-i', input_path,
        '-c:v', 'libx264',
        '-preset', 'ultrafast',  # Use ultrafast preset for speed
        '-crf', '23',           # Reasonable quality
        '-c:a', 'aac',          # AAC audio for web compatibility
        '-movflags', '+faststart',  # Enable fast start for web playback
        '-loglevel', 'error',   # Suppress output
        output_path
    ]
    subprocess.run(convert_cmd, check=True)
    return output_path

def concatenate_videos(video_paths, output_path, has_audio=False, audio_source=None, audio_delay=0):
    """Concatenate multiple videos and optionally add delayed audio."""
    # Create concat file
    concat_file = os.path.join(os.path.dirname(output_path), 'concat_list.txt')
    with open(concat_file, 'w') as f:
        for path in video_paths:
            f.write(f"file '{os.path.abspath(path)}'\n")
    
    # Concatenate videos
    concat_output = output_path + '.concat.mp4'
    concat_cmd = [
        'ffmpeg', '-y',
        '-f', 'concat',
        '-safe', '0',
        '-i', concat_file,
        '-c', 'copy',
        '-loglevel', 'error',
        concat_output
    ]
    subprocess.run(concat_cmd, check=True)
    
    # Add audio if needed
    if has_audio and audio_source:
        cmd = [
            'ffmpeg', '-y',
            '-i', concat_output,      # Input concatenated video
            '-i', audio_source,       # Input original video (for audio)
            '-filter_complex', f'[1:a]adelay={int(audio_delay*1000)}|{int(audio_delay*1000)}[delayed_audio]',
            '-c:v', 'copy',           # Copy video stream as is
            '-map', '0:v',            # Use video from first input
            '-map', '[delayed_audio]', # Use delayed audio
            '-loglevel', 'error',
            output_path
        ]
    else:
        cmd = [
            'ffmpeg', '-y',
            '-i', concat_output,
            '-c:v', 'copy',
            '-loglevel', 'error',
            output_path
        ]
    
    subprocess.run(cmd, check=True)
    
    # Clean up temporary files
    os.remove(concat_file)
    os.remove(concat_output)
    
    return output_path

def extract_audio(video_path):
    """Extract audio from video to MP3."""
    audio_path = os.path.join('outputs', f'temp_audio_{int(time.time())}.mp3')
    os.makedirs('outputs', exist_ok=True)
    
    try:
        cmd = [
            'ffmpeg', '-y',
            '-i', video_path,
            '-vn',  # No video
            '-loglevel', 'error',
            audio_path
        ]
        subprocess.run(cmd, check=True)
        return audio_path
    except Exception as e:
        print(f"Error extracting audio: {str(e)}")
        return None
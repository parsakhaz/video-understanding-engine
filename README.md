# Video Summarizer

> **⚠️ IMPORTANT:** This project uses Moondream2 (2025-01-09 release), CLIP, Llama 3.1 8B Instruct, and Whisper large-v3-turbo via the Hugging Face Transformers library.

> **💡 NOTE:** This project offers two options for content synthesis:
> 1. OpenAI GPT-4o API (Default, recommended if you don't have access to LLama 3.1 8B Instruct yet)
> 2. Local Meta-Llama-3.1-8B-Instruct (Recommended if you want to run everything locally, requires requesting access to the model from Meta on Hugging Face repository [here](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct))
>
> **⚠️ AUTHENTICATION:** When using OpenAI, make sure to set your API key in the `.env` file with the key `OPENAI_API_KEY`.

## Table of Contents

- [Quick Start](#quick-start)
- [Features](#features)
  - [Core Features](#core-features)
  - [Video Output Features](#video-output-features)
  - [Caption Types](#caption-types)
  - [Accessibility Features](#accessibility-features)
  - [Content Synthesis Options](#content-synthesis-options)
- [Architecture Overview](#architecture-overview)
- [Process Flow](#process-flow)
- [Installation](#installation)
- [Usage](#usage)
  - [Web Interface](#web-interface-recommended)
  - [Command Line Interface](#command-line-interface)
  - [Local LLM Support](#local-llm-support)
- [Model Prompts](#model-prompts)
- [Output Format](#output-format)
- [Frame Analysis Visualization](#frame-analysis-visualization)
- [Requirements](#requirements)
- [Performance Considerations](#performance-considerations)
- [Troubleshooting](#troubleshooting)
- [Video Output Format](#video-output-format)
- [Recent Updates](#recent-updates)
  - [Audio Processing Improvements](#audio-processing-improvements)
  - [Caption Enhancements](#caption-enhancements)
  - [Technical Improvements](#technical-improvements)
- [Advanced Features & Limitations](#advanced-features--limitations)
  - [Hallucination Detection Parameters](#hallucination-detection-parameters)
  - [Transcript Cleaning System](#transcript-cleaning-system)
  - [Caption Timing Specifications](#caption-timing-specifications)
  - [Error Recovery System](#error-recovery-system)
  - [Resolution-based Adaptations](#resolution-based-adaptations)
  - [Video Processing Limitations](#video-processing-limitations)

A powerful video summarization tool that combines multiple AI models to provide comprehensive video understanding through audio transcription, intelligent frame selection, visual description, and content summarization.

## Quick Start (Mac/Linux)

Reference [Installation](#installation) for more a detailed guide on how to install the dependencies on different devices (ffmpeg, pyvips, etc) and get things running.

```bash
# Create a virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
.\venv\Scripts\activate   # Windows

# Install dependencies and run with default settings
pip install -r requirements.txt

# Login to Hugging Face (required for model access)
huggingface-cli login

# Install Pyvips and ffmpeg to device (linux)
apt-get install libvips libvips-dev ffmpeg

# Run with web interface (recommended)
python main.py --web

# Run with all features enabled (including local LLM) on 'video.mp4'
python main.py video.mp4 --frame-selection --local --save --synthesis-captions --transcribe --debug

# Run with all features enabled (including local LLM) on 'inputs' folder
python main.py inputs --frame-selection --local --save --synthesis-captions --transcribe --debug
```

## Features

### Core Features
- Intelligent frame selection using CLIP embeddings and clustering
- High-quality audio transcription using Whisper large-v3-turbo
- Visual scene description using Moondream2
- Dynamic content synthesis with GPT-4o/Llama 3.1
- Interactive web interface
- Accessible video output with captions
- Local LLM support with Meta-Llama-3.1-8B-Instruct

### Video Output Features
- Intelligent frame descriptions with timestamps
- High-accuracy speech transcription
- Adaptive text wrapping and positioning
- Accessibility-focused caption design
- Choice between detailed or synthesized captions
- Original audio preservation with proper synchronization

### Caption Types
1. **Full Frame Descriptions**
   - Detailed descriptions for every key frame
   - Technical visual details
   - More frequent updates
   - Timestamps included
   
2. **Synthesis Captions** (Highly recommended)
   - Context-aware, narrative-focused captions
   - Dynamic quantity based on video length
   - Automatic deduplication of close captions (< 1.2s apart)
   - Better for storytelling and overview
   - Clean text without timestamps

### Accessibility Features
- High contrast caption backgrounds (70% opacity)
- Responsive font sizing
  - Frame descriptions: Standard size
  - Speech transcriptions: 4 sizes larger
- Automatic text wrapping
- Minimum readable text size
- Caption persistence between transitions
- Clear timestamp indicators
- Separated visual and speech captions
  - Frame descriptions at top
  - Speech transcriptions centered near bottom
  - Tight background boxes for speech transcriptions
- Original audio track preservation

### Content Synthesis Options
1. **Hosted LLM (Default)**
   - Uses OpenAI's GPT-4o
   - Requires API key
   - Faster processing
   - Higher reliability
   - Integrates video metadata

2. **Local LLM (New!)**
   - Uses Meta-Llama-3.1-8B-Instruct
   - No API key required
   - Full offline operation
   - Automatic fallback to hosted LLM
   - ~8GB GPU memory required
   - Compatible response format
   - Supports metadata context
   - Chunk-level metadata integration

## Architecture Overview

```mermaid
graph TD
    A[Input Video] --> B[Frame Selection]
    A --> C[Audio Processing]
    A --> MD[Metadata Extraction]
    
    subgraph "Frame Selection Pipeline"
    B --> D[CLIP Embeddings]
    D --> E[Similarity Analysis]
    E --> F[Key Frame Detection]
    F --> G[Frame Clustering]
    G --> H[Cache Management]
    end
    
    subgraph "Audio Pipeline"
    C --> AA[Audio Stream Check]
    AA -->|Has Audio| AE[Audio Extraction]
    AA -->|No Audio| AB[Empty Transcript]
    AE --> I[Whisper large-v3-turbo]
    I --> J[Timestamped Transcript]
    AB --> J
    end
    
    subgraph "Metadata Pipeline"
    MD --> ME[FFprobe Extraction]
    ME --> MF[Parse Metadata]
    MF --> MG[Format Context]
    end
    
    F --> L[Selected Frames]
    G --> L
    L --> M[Moondream VLM]
    M --> N[Frame Descriptions]
    
    J --> O[Content Synthesis]
    N --> O
    MG --> O
    
    subgraph "Summarization Pipeline"
    O --> P{Model Selection}
    P -->|Local| Q[Llama 3.1]
    P -->|Hosted| R[gpt4o]
    Q --> S[XML Validation]
    R --> S
    S --> T[Summary + Captions]
    end
    
    subgraph "Video Generation"
    T --> U[Frame Description Overlay]
    J --> V[Speech Transcript Overlay]
    U --> AI[Background Layer]
    AI --> AJ[Text Layer]
    V --> AI
    AJ --> W[Video Assembly]
    A --> X[Audio Stream]
    X --> AC[Audio Check]
    AC -->|Has Audio| Y[FFmpeg Merge]
    AC -->|No Audio| AD[Direct Output]
    W --> Y
    W --> AD
    Y --> AK[Final Video]
    AD --> AK
    end
```

## Process Flow

```mermaid
sequenceDiagram
    participant User
    participant Main
    participant FrameSelection
    participant AudioProc
    participant Whisper
    participant Moondream
    participant LLM
    participant VideoGen

    User->>Main: process_video_web(video_file)
    activate Main
    
    Main->>Main: Check audio stream
    
    par Frame Analysis
        Main->>FrameSelection: process_video(video_path)
        activate FrameSelection
        FrameSelection->>FrameSelection: load_model()
        FrameSelection->>FrameSelection: get_or_compute_embeddings()
        Note over FrameSelection: Check cache first
        FrameSelection->>FrameSelection: process_batch()
        FrameSelection->>FrameSelection: sliding_window_filter()
        FrameSelection->>FrameSelection: find_interesting_frames()
        Note over FrameSelection: novelty_threshold=0.08<br/>min_skip=10<br/>n_clusters=15
        FrameSelection-->>Main: key_frame_numbers
        deactivate FrameSelection
    and Audio Processing
        Main->>AudioProc: extract_audio()
        activate AudioProc
        AudioProc->>AudioProc: Check audio stream
        AudioProc-->>Main: audio_path
        deactivate AudioProc
        Main->>Whisper: model.transcribe(audio_path)
        activate Whisper
        Note over Whisper: Raw MP3 input<br/>No preprocessing<br/>Direct transcription
        Whisper-->>Main: timestamped_transcript
        deactivate Whisper
    end

    Main->>Moondream: describe_frames(video_path, frame_numbers)
    activate Moondream
    Note over Moondream: Batch process (8 frames)<br/>Generate descriptions
    Moondream-->>Main: frame_descriptions
    deactivate Moondream

    Main->>LLM: summarize_with_hosted_llm(transcript, descriptions)
    activate LLM
    Note over LLM: Generate summary<br/>Create synthesis captions<br/>Validate XML format<br/>Use earliest timestamps
    LLM-->>Main: synthesis_output
    deactivate LLM

    Main->>Main: parse_synthesis_output()
    Note over Main: Extract summary<br/>Parse captions<br/>Validate format

    Main->>VideoGen: create_captioned_video()
    activate VideoGen
    Note over VideoGen: Create summary intro (5s)<br/>Process main video<br/>Add overlays<br/>Handle captions
    VideoGen->>VideoGen: Filter close captions
    Note over VideoGen: min_time_gap=1.2s<br/>Adjust timing -0.5s
    VideoGen->>VideoGen: Create background layer
    VideoGen->>VideoGen: Add opaque text layer
    Note over VideoGen: Separate rendering passes<br/>70% background opacity<br/>100% text opacity
    VideoGen->>VideoGen: Add frame descriptions
    VideoGen->>VideoGen: Add speech transcripts
    Note over VideoGen: Position & style<br/>Handle hallucinations<br/>ASCII text normalization
    VideoGen->>VideoGen: FFmpeg concat & merge
    Note over VideoGen: Handle missing audio<br/>Proper audio sync
    VideoGen-->>Main: output_video_path
    deactivate VideoGen

    Main-->>User: summary, gallery, video_path
    deactivate Main
```

## Directory Structure
```
video-understanding-engine/
├── main.py              # Main entry point
├── frame_selection.py   # CLIP-based frame selection
├── requirements.txt     # Dependencies
├── .env                # Environment variables
├── prompts/            # Model prompts
│   ├── moondream_prompt.md
│   └── synthesis_prompt.md
├── logs/               # Output logs
├── frame_analysis_plots/ # Frame analysis visualizations
└── embedding_cache/    # Cached CLIP embeddings
```

## Process Flow

1. **Video Input Processing**
   - Input: Video file
   - Output: Frame data, audio stream, and metadata
   - Supported formats: MP4, AVI, MOV, MKV
   - Maximum recommended length: 6-8 minutes
   - Automatic audio stream detection
   - Metadata extraction using ffprobe:
     - Title, artist, duration
     - Format tags and properties
     - Custom metadata fields

2. **Frame Selection** (when using `--frame-selection`)
   - Model: CLIP (ViT-SO400M-14-SigLIP-384)
   - Process:
     1. Extract frames from video
     2. Generate CLIP embeddings for each frame (bfloat16 precision)
     3. Calculate frame similarities
     4. Detect novel frames using sliding window analysis (window_size=30)
     5. Cluster similar frames using KMeans (n_clusters=15)
     6. Select representative frames with stricter novelty threshold
   - Output: List of key frame numbers (~20% fewer than previous version)
   - Cache: Embeddings stored in `embedding_cache/<video_name>.npy`
   - Default mode: Samples every 50 frames when not using CLIP selection

3. **Audio Transcription**
   - Model: Whisper (large)
   - Process:
     1. Extract audio from video
     2. Generate transcript with timestamps
     3. Apply hallucination detection (words/second analysis)
   - Output: JSON with `{start, end, text}` segments
   - Supported languages: 100+ languages (auto-detected)
   - Accuracy: ~95% for clear English speech

4. **Frame Description**
   - Model: Moondream2 (vikhyatk/moondream2)
   - Input: Selected frames (either from CLIP selection or regular sampling)
   - Process:
     1. Batch process frames (8 at a time)
     2. Generate detailed descriptions
   - Output: Frame descriptions with timestamps
   - Batch size: Configurable (default: 8)
   - Memory usage: ~4GB GPU RAM

5. **Content Synthesis**
   - Models: 
     - Local (to be implemented): Meta-Llama-3.1-8B-Instruct
     - Hosted: gpt-4o
   - Input: 
     - Timestamped transcript
     - Frame descriptions
     - Video metadata context
   - Process:
     1. Extract and format metadata context
     2. For long videos:
        - Process chunks with metadata context
        - Generate chunk summaries
        - Create final summary with metadata
     3. For short videos:
        - Direct synthesis with metadata context
   - Output:
     - Metadata-enriched video summary
     - Dynamic number of synthesized captions
   - Strict XML format with validation
   - Error handling with fallback to frame descriptions
   - Token limit: 4096 tokens
   - Temperature: 0.3

6. **Video Generation**
   - Input: Original video, descriptions, summary, transcript
   - Process:
     1. Create 5-second summary intro
     2. Process main video with captions
     3. Concatenate using FFmpeg
   - Features:
     - Predictive caption timing (0.5s early display)
     - Minimum 1.2s gap between captions
     - ASCII-compatible text rendering
     - Graceful handling of missing audio
     - Smart timestamp selection
     - Adaptive font scaling
     - High-contrast overlays
     - Centered transcript positioning
     - Individual background boxes
     - Automatic audio stream detection

7. **Recent Improvements**
   - Enhanced timing and synchronization:
     - Predictive caption timing with 0.5s early display
     - Smart timestamp selection using earliest scene detection
     - Minimum 1.2s gap between captions
     - Automatic audio stream detection
   - Visual enhancements:
     - Resolution-based font scaling (SD/HD/2K+)
     - Improved contrast with semi-transparent overlays (70% opacity)
     - Centered transcript text with individual boxes
     - Dynamic padding and margins based on resolution
   - Robustness improvements:
     - Graceful handling of missing audio streams
     - Fallback to frame descriptions if synthesis fails
     - Smart handling of caption timing edge cases
     - Comprehensive error handling
     - ASCII-compatible text normalization
   - Quality of life:
     - Automatic video property detection
     - Progress tracking
     - Debug logging
     - Informative status messages
     - Web-compatible output format (h264)

8. **Gallery View** (Web Interface)
   - Always shows frame descriptions (not synthesis captions)
   - Includes frame numbers and timestamps
   - Used for debugging frame selection
   - Helps visualize key frame detection
   - Independent from video output format

## Installation

### Initial Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/video-understanding-engine.git
   cd video-understanding-engine
   ```

2. Create a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # Linux/Mac
   .\venv\Scripts\activate   # Windows
   ```

3. Login to Hugging Face:
   ```bash
   # Install Hugging Face CLI if not already installed
   pip install huggingface_hub

   # Login with your token from https://huggingface.co/settings/tokens
   huggingface-cli login
   ```

### System Dependencies
```bash
# Linux/Ubuntu
sudo apt-get update
sudo apt-get install ffmpeg libvips libvips-dev

# macOS with Homebrew
brew install ffmpeg vips

# Windows
# 1. Download and install FFmpeg:
#    - Go to https://ffmpeg.org/download.html
#    - Download the Windows build
#    - Extract the ZIP file
#    - Add the bin folder to your system PATH
#
# 2. Install libvips:
#    - Go to https://github.com/libvips/build-win64/releases
#    - Download vips-dev-w64-all-8.16.0.zip for 64-bit or vips-dev-w32-all-8.16.0.zip for 32-bit
#    - Extract the ZIP file
#    - Copy all DLL files from vips-dev-8.16\bin to either:
#      - Your project's root directory OR
#      - C:\Windows\System32 (requires admin privileges)
#    - Add to PATH:
#      - Open System Properties -> Advanced -> Environment Variables
#      - Under System Variables, find PATH
#      - Add the full path to the vips-dev-8.16\bin directory
```

### Python Dependencies
```bash
pip install -r requirements.txt
```

### Moondream Setup
For detailed setup instructions and model information, visit [docs.moondream.ai/quick-start](https://docs.moondream.ai/quick-start). The Moondream model requires specific configurations and dependencies that are documented there.

### Environment Setup (if you want to use OpenAI API for frame synthesis)
Create a `.env` file:
```
OPENAI_API_KEY=your_api_key_here  # Only needed if not using local LLM
```

## Usage

### Web Interface (Recommended)
```bash
python main.py --web
```

The web interface provides:
- Drag-and-drop video upload
- Progress tracking
- Interactive results viewing
- Download options for JSON output
- Caption style selection:
  - Full frame descriptions
  - Synthesized captions
- Frame selection toggle

### Command Line Interface

Basic usage:
```bash
python main.py <video_path>
```

Advanced options:
```bash
python main.py <video_path> [--save] [--local] [--frame-selection] [--web] [--synthesis-captions] [--transcribe] [--debug]
```

Options explained:
- `--save`: Save all outputs to JSON (includes transcript, descriptions, summary)
- `--local`: (under development, not working yet) Use local Llama model instead of hosted LLM
- `--frame-selection`: Use CLIP-based intelligent frame selection
- `--web`: Launch web interface (highly recommended)
- `--synthesis-captions`: Use synthesized narrative captions (recommended for better viewing experience)
- `--transcribe`: Show speech transcriptions in the output video (optional, transcripts are still generated for captions)
- `--debug`: Enable debug mode with additional metadata overlay and detailed logging

Example commands:
```bash
# Launch web interface (recommended for single video processing, easy to use)
python main.py --web

# Process single video with all features including debug mode
python main.py video.mp4 --frame-selection --local --save --synthesis-captions --transcribe --debug

# Process video with debug mode for troubleshooting
python main.py video.mp4 --debug

# Process video with captions but without visible transcriptions
python main.py video.mp4 --frame-selection --synthesis-captions

# Process all videos in a folder with transcriptions
python main.py /path/to/folder --frame-selection --transcribe

# Process 'inputs' folder with all features
python main.py inputs --frame-selection --local --save --synthesis-captions --transcribe --debug

# Quick processing with hosted LLM and synthesis captions
python main.py video.mp4 --save --synthesis-captions
```

**Folder Processing Features:**
- Automatically processes all videos in the specified folder
- Supports common video formats (.mp4, .avi, .mov, .mkv, .webm)
- Shows progress (e.g., "Processing video 1/5")
- Retries up to 10 times if a video fails
- Continues to next video even if one fails
- Maintains consistent output structure in `outputs/` directory

### Local LLM Support
Now supports Meta-Llama-3.1-8B-Instruct for local content synthesis:
```bash
# Run with local LLM
python main.py video.mp4 --local

# Or enable in web interface
python main.py --web  # Then check "Use Local LLM" option
```

## Model Prompts

### Moondream Frame Description Prompt
Location: `prompts/moondream_prompt.md`
```
Describe this frame in detail, focusing on key visual elements, actions, and any text visible in the frame. Be specific but concise.
```

### Content Synthesis Prompt
Location: `prompts/synthesis_prompt.md`
```
You are a video content analyzer. Your task is to create a comprehensive summary of a video based on its transcript and frame descriptions.

The input will be provided in two sections:
<transcript>
Timestamped transcript of the video's audio
</transcript>

<frame_descriptions>
Timestamped descriptions of key frames from the video
</frame_descriptions>

Create a detailed summary that:
1. Captures the main topics and themes
2. Highlights key visual and auditory information
3. Maintains chronological flow
4. Integrates both visual and audio elements coherently

Be concise but comprehensive. Focus on the most important information.
```

## Output Format

When using `--save` or `--debug`, outputs are saved in `logs/` with the following structure:
```json
{
    "video_path": "path/to/video",
    "frame_count": 1000,
    "metadata": {
        "title": "Video Title",
        "artist": "Creator Name",
        "duration": 120.5,
        "format_tags": {
            "key1": "value1",
            "key2": "value2"
        }
    },
    "transcript": [
        {
            "start": 0.0,
            "end": 2.5,
            "text": "Transcribed text..."
        }
    ],
    "moondream_prompt": "Prompt used for frame description",
    "frame_descriptions": [
        {
            "frame_number": 100,
            "timestamp": 4.0,
            "description": "Frame description..."
        }
    ],
    "summary": "Final video summary...",
    "total_run_time": 120.5,
    "synthesis": {
        "raw_output": "Raw synthesis output with XML tags",
        "captions": [
            {
                "timestamp": 0.0,
                "text": "Caption text..."
            }
        ]
    }
}
```

## Frame Analysis Visualization

When using `--frame-selection`, analysis plots are saved in `frame_analysis_plots/` showing:
- Sequential frame similarities
- Sliding window differences
- Identified key frames
- Cluster representative frames

Plot interpretation:
- Blue line: Frame-to-frame similarity
- Orange line: Sliding window differences
- Red dots: Novel frames
- Green dots: Cluster representatives

## Requirements

### Hardware Requirements
- Python 3.10+
- CUDA-compatible GPU (recommended)
- Minimum 16GB RAM
- Storage:
  - ~10GB for models
  - Additional space for video processing
- FFmpeg (for video processing)
- Internet connection (for hosted LLM)

### Key Dependencies
- `torch`: GPU acceleration
- `whisper`: Audio transcription
- `transformers`: LLM models
- `open_clip`: Frame analysis
- `gradio`: Web interface
- `opencv-python`: Video processing
- `numpy`: Numerical operations
- `matplotlib`: Visualization
- See `requirements.txt` for complete list

## Performance Considerations

1. **Frame Selection**
   - CLIP embeddings are cached in `embedding_cache/`
   - First run on a video will be slower
   - Subsequent runs use cached embeddings
   - Cache format: NumPy arrays (.npy)
   - Cache size: ~2MB per minute of video

2. **Model Loading**
   - Models are loaded/unloaded to manage memory
   - GPU memory requirements:
     - Whisper large-v3-turbo: ~4GB
     - Moondream: ~4GB
     - Llama (if local): ~8GB
     - CLIP: ~2GB
   - Total peak memory: ~12GB (with local LLM)

## Troubleshooting

1. **Out of Memory**
   - Reduce batch size in `describe_frames()`

## Recent Updates

### Debug Mode Addition
- New `--debug` flag for enhanced troubleshooting
- Comprehensive metadata overlay
- Real-time processing information
- Visual debug indicators
- Improved error tracking

### Audio Processing Improvements
- Enhanced timing and synchronization:
  - Predictive caption timing with 0.5s early display
  - Smart timestamp selection using earliest scene detection
  - Minimum 1.2s gap between captions
  - Automatic audio stream detection
- **NEW: Simplified Audio Pipeline**
  - Removed all audio manipulation logic for better reliability
  - Direct MP3 extraction from video
  - Raw MP3 passed to Whisper model
  - Cleaner transcription results
  - Reduced complexity and potential failure points
  - Better handling of audio streams

### Synthesis Improvements
- **NEW: Enhanced Retry Logic**
  - Increased retry attempts from 3 to 5 for all synthesis operations
  - Better exponential backoff timing
  - Improved error handling and recovery
  - Separate retries for:
    - Chunk synthesis
    - Final summary generation
    - Short video synthesis
    - API calls
  - More reliable caption generation
  - Better handling of synthesis failures

### Technical Improvements
- **NEW: Better Summary Extraction**
  - Improved parsing of synthesis output
  - Proper extraction of summary from XML format
  - Cleaner logging of summaries
  - Fixed summary display in web interface
  - Better error messages for synthesis failures
  - Proper validation of output format

## Advanced Features & Limitations

### Hallucination Detection Parameters
- Detection threshold: 0.08 words/second
- Minimum skip: 10 frames
- Number of clusters: 15

### Transcript Cleaning System
- Automatic removal of non-speech segments
- Post-processing for better transcription quality

### Caption Timing Specifications
- Predictive caption timing: 0.5s early display
- Minimum 1.2s gap between captions

### Error Recovery System
- Fallback to frame descriptions if synthesis fails
- Smart handling of caption timing edge cases

### Resolution-based Adaptations
- Adaptive font scaling based on video resolution

### Video Processing Limitations
- Maximum recommended video length: 6-8 minutes
- Automatic audio stream detection

### Debug Mode Features
- Detailed metadata overlay in video output:
  - Video Info: Duration, Resolution, FPS, Size
  - Caption Info: Video Type, Target Captions, Captions/Second, Calculation
- Enhanced logging of processing steps
- Visual indicators in debug mode:
  - "DEBUG" prefix in attribution text
  - Metadata overlay in top-left corner
- Automatic metadata extraction and display
- Real-time calculation visualization
- Helps with:
  - Troubleshooting caption timing
  - Verifying video properties
  - Understanding caption calculations
  - Monitoring processing flow

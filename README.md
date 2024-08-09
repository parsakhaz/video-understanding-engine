# Video Summarizer

A powerful video summarization tool using Moondream, CLIP, Llama 3.1, and Whisper.

## Features

- Transcribe video audio using Whisper
- Extract key frames using CLIP-based selection algorithm (optional)
- Describe frames using Moondream vision-language model
- Summarize video content using Llama 3.1 or gpt4o-mini

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/yourusername/video-summarizer.git
   cd video-summarizer
   ```

2. Create a virtual environment (optional but recommended):
   ```
   python -m venv venv
   source venv/bin/activate
   ```

3. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

4. Set up environment variables if needed:
   Create a `.env` file in the project root and add your OpenAI API key:
   ```
   OPENAI_API_KEY=your_api_key_here
   ```

## Usage

Run the main script with the following command:

```
python main.py <path_to_video_file> [options]
```

Options:
- `--debug`: Display moondream labeledframes with descriptions
- `--save`: Save output data to a JSON file
- `--local`: Use local Llama model for summarization instead of gpt4o-mini (not recommended due to quality issues)
- `--frame-selection`: Use CLIP-based key frame selection algorithm (requires CLIPing all frames and is mainly useful for reducing frames labeled in mundane videos)

Example:
```
python main.py path/to/your/video.mp4 --save --debug
```

## Requirements

- Python 3.10+
- CUDA-compatible GPU (for optimal performance)

See `requirements.txt` for a full list of Python package dependencies.

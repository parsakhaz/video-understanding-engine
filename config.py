import cv2

# =============================================================================
# CONFIGURATION CONSTANTS
# =============================================================================

# Video Processing Settings
VIDEO_SETTINGS = {
    'FRAME_SAMPLING_INTERVAL': 50,  # Interval for regular frame sampling
    'MIN_VIDEO_DURATION': 30,       # Seconds, threshold for short videos
    'MEDIUM_VIDEO_DURATION': 90,    # Seconds, threshold for medium videos
    'LONG_VIDEO_DURATION': 150,     # Seconds, threshold for long videos
    'OUTPUT_DIR': 'outputs',        # Directory for output files
    'SUPPORTED_EXTENSIONS': ('.mp4', '.avi', '.mov', '.mkv', '.webm')
}

# Frame Selection Settings
FRAME_SELECTION = {
    'NOVELTY_THRESHOLD': 0.08,      # Threshold for frame uniqueness
    'MIN_SKIP_FRAMES': 10,          # Minimum frames to skip
    'NUM_CLUSTERS': 15,             # Number of frame clusters
    'BATCH_SIZE': 8                 # Batch size for frame processing
}

# Model Settings
MODEL_SETTINGS = {
    'VISION_MODEL': "vikhyatk/moondream2",
    'VISION_MODEL_REVISION': "2025-01-09",
    'WHISPER_MODEL': "large",
    'LOCAL_LLM_MODEL': "meta-llama/Meta-Llama-3.1-8B-Instruct",
    'GPT_MODEL': "gpt-4o",
    'TEMPERATURE': 0.3
}

# UI/Display Settings
DISPLAY = {
    'FONT': cv2.FONT_HERSHEY_SIMPLEX,
    'FONT_SCALE': {
        'CAPTION': 0.7,
        'ATTRIBUTION': 0.6,
        'DEBUG': 0.5
    },
    'MARGIN': {
        'DEFAULT': 10,
        'TEXT': 40,
        'CAPTION': 8
    },
    'PADDING': 10,
    'LINE_SPACING': 10,
    'SECTION_SPACING': 35,
    'TEXT_COLOR': {
        'WHITE': (255, 255, 255),
        'GRAY': (200, 200, 200),
        'RED': (0, 0, 255)
    },
    'OVERLAY_ALPHA': 0.7,
    'MAX_WIDTH_RATIO': 0.9,         # Percentage of frame width for text
    'MIN_LINE_HEIGHT': 20
}

# Retry Settings
RETRY = {
    'MAX_RETRIES': 5,
    'MAX_RECONTEXTUALIZE_RETRIES': 3,
    'MAX_METADATA_RETRIES': 3,
    'INITIAL_DELAY': 1,             # Initial retry delay in seconds
    'MAX_VIDEO_RETRIES': 10         # Maximum retries for video processing
}

# Caption Generation Settings
CAPTION = {
    'SHORT_VIDEO': {
        'INTERVAL': 3.5,            # Seconds between captions for short videos
        'MIN_CAPTIONS': 4,
        'MAX_CAPTIONS': 100
    },
    'MEDIUM_VIDEO': {
        'MIN_CAPTIONS': 8,
        'BASE_INTERVAL': 5.0,       # Base interval for medium videos
        'MAX_INTERVAL': 7.0         # Maximum interval for medium videos
    },
    'LONG_VIDEO': {
        'MIN_CAPTIONS': 15,
        'INTERVAL_RATIO': 6.0       # One caption every 6 seconds
    },
    'TIMESTAMP_OFFSET': 0.1         # Seconds to offset timestamps earlier
}

# String Similarity Settings
SIMILARITY = {
    'THRESHOLD': 0.90,              # Threshold for string similarity comparison
    'NOISE_PHRASES': [
        "ambient music",
        "music playing",
        "background music",
        "music",
        "captions",
        "[Music]",
        "♪",
        "♫"
    ],
    'BOILERPLATE_PHRASES': [
        "work of fiction",
        "any resemblance",
        "coincidental",
        "unintentional",
        "all rights reserved",
        "copyright",
        "trademark"
    ]
} 
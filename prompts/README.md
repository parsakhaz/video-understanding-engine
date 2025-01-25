# Prompts

This directory contains the prompts used by the video understanding engine.

## Directory Structure

```
prompts/
├── synthesis/
│   ├── base.jsx         - Main synthesis prompt for generating summaries and captions
│   ├── recontextualize.jsx - Prompt for enriching summaries with metadata context
│   └── chunk.jsx        - Prompt for synthesizing summaries from video chunks
├── frame_description/
│   └── moondream.jsx    - Prompt for Moondream model to describe video frames
└── utils/
    └── prompt_loader.py - Utilities for loading prompts
```

## Usage

The prompts are stored in JSX format for better syntax highlighting and maintainability. They can be loaded using the utilities in `prompt_loader.py`:

```python
from prompts.utils.prompt_loader import (
    load_synthesis_base,
    load_synthesis_recontextualize,
    load_synthesis_chunk
)

# Load prompts
base_prompt = load_synthesis_base()
recontextualize_prompt = load_synthesis_recontextualize()
chunk_prompt = load_synthesis_chunk()
```

## Prompt Guidelines

1. All prompts should be stored in JSX format using template literals
2. Use descriptive variable names in template strings (e.g. `{num_captions}`)
3. Include clear formatting instructions and examples
4. Document any special requirements or constraints
5. Keep prompts focused and single-purpose
6. Use consistent XML-style tags for input/output sections 
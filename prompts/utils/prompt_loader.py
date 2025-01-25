"""Utility functions for loading prompts."""

import os
import re

def load_prompt(prompt_path: str) -> callable:
    """Load a prompt from a file.
    
    Args:
        prompt_path: Path to the prompt file relative to the prompts directory
        
    Returns:
        A function that takes a dict of variables and returns the formatted prompt
    """
    # Get the absolute path to the prompts directory
    prompts_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Construct full path to prompt file
    full_path = os.path.join(prompts_dir, prompt_path)
    
    def prompt_formatter(variables: dict = None) -> str:
        """Format the prompt with the given variables."""
        if variables is None:
            variables = {}
            
        # Read the JSX file
        with open(full_path, 'r', encoding='utf-8') as f:
            content = f.read()
            
        # Extract the template string
        template_match = re.search(r'`([^`]+)`', content)
        if not template_match:
            raise ValueError(f"No template string found in {prompt_path}")
            
        template = template_match.group(1)
        
        # If we have variables, replace ${var} with their values
        if variables:
            for key, value in variables.items():
                template = template.replace(f"${{{key}}}", str(value))
                
        return template

    return prompt_formatter

def load_synthesis_base() -> callable:
    """Load the base synthesis prompt."""
    return load_prompt('synthesis/base.jsx')

def load_metadata_context() -> callable:
    """Load the metadata context template."""
    return load_prompt('synthesis/metadata_context.jsx')

def load_synthesis_recontextualize() -> callable:
    """Load the recontextualization prompt."""
    return load_prompt('synthesis/recontextualize.jsx')

def load_synthesis_chunk() -> callable:
    """Load the chunk synthesis prompt."""
    return load_prompt('synthesis/chunk.jsx')

def load_moondream() -> callable:
    """Load the Moondream frame description prompt."""
    return load_prompt('frame_description/moondream.jsx') 
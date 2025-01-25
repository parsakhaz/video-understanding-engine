#!/usr/bin/env python3

import torch
import whisper
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import open_clip
from PIL import Image
import cv2
import os
from config import MODEL_SETTINGS
from contextlib import contextmanager

class ModelLoadError(Exception):
    """Custom exception for model loading errors."""
    pass

@contextmanager
def model_context(model_type: str):
    """Context manager for safely loading and cleaning up models."""
    model = None
    try:
        if model_type == "whisper":
            model = whisper.load_model(MODEL_SETTINGS['WHISPER_MODEL'])
        elif model_type == "moondream":
            model = (
                AutoModelForCausalLM.from_pretrained(
                    MODEL_SETTINGS['VISION_MODEL'],
                    revision=MODEL_SETTINGS['VISION_MODEL_REVISION'],
                    trust_remote_code=True,
                    device_map={"": "cuda"}
                ),
                AutoTokenizer.from_pretrained(
                    MODEL_SETTINGS['VISION_MODEL'], 
                    revision=MODEL_SETTINGS['VISION_MODEL_REVISION']
                )
            )
        elif model_type == "clip":
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model, _, preprocess = open_clip.create_model_and_transforms(
                'ViT-SO400M-14-SigLIP-384', 
                pretrained='webli', 
                device=device
            )
            model = (model.eval(), preprocess, device)
        elif model_type == "llama":
            if not check_gpu_memory(8):  # Check if we have enough GPU memory (8GB)
                raise ModelLoadError("Not enough GPU memory for LLama model")
            model = transformers.pipeline(
                "text-generation",
                model=MODEL_SETTINGS['LOCAL_LLM_MODEL'],
                model_kwargs={"torch_dtype": torch.bfloat16},
                device_map="auto",
            )
        else:
            raise ValueError(f"Unknown model type: {model_type}")

        if model is None:
            raise ModelLoadError(f"Failed to load {model_type} model")

        yield model

    except Exception as e:
        print(f"Error loading {model_type} model: {e}")
        yield None
    finally:
        cleanup_model(model)

def cleanup_model(model):
    """Clean up model resources."""
    try:
        if isinstance(model, tuple):
            for component in model:
                if component is not None:
                    try:
                        del component
                    except:
                        pass
        elif model is not None:
            del model
            
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
    except Exception as e:
        print(f"Error cleaning up model: {e}")

def get_total_gpu_memory():
    """Get total GPU memory available."""
    if not torch.cuda.is_available():
        return 0
    return torch.cuda.get_device_properties(0).total_memory

def get_used_gpu_memory():
    """Get currently used GPU memory."""
    if not torch.cuda.is_available():
        return 0
    return torch.cuda.memory_allocated()

def check_gpu_memory(required_memory_gb):
    """Check if there's enough GPU memory available."""
    if not torch.cuda.is_available():
        return False
    
    total_memory = get_total_gpu_memory()
    used_memory = get_used_gpu_memory()
    available_memory = total_memory - used_memory
    
    required_memory = required_memory_gb * (1024 ** 3)  # Convert GB to bytes
    return available_memory >= required_memory 
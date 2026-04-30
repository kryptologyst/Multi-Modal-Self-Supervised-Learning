"""Utility functions for device management and deterministic behavior."""

import os
import random
from typing import Optional, Union

import numpy as np
import torch


def get_device(device_type: str = "auto") -> torch.device:
    """
    Get the appropriate device for computation.
    
    Args:
        device_type: Device type ("auto", "cuda", "mps", "cpu")
        
    Returns:
        torch.device: The selected device
    """
    if device_type == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")
    else:
        return torch.device(device_type)


def set_seed(seed: int = 42, deterministic: bool = True) -> None:
    """
    Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value
        deterministic: Whether to use deterministic algorithms
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.use_deterministic_algorithms(True, warn_only=True)


def setup_device_and_seed(
    device_type: str = "auto", 
    seed: int = 42, 
    deterministic: bool = True
) -> torch.device:
    """
    Setup device and set random seeds.
    
    Args:
        device_type: Device type ("auto", "cuda", "mps", "cpu")
        seed: Random seed value
        deterministic: Whether to use deterministic algorithms
        
    Returns:
        torch.device: The selected device
    """
    device = get_device(device_type)
    set_seed(seed, deterministic)
    
    print(f"Using device: {device}")
    if device.type == "cuda":
        print(f"CUDA device: {torch.cuda.get_device_name()}")
        print(f"CUDA memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    elif device.type == "mps":
        print("Using Apple Silicon GPU (MPS)")
    
    return device


def move_to_device(data: Union[torch.Tensor, dict, list], device: torch.device) -> Union[torch.Tensor, dict, list]:
    """
    Move data to the specified device.
    
    Args:
        data: Data to move (tensor, dict, or list)
        device: Target device
        
    Returns:
        Data moved to the target device
    """
    if isinstance(data, torch.Tensor):
        return data.to(device)
    elif isinstance(data, dict):
        return {k: move_to_device(v, device) for k, v in data.items()}
    elif isinstance(data, list):
        return [move_to_device(item, device) for item in data]
    else:
        return data


def get_model_size(model: torch.nn.Module) -> dict:
    """
    Get model size information.
    
    Args:
        model: PyTorch model
        
    Returns:
        Dictionary with model size information
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    # Estimate memory usage (rough approximation)
    param_size = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.numel() * b.element_size() for b in model.buffers())
    
    return {
        "total_parameters": total_params,
        "trainable_parameters": trainable_params,
        "non_trainable_parameters": total_params - trainable_params,
        "estimated_memory_mb": (param_size + buffer_size) / 1024 / 1024,
    }

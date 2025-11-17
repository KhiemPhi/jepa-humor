import os, random
import numpy as np
import torch

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def device(gpu_id=None):
    """
    Get the device to use for training.
    
    Args:
        gpu_id: Optional GPU ID (int). If None, uses CUDA:0 if available, else CPU.
                If specified, uses CUDA:gpu_id if available, else CPU.
    
    Returns:
        torch.device: The device to use
    """
    if not torch.cuda.is_available():
        return torch.device("cpu")
    
    if gpu_id is None:
        return torch.device("cuda:0")
    
    if isinstance(gpu_id, int):
        if gpu_id < 0:
            return torch.device("cpu")
        if gpu_id >= torch.cuda.device_count():
            print(f"Warning: GPU {gpu_id} not available. Using GPU 0 instead.")
            return torch.device("cuda:0")
        return torch.device(f"cuda:{gpu_id}")
    
    # If gpu_id is a string like "cuda:1", use it directly
    return torch.device(gpu_id)

def save_ckpt(state, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(state, path)

def load_encoder_state(model, ckpt_path: str, map_location="cpu"):
    state = torch.load(ckpt_path, map_location=map_location)
    # supports either full JEPA state or encoder-only state
    if "encoder_state_dict" in state:
        model.load_state_dict(state["encoder_state_dict"])
    else:
        model.load_state_dict(state)
    return model

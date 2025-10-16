import os, random
import numpy as np
import torch

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

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

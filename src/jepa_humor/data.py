from datasets import load_dataset
from torch.utils.data import DataLoader
import torch
import numpy as np
import random



def load_hf_splits(name: str, test_size=0.2, seed=42):
    ds = load_dataset(name, 'mrpc')
   
    return ds

def make_dataloaders(train_ds, test_ds, batch_size=64, num_workers=4, shuffle=True):
    # Both datasets must yield dicts with at least {"text", "humor"}
    collate = lambda batch: {
        "sent1": [b["sent1"] for b in batch],
        "sent2": [b["sent2"] for b in batch],
        "label": [bool(b["label"]) for b in batch]
    }
    # fixed generator for deterministic shuffling
    seed = 42
    g = torch.Generator()
    g.manual_seed(seed)

    # ensure each worker is seeded deterministically
    def seed_worker(worker_id):
        worker_seed = seed + worker_id
        np.random.seed(worker_seed)
        random.seed(worker_seed)
        torch.manual_seed(worker_seed)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate,
        pin_memory=True,
        generator=g,
        worker_init_fn=seed_worker
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate,
        pin_memory=True
    )

    return train_loader, test_loader

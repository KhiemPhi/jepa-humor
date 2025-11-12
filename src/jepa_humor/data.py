from datasets import load_dataset
from torch.utils.data import DataLoader
import torch
import numpy as np
import random
from torch.utils.data import WeightedRandomSampler


def load_hf_splits(name: str, test_size=0.2, seed=42):
    ds = load_dataset(name, 'rte')
   
    return ds

def load_hf_splits_test(name: str, test_size=0.2, seed=42):
    ds = load_dataset(name, 'adv_rte')
   
    return ds

def make_dataloaders(train_ds, test_ds, batch_size=64, num_workers=4, shuffle=True, tokenizer=None):
    # Both datasets must yield dicts with at least {"text", "humor"}
    collate = lambda batch: {
        "sentence1": [b["sentence1"] for b in batch],
        "sentence2": [b["sentence2"] for b in batch],
        "label": [bool(b["label"]) for b in batch]
    }
    # fixed generator for deterministic shuffling
    seed = 42
    g = torch.Generator()
    g.manual_seed(seed)
    
        
    train_labels = np.asarray(train_ds["label"])
    class_counts = np.bincount(train_labels)
    weights = 1.0 / class_counts
    sample_weights = weights[train_labels]
    sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)

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
        num_workers=0,
        collate_fn=collate,
        pin_memory=True
    )

    return train_loader, test_loader

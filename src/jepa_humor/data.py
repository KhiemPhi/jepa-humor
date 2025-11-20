from datasets import load_dataset
from torch.utils.data import DataLoader
import torch
import numpy as np
import random
from torch.utils.data import WeightedRandomSampler


def load_hf_splits(name: str, test_size=0.2, seed=42):
    ds = load_dataset(name)
   
    return ds


def make_dataloaders(train_ds, test_ds, batch_size=64, num_workers=4, shuffle=True, tokenizer=None):
    # Both datasets must yield dicts with at least {"text", "humor"}
    collate_train = lambda batch: {
        "input_ids": torch.stack([torch.tensor(b["input_ids"]).t() for b in batch]),
        "token_type_ids": torch.stack([torch.tensor(b["token_type_ids"]) for b in batch]),
        "attention_mask": torch.stack([torch.tensor(b["attention_mask"]).t() for b in batch]),
        "start_positions": [b["start_positions"] for b in batch],
        "end_positions": [b["end_positions"] for b in batch]
    }

    collate_test = lambda batch: {
        "input_ids": torch.stack([torch.tensor(b["input_ids"]).t() for b in batch]),
        "token_type_ids": torch.stack([torch.tensor(b["token_type_ids"]) for b in batch]),
        "attention_mask": torch.stack([torch.tensor(b["attention_mask"]).t() for b in batch]),
        "offset_mapping": [b["offset_mapping"] for b in batch],
        "example_id": [b["example_id"] for b in batch]
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
        collate_fn=collate_train,
        pin_memory=True,
        generator=g,
        worker_init_fn=seed_worker
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_test,
        pin_memory=True
    )

    return train_loader, test_loader
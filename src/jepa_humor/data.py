from datasets import load_dataset, Dataset
from torch.utils.data import DataLoader
import torch
import numpy as np
import random
import json
import os



def load_hf_splits(name: str, test_size=0.2, seed=42):
    ds = load_dataset(name, 'qnli')
   
    return ds
    
def load_hf_splits_test(name: str, test_size=0.2, seed=42):
    ds = load_dataset(name, 'adv_qnli')
   
    return ds

def load_hf_splits_test_mismatched(name: str, test_size=0.2, seed=42):
    ds = load_dataset(name, 'adv_qnli')
   
    return ds

def load_mnli_test_from_json(json_path: str = None):
    """Load MNLI test set from test_ann.json file."""
    if json_path is None:
        # Default path relative to this file
        json_path = os.path.join(os.path.dirname(__file__), "test_ann.json")
    
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Extract MNLI test data
    mnli_data = data.get("mnli", [])
    
    # Convert to format expected by the model (premise, hypothesis, label)
    # Filter out any entries missing required fields and exclude items with method="glue"
    processed_data = []
    for item in mnli_data:
        if "premise" in item and "hypothesis" in item and "label" in item:
            # Exclude items with method="glue"
            if item.get("method") != "glue":
                processed_data.append({
                    "premise": item["premise"],
                    "hypothesis": item["hypothesis"],
                    "label": item["label"]
                })
    
    # Create Dataset from list
    dataset = Dataset.from_list(processed_data)
    
    # Return in the same format as load_dataset (with "test" key)
    return {"test": dataset}

def load_mnli_mm_test_from_json(json_path: str = None):
    """Load MNLI-MM (mismatched) test set from test_ann.json file."""
    if json_path is None:
        # Default path relative to this file
        json_path = os.path.join(os.path.dirname(__file__), "test_ann.json")
    
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Extract MNLI-MM test data
    mnli_mm_data = data.get("mnli-mm", [])
    
    # Convert to format expected by the model (premise, hypothesis, label)
    # Filter out any entries missing required fields and exclude items with method="glue"
    processed_data = []
    for item in mnli_mm_data:
        if "premise" in item and "hypothesis" in item and "label" in item:
            # Exclude items with method="glue"
            if item.get("method") != "glue":
                processed_data.append({
                    "premise": item["premise"],
                    "hypothesis": item["hypothesis"],
                    "label": item["label"]
                })
    
    # Create Dataset from list
    dataset = Dataset.from_list(processed_data)
    
    # Return in the same format as load_dataset (with "test" key)
    return {"test": dataset}

def load_qqp_test_from_json(json_path: str = None):
    """Load QQP test set from test_ann.json file."""
    if json_path is None:
        # Default path relative to this file
        json_path = os.path.join(os.path.dirname(__file__), "test_ann.json")
    
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Extract QQP test data
    qqp_data = data.get("qqp", [])
    
    # Convert to format expected by the model (question1, question2, label)
    # Filter out any entries missing required fields and exclude items with method="glue"
    processed_data = []
    for item in qqp_data:
        if "question1" in item and "question2" in item and "label" in item:
            # Exclude items with method="glue"
            if item.get("method") != "glue":
                processed_data.append({
                    "question1": item["question1"],
                    "question2": item["question2"],
                    "label": item["label"]
                })
    
    # Create Dataset from list
    dataset = Dataset.from_list(processed_data)
    
    # Return in the same format as load_dataset (with "test" key)
    return {"test": dataset}

def load_qnli_test_from_json(json_path: str = None):
    """Load QNLI test set from test_ann.json file."""
    if json_path is None:
        # Default path relative to this file
        json_path = os.path.join(os.path.dirname(__file__), "test_ann.json")
    
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Extract QQP test data
    qnli_data = data.get("qnli", [])
    
    # Convert to format expected by the model (question1, question2, label)
    # Filter out any entries missing required fields and exclude items with method="glue"
    processed_data = []
    for item in qnli_data:
        if "question" in item and "sentence" in item and "label" in item:
            # Exclude items with method="glue"
            if item.get("method") != "glue":
                processed_data.append({
                    "question": item["question"],
                    "sentence": item["sentence"],
                    "label": item["label"]
                })
    
    # Create Dataset from list
    dataset = Dataset.from_list(processed_data)
    
    # Return in the same format as load_dataset (with "test" key)
    return {"test": dataset}

def preprocess_dataset(ds, tokenizer, max_length=256):
    """Pre-tokenize dataset to avoid tokenizing during training."""
    def tokenize_function(examples):
        return tokenizer(
            examples["question"],
            examples["sentence"],
            truncation=True,
            padding="max_length",
            max_length=max_length,
        )
    
    ds = ds.map(
        tokenize_function,
        batched=True,
        desc="Tokenizing dataset",
        remove_columns=[col for col in ds.column_names if col not in ["input_ids", "attention_mask", "label"]]
    )
    return ds

def preprocess_dataset_ssl(ds, tokenizer, max_length=256):
    """Pre-tokenize dataset for SSL training (tokenizes sent1 and sent2 separately)."""
    def tokenize_function(examples):
        # Tokenize question1 (sent1) separately
        keep_tok = tokenizer(
            examples["question"],
            truncation=True,
            padding="max_length",
            max_length=max_length,
        )
        # Tokenize question2 (sent2) separately
        target_tok = tokenizer(
            examples["sentence"],
            truncation=True,
            padding="max_length",
            max_length=max_length,
        )
        # Return both tokenizations with prefixed keys
        return {
            "keep_input_ids": keep_tok["input_ids"],
            "keep_attention_mask": keep_tok["attention_mask"],
            "target_input_ids": target_tok["input_ids"],
            "target_attention_mask": target_tok["attention_mask"],
        }
    
    ds = ds.map(
        tokenize_function,
        batched=True,
        desc="Tokenizing SSL dataset",
        remove_columns=[col for col in ds.column_names if col not in ["keep_input_ids", "keep_attention_mask", "target_input_ids", "target_attention_mask", "label"]]
    )
    return ds

def make_dataloaders(train_ds, test_ds, batch_size=64, num_workers=4, shuffle=True, pre_tokenized=False, ssl_mode=False):
    # Collate function for pre-tokenized SSL data
    if pre_tokenized and ssl_mode:
        def collate(batch):
            # For SSL: keep_tok and target_tok are separate
            keep_input_ids = torch.tensor([b["keep_input_ids"] for b in batch], dtype=torch.long)
            keep_attention_mask = torch.tensor([b["keep_attention_mask"] for b in batch], dtype=torch.long)
            target_input_ids = torch.tensor([b["target_input_ids"] for b in batch], dtype=torch.long)
            target_attention_mask = torch.tensor([b["target_attention_mask"] for b in batch], dtype=torch.long)
            labels = torch.tensor([b["label"] for b in batch], dtype=torch.long)
            return {
                "keep_input_ids": keep_input_ids,
                "keep_attention_mask": keep_attention_mask,
                "target_input_ids": target_input_ids,
                "target_attention_mask": target_attention_mask,
                "label": labels
            }
    # Collate function for pre-tokenized data (finetune mode)
    elif pre_tokenized:
        def collate(batch):
            # input_ids and attention_mask are already lists from tokenizer
            input_ids = torch.tensor([b["input_ids"] for b in batch], dtype=torch.long)
            attention_mask = torch.tensor([b["attention_mask"] for b in batch], dtype=torch.long)
            labels = torch.tensor([b["label"] for b in batch], dtype=torch.long)
            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "label": labels
            }
    else:
        # Original collate for non-tokenized data
        collate = lambda batch: {
            "question": [b["question"] for b in batch],
            "sentence": [b["sentence"] for b in batch],
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

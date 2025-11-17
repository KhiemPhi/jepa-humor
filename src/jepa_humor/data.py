from datasets import load_dataset, Dataset
from torch.utils.data import DataLoader
import torch
import numpy as np
import random
import json
import os



def load_hf_splits(name: str, config: str = None, test_size=0.2, seed=42):
    """
    Load HuggingFace dataset splits.
    
    Args:
        name: Dataset name (e.g., "nyu-mll/glue")
        config: Dataset configuration (e.g., "qnli", "mnli"). If None, will try to infer from name.
        test_size: Not used, kept for compatibility
        seed: Not used, kept for compatibility
    
    Returns:
        Dataset dict with train/validation/test splits
    """
    if config is None:
        # Try to infer config from name or default to qnli
        if 'mnli' in name.lower():
            config = 'mnli'
        else:
            config = 'qnli'
    
    ds = load_dataset(name, config)
    return ds
    
def load_hf_splits_test(name: str, config: str = None, test_size=0.2, seed=42):
    """
    Load test/validation dataset for evaluation.
    For MNLI, returns matched validation set.
    """
    if config is None:
        if 'mnli' in name.lower():
            config = 'mnli'
        else:
            config = 'adv_qnli'
    
    if config == 'mnli':
        # For MNLI, return matched validation set
        # MNLI has separate validation_matched and validation_mismatched splits
        ds = load_dataset(name, config)
        if "validation_matched" in ds:
            return {"validation": ds["validation_matched"]}
        elif "validation" in ds:
            return {"validation": ds["validation"]}
        else:
            raise ValueError(f"MNLI dataset does not have expected validation splits. Available: {list(ds.keys())}")
    else:
        ds = load_dataset(name, config)
        return ds

def load_hf_splits_test_mismatched(name: str, config: str = None, test_size=0.2, seed=42):
    """
    Load mismatched test/validation dataset.
    For MNLI, returns mismatched validation set.
    """
    if config is None:
        if 'mnli' in name.lower():
            config = 'mnli'
        else:
            config = 'adv_qnli'
    
    if config == 'mnli':
        # For MNLI, return mismatched validation set
        # MNLI has separate validation_matched and validation_mismatched splits
        ds = load_dataset(name, config)
        if "validation_mismatched" in ds:
            return {"validation": ds["validation_mismatched"]}
        else:
            raise ValueError(f"MNLI dataset does not have validation_mismatched split. Available: {list(ds.keys())}")
    else:
        ds = load_dataset(name, config)
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
    
    # Extract QNLI test data
    qnli_data = data.get("qnli", [])
    
    # Convert to format expected by the model (question, sentence, label)
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

def load_test_set_from_json(dataset_config: str, json_path: str = None):
    """
    Load test set from JSON file based on dataset configuration.
    
    Args:
        dataset_config: Dataset configuration name (e.g., "qnli", "mnli", "qqp", "rte")
        json_path: Path to test_ann.json file. If None, uses default path.
    
    Returns:
        Dictionary with "test" key containing the test dataset, or None if not found
    """
    if json_path is None:
        json_path = os.path.join(os.path.dirname(__file__), "test_ann.json")
    
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Warning: test_ann.json not found at {json_path}")
        return None
    
    # Map dataset config to JSON key and expected fields
    config_map = {
        "qnli": ("qnli", ["question", "sentence"]),
        "mnli": ("mnli", ["premise", "hypothesis"]),
        "mnli-mm": ("mnli-mm", ["premise", "hypothesis"]),
        "qqp": ("qqp", ["question1", "question2"]),
        "rte": ("rte", ["sentence1", "sentence2"]),  # RTE typically uses sentence1/sentence2
        "sst2": ("sst2", ["sentence"]),
    }
    
    if dataset_config not in config_map:
        print(f"Warning: No test set loader for dataset_config '{dataset_config}'")
        return None
    
    json_key, expected_fields = config_map[dataset_config]
    test_data = data.get(json_key, [])
    
    if not test_data:
        print(f"Warning: No test data found for '{json_key}' in JSON file")
        return None
    
    # Convert to format expected by the model
    processed_data = []
    for item in test_data:
        # Check if all expected fields are present
        if all(field in item for field in expected_fields) and "label" in item:
            # Exclude items with method="glue"
            if item.get("method") != "glue":
                processed_item = {field: item[field] for field in expected_fields}
                processed_item["label"] = item["label"]
                processed_data.append(processed_item)
    
    if not processed_data:
        print(f"Warning: No valid test data found for '{json_key}' after filtering")
        return None
    
    # Create Dataset from list
    dataset = Dataset.from_list(processed_data)
    
    # Return in the same format as load_dataset (with "test" key)
    return {"test": dataset}

def preprocess_dataset(ds, tokenizer, max_length=256, text_keys=None):
    """
    Pre-tokenize dataset to avoid tokenizing during training.
    
    Args:
        ds: Dataset to preprocess
        tokenizer: Tokenizer to use
        max_length: Maximum sequence length
        text_keys: List of column names to tokenize (e.g., ["question", "sentence"] or ["premise", "hypothesis"])
                   If None, defaults to ["question", "sentence"]
    """
    if text_keys is None:
        text_keys = ["question", "sentence"]
    
    def tokenize_function(examples):
        # Get the text columns to tokenize
        texts = [examples[key] for key in text_keys]
        return tokenizer(
            *texts,
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

def preprocess_dataset_ssl(ds, tokenizer, max_length=256,
                           mask_probability=0.15, span_mean=3, text_keys=None):
    """
    Tokenize text fields together, then create masked target versions
    using SpanBERT-style random span masking.
    
    Args:
        ds: Dataset to preprocess
        tokenizer: Tokenizer to use
        max_length: Maximum sequence length
        mask_probability: Probability of masking tokens
        span_mean: Mean span length for masking
        text_keys: List of column names to tokenize (e.g., ["question", "sentence"] or ["premise", "hypothesis"])
                   If None, defaults to ["question", "sentence"]
    """

    import numpy as np

    if text_keys is None:
        text_keys = ["question", "sentence"]

    MASK_ID = tokenizer.mask_token_id
    PAD_ID = tokenizer.pad_token_id

    def random_span_mask(ids, mask_prob, span_mean):
        """SpanBERT-style random contiguous span masking."""
        ids = ids.copy()
        length = len(ids)

        # how many tokens to mask total
        num_to_mask = int(length * mask_prob)
        if num_to_mask == 0:
            return ids

        masked = 0
        while masked < num_to_mask:
            # choose random span length (Poisson)
            span_len = max(1, np.random.poisson(span_mean))
            start = np.random.randint(0, max(1, length - span_len))

            for i in range(start, min(start + span_len, length)):
                if masked >= num_to_mask:
                    break
                if ids[i] != PAD_ID:   # don't mask padding
                    ids[i] = MASK_ID
                    masked += 1

        return ids

    def tokenize_function(examples):
        # Tokenize text fields together
        texts = [examples[key] for key in text_keys]
        encoded = tokenizer(
            *texts,
            truncation=True,
            padding="max_length",
            max_length=max_length,
        )

        keep_ids = encoded["input_ids"]
        keep_mask = encoded["attention_mask"]

        # Now create masked copies
        target_ids = [
            random_span_mask(seq, mask_probability, span_mean)
            for seq in keep_ids
        ]

        return {
            "keep_input_ids": keep_ids,      # original text
            "keep_attention_mask": keep_mask,
            "target_input_ids": target_ids,  # masked version for SSL
            "target_attention_mask": keep_mask,
        }

    ds = ds.map(
        tokenize_function,
        batched=True,
        desc="Tokenizing SSL dataset",
        remove_columns=[
            col for col in ds.column_names
            if col not in [
                "keep_input_ids",
                "keep_attention_mask",
                "target_input_ids",
                "target_attention_mask",
                "label",
            ]
        ]
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
        # This is a fallback and should not be used with the new configurable keys
        # If needed, text_keys should be passed to make_dataloaders
        collate = lambda batch: {
            "question": [b.get("question", b.get("premise", "")) for b in batch],
            "sentence": [b.get("sentence", b.get("hypothesis", "")) for b in batch],
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

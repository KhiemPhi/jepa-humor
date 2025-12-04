from datasets import load_dataset, Dataset
from torch.utils.data import DataLoader
import torch
import numpy as np
import random
import json
import os


def sample_dataset(ds, sample_percent=10.0, seed=42):
    """
    Sample a percentage of a dataset for quick prototyping.
    
    Args:
        ds: HuggingFace Dataset or DatasetDict to sample from
        sample_percent: Percentage of data to keep (default 10.0)
        seed: Random seed for reproducibility
    
    Returns:
        Sampled dataset (same type as input)
    """
    if sample_percent is None or sample_percent >= 100.0:
        return ds
    
    if isinstance(ds, dict):
        # DatasetDict: sample each split
        sampled = {}
        for split_name, split_ds in ds.items():
            if len(split_ds) == 0:
                sampled[split_name] = split_ds
                continue
            num_samples = max(1, int(len(split_ds) * sample_percent / 100.0))
            sampled[split_name] = split_ds.shuffle(seed=seed).select(range(num_samples))
        return type(ds)(sampled)
    else:
        # Single Dataset
        if len(ds) == 0:
            return ds
        num_samples = max(1, int(len(ds) * sample_percent / 100.0))
        return ds.shuffle(seed=seed).select(range(num_samples))


def load_anli_splits(name: str = "facebook/anli", train_rounds: list = [1, 2, 3], seed: int = 42):
    """
    Load ANLI dataset and aggregate training sets from specified rounds.
    
    Args:
        name: Dataset name (default: "facebook/anli")
        train_rounds: List of rounds to aggregate (e.g., [1, 2, 3] for all, [1] for only r1)
        seed: Random seed for shuffling
    
    Returns:
        Dataset dict with aggregated train split and validation split
    """
    ds = load_dataset(name)
    
    # Aggregate training sets from specified rounds
    train_datasets = []
    for r in train_rounds:
        split_name = f"train_r{r}"
        if split_name in ds:
            train_datasets.append(ds[split_name])
            print(f"Loaded {split_name}: {len(ds[split_name])} examples")
        else:
            print(f"Warning: {split_name} not found in dataset")
    
    if not train_datasets:
        raise ValueError(f"No training data found for rounds {train_rounds}")
    
    # Concatenate all training datasets
    from datasets import concatenate_datasets
    aggregated_train = concatenate_datasets(train_datasets)
    print(f"Aggregated training set size: {len(aggregated_train)} examples")
    
    # Use dev_r1 as validation (or first available dev split)
    validation_split = None
    for r in [1, 2, 3]:
        if f"dev_r{r}" in ds:
            validation_split = ds[f"dev_r{r}"]
            print(f"Using dev_r{r} as validation: {len(validation_split)} examples")
            break
    
    if validation_split is None:
        raise ValueError("No validation split found in ANLI dataset")
    
    return {
        "train": aggregated_train,
        "validation": validation_split
    }


def load_anli_test_splits(name: str = "facebook/anli"):
    """
    Load ANLI test sets for all rounds separately.
    
    Args:
        name: Dataset name (default: "facebook/anli")
    
    Returns:
        Dictionary with test_r1, test_r2, test_r3 (if available)
    """
    ds = load_dataset(name)
    
    test_splits = {}
    for r in [1, 2, 3]:
        test_split_name = f"test_r{r}"
        if test_split_name in ds:
            test_splits[test_split_name] = ds[test_split_name]
            print(f"Loaded {test_split_name}: {len(ds[test_split_name])} examples")
    
    if not test_splits:
        raise ValueError("No test splits found in ANLI dataset")
    
    return test_splits


def load_composite_nli_dataset(
    snli_subset: str = None,
    mnli_splits: list = None,
    anli_train_rounds: list = None,
    anli_val_rounds: list = None,
    anli_val_type: str = "test",  # "test" or "dev"
):
    """
    Build a composite dataset from any combination of SNLI, MNLI, and ANLI rounds.
    
    Args:
        snli_subset: SNLI split to include (e.g., "train", None to skip)
        mnli_splits: List of MNLI splits to include (e.g., ["validation_matched"], None to skip)
        anli_train_rounds: List of ANLI train rounds to include (e.g., [1, 2, 3], None to skip)
        anli_val_rounds: List of ANLI rounds to use for validation (e.g., [1, 2, 3])
        anli_val_type: Type of ANLI splits for validation ("test" or "dev")
    
    Returns:
        Dictionary with "train" and "validation" datasets
    """
    from datasets import concatenate_datasets

    train_parts = []
    
    # Load SNLI if specified
    if snli_subset is not None:
        snli = load_dataset("snli")
        if snli_subset not in snli:
            raise ValueError(f"SNLI split '{snli_subset}' not found. Available: {list(snli.keys())}")
        snli_split = snli[snli_subset]
        snli_split = snli_split.filter(lambda x: x["label"] != -1)
        train_parts.append(snli_split)
        print(f"Loaded SNLI {snli_subset}: {len(snli_split)} examples")

    # Load MNLI splits if specified
    if mnli_splits is not None and len(mnli_splits) > 0:
        mnli = load_dataset("nyu-mll/glue", "mnli")
        mnli_parts = []
        for split_name in mnli_splits:
            if split_name not in mnli:
                print(f"Warning: MNLI split '{split_name}' not found. Available: {list(mnli.keys())}")
                continue
            mnli_part = mnli[split_name]
            mnli_parts.append(mnli_part)
            print(f"Loaded MNLI {split_name}: {len(mnli_part)} examples")
        
        if mnli_parts:
            if len(mnli_parts) > 1:
                mnli_combined = concatenate_datasets(mnli_parts)
            else:
                mnli_combined = mnli_parts[0]
            train_parts.append(mnli_combined)
            print(f"Combined MNLI size: {len(mnli_combined)} examples")

    # Load ANLI train rounds if specified
    if anli_train_rounds is not None and len(anli_train_rounds) > 0:
        anli = load_dataset("facebook/anli")
        anli_train_parts = []
        for r in anli_train_rounds:
            split_name = f"train_r{r}"
            if split_name in anli:
                anli_train_parts.append(anli[split_name])
                print(f"Loaded ANLI {split_name}: {len(anli[split_name])} examples")
            else:
                print(f"Warning: {split_name} not found in ANLI dataset")
        
        if anli_train_parts:
            if len(anli_train_parts) > 1:
                anli_train_combined = concatenate_datasets(anli_train_parts)
            else:
                anli_train_combined = anli_train_parts[0]
            train_parts.append(anli_train_combined)
            print(f"Combined ANLI train size: {len(anli_train_combined)} examples")

    # Combine all training parts
    if not train_parts:
        raise ValueError("No training data specified. At least one of snli_subset, mnli_splits, or anli_train_rounds must be provided.")
    
    if len(train_parts) > 1:
        train_concat = concatenate_datasets(train_parts)
    else:
        train_concat = train_parts[0]
    print(f"Total combined training size: {len(train_concat)} examples")

    # Load validation from ANLI
    if anli_val_rounds is not None and len(anli_val_rounds) > 0:
        anli = load_dataset("facebook/anli")
        val_parts = []
        for r in anli_val_rounds:
            split_name = f"{anli_val_type}_r{r}"
            if split_name in anli:
                val_parts.append(anli[split_name])
                print(f"Included ANLI {split_name} for validation: {len(anli[split_name])} examples")
            else:
                print(f"Warning: {split_name} not found in ANLI dataset")

        if not val_parts:
            raise ValueError(f"No ANLI {anli_val_type} splits were found to use as validation.")

        validation_concat = concatenate_datasets(val_parts) if len(val_parts) > 1 else val_parts[0]
        print(f"Validation (ANLI {anli_val_type}) size: {len(validation_concat)} examples")
    else:
        # Fallback: use first available dev split from one of the datasets
        print("Warning: No ANLI validation rounds specified. Using first available dev split.")
        validation_concat = None

    return {
        "train": train_concat,
        "validation": validation_concat,
    }


def expand_swag_for_pretraining(ds_split):
    """
    Expand SWAG multiple-choice examples for pretraining.
    Creates one example per ending (context + ending) for SSL pretraining.
    Labels are preserved for reference but not used in pretraining.
    """
    records = []
    for ex in ds_split:
        sent1 = ex.get("sent1", "")
        sent2 = ex.get("sent2", "")
        prompt = f"{sent1} {sent2}".strip()
        endings = [ex.get(f"ending{i}", "") for i in range(4)]
        label = ex.get("label", -1)
        for idx, ending in enumerate(endings):
            records.append({
                "question": prompt,
                "sentence": ending,
                "label": label,  # Keep original label (0-3) for reference
            })
    return Dataset.from_list(records)


def load_swag():
    """
    Load SWAG dataset in native multiple-choice format.
    Returns train/validation/test splits with original SWAG fields: sent1, sent2, ending0-3, label.
    """
    swag = load_dataset("swag", "regular")
    data = {}
    for split_name in ["train", "validation", "test"]:
        if split_name in swag:
            print(f"Loaded SWAG {split_name} split: {len(swag[split_name])} examples")
            data[split_name] = swag[split_name]
        elif split_name in ["train", "validation"]:
            # train and validation are required
            raise ValueError(f"SWAG dataset missing '{split_name}' split.")
        # test split is optional, so we don't raise an error if it's missing
    return data


def load_snli_mnli_for_anli(
    snli_subset: str = "train",
    mnli_splits: list = ["validation_matched", "validation_mismatched"],
    anli_rounds = (1, 2, 3),
):
    """
    Build a composite dataset where training data comes from SNLI (train)
    concatenated with MNLI dev (both matched and mismatched), and validation data comes from ANLI dev rounds.
    This is handy for pretraining on SNLI+MNLI dev while monitoring on ANLI.
    """
    from datasets import concatenate_datasets

    # Load SNLI and drop unlabeled items (label == -1)
    snli = load_dataset("snli")
    snli_split = snli[snli_subset]
    snli_split = snli_split.filter(lambda x: x["label"] != -1)
    print(f"Loaded SNLI {snli_subset}: {len(snli_split)} examples")

    # Load MNLI dev splits (both matched and mismatched by default)
    mnli = load_dataset("nyu-mll/glue", "mnli")
    mnli_parts = []
    for split_name in mnli_splits:
        if split_name not in mnli:
            print(f"Warning: MNLI split '{split_name}' not found. Available: {list(mnli.keys())}")
            continue
        mnli_part = mnli[split_name]
        mnli_parts.append(mnli_part)
        print(f"Loaded MNLI {split_name}: {len(mnli_part)} examples")
    
    if not mnli_parts:
        raise ValueError(f"No valid MNLI splits found. Requested: {mnli_splits}, Available: {list(mnli.keys())}")
    
    # Concatenate all MNLI parts
    if len(mnli_parts) > 1:
        mnli_dev = concatenate_datasets(mnli_parts)
    else:
        mnli_dev = mnli_parts[0]
    print(f"Combined MNLI dev size: {len(mnli_dev)} examples")

    train_concat = concatenate_datasets([snli_split, mnli_dev])
    print(f"Combined SNLI+MNLI train size: {len(train_concat)} examples")

    # Validation from ANLI dev rounds
    anli = load_dataset("facebook/anli")
    val_parts = []
    for r in anli_rounds:
        split_name = f"dev_r{r}"
        if split_name in anli:
            val_parts.append(anli[split_name])
            print(f"Included ANLI {split_name}: {len(anli[split_name])} examples")
        else:
            print(f"Warning: {split_name} not found in ANLI dataset")

    if not val_parts:
        raise ValueError("No ANLI dev splits were found to use as validation.")

    validation_concat = concatenate_datasets(val_parts) if len(val_parts) > 1 else val_parts[0]
    print(f"Validation (ANLI dev) size: {len(validation_concat)} examples")

    return {
        "train": train_concat,
        "validation": validation_concat,
    }


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

def preprocess_swag_multiple_choice(ds, tokenizer, max_length=256):
    """
    Preprocess SWAG dataset for multiple-choice classification.
    Expands each example into 4 examples (one per ending) for training.
    Each example has: context (sent1 + sent2) + ending, with label indicating correct choice (0-3).
    Also includes example_idx to group choices back together during evaluation.
    """
    from datasets import Dataset
    
    records = []
    for ex_idx, ex in enumerate(ds):
        sent1 = ex.get("sent1", "")
        sent2 = ex.get("sent2", "")
        context = f"{sent1} {sent2}".strip()
        endings = [ex.get(f"ending{i}", "") for i in range(4)]
        label = ex.get("label", -1)
        
        # Create 4 examples, one per ending
        for choice_idx, ending in enumerate(endings):
            # Tokenize context + ending
            tokenized = tokenizer(
                context,
                ending,
                truncation=True,
                padding="max_length",
                max_length=max_length,
            )
            records.append({
                "input_ids": tokenized["input_ids"],
                "attention_mask": tokenized["attention_mask"],
                "label": label,  # Original label (0-3) indicating correct choice
                "choice_idx": choice_idx,  # Which choice this is (0-3)
                "example_idx": ex_idx,  # Original example index for grouping
            })
    
    return Dataset.from_list(records)


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

def preprocess_dataset_ssl(
    ds,
    tokenizer,
    max_length=256,
    mask_probability=0.15,
    span_mean=3,
    text_keys=None,
    mask_type="random_span",
):
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

    def random_token_mask(ids, mask_prob, span_mean=None):
        """
        BERT-style random token masking.
        Randomly selects individual tokens to mask (not contiguous spans).
        """
        ids = ids.copy()
        length = len(ids)

        # Get maskable token indices (exclude padding and special tokens)
        CLS_ID = tokenizer.cls_token_id if tokenizer.cls_token_id is not None else -1
        SEP_ID = tokenizer.sep_token_id if tokenizer.sep_token_id is not None else -1
        maskable_indices = [
            i for i in range(length)
            if ids[i] not in (PAD_ID, CLS_ID, SEP_ID)
        ]

        if not maskable_indices:
            return ids

        # Calculate how many tokens to mask
        num_to_mask = max(1, int(len(maskable_indices) * mask_prob))
        num_to_mask = min(num_to_mask, len(maskable_indices))

        # Randomly sample tokens to mask
        selected_indices = np.random.choice(maskable_indices, size=num_to_mask, replace=False)

        # Mask the selected tokens
        for idx in selected_indices:
            ids[idx] = MASK_ID

        return ids

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

    def sentiment_aware_mask(ids, tokenizer, mask_prob, span_mean):
        """
        Mask tokens using a sentiment-aware strategy.
        Higher probability to mask sentiment-bearing tokens.
        Avoids masking negations and uninformative tokens.
        Designed for SST-2 sentiment classification tasks.
        """
        # Simple sentiment lexicon (extend with your own)
        SENTIMENT_WORDS = {
            "good", "great", "excellent", "amazing", "positive", "wonderful",
            "bad", "terrible", "awful", "horrible", "negative", "worst",
            "love", "hate", "like", "dislike", "boring", "fantastic"
        }

        NEGATION_WORDS = {"not", "never", "no", "n't"}

        ids = ids.copy()
        length = len(ids)

        # identify which tokens correspond to sentiment words
        tokens = tokenizer.convert_ids_to_tokens(ids)

        # assign masking weights
        weights = []
        for tok in tokens:
            w = 1.0
            base = tok.lower().strip()

            if base in SENTIMENT_WORDS:
                w = 5.0   # strong mask probability
            elif base in NEGATION_WORDS:
                w = 0.2   # rarely mask negations
            elif base.startswith("##"):
                w = 0.7   # subword pieces: moderate
            else:
                w = 1.0   # normal

            # don't mask pad tokens or special tokens
            if tok == tokenizer.pad_token or tok in (tokenizer.cls_token, tokenizer.sep_token):
                w = 0.0

            weights.append(w)

        weights = np.array(weights)
        if weights.sum() == 0:
            return ids

        # Get indices with non-zero weights (maskable tokens)
        maskable_indices = np.where(weights > 0)[0]
        if len(maskable_indices) == 0:
            return ids

        # Normalize weights only for maskable tokens
        maskable_weights = weights[maskable_indices]
        maskable_weights = maskable_weights / maskable_weights.sum()

        # Calculate how many tokens to mask
        num_to_mask = max(1, int(mask_prob * length))
        # Don't try to mask more tokens than we have maskable tokens
        num_to_mask = min(num_to_mask, len(maskable_indices))
        
        if num_to_mask == 0:
            return ids

        # Sample from maskable indices only
        chosen_maskable = np.random.choice(
            len(maskable_indices), 
            size=num_to_mask, 
            replace=False, 
            p=maskable_weights
        )
        chosen = maskable_indices[chosen_maskable]

        # expand into small spans
        MASK_ID = tokenizer.mask_token_id
        PAD_ID = tokenizer.pad_token_id
        CLS_ID = tokenizer.cls_token_id
        SEP_ID = tokenizer.sep_token_id

        for idx in chosen:
            if ids[idx] not in (PAD_ID, CLS_ID, SEP_ID):
                ids[idx] = MASK_ID

            # contiguous span around sentiment words
            span = max(1, np.random.poisson(span_mean))
            for j in range(1, span + 1):
                if idx + j < length and ids[idx + j] not in (PAD_ID, CLS_ID, SEP_ID):
                    ids[idx + j] = MASK_ID
                if idx - j >= 0 and ids[idx - j] not in (PAD_ID, CLS_ID, SEP_ID):
                    ids[idx - j] = MASK_ID

        return ids

    def adversarial_mask(ids, tokenizer, mask_prob, span_mean):
        """
        ADVGLUE-style adversarial masking:
        - span masking
        - typos
        - synonym replacement
        - contextual MLM replacement
        - distraction appending (URL/noise phrase)
        """

        MASK = tokenizer.mask_token_id
        PAD = tokenizer.pad_token_id
        CLS = tokenizer.cls_token_id
        SEP = tokenizer.sep_token_id
        vocab_size = tokenizer.vocab_size

        ids = ids.copy()
        L = len(ids)

        num_to_modify = max(1, int(L * mask_prob))

        for _ in range(num_to_modify):
            i = random.randint(1, max(1, L - 2))  # avoid CLS/SEP as much as possible
            tok = ids[i]

            if tok in (PAD, CLS, SEP):
                continue

            r = random.random()

            if r < 0.40:
                span_len = max(1, np.random.poisson(span_mean))
                for j in range(i, min(i + span_len, L)):
                    if ids[j] not in (PAD, CLS, SEP):
                        ids[j] = MASK

            elif r < 0.55: # TextBugger 
                token_str = tokenizer.decode([tok])
                if len(token_str) >= 2:
                    idx = np.random.randint(0, len(token_str) - 1)
                    typo = list(token_str)
                    typo[idx], typo[idx + 1] = typo[idx + 1], typo[idx]
                    new_tok = tokenizer.encode("".join(typo), add_special_tokens=False)
                    if new_tok:
                        ids[i] = new_tok[0]

            elif r < 0.75: # TextFooler
                ids[i] = random.randint(100, max(100, vocab_size - 10))

            elif r < 0.95: #  BERT-ATTACK
                ids[i] = MASK

            else: # StressTest
                distraction_tokens = tokenizer.encode(
                    random.choice(
                        [
                            "and true is true",
                            "and false is not true",
                            "https://t.co/abc123",
                            "@random_user",
                        ]
                    ),
                    add_special_tokens=False,
                )
                tail_start = max(1, L - len(distraction_tokens) - 1)
                for j, tok2 in enumerate(distraction_tokens):
                    pos = tail_start + j
                    if pos >= L - 1:
                        break
                    if ids[pos] in (PAD, CLS, SEP):
                        continue
                    ids[pos] = tok2
        # SCPN
        return ids

    mask_type = (mask_type or "random_span").lower()
    if mask_type == "adversarial":
        def mask_function(seq):
            return adversarial_mask(seq, tokenizer, mask_probability, span_mean)
    elif mask_type == "sentiment_aware":
        def mask_function(seq):
            return sentiment_aware_mask(seq, tokenizer, mask_probability, span_mean)
    elif mask_type == "random_token":
        def mask_function(seq):
            return random_token_mask(seq, mask_probability, span_mean)
    elif mask_type in ("random_span", "span"):
        def mask_function(seq):
            return random_span_mask(seq, mask_probability, span_mean)
    else:
        raise ValueError(f"Unknown mask_type '{mask_type}'. Expected 'random_token', 'random_span', 'adversarial', or 'sentiment_aware'.")

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
        target_ids = [mask_function(seq) for seq in keep_ids]

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
            result = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "label": labels
            }
            # Preserve example_idx for SWAG multiple-choice grouping
            if "example_idx" in batch[0]:
                result["example_idx"] = torch.tensor([b["example_idx"] for b in batch], dtype=torch.long)
            return result
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

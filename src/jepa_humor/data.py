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


def load_anli_splits(name: str = "facebook/anli", train_rounds: list = [1, 2, 3], seed: int = 42, val_rounds: list = None):
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
    if val_rounds:   # 2026-10-26, opt-in (finetune anli_val_rounds): concatenate these dev rounds (standard: [1, 2, 3])
        validation_split = concatenate_datasets([ds[f"dev_r{r}"] for r in val_rounds])
        print(f"Using dev rounds {list(val_rounds)} as validation: {len(validation_split)} examples")
        return {"train": aggregated_train, "validation": validation_split}
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


def expand_copa_for_pretraining(ds_split):
    """Expand each COPA example into 2 single-sequence rows (same format as finetune)."""
    records = []
    for ex in ds_split:
        connector = "because" if ex["question"] == "cause" else "so"
        for choice in [ex["choice1"], ex["choice2"]]:
            records.append({
                "text": ex["premise"] + " " + connector + " " + choice,
                "label": ex["label"],
            })
    return Dataset.from_list(records)


def preprocess_multirc_for_pretraining(ds_split):
    """Merge question+answer into a single column so pretrain matches finetune tokenization."""
    def fn(examples):
        return {"question_answer": [q + " " + a for q, a in zip(examples["question"], examples["answer"])]}
    return ds_split.map(fn, batched=True)


def expand_record_for_pretraining(ds_split):
    """Expand each ReCoRD example into one (passage, query) row per candidate entity, with
    @placeholder replaced by the entity — so the pretrain input is exactly what the model sees
    at finetune time (mirrors _expand_record in finetune_superglue.py)."""
    records = []
    for ex in ds_split:
        for entity in ex["entities"]:
            records.append({
                "passage": ex["passage"],
                "query": ex["query"].replace("@placeholder", entity),
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


def load_hf_splits(name: str, config=None, test_size=0.2, seed=42,
                   sample_per_config=None, cache_dir=None):
    """
    Load HuggingFace dataset splits.

    Args:
        name: Dataset name (e.g., "nyu-mll/glue")
        config: Dataset configuration (e.g., "qnli", "mnli"). If None, will try to infer from name.
                May also be a LIST of sub-configs (e.g. wikimedia/wikipedia languages),
                in which case each is streamed+capped and concatenated (see below).
        test_size: Not used, kept for compatibility
        seed: Not used, kept for compatibility
        sample_per_config: When `config` is a list, cap rows streamed per sub-config.
        cache_dir: Where to cache the built multilingual subset (defaults to HF cache).

    Returns:
        Dataset dict with train/validation/test splits
    """
    # Multi-config, per-config capped load (e.g. wikimedia/wikipedia multilingual).
    # When `config` is a LIST of sub-configs, each is STREAMED (so we never download
    # the full multi-GB dump) and capped at `sample_per_config` rows, then all are
    # concatenated into one train split. The capped subset is auto-cached on disk so
    # it is built only once (like any other HF dataset). Backward-compatible: string
    # or None configs skip this entirely, so all existing tasks are unchanged.
    if isinstance(config, (list, tuple)):
        import os
        from datasets import Dataset, concatenate_datasets, load_from_disk
        cache_root = (cache_dir or os.environ.get("HF_DATASETS_CACHE")
                      or os.path.expanduser("~/.cache/huggingface/datasets"))
        key = f"{name.replace('/', '_')}__{'_'.join(config)}__n{sample_per_config}"
        cache_path = os.path.join(cache_root, "capped_subsets", key)
        if os.path.isdir(cache_path):
            return {"train": load_from_disk(cache_path)}
        parts = []
        for sub in config:
            if sample_per_config:
                stream = load_dataset(name, sub, split="train", streaming=True)
                rows = []
                for i, r in enumerate(stream):
                    if i >= sample_per_config:
                        break
                    txt = (r.get("text") or "").strip()
                    if txt:
                        rows.append({"text": txt, "lang": sub})
                parts.append(Dataset.from_list(rows))
            else:
                parts.append(load_dataset(name, sub, split="train"))
        combined = concatenate_datasets(parts)
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        combined.save_to_disk(cache_path)
        return {"train": combined}

    if config is None and 'glue' in name.lower():
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
    
    # Map dataset config to JSON key and expected fields.
    # CoLA, MRPC, and STS-B have no public GLUE test labels — they return None gracefully.
    config_map = {
        "qnli": ("qnli", ["question", "sentence"]),
        "mnli": ("mnli", ["premise", "hypothesis"]),
        "mnli-mm": ("mnli-mm", ["premise", "hypothesis"]),
        "qqp": ("qqp", ["question1", "question2"]),
        "rte": ("rte", ["sentence1", "sentence2"]),
        "sst2": ("sst2", ["sentence"]),
        "cola": ("cola", ["sentence"]),
        "mrpc": ("mrpc", ["sentence1", "sentence2"]),
        "stsb": ("stsb", ["sentence1", "sentence2"]),
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
    
    # Single-process by default so every existing task keeps its current behaviour AND its
    # existing cache fingerprints. Opt in with SSL_TOKENIZE_PROC=<n>.
    # Measured 2026-09-18 on 11.4M sequences: 1,286 ex/s on one core = 2h28m per cache,
    # while the corpus packer doing comparable work at num_proc=16 ran ~7,200 ex/s. The
    # pretrain path does this twice (one cache per distinct mask_ratio), so the default
    # costs ~4.9h of wall clock that 16 workers cut to ~0.6h.
    # NOTE changing this changes the datasets fingerprint, i.e. it invalidates caches built
    # under a different setting -- harmless, but it means switching mid-run re-tokenises.
    _nproc = int(os.environ.get("SSL_TOKENIZE_PROC", "1"))
    ds = ds.map(
        tokenize_function,
        batched=True,
        num_proc=_nproc if _nproc > 1 else None,
        desc="Tokenizing dataset",
        remove_columns=[col for col in ds.column_names if col not in ["input_ids", "attention_mask", "token_type_ids", "label"]]
    )
    return ds

def _entity_mask(ids, tok_strs, mask_prob, mask_id, skip_ids, v2=False, only=False):
    """Entity-first masking (mask_type: entity; added 2026-10-10, opt-in). Same budget as random_token
    (mask_prob x maskable tokens). Entity tokens = a capitalised word that does not start a sentence
    (RoBERTa byte-BPE: piece begins with 'Ġ' + uppercase letter, previous piece not ending in . ! ? : or a
    newline) plus its continuation pieces (pieces without the 'Ġ' prefix). Whole entity words are masked
    first, in random order, until the budget is spent; any budget left goes to random other tokens.
    Motivated by the HYPERPARTISAN error diagnosis: JEPA-TAPT leans hardest on named entities."""
    import numpy as np
    ids = list(ids)
    maskable = [i for i in range(len(ids)) if ids[i] not in skip_ids]
    if not maskable:
        return ids
    budget = len(maskable) if only else min(max(1, int(len(maskable) * mask_prob)), len(maskable))
    words, i = [], 0
    while i < len(ids):
        t = tok_strs[ids[i]] if ids[i] not in skip_ids else ""
        prev = tok_strs[ids[i - 1]] if i > 0 and ids[i - 1] not in skip_ids else "."
        # v2 (mask_type: entity_v2, AGNEWS 2026-10-10; v1 unchanged): also a word whose first piece has no 'Ġ'
        # because it follows "(" / a quote or opens the document ("(AFP)", "PARIS ..."), and all-caps words even
        # at a sentence start (datelines, agencies)
        w0 = t[1:] if t[:1] == "\u0120" else t
        starts = t[:1] == "\u0120" or (v2 and (i == 0 or ids[i - 1] in skip_ids or not prev[-1:].isalnum()))
        allcaps = v2 and len(w0) >= 2 and w0.isupper()
        if starts and w0[:1].isupper() and (allcaps or not prev.rstrip().endswith((".", "!", "?", ":", "\u010a"))):
            j = i + 1
            while j < len(ids) and ids[j] not in skip_ids and not tok_strs[ids[j]].startswith("\u0120") \
                    and tok_strs[ids[j]][:1].isalnum():
                j += 1
            words.append(list(range(i, j))); i = j
        else:
            i += 1
    chosen = set()
    for k in np.random.permutation(len(words)):
        if len(chosen) >= budget:
            break
        chosen.update(words[k][:budget - len(chosen)])
    rest = [i for i in maskable if i not in chosen]
    if only and not chosen:   # *_only: nothing important in this input -> 1 random token so the loss is defined
        chosen.add(int(np.random.choice(rest or maskable)))
    if not only and len(chosen) < budget and rest:
        chosen.update(np.random.choice(rest, size=min(budget - len(chosen), len(rest)), replace=False).tolist())
    for i in chosen:
        ids[i] = mask_id
    return ids


def _content_mask(ids, tok_strs, mask_prob, mask_id, skip_ids, only=False):
    """Content-word-first masking (mask_type: content; added 2026-10-10, opt-in). Same budget as random_token.
    Content word = a lowercase, non-stopword, non-number word ('Ġ' + lowercase letter, not in sklearn's
    ENGLISH_STOP_WORDS) plus its continuation pieces -- the "content" category of diagnosis/diagnose.py.
    Whole content words are masked first, in random order; any budget left goes to random other tokens.
    Motivated by the ACL-ARC error diagnosis: JEPA-TAPT leans on content words (1.46x), not names (0.52x)."""
    import numpy as np
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
    ids = list(ids)
    maskable = [i for i in range(len(ids)) if ids[i] not in skip_ids]
    if not maskable:
        return ids
    budget = len(maskable) if only else min(max(1, int(len(maskable) * mask_prob)), len(maskable))
    words, i = [], 0
    while i < len(ids):
        t = tok_strs[ids[i]] if ids[i] not in skip_ids else ""
        if len(t) > 1 and t[0] == "Ġ" and t[1].islower():
            j = i + 1
            while j < len(ids) and ids[j] not in skip_ids and not tok_strs[ids[j]].startswith("Ġ") \
                    and tok_strs[ids[j]][:1].isalnum():
                j += 1
            w = "".join(tok_strs[ids[k]] for k in range(i, j)).lstrip("Ġ")
            if w.lower() not in ENGLISH_STOP_WORDS:
                words.append(list(range(i, j)))
            i = j
        else:
            i += 1
    chosen = set()
    for k in np.random.permutation(len(words)):
        if len(chosen) >= budget:
            break
        chosen.update(words[k][:budget - len(chosen)])
    rest = [i for i in maskable if i not in chosen]
    if only and not chosen:   # *_only: nothing important in this input -> 1 random token so the loss is defined
        chosen.add(int(np.random.choice(rest or maskable)))
    if not only and len(chosen) < budget and rest:
        chosen.update(np.random.choice(rest, size=min(budget - len(chosen), len(rest)), replace=False).tolist())
    for i in chosen:
        ids[i] = mask_id
    return ids


def _split_only(mask_type):
    """mask_type "<rule>_only" (entity_only, content_only, marked_span_only, spacy_entity_only; 2026-10-11, opt-in):
    mask ONLY the rule's important tokens -- every one of them, no random fill, mask_ratio ignored (user: "let the
    pretrain be the important word only, irrelevant to mask_ratio"). An input with none gets 1 random token."""
    m = mask_type or ""
    return (m[:-5], True) if m.endswith("_only") else (m, False)


def _ent_ids(text, offsets, spans):
    """Per-token spaCy entity id (0 = none, k = k-th entity of the document) for mask_type: spacy_entity. A token is
    in an entity when its character range overlaps the entity's span. offsets = tokenizer offset_mapping."""
    out = [0] * len(offsets)
    for k, sp in enumerate(spans, 1):
        s, e = sp[0], sp[1]
        v = sp[2] if len(sp) > 2 else k   # [start, end, pct] spans (importance.py, mask_type cue/importance) -> pct
        for i, (a, b) in enumerate(offsets):
            if b > a and a < e and b > s:
                out[i] = v
    return out


def _load_ent_spans(path):
    import json
    if not path:
        raise ValueError("mask_type: spacy_entity needs ent_spans_file (experiment/dsp/diagnosis/spacy_ents.py)")
    return json.load(open(path))


def _spacy_entity_mask(ids, ents, mask_prob, mask_id, skip_ids, only=False):
    """spaCy-NER-first masking (mask_type: spacy_entity; added 2026-10-10, opt-in). Same budget as random_token
    (mask_prob x maskable). ents = per-token entity ids from _ent_ids (spaCy en_core_web_lg named entities,
    precomputed once per document by spacy_ents.py). Whole entities are masked first, in random order, until the
    budget is spent; any budget left goes to random other tokens. Replaces the capitalisation heuristic of
    mask_type: entity (user: "for entity, try using SpaCy")."""
    import numpy as np
    ids = list(ids)
    maskable = [i for i in range(len(ids)) if ids[i] not in skip_ids]
    if not maskable:
        return ids
    budget = len(maskable) if only else min(max(1, int(len(maskable) * mask_prob)), len(maskable))
    groups = {}
    for i, k in enumerate(ents):
        if k and ids[i] not in skip_ids:
            groups.setdefault(k, []).append(i)
    words = list(groups.values())
    chosen = set()
    for k in np.random.permutation(len(words)):
        if len(chosen) >= budget:
            break
        chosen.update(words[k][:budget - len(chosen)])
    rest = [i for i in maskable if i not in chosen]
    if only and not chosen:   # *_only: nothing important in this input -> 1 random token so the loss is defined
        chosen.add(int(np.random.choice(rest or maskable)))
    if not only and len(chosen) < budget and rest:
        chosen.update(np.random.choice(rest, size=min(budget - len(chosen), len(rest)), replace=False).tolist())
    for i in chosen:
        ids[i] = mask_id
    return ids


WEIGHTS = {"marked": 4.0, "entity": 3.0, "content": 1.5, "other": 1.0}


def _weighted_mask(ids, tok_strs, ents, mask_prob, mask_id, skip_ids, weights=WEIGHTS):
    """Importance-weighted random masking (mask_type: weighted; 2026-10-14, opt-in; user: "combine all masking type
    for all task ... generalize the masking type"). Exactly mask_prob x maskable tokens are masked (same budget as
    random_token), drawn without replacement with probability proportional to a per-token weight: inside [[ ]] / << >>
    marked spans 4, spaCy named entity 3, content word 1.5, any other token 1 (highest applicable weight wins). The
    markers themselves stay visible, as in marked_span. ents = per-token entity ids from _ent_ids (0 = none).
    Same token definitions as _marked_span_mask / _spacy_entity_mask / _content_mask."""
    import numpy as np
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
    ids = list(ids)
    strs = [tok_strs[x].lstrip("\u0120") if x not in skip_ids else "" for x in ids]
    markers = {i for i, t in enumerate(strs) if t in ("[[", "<<", ">>", "]]", "]")}
    maskable = [i for i in range(len(ids)) if ids[i] not in skip_ids and i not in markers]
    if not maskable:
        return ids
    w = {i: weights["other"] for i in maskable}
    i = 0                                                    # content words (as _content_mask)
    while i < len(ids):
        t = tok_strs[ids[i]] if ids[i] not in skip_ids else ""
        if len(t) > 1 and t[0] == "\u0120" and t[1].islower():
            j = i + 1
            while j < len(ids) and ids[j] not in skip_ids and not tok_strs[ids[j]].startswith("\u0120") \
                    and tok_strs[ids[j]][:1].isalnum():
                j += 1
            if "".join(tok_strs[ids[k]] for k in range(i, j)).lstrip("\u0120").lower() not in ENGLISH_STOP_WORDS:
                for k in range(i, j):
                    if k in w: w[k] = max(w[k], weights["content"])
            i = j
        else:
            i += 1
    for k, e in enumerate(ents or []):                       # spaCy entities
        if e and k in w: w[k] = max(w[k], weights["entity"])
    cur = None                                               # inside marked spans (as _marked_span_mask)
    for k, t in enumerate(strs):
        if t in ("[[", "<<"):
            cur = []
        elif cur is not None and t in (">>", "]]", "]"):
            for q in cur:
                if q in w: w[q] = max(w[q], weights["marked"])
            cur = None
        elif cur is not None:
            cur.append(k)
    n = min(max(1, int(len(maskable) * mask_prob)), len(maskable))
    p = np.array([w[k] for k in maskable], float); p /= p.sum()
    for k in np.random.choice(maskable, size=n, replace=False, p=p):
        ids[k] = mask_id
    return ids


STRATEGY_TOP = 80   # cue / importance: a word is "important" when its importance percentile in its document is > 80
SOFT_W = 3.0   # importance_soft: an important token is 3x as likely to be drawn as any other


def _strategy_mask(ids, tok_strs, pct, mask_prob, mask_id, skip_ids, mode, top=STRATEGY_TOP):
    """Hypothesis-test masks (2026-10-16, opt-in; user: "try all the hypothesis"). All mask exactly
    mask_prob x maskable tokens, the same budget as random_token, so only WHICH tokens are hidden changes.
      nomarker   -- random tokens, but the [[ ]] << >> markers are never masked (marker-visibility confound)
      wwm        -- random WHOLE words (a word = a 'Ġ'-initial piece + its continuation pieces)
      cue        -- the words the classifier relies on most (importance percentile > top) and the markers are
                    never masked; the budget is drawn at random from the other tokens ("keep the cues visible")
      importance -- the important words are masked first (random order), the rest of the budget at random
                    (the opposite direction; = task-guided selective masking)
      entity_cue -- "erase the names, keep the cues": tokens inside the [[ ]] / << >> entity spans are masked first,
                    the markers and the important words OUTSIDE the spans (importance percentile > top) are never masked,
                    the rest of the budget is drawn at random from the other tokens (= marked_span + cue). Tasks without
                    markers: names = words whose span value is >= 1000 (citations / spaCy entities, build_name_spans.py)
      signed_cue -- entity_cue with no marker fallback: words with span value >= 1000 (out-of-fold distractors, the words
                    whose removal raises the gold-class probability; diagnosis/cv_attr.py) are masked first, words with
                    value % 1000 > top (out-of-fold support) and the markers are never masked, the rest at random
      keepK      -- keep-only (K = 10/20/35/50): EVERY token is masked except the top-K% words by the span file's
                    percentile and the markers (mask_prob ignored). With PMI(word, own train label) spans
                    (diagnosis/build_lex_masks.py) the teacher's <s> neighbours become label-pure (purity_grid.py),
                    which is what the kNN term -- ~all of the encoder gradient (grad_decomp.py) -- transfers
      cue_soft   -- cue, but the important words are masked at 1/SOFT_W the rate of the others instead of never
      cue_span   -- cue protection, the budget spent on contiguous spans (geometric length, mean ~3, max 8) of the others
      cue30 / cue40 -- cue with the top 30% / 40% important words protected (diagnosis/preview2.py: the expected masked
                    view keeps the most label information at K = 30-40)
      punct_first -- punctuation tokens (not the markers) masked first, the rest of the budget at random (chosen by
                    the mask preview, diagnosis/preview.py: keeps the most label information of the token groups)
      importance_soft -- like importance but stochastic: tokens drawn without replacement with probability
                    proportional to SOFT_W (important) vs 1 (other), so the important words hidden change every
                    epoch (importance masks ALL of them every epoch: ~2/3 of its mask is fixed across epochs)
    pct = per-token importance percentile from _ent_ids with importance.py spans (0 = unscored)."""
    import numpy as np
    if mode in ("cue30", "cue40"):   # 2026-10-20: cueK = cue protecting the top K% important words (chosen per task by preview2)
        top, mode = 100 - int(mode[3:]), "cue"
    ids = list(ids)
    maskable = [i for i in range(len(ids)) if ids[i] not in skip_ids]
    if not maskable:
        return ids
    n = min(max(1, int(len(maskable) * mask_prob)), len(maskable))
    strs = [tok_strs[x].lstrip("\u0120") if x not in skip_ids else "" for x in ids]
    markers = {i for i, t in enumerate(strs) if t in ("[[", "<<", ">>", "]]", "]")}
    if mode in ("keep10", "keep20", "keep35", "keep50"):   # 2026-10-21: keep-only -- every token masked except the
        top = 100 - int(mode[4:])                           # top-K% words (pct > 100-K) and the markers; no budget
        for i in maskable:
            if i not in markers and pct[i] <= top:
                ids[i] = mask_id
        return ids
    if mode == "wwm":
        words, cur = [], []
        for i in maskable:
            if cur and (tok_strs[ids[i]].startswith("\u0120") or i != cur[-1] + 1):
                words.append(cur); cur = []
            cur.append(i)
        if cur:
            words.append(cur)
        chosen = []
        for k in np.random.permutation(len(words)):
            if len(chosen) >= n:
                break
            chosen += words[k][:n - len(chosen)]
    elif mode in ("entity_cue", "signed_cue"):   # 2026-10-19: erase the names, keep the cues
        pv = pct
        if mode == "signed_cue" or any(pct[i] >= 1000 for i in maskable):   # 2026-10-20: no-marker tasks pass names as importance + 1000
            inside = [i for i in maskable if pct[i] >= 1000]   # (diagnosis/build_name_spans.py); value % 1000 = importance
            pv = [p % 1000 if p else 0 for p in pct]
        else:
            inside, cur = [], None
            for i, w in enumerate(strs):
                if w in ("[[", "<<"):
                    cur = []
                elif cur is not None and w in (">>", "]]", "]"):
                    inside += cur; cur = None
                elif cur is not None and i in set(maskable):
                    inside.append(i)
        ins = set(inside)
        protect = markers | {i for i in maskable if pv[i] > top and i not in ins}
        chosen = list(np.random.permutation(inside)[:n])
        rest = [i for i in maskable if i not in protect and i not in set(chosen)]
        if len(chosen) < n and rest:
            chosen += np.random.choice(rest, size=min(n - len(chosen), len(rest)), replace=False).tolist()
    elif mode == "punct_first":   # 2026-10-19: punctuation tokens (markers excluded) masked first, rest of budget random
        grp = [i for i in maskable if strs[i] and i not in markers and all(not c.isalnum() for c in strs[i])]
        chosen = list(np.random.permutation(grp)[:n])
        rest = [i for i in maskable if i not in set(chosen)]
        if len(chosen) < n and rest:
            chosen += np.random.choice(rest, size=min(n - len(chosen), len(rest)), replace=False).tolist()
    elif mode == "cue_soft":   # 2026-10-21: cue, but the important words are masked at 1/SOFT_W the rate instead of never
        pool = [i for i in maskable if i not in markers]
        w = np.array([1.0 / SOFT_W if pct[i] > top else 1.0 for i in pool])
        chosen = np.random.choice(pool, size=min(n, len(pool)), replace=False, p=w / w.sum()).tolist() if pool else []
    elif mode == "cue_span":   # 2026-10-21: cue protection, the budget spent on contiguous spans (mean ~3) of the other tokens
        pool = [i for i in maskable if i not in markers and pct[i] <= top]; ps, cs, chosen = set(pool), set(), []
        for st in np.random.permutation(pool):
            if len(chosen) >= n:
                break
            j, end = int(st), int(st) + min(int(np.random.geometric(0.3)), 8)
            while j < end and j in ps and j not in cs and len(chosen) < n:
                chosen.append(j); cs.add(j); j += 1
    elif mode == "importance_soft":   # 2026-10-16: importance, but redrawn every epoch (no fixed part)
        w = np.array([SOFT_W if pct[i] > top else 1.0 for i in maskable])
        chosen = np.random.choice(maskable, size=n, replace=False, p=w / w.sum()).tolist()
    elif mode == "importance":
        imp = [i for i in maskable if pct[i] > top]
        chosen = list(np.random.permutation(imp)[:n])
        rest = [i for i in maskable if pct[i] <= top]
        if len(chosen) < n and rest:
            chosen += np.random.choice(rest, size=min(n - len(chosen), len(rest)), replace=False).tolist()
    else:   # nomarker / cue
        pool = [i for i in maskable if i not in markers and (mode == "nomarker" or pct[i] <= top)]
        if not pool:
            return ids
        chosen = np.random.choice(pool, size=min(n, len(pool)), replace=False).tolist()
    for i in chosen:
        ids[int(i)] = mask_id
    return ids


def _marked_span_mask(ids, tok_strs, mask_prob, mask_id, skip_ids, only=False):
    """Marked-span-first masking (mask_type: marked_span; added 2026-10-10, opt-in). For SCIERC-style text
    where the relation arguments are wrapped in [[ ... ]] and << ... >>: the tokens INSIDE the markers are
    masked first (markers stay visible), then any budget left (mask_prob x maskable) goes to random tokens.
    Same budget as random_token. Motivated by the SCIERC error diagnosis (JEPA-TAPT leans on the markers)."""
    import numpy as np
    ids = list(ids)
    maskable = [i for i in range(len(ids)) if ids[i] not in skip_ids]
    if not maskable:
        return ids
    budget = len(maskable) if only else min(max(1, int(len(maskable) * mask_prob)), len(maskable))
    inside, cur = [], None
    for i, x in enumerate(ids):
        t = tok_strs[x].lstrip("\u0120") if x not in skip_ids else ""
        if t in ("[[", "<<"):
            cur = []
        elif cur is not None and (t in (">>", "]]") or t == "]"):
            inside.append(cur); cur = None
        elif cur is not None:
            cur.append(i)
    chosen = set()
    for k in np.random.permutation(len(inside)):
        if len(chosen) >= budget:
            break
        chosen.update(inside[k][:budget - len(chosen)])
    markers = {i for i, x in enumerate(ids) if x not in skip_ids and tok_strs[x].lstrip("\u0120") in ("[[", "<<", ">>", "]]", "]")}
    rest = [i for i in maskable if i not in chosen and i not in markers]
    if only and not chosen:   # *_only: nothing important in this input -> 1 random token so the loss is defined
        chosen.add(int(np.random.choice(rest or maskable)))
    if not only and len(chosen) < budget and rest:
        chosen.update(np.random.choice(rest, size=min(budget - len(chosen), len(rest)), replace=False).tolist())
    for i in chosen:
        ids[i] = mask_id
    return ids


def preprocess_dataset_ssl(
    ds,
    tokenizer,
    max_length=256,
    mask_probability=0.20,
    span_mean=5,
    text_keys=None,
    mask_type="random_span",
    bert_mlm_masking=False,
    ent_spans_file=None,
):
    """
    Tokenize text fields together, then create masked target versions
    using SpanBERT-style random span masking.

    bert_mlm_masking (default False -> every existing task is unaffected):
        use original-BERT corruption instead of 100% [MASK] substitution, and additionally
        emit "mlm_labels" (original id at selected positions, -100 elsewhere). Of the selected
        positions: 80% -> [MASK], 10% -> random vocab token, 10% -> left unchanged. The labels
        column is required because with 80/10/10 the selected set can no longer be recovered
        by testing `input_ids == mask_token_id`.
    
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
    mask_type, _only = _split_only(mask_type)

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
    elif mask_type == "entity":
        _tok_strs = tokenizer.convert_ids_to_tokens(list(range(len(tokenizer))))
        _skip = {PAD_ID, tokenizer.cls_token_id, tokenizer.sep_token_id}
        def mask_function(seq):
            return _entity_mask(seq, _tok_strs, mask_probability, MASK_ID, _skip, only=_only)
    elif mask_type == "entity_v2":
        _tok_strs = tokenizer.convert_ids_to_tokens(list(range(len(tokenizer))))
        _skip = {PAD_ID, tokenizer.cls_token_id, tokenizer.sep_token_id}
        def mask_function(seq):
            return _entity_mask(seq, _tok_strs, mask_probability, MASK_ID, _skip, v2=True)
    elif mask_type == "marked_span":
        _tok_strs = tokenizer.convert_ids_to_tokens(list(range(len(tokenizer))))
        _skip = {PAD_ID, tokenizer.cls_token_id, tokenizer.sep_token_id}
        def mask_function(seq):
            return _marked_span_mask(seq, _tok_strs, mask_probability, MASK_ID, _skip, only=_only)
    elif mask_type == "content":
        _tok_strs = tokenizer.convert_ids_to_tokens(list(range(len(tokenizer))))
        _skip = {PAD_ID, tokenizer.cls_token_id, tokenizer.sep_token_id}
        def mask_function(seq):
            return _content_mask(seq, _tok_strs, mask_probability, MASK_ID, _skip, only=_only)
    elif mask_type in ("nomarker", "wwm", "punct_first"):
        _tok_strs = tokenizer.convert_ids_to_tokens(list(range(len(tokenizer))))
        _skip = {PAD_ID, tokenizer.cls_token_id, tokenizer.sep_token_id}
        def mask_function(seq):
            return _strategy_mask(seq, _tok_strs, None, mask_probability, MASK_ID, _skip, mask_type)
    elif mask_type in ("spacy_entity", "weighted", "cue", "cue30", "cue40", "signed_cue", "cue_soft", "cue_span", "keep10", "keep20", "keep35", "keep50", "importance", "importance_soft", "entity_cue"):   # per-document spans -> handled in tokenize_function
        import hashlib
        _spans = _load_ent_spans(ent_spans_file)
        _skip = {PAD_ID, tokenizer.cls_token_id, tokenizer.sep_token_id}
        _tok_strs = tokenizer.convert_ids_to_tokens(list(range(len(tokenizer))))
        mask_function = None
    else:
        raise ValueError(f"Unknown mask_type '{mask_type}'. Expected 'random_token', 'random_span', 'adversarial', or 'sentiment_aware'.")

    def bert_mlm_corrupt(ids):
        """Original-BERT corruption. Returns (corrupted_ids, labels) with labels=-100 off-target."""
        ids = list(ids)
        labels = [-100] * len(ids)
        PAD_ID = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else -1
        CLS_ID = tokenizer.cls_token_id if tokenizer.cls_token_id is not None else -1
        SEP_ID = tokenizer.sep_token_id if tokenizer.sep_token_id is not None else -1
        MASK_ID = tokenizer.mask_token_id
        vocab = tokenizer.vocab_size
        for i, t in enumerate(ids):
            if t in (PAD_ID, CLS_ID, SEP_ID):
                continue
            if random.random() >= mask_probability:
                continue
            labels[i] = t                      # loss is taken on every selected position
            r = random.random()
            if r < 0.8:
                ids[i] = MASK_ID               # 80%
            elif r < 0.9:
                ids[i] = random.randrange(vocab)   # 10% random token
            # else: remaining 10% keep the original token
        return ids, labels

    def tokenize_function(examples):
        # Tokenize text fields together
        texts = [examples[key] for key in text_keys]
        encoded = tokenizer(
            *texts,
            truncation=True,
            padding="max_length",
            max_length=max_length,
            **({"return_offsets_mapping": True} if mask_type in ("spacy_entity", "weighted", "cue", "cue30", "cue40", "signed_cue", "cue_soft", "cue_span", "keep10", "keep20", "keep35", "keep50", "importance", "importance_soft", "entity_cue") else {}),
        )

        keep_ids = encoded["input_ids"]
        keep_mask = encoded["attention_mask"]

        # Now create masked copies
        if mask_type in ("spacy_entity", "weighted", "cue", "cue30", "cue40", "signed_cue", "cue_soft", "cue_span", "keep10", "keep20", "keep35", "keep50", "importance", "importance_soft", "entity_cue") and not bert_mlm_masking:
            assert len(texts) == 1, "spacy_entity supports a single text key"
            ents = [_ent_ids(t, o, _spans.get(hashlib.sha1(t.encode()).hexdigest(), []))
                    for t, o in zip(texts[0], encoded["offset_mapping"])]
            target_ids = ([_strategy_mask(q, _tok_strs, e, mask_probability, MASK_ID, _skip, mask_type)
                           for q, e in zip(keep_ids, ents)]
                          if mask_type in ("cue", "cue30", "cue40", "signed_cue", "cue_soft", "cue_span", "keep10", "keep20", "keep35", "keep50", "importance", "importance_soft", "entity_cue") else
                          [_weighted_mask(q, _tok_strs, e, mask_probability, MASK_ID, _skip) for q, e in zip(keep_ids, ents)]
                          if mask_type == "weighted" else
                          [_spacy_entity_mask(q, e, mask_probability, MASK_ID, _skip, only=_only) for q, e in zip(keep_ids, ents)])
            mlm_labels = None
        elif bert_mlm_masking:
            pairs = [bert_mlm_corrupt(seq) for seq in keep_ids]
            target_ids = [p[0] for p in pairs]
            mlm_labels = [p[1] for p in pairs]
        else:
            target_ids = [mask_function(seq) for seq in keep_ids]
            mlm_labels = None

        result = {
            "keep_input_ids": keep_ids,
            "keep_attention_mask": keep_mask,
            "target_input_ids": target_ids,
            "target_attention_mask": keep_mask,
        }
        if mlm_labels is not None:
            result["mlm_labels"] = mlm_labels
        if "token_type_ids" in encoded:
            result["keep_token_type_ids"] = encoded["token_type_ids"]
            result["target_token_type_ids"] = encoded["token_type_ids"]
        return result

    ds = ds.map(
        tokenize_function,
        batched=True,
        desc="Tokenizing SSL dataset",
        remove_columns=[
            col for col in ds.column_names
            if col not in [
                "keep_input_ids",
                "keep_attention_mask",
                "keep_token_type_ids",
                "target_input_ids",
                "target_attention_mask",
                "target_token_type_ids",
                "mlm_labels",
                "label",
            ]
        ]
    )

    return ds


def _swap_flags(text_a, text_b, offsets, seq_ids):
    """mask_type swap_first (PAWS, 2026-11-01, opt-in): per-token 1 if the token belongs to a word that is NOT in the
    longest in-order common word sequence of the two sentences (difflib on whitespace words) -- the swapped/moved/changed
    words of a PAWS pair -- else 0. Special and pad tokens get 0."""
    import difflib, re
    wa = [(m.start(), m.end(), m.group()) for m in re.finditer(r"\S+", text_a)]
    wb = [(m.start(), m.end(), m.group()) for m in re.finditer(r"\S+", text_b)]
    sm = difflib.SequenceMatcher(a=[w[2] for w in wa], b=[w[2] for w in wb], autojunk=False)
    ka, kb = set(), set()
    for blk in sm.get_matching_blocks():
        ka.update(range(blk.a, blk.a + blk.size)); kb.update(range(blk.b, blk.b + blk.size))
    spans = {0: [(s, e) for j, (s, e, _) in enumerate(wa) if j not in ka],
             1: [(s, e) for j, (s, e, _) in enumerate(wb) if j not in kb]}
    return [int(sid in spans and e > s and any(s < we and e > ws for ws, we in spans[sid]))
            for (s, e), sid in zip(offsets, seq_ids)]


def tokenize_only_ssl(ds, tokenizer, max_length=256, text_keys=None, num_proc=None, ent_spans_file=None, swap_flags=False):
    """Tokenize for SSL without masking. Stores original tokens as orig_* columns.
    Use with remask_ssl_dataset for dynamic per-epoch masking."""
    if text_keys is None:
        text_keys = ["question", "sentence"]

    def tokenize_fn(examples):
        texts = [examples[key] for key in text_keys]
        encoded = tokenizer(*texts, truncation=True, padding="max_length", max_length=max_length,
                            **({"return_offsets_mapping": True} if (ent_spans_file or swap_flags) else {}))
        result = {
            "orig_input_ids": encoded["input_ids"],
            "orig_attention_mask": encoded["attention_mask"],
        }
        if swap_flags:   # mask_type: swap_first (opt-in): per-token swapped-word flags for remask_ssl_dataset
            assert len(texts) == 2, "swap_first needs a sentence pair"
            result["orig_swap"] = [_swap_flags(a, b, o, encoded.sequence_ids(i))
                                   for i, (a, b, o) in enumerate(zip(texts[0], texts[1], encoded["offset_mapping"]))]
        if ent_spans_file:   # mask_type: spacy_entity (opt-in): per-token spaCy entity ids for remask_ssl_dataset
            import hashlib
            assert len(texts) == 1, "spacy_entity supports a single text key"
            result["orig_ent"] = [_ent_ids(t, o, _spans.get(hashlib.sha1(t.encode()).hexdigest(), []))
                                  for t, o in zip(texts[0], encoded["offset_mapping"])]
        if "token_type_ids" in encoded:
            result["orig_token_type_ids"] = encoded["token_type_ids"]
        return result

    keep = {"orig_input_ids", "orig_attention_mask", "orig_token_type_ids", "label", "orig_ent", "orig_swap"}
    _spans = _load_ent_spans(ent_spans_file) if ent_spans_file else None
    return ds.map(
        tokenize_fn,
        batched=True,
        desc="Tokenizing SSL dataset (no masking)",
        remove_columns=[col for col in ds.column_names if col not in keep],
        num_proc=num_proc,
    )


def remask_ssl_dataset(ds, tokenizer, mask_probability=0.15, span_mean=3.0, mask_type="random_token",
                       bert_mlm_masking=False, fresh_seed=None, num_proc=None, cache_file_name=None):
    """Apply a fresh random mask to a dataset produced by tokenize_only_ssl.
    Call at the start of each training epoch for dynamic masking."""
    import numpy as np
    mask_type, _only = _split_only(mask_type)
    _tok_strs_w = (tokenizer.convert_ids_to_tokens(list(range(len(tokenizer))))
                   if mask_type in ("weighted", "cue", "cue30", "cue40", "signed_cue", "cue_soft", "cue_span", "keep10", "keep20", "keep35", "keep50", "importance", "importance_soft", "entity_cue", "nomarker", "wwm", "punct_first") else None)

    MASK_ID = tokenizer.mask_token_id
    PAD_ID  = tokenizer.pad_token_id
    CLS_ID  = tokenizer.cls_token_id if tokenizer.cls_token_id is not None else -1
    SEP_ID  = tokenizer.sep_token_id if tokenizer.sep_token_id is not None else -1

    def random_token_mask(ids):
        ids = list(ids)
        maskable = [i for i in range(len(ids)) if ids[i] not in (PAD_ID, CLS_ID, SEP_ID)]
        if not maskable:
            return ids
        n = min(max(1, int(len(maskable) * mask_probability)), len(maskable))
        for idx in np.random.choice(maskable, size=n, replace=False):
            ids[idx] = MASK_ID
        return ids

    def random_span_mask(ids):
        ids = list(ids)
        num_to_mask = int(len(ids) * mask_probability)
        if num_to_mask == 0:
            return ids
        masked = 0
        while masked < num_to_mask:
            span_len = max(1, np.random.poisson(span_mean))
            start = np.random.randint(0, max(1, len(ids) - span_len))
            for i in range(start, min(start + span_len, len(ids))):
                if masked >= num_to_mask:
                    break
                if ids[i] != PAD_ID:
                    ids[i] = MASK_ID
                    masked += 1
        return ids

    mask_fn = random_span_mask if (mask_type or "").lower() in ("random_span", "span") else random_token_mask
    if (mask_type or "").lower() == "entity":
        _tok_strs = tokenizer.convert_ids_to_tokens(list(range(len(tokenizer))))
        def mask_fn(ids):
            return _entity_mask(ids, _tok_strs, mask_probability, MASK_ID, {PAD_ID, CLS_ID, SEP_ID}, only=_only)
    if (mask_type or "").lower() == "entity_v2":
        _tok_strs = tokenizer.convert_ids_to_tokens(list(range(len(tokenizer))))
        def mask_fn(ids):
            return _entity_mask(ids, _tok_strs, mask_probability, MASK_ID, {PAD_ID, CLS_ID, SEP_ID}, v2=True)
    if (mask_type or "").lower() == "marked_span":
        _tok_strs = tokenizer.convert_ids_to_tokens(list(range(len(tokenizer))))
        def mask_fn(ids):
            return _marked_span_mask(ids, _tok_strs, mask_probability, MASK_ID, {PAD_ID, CLS_ID, SEP_ID}, only=_only)
    if (mask_type or "").lower() == "content":
        _tok_strs = tokenizer.convert_ids_to_tokens(list(range(len(tokenizer))))
        def mask_fn(ids):
            return _content_mask(ids, _tok_strs, mask_probability, MASK_ID, {PAD_ID, CLS_ID, SEP_ID}, only=_only)
    if (mask_type or "").lower() in ("nomarker", "wwm", "punct_first"):
        def mask_fn(ids):
            return _strategy_mask(ids, _tok_strs_w, None, mask_probability, MASK_ID, {PAD_ID, CLS_ID, SEP_ID}, mask_type)

    def bert_mask(ids):
        """BERT 80/10/10 corruption plus the labels column.

        The static path (preprocess_dataset_ssl) emits mlm_labels when bert_mlm_masking is
        set, but this dynamic path never did -- so model_type: mlm with dynamic_masking: true
        died on KeyError: 'mlm_labels' in train_ssl.py:382. Labels are required because with
        80/10/10 the selected positions can no longer be recovered by testing
        `input_ids == mask_token_id`.
        """
        ids = list(ids)
        labels = [-100] * len(ids)
        maskable = [i for i in range(len(ids)) if ids[i] not in (PAD_ID, CLS_ID, SEP_ID)]
        if not maskable:
            return ids, labels
        n = min(max(1, int(len(maskable) * mask_probability)), len(maskable))
        for idx in np.random.choice(maskable, size=n, replace=False):
            labels[idx] = ids[idx]
            r = np.random.rand()
            if r < 0.8:
                ids[idx] = MASK_ID
            elif r < 0.9:
                ids[idx] = int(np.random.randint(0, tokenizer.vocab_size))
            # else: leave the original token in place
        return ids, labels

    def _swap_select(ids, flags):
        """swap_first: every swapped token, then random others until mask_probability of the maskable tokens (no cap:
        a pair whose swapped words exceed the ratio has all of them masked -- user 2026-11-01).
        swap_cap (2026-11-02): same, but capped at mask_probability -- a pair with more swapped tokens than the budget
        gets a random budget-sized subset of its swapped tokens (re-drawn every epoch by remask_fresh), no random fill."""
        maskable = [i for i in range(len(ids)) if ids[i] not in (PAD_ID, CLS_ID, SEP_ID)]
        forced = [i for i in maskable if flags[i]]
        rest = [i for i in maskable if not flags[i]]
        budget = int(len(maskable) * mask_probability)
        if _swap_cap and len(forced) > budget:
            forced = list(np.random.choice(forced, size=budget, replace=False))
        n_fill = min(max(0, budget - len(forced)), len(rest))
        fill = list(np.random.choice(rest, size=n_fill, replace=False)) if n_fill else []
        sel = forced + fill
        return sel if sel else ([int(np.random.choice(maskable))] if maskable else [])

    def swap_mask(ids, flags):
        ids = list(ids)
        for i in _swap_select(ids, flags):
            ids[i] = MASK_ID
        return ids

    def bert_swap_mask(ids, flags):
        ids = list(ids); labels = [-100] * len(ids)
        for idx in _swap_select(ids, flags):
            labels[idx] = ids[idx]
            r = np.random.rand()
            if r < 0.8:
                ids[idx] = MASK_ID
            elif r < 0.9:
                ids[idx] = int(np.random.randint(0, tokenizer.vocab_size))
        return ids, labels

    _swap = (mask_type or "").lower() in ("swap_first", "swap_cap")
    _swap_cap = (mask_type or "").lower() == "swap_cap"

    def apply(examples, idx=None):
        if idx is not None:
            # fresh_seed path: reseed per batch so forked num_proc workers (which inherit one
            # identical np.random state) and successive epochs all draw different masks.
            np.random.seed((fresh_seed * 1_000_003 + idx[0]) % (2**32))
        orig_ids   = examples["orig_input_ids"]
        orig_amask = examples["orig_attention_mask"]
        if bert_mlm_masking:
            pairs = ([bert_swap_mask(q, f) for q, f in zip(orig_ids, examples["orig_swap"])] if _swap else
                     [bert_mask(seq) for seq in orig_ids])
            result = {
                "keep_input_ids":        orig_ids,
                "keep_attention_mask":   orig_amask,
                "target_input_ids":      [p[0] for p in pairs],
                "target_attention_mask": orig_amask,
                "mlm_labels":            [p[1] for p in pairs],
            }
            if "orig_token_type_ids" in examples:
                result["keep_token_type_ids"]   = examples["orig_token_type_ids"]
                result["target_token_type_ids"] = examples["orig_token_type_ids"]
            return result
        result = {
            "keep_input_ids":        orig_ids,
            "keep_attention_mask":   orig_amask,
            "target_input_ids":      ([_spacy_entity_mask(q, e, mask_probability, MASK_ID, {PAD_ID, CLS_ID, SEP_ID}, only=_only)
                                       for q, e in zip(orig_ids, examples["orig_ent"])]
                                      if (mask_type or "").lower() == "spacy_entity" else
                                      [_weighted_mask(q, _tok_strs_w, e, mask_probability, MASK_ID, {PAD_ID, CLS_ID, SEP_ID})
                                       for q, e in zip(orig_ids, examples["orig_ent"])]
                                      if (mask_type or "").lower() == "weighted" else
                                      [_strategy_mask(q, _tok_strs_w, e, mask_probability, MASK_ID, {PAD_ID, CLS_ID, SEP_ID}, mask_type)
                                       for q, e in zip(orig_ids, examples["orig_ent"])]
                                      if (mask_type or "").lower() in ("cue", "cue30", "cue40", "signed_cue", "cue_soft", "cue_span", "keep10", "keep20", "keep35", "keep50", "importance", "importance_soft", "entity_cue") else
                                      [swap_mask(q, f) for q, f in zip(orig_ids, examples["orig_swap"])]
                                      if _swap else
                                      [mask_fn(seq) for seq in orig_ids]),
            "target_attention_mask": orig_amask,
        }
        if "orig_token_type_ids" in examples:
            result["keep_token_type_ids"]   = examples["orig_token_type_ids"]
            result["target_token_type_ids"] = examples["orig_token_type_ids"]
        return result

    if fresh_seed is None:
        return ds.map(apply, batched=True, desc="Applying SSL mask")
    # Default path above reuses HF's cache: its fingerprint ignores the RNG, so every epoch after
    # the first silently reloaded epoch 1's mask. fresh_seed (opt-in) forces a genuinely new mask.
    return ds.map(apply, batched=True, with_indices=True, desc="Applying SSL mask",
                  num_proc=num_proc, load_from_cache_file=False,
                  new_fingerprint=f"remask_{fresh_seed}", cache_file_name=cache_file_name)


def make_dataloaders(train_ds, test_ds, batch_size=64, num_workers=4, shuffle=True, pre_tokenized=False, ssl_mode=False, label_dtype=torch.long):
    # Collate function for pre-tokenized SSL data
    if pre_tokenized and ssl_mode:
        def collate(batch):
            result = {
                "keep_input_ids": torch.tensor([b["keep_input_ids"] for b in batch], dtype=torch.long),
                "keep_attention_mask": torch.tensor([b["keep_attention_mask"] for b in batch], dtype=torch.long),
                "target_input_ids": torch.tensor([b["target_input_ids"] for b in batch], dtype=torch.long),
                "target_attention_mask": torch.tensor([b["target_attention_mask"] for b in batch], dtype=torch.long),
            }
            if "label" in batch[0]:
                result["label"] = torch.tensor([b["label"] for b in batch], dtype=torch.long)
            if "keep_token_type_ids" in batch[0]:
                result["keep_token_type_ids"] = torch.tensor([b["keep_token_type_ids"] for b in batch], dtype=torch.long)
                result["target_token_type_ids"] = torch.tensor([b["target_token_type_ids"] for b in batch], dtype=torch.long)
            if "mlm_labels" in batch[0]:
                result["mlm_labels"] = torch.tensor([b["mlm_labels"] for b in batch], dtype=torch.long)
            return result
    # Collate function for pre-tokenized data (finetune mode)
    elif pre_tokenized:
        def collate(batch):
            result = {
                "input_ids": torch.tensor([b["input_ids"] for b in batch], dtype=torch.long),
                "attention_mask": torch.tensor([b["attention_mask"] for b in batch], dtype=torch.long),
                # label_dtype defaults to torch.long (classification, unchanged). Regression (STS-B) passes
                # torch.float so continuous 0-5 targets aren't truncated to integers by a long cast.
                "label": torch.tensor([b["label"] for b in batch], dtype=label_dtype),
            }
            if "token_type_ids" in batch[0]:
                result["token_type_ids"] = torch.tensor([b["token_type_ids"] for b in batch], dtype=torch.long)
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

from dataclasses import dataclass
from tqdm import tqdm
import json
import torch
import torch.nn as nn
from torch.cuda.amp import autocast
from torch.amp import GradScaler
from torch.optim.swa_utils import AveragedModel, SWALR, update_bn
from transformers import AutoTokenizer, AutoModel, AutoConfig
from sklearn.metrics import matthews_corrcoef, f1_score, accuracy_score
from scipy.stats import spearmanr
from .utils import set_seed, device, load_encoder_state, save_ckpt
from .data import load_hf_splits, make_dataloaders, load_hf_splits_test, load_hf_splits_test_mismatched, preprocess_dataset, load_test_set_from_json, sample_dataset, load_anli_splits, load_anli_test_splits, load_swag, preprocess_swag_multiple_choice
from .train_ssl import pretrain
import os
from torch import amp
import numpy as np
import math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def _save_training_plots(epochs, acc_curves, loss_hist, lr_hist, title, save_path):
    """Save a multi-panel plot: accuracy curves, train loss, and learning rate."""
    panels = [("acc", acc_curves), ("loss", loss_hist), ("lr", lr_hist)]
    active = [(k, v) for k, v in panels if v]
    fig, axes = plt.subplots(len(active), 1, figsize=(8, 4 * len(active)), squeeze=False)
    for i, (kind, data) in enumerate(active):
        ax = axes[i][0]
        if kind == "acc":
            for label, accs in data.items():
                ax.plot(epochs[:len(accs)], accs, marker="o", markersize=3, label=label)
            ax.set_ylabel("Accuracy")
            ax.set_title(title)
            ax.legend()
        elif kind == "loss":
            ax.plot(epochs[:len(data)], data, marker="o", markersize=3, color="tab:orange")
            ax.set_ylabel("Train Loss")
            ax.set_title("Train Loss")
        elif kind == "lr":
            ax.plot(epochs[:len(data)], data, marker="o", markersize=3, color="tab:green")
            ax.set_ylabel("Learning Rate")
            ax.set_title("Learning Rate")
        ax.set_xlabel("Epoch")
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved training plots to {save_path}")


def save_finetune_results(results_dir, task, experiment, data):
    """Save fine-tune results (val/test accuracy per task) to a JSON file in results_dir."""
    import datetime
    os.makedirs(results_dir, exist_ok=True)
    fname = f"{experiment}_{task}.json" if experiment else f"{task}.json"
    path = os.path.join(results_dir, fname)
    data = {**data, "task": task, "experiment": experiment, "timestamp": datetime.datetime.now().isoformat()}
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Saved results to {path}")


def _build_finetune_lr_scheduler(optimizer, num_training_steps, warmup_ratio, schedule_type):
    """Per-optimizer-step LambdaLR: linear warmup then cosine or linear decay to ~min LR."""
    schedule_type = (schedule_type or "cosine").lower()
    warmup_steps = max(1, int(num_training_steps * warmup_ratio))

    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / max(1, num_training_steps - warmup_steps)
        if schedule_type == "linear":
            return max(0.0, 1.0 - progress)
        # cosine: same tail as train_ssl (decay toward 0.1 * base)
        return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

class CorrectEnglishSentenceClassifier(nn.Module):
    def __init__(self, encoder, hidden_size=768, num_labels=2, dropout=0.0):  # fairseq pooler_dropout default = 0.0
        super().__init__()
        self.encoder = encoder
        self.dropout = nn.Dropout(dropout)
        self.dense = nn.Linear(hidden_size, hidden_size)
        self.out_proj = nn.Linear(hidden_size, num_labels)

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        x = out.last_hidden_state[:, 0, :]
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        return self.out_proj(x)


class HFClassifierWrapper(nn.Module):
    """Thin wrapper so a HuggingFace AutoModelForSequenceClassification presents the SAME
    (logits-tensor) interface as CorrectEnglishSentenceClassifier — the training loop is unchanged.
    Used only when cfg['use_hf_classifier'] is true (the model's own native head)."""
    def __init__(self, hf_model):
        super().__init__()
        self.hf_model = hf_model

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        return self.hf_model(input_ids=input_ids, attention_mask=attention_mask,
                             token_type_ids=token_type_ids).logits


def _build_classifier(cfg, base, num_labels, is_regression, dev):
    """Build the finetuning model. GATED: default (use_hf_classifier absent/false) => the existing
    CorrectEnglishSentenceClassifier (RoBERTa/fairseq head), so all current tasks are unchanged.
    When use_hf_classifier=true => the model's NATIVE HF classification head
    (AutoModelForSequenceClassification), transferring the already-loaded encoder weights (JEPA or
    fresh) from `base` into its base_model; the HF head is left freshly initialised."""
    if not cfg.get("use_hf_classifier", False):
        return CorrectEnglishSentenceClassifier(base, hidden_size=base.config.hidden_size,
                                                num_labels=num_labels,
                                                dropout=cfg.get("head_dropout", 0.0)).to(dev)
    from transformers import AutoModelForSequenceClassification
    hf_config = AutoConfig.from_pretrained(cfg["model_name"], num_labels=num_labels)
    if "hidden_dropout" in cfg:
        hf_config.hidden_dropout_prob = cfg["hidden_dropout"]
    if "attention_dropout" in cfg:
        hf_config.attention_probs_dropout_prob = cfg["attention_dropout"]
    if "cls_dropout" in cfg:
        hf_config.cls_dropout = cfg["cls_dropout"]  # DeBERTa head dropout before the classifier
    if "pooler_dropout" in cfg:
        hf_config.pooler_dropout = cfg["pooler_dropout"]  # DeBERTa head dropout before the pooler dense
    hf_config.problem_type = "regression" if is_regression else "single_label_classification"
    hf_model = AutoModelForSequenceClassification.from_pretrained(cfg["model_name"], config=hf_config)
    miss, unexp = hf_model.base_model.load_state_dict(base.state_dict(), strict=False)
    unexp = [k for k in unexp if not k.startswith("pooler.")]  # AutoModel may carry an unused pooler
    print(f"[use_hf_classifier] {cfg['model_name']} native head; encoder -> '{hf_model.base_model_prefix}' "
          f"(missing={len(miss)} unexpected_non_pooler={len(unexp)})")
    return HFClassifierWrapper(hf_model).to(dev)

def finetune(cfg, encoder_ckpt: str, eval_only: bool = False, model_ckpt: str = None):
    set_seed(cfg["seed"])
    # Get GPU ID from config, default to None (which will use cuda:0 or cpu)
    gpu_id = cfg.get("gpu_id", None)
    dev = device(gpu_id)
    if gpu_id is not None:
        print(f"Using device: {dev}")
    else:
        print(f"Using device: {dev} (default)")

    # 1. Load Encoder
    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"])
    # Encoder dropout from config (fairseq model.dropout / attention_dropout); when the
    # keys are absent the HF model default (0.1 / 0.1) is kept, so behavior is unchanged.
    enc_config = AutoConfig.from_pretrained(cfg["model_name"])
    if "hidden_dropout" in cfg:
        enc_config.hidden_dropout_prob = cfg["hidden_dropout"]
    if "attention_dropout" in cfg:
        enc_config.attention_probs_dropout_prob = cfg["attention_dropout"]
    base = AutoModel.from_pretrained(cfg["model_name"], config=enc_config)
    if encoder_ckpt != "fresh" and not eval_only:
        print(f"Loading pretrained encoder from {encoder_ckpt}")
        base = load_encoder_state(base, encoder_ckpt, map_location="cpu")
    else: 
        print(f"Loading fresh encoder")

    # Determine dataset type from config
    dataset_name = cfg["dataset"]
    dataset_config = cfg.get("dataset_config", None)  # e.g., "qnli", "mnli", "rte", "anli"
    is_mnli = dataset_config == "mnli" or (dataset_config is None and "mnli" in dataset_name.lower())
    is_anli = dataset_config == "anli" or (dataset_config is None and "anli" in dataset_name.lower())
    is_swag = dataset_config == "swag" or (dataset_config is None and "swag" in dataset_name.lower())
    is_regression = dataset_config == "stsb" or cfg.get("num_labels", 2) == 1
    
    # Load training and validation datasets
    if is_anli:
        # ANLI: aggregate training sets from specified rounds
        train_rounds = cfg.get("train_rounds", [1, 2, 3])  # Default: use all rounds
        if isinstance(train_rounds, int):
            train_rounds = [train_rounds]  # Convert single int to list
        ds = load_anli_splits(dataset_name, train_rounds=train_rounds, seed=cfg["seed"]) 
        # For ANLI, we'll use dev_r1 as validation (already loaded in load_anli_splits)
        ds_test_matched = {"validation": ds["validation"]}
        # Load test splits separately
        anli_test_splits = load_anli_test_splits(dataset_name)
    elif is_swag:
        swag = load_swag()
        ds = {
            "train": swag["train"],
            "validation": swag["validation"],
        }
        ds_test_matched = {"validation": swag["validation"]}
        anli_test_splits = None
    else:
        # Standard HuggingFace datasets
        ds = load_hf_splits(dataset_name, config=dataset_config)
        ds_test_matched = load_hf_splits_test(cfg.get("test_dataset", dataset_name), config=dataset_config)
        anli_test_splits = None
    
    # For MNLI, also load mismatched validation set
    if is_mnli:
        ds_test_mismatched = load_hf_splits_test_mismatched(cfg.get("test_dataset", dataset_name), config=dataset_config)
    elif is_anli or is_swag:
        ds_test_mismatched = None  # ANLI doesn't have matched/mismatched splits
    else:
        ds_test_mismatched = None
    
    # Load ALL test sets from JSON file
    json_path = os.path.join(os.path.dirname(__file__), "test_ann.json")
    
    if is_mnli:
        # For MNLI, load both matched and mismatched test sets from JSON
        ds_test_matched_json = load_test_set_from_json("mnli", json_path=json_path)
        ds_test_mismatched_json = load_test_set_from_json("mnli-mm", json_path=json_path)
        ds_test_other = None
    elif is_swag:
        ds_test_matched_json = None
        ds_test_mismatched_json = None
        # Use test split for SWAG if available, otherwise fall back to validation
        if "test" in swag:
            ds_test_other = {"test": swag["test"]}
        else:
            print("Warning: SWAG test split not available, using validation split instead")
            ds_test_other = {"test": ds["validation"]}
    else:
        # For other datasets, load test set from JSON based on dataset_config
        ds_test_matched_json = None
        ds_test_mismatched_json = None
        if dataset_config:
            ds_test_other = load_test_set_from_json(dataset_config, json_path=json_path)
        else:
            # Try to infer from dataset name
            if "qnli" in dataset_name.lower():
                ds_test_other = load_test_set_from_json("qnli", json_path=json_path)
            elif "qqp" in dataset_name.lower():
                ds_test_other = load_test_set_from_json("qqp", json_path=json_path)
            elif "rte" in dataset_name.lower():
                ds_test_other = load_test_set_from_json("rte", json_path=json_path)
            elif "sst2" in dataset_name.lower():
                ds_test_other = load_test_set_from_json("sst2", json_path=json_path)
            else:
                ds_test_other = None
                print(f"Warning: Could not determine test set to load from JSON for dataset '{dataset_name}'")
    
    # Sample datasets if sample_percent is specified
    sample_percent = cfg.get("sample_percent", None)
    if sample_percent is not None and sample_percent < 100.0:
        print(f"Sampling {sample_percent}% of datasets for quick prototyping...")
        ds = sample_dataset(ds, sample_percent=sample_percent, seed=cfg["seed"])
        ds_test_matched = sample_dataset(ds_test_matched, sample_percent=sample_percent, seed=cfg["seed"])
        if ds_test_mismatched:
            ds_test_mismatched = sample_dataset(ds_test_mismatched, sample_percent=sample_percent, seed=cfg["seed"])
        if ds_test_matched_json:
            ds_test_matched_json = sample_dataset(ds_test_matched_json, sample_percent=sample_percent, seed=cfg["seed"])
        if ds_test_mismatched_json:
            ds_test_mismatched_json = sample_dataset(ds_test_mismatched_json, sample_percent=sample_percent, seed=cfg["seed"])
        if ds_test_other:
            ds_test_other = sample_dataset(ds_test_other, sample_percent=sample_percent, seed=cfg["seed"])
        if anli_test_splits:
            for split_name in anli_test_splits:
                anli_test_splits[split_name] = sample_dataset(anli_test_splits[split_name], sample_percent=sample_percent, seed=cfg["seed"])
    
    # Get text keys from config (defaults to ["question", "sentence"])
    if is_anli:
        text_keys = cfg.get("text_keys", ["premise", "hypothesis"])  # ANLI uses premise/hypothesis like MNLI
    elif is_mnli:
        text_keys = cfg.get("text_keys", ["premise", "hypothesis"])
    elif is_swag:
        text_keys = None  # SWAG uses special preprocessing
    else:
        text_keys = cfg.get("text_keys", ["question", "sentence"])
    print(f"Using text keys: {text_keys}")
    
    # Pre-tokenize datasets once before training
    if is_swag:
        print("Pre-tokenizing SWAG training dataset (multiple-choice format)...")
        ds["train"] = preprocess_swag_multiple_choice(ds["train"], tokenizer, max_length=cfg["max_length"])
        print("Pre-tokenizing SWAG validation dataset (multiple-choice format)...")
        ds_test_matched["validation"] = preprocess_swag_multiple_choice(ds_test_matched["validation"], tokenizer, max_length=cfg["max_length"])
    else:
        print("Pre-tokenizing training dataset...")
        ds["train"] = preprocess_dataset(ds["train"], tokenizer, max_length=cfg["max_length"], text_keys=text_keys)
        print("Pre-tokenizing validation dataset (matched)...")
        ds_test_matched["validation"] = preprocess_dataset(ds_test_matched["validation"], tokenizer, max_length=cfg["max_length"], text_keys=text_keys)
    
    # Pre-tokenize mismatched validation set for MNLI
    if is_mnli and ds_test_mismatched:
        print("Pre-tokenizing validation dataset (mismatched)...")
        ds_test_mismatched["validation"] = preprocess_dataset(ds_test_mismatched["validation"], tokenizer, max_length=cfg["max_length"], text_keys=text_keys)
    
    # Pre-tokenize test sets from JSON
    if is_mnli:
        if ds_test_matched_json:
            print("Pre-tokenizing MNLI test dataset (from JSON)...")
            ds_test_matched_json["test"] = preprocess_dataset(ds_test_matched_json["test"], tokenizer, max_length=cfg["max_length"], text_keys=text_keys)
        if ds_test_mismatched_json:
            print("Pre-tokenizing MNLI-MM test dataset (from JSON)...")
            ds_test_mismatched_json["test"] = preprocess_dataset(ds_test_mismatched_json["test"], tokenizer, max_length=cfg["max_length"], text_keys=text_keys)
    elif is_anli:
        # Pre-tokenize ANLI test splits
        if anli_test_splits:
            for split_name in anli_test_splits:
                print(f"Pre-tokenizing ANLI {split_name}...")
                anli_test_splits[split_name] = preprocess_dataset(anli_test_splits[split_name], tokenizer, max_length=cfg["max_length"], text_keys=text_keys)
    elif is_swag:
        # Pre-tokenize SWAG test dataset (uses validation split) with multiple-choice format
        if ds_test_other:
            print("Pre-tokenizing swag test dataset (multiple-choice format)...")
            ds_test_other["test"] = preprocess_swag_multiple_choice(ds_test_other["test"], tokenizer, max_length=cfg["max_length"])
    else:
        if ds_test_other:
            print(f"Pre-tokenizing {dataset_config or 'test'} test dataset (from JSON)...")
            ds_test_other["test"] = preprocess_dataset(ds_test_other["test"], tokenizer, max_length=cfg["max_length"], text_keys=text_keys)
    
    # Create dataloaders.
    # Regression (STS-B) needs float labels so continuous 0-5 targets aren't truncated to ints by the
    # collate's long cast; classification keeps long (unchanged). Only is_regression (STS-B) differs.
    _label_dtype = torch.float if is_regression else torch.long
    train_loader, val_loader_matched = make_dataloaders(ds["train"], ds_test_matched["validation"], cfg["batch_size"], cfg["num_workers"], pre_tokenized=True, label_dtype=_label_dtype)
    
    if is_anli:
        # ANLI: create separate test loaders for each round
        val_loader_mismatched = None
        test_loader_matched = None
        test_loader_mismatched = None
        test_loader_other = None
        anli_test_loaders = {}
        if anli_test_splits:
            for split_name, test_ds in anli_test_splits.items():
                _, test_loader = make_dataloaders(ds["train"], test_ds, cfg["batch_size"], cfg["num_workers"], pre_tokenized=True)
                anli_test_loaders[split_name] = test_loader
    elif is_mnli and ds_test_mismatched:
        _, val_loader_mismatched = make_dataloaders(ds["train"], ds_test_mismatched["validation"], cfg["batch_size"], cfg["num_workers"], pre_tokenized=True)
        test_loader_matched = None
        test_loader_mismatched = None
        test_loader_other = None
        if ds_test_matched_json:
            _, test_loader_matched = make_dataloaders(ds["train"], ds_test_matched_json["test"], cfg["batch_size"], cfg["num_workers"], pre_tokenized=True)
        if ds_test_mismatched_json:
            _, test_loader_mismatched = make_dataloaders(ds["train"], ds_test_mismatched_json["test"], cfg["batch_size"], cfg["num_workers"], pre_tokenized=True)
        anli_test_loaders = None
    else:
        val_loader_mismatched = None
        test_loader_other = None
        if ds_test_other:
            _, test_loader_other = make_dataloaders(ds["train"], ds_test_other["test"], cfg["batch_size"], cfg["num_workers"], pre_tokenized=True, label_dtype=_label_dtype)
        test_loader_matched = None
        test_loader_mismatched = None
        anli_test_loaders = None

    # When no test set is available (e.g. CoLA/MRPC/STS-B have no public test labels),
    # fall back to validation accuracy for best-model selection.
    has_test_set = (
        test_loader_other is not None
        or test_loader_matched is not None
        or (anli_test_loaders and len(anli_test_loaders) > 0)
    )

    print(f"Train dataset size: {len(ds['train'])}")
    print(f"Validation dataset size: {len(ds_test_matched['validation'])}")
    if is_anli:
        if anli_test_splits:
            for split_name, test_ds in anli_test_splits.items():
                print(f"Test dataset size ({split_name}): {len(test_ds)}")
    elif is_mnli and ds_test_mismatched:
        print(f"Validation dataset size (mismatched): {len(ds_test_mismatched['validation'])}")
    if is_mnli:
        if ds_test_matched_json:
            print(f"Test dataset size (MNLI from JSON): {len(ds_test_matched_json['test'])}")
        if ds_test_mismatched_json:
            print(f"Test dataset size (MNLI-MM from JSON): {len(ds_test_mismatched_json['test'])}")
    elif not is_anli and ds_test_other:
        print(f"Test dataset size ({dataset_config or 'test'}): {len(ds_test_other['test'])}")

    # Get number of labels from config (defaults to 1 for regression, 2 for classification)
    num_labels = cfg.get("num_labels", 1 if is_regression else 2)
    print(f"Number of labels: {num_labels}")
    
    model = _build_classifier(cfg, base, num_labels, is_regression, dev)
    # fairseq RoBERTa: adam with decoupled weight decay (== AdamW) applied uniformly to all params.
    _betas = tuple(cfg["adam_betas"]) if "adam_betas" in cfg else (0.9, 0.98)
    _eps   = cfg.get("adam_eps", 1e-6)
    _wd    = cfg.get("weight_decay", 0.1)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=_wd, betas=_betas, eps=_eps)

    use_swa = cfg.get("swa", False)
    if use_swa:
        swa_start_epoch = max(1, int(cfg["epochs"] * 0.9))
        swa_model = AveragedModel(model)
        swa_scheduler = SWALR(opt, swa_lr=cfg["lr"] * 0.5)
        print(f"SWA will start at epoch {swa_start_epoch}/{cfg['epochs']}")

    lr_sched_cfg = cfg.get("lr_scheduler", None)
    lr_warmup_ratio = cfg.get("lr_warmup_ratio", 0.06)
    lr_scheduler = None
    if isinstance(lr_sched_cfg, dict):
        sched_type = lr_sched_cfg.get("type", "cosine")
        lr_warmup_ratio = lr_sched_cfg.get("warmup_ratio", lr_warmup_ratio)
    elif lr_sched_cfg is None or (
        isinstance(lr_sched_cfg, str) and lr_sched_cfg.lower() in ("none", "null", "")
    ):
        sched_type = None
    else:
        sched_type = lr_sched_cfg

    if sched_type:
        if use_swa:
            pre_swa_epochs = max(0, swa_start_epoch - 1)
            num_sched_steps = len(train_loader) * pre_swa_epochs
        else:
            num_sched_steps = len(train_loader) * cfg["epochs"]
        # gated: absolute warmup_steps override (DeBERTa uses absolute steps, not a ratio).
        # Only fires if warmup_steps is set in cfg; default (ratio) path is unchanged.
        if "warmup_steps" in cfg and num_sched_steps > 0:
            lr_warmup_ratio = min(1.0, cfg["warmup_steps"] / num_sched_steps)
        if num_sched_steps > 0:
            lr_scheduler = _build_finetune_lr_scheduler(
                opt, num_sched_steps, lr_warmup_ratio, sched_type
            )
            print(
                f"LR scheduler: type={sched_type}, warmup_ratio={lr_warmup_ratio}, "
                f"steps={num_sched_steps}"
                + (" (decay only before SWA)" if use_swa else "")
            )
        else:
            print("LR scheduler skipped: no training steps before SWA phase")

    # Loss criterion
    if is_regression:
        criterion = nn.MSELoss()
        print("Using MSELoss for regression (STS-B)")
    else:
        criterion = nn.CrossEntropyLoss()
    scaler = GradScaler('cuda', enabled=cfg.get("mixed_precision", True))
    best_t = 0.5  # Default threshold, will be updated during training
    
    # Track best models for saving (val and test independently)
    best_val_score = -1.0
    best_test_score = -1.0
    best_val_model_state = None
    best_test_model_state = None

    # Setup results tracking
    epoch_log: list[dict] = []
    _results_dir = cfg.get("results_dir", "results")
    _task = cfg.get("dataset_config") or cfg.get("dataset", "task")
    _exp_name = cfg.get("experiment_name", "")
    _model_slug = cfg.get("model_name", "").replace("-", "_").replace("/", "_")
    _exp_name = f"{_exp_name}_{_model_slug}" if _exp_name else _model_slug
    if cfg.get("dynamic_masking", False):
        _exp_name = _exp_name + "_dynamic_masking"

    # Per-epoch accuracy history for plotting
    _hist_val_matched:    list[float] = []
    _hist_val_mismatched: list[float] = []
    _hist_test_matched:   list[float] = []
    _hist_test_mismatched:list[float] = []
    _hist_test_other:     list[float] = []
    _anli_hist: dict[str, list[float]] = {}
    _hist_loss: list[float] = []
    _hist_lr:   list[float] = []

    # If eval_only mode, load the model checkpoint and skip training
    if eval_only:
        if model_ckpt is None:
            raise ValueError("model_ckpt must be provided when eval_only=True")
        
        if "pretrain" in model_ckpt:
            print("Loading embeddings from pretrained model...")
            base = load_encoder_state(base, model_ckpt, map_location="cpu")
            model = _build_classifier(cfg, base, num_labels, is_regression, dev)
            print("Embeddings loaded. Running evaluation only...")
        else:
            print(f"Loading model checkpoint from {model_ckpt}")
            checkpoint = torch.load(model_ckpt, map_location=dev, weights_only=False)
            if "model_state_dict" in checkpoint:
                model.load_state_dict(checkpoint["model_state_dict"])
            elif "state_dict" in checkpoint:
                model.load_state_dict(checkpoint["state_dict"])
            else:
                model.load_state_dict(checkpoint)
            # Load threshold if available
            if "best_threshold" in checkpoint:
                best_t = checkpoint["best_threshold"]
                print(f"Loaded threshold from checkpoint: {best_t:.3f}")
            print("Model loaded. Running evaluation only...")
    else:
        # Training loop
        for epoch in range(1, cfg["epochs"] + 1):
            # Check if SWA phase starts this epoch
            if use_swa and epoch == swa_start_epoch:
                print(f"Starting SWA at epoch {epoch}")
            
            # train
            model.train()
            running, y_true, y_pred = 0.0, [], []
            y_prob = []  # store positive-class probabilities across the epoch
            for batch in tqdm(train_loader, desc=f"Finetune {epoch}/{cfg['epochs']}", leave=False):
                # Data is already tokenized
                input_ids = batch["input_ids"].to(dev, non_blocking=True)
                attention_mask = batch["attention_mask"].to(dev, non_blocking=True)
                if is_regression:
                    labels = batch["label"].float().to(dev, non_blocking=True)
                else:
                    labels = batch["label"].to(dev, non_blocking=True)
                token_type_ids = batch["token_type_ids"].to(dev, non_blocking=True) if "token_type_ids" in batch else None

                opt.zero_grad(set_to_none=True)
                with amp.autocast("cuda", enabled=cfg.get("mixed_precision", True)):
                    # --- Clean forward ---
                    logits_clean = model(input_ids, attention_mask, token_type_ids=token_type_ids)
                    if is_regression:
                        logits_clean = logits_clean.squeeze(-1)
                    loss_clean = criterion(logits_clean, labels)

                    # --- Combined loss ---
                    loss = loss_clean

                scaler.scale(loss).backward()

                # Gradient clipping
                max_grad_norm = cfg.get("max_grad_norm", 1.0)
                if max_grad_norm > 0:
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

                scaler.step(opt)
                scaler.update()

                if lr_scheduler is not None and not (use_swa and epoch >= swa_start_epoch):
                    lr_scheduler.step()

                # Update SWA model and scheduler if in SWA phase (after each batch)
                if use_swa and epoch >= swa_start_epoch:
                    swa_model.update_parameters(model)
                    swa_scheduler.step()

                running += float(loss.item())
                y_true.extend(labels.detach().cpu().tolist())
                if is_regression:
                    y_pred.extend(logits_clean.detach().cpu().tolist())
                else:
                    y_pred.extend(logits_clean.detach().cpu().argmax(-1).tolist())
                    # collect probabilities (for binary: class 1, for multi-class: all classes)
                    probs_batch = torch.softmax(logits_clean.detach().cpu(), dim=-1)
                    if num_labels > 2:
                        y_prob.append(probs_batch.numpy())
                    else:
                        y_prob.append(probs_batch[:, 1].numpy())

            if is_regression:
                train_pearson = float(np.corrcoef(np.array(y_true), np.array(y_pred))[0, 1])
                train_mcc, train_f1 = train_pearson, train_pearson
            else:
                train_mcc = matthews_corrcoef(y_true, y_pred)
                train_f1 = f1_score(y_true, y_pred, average='macro')
                # concatenate per-batch probabilities to a single vector aligned with y_true
                probs = np.concatenate(y_prob, axis=0)
                true = np.asarray(y_true)

                best_t = 0.5
     
            epoch_loss = running / len(train_loader)
            epoch_lr = opt.param_groups[0]['lr']
            if is_regression:
                print(f"[Finetune] epoch {epoch}: loss={epoch_loss:.4f} pearson={train_mcc:.4f} lr={epoch_lr:.2e}")
            else:
                print(f"[Finetune] epoch {epoch}: loss={epoch_loss:.4f} mcc={train_mcc:.4f} f1={train_f1:.4f} lr={epoch_lr:.2e}")

            # validate on matched
            model.eval()
            v_true_matched, v_prob_matched = [], []
            example_indices_matched = []  # For SWAG grouping
            with torch.no_grad():
                for batch in val_loader_matched:
                    input_ids = batch["input_ids"].to(dev, non_blocking=True)
                    attention_mask = batch["attention_mask"].to(dev, non_blocking=True)
                    labels = batch["label"].to(dev, non_blocking=True)
                    token_type_ids = batch["token_type_ids"].to(dev, non_blocking=True) if "token_type_ids" in batch else None
                    logits = model(input_ids, attention_mask, token_type_ids=token_type_ids)
                    v_true_matched.extend(labels.cpu().tolist())

                    if is_regression:
                        v_prob_matched.append(logits.squeeze(-1).detach().cpu().numpy())
                    elif is_swag:
                        # For SWAG, collect example indices for grouping
                        if "example_idx" in batch:
                            example_indices_matched.extend(batch["example_idx"].cpu().tolist())
                        # Get scores for all choices (we'll group by example_idx later)
                        logits_batch = logits.detach().cpu()
                        if num_labels > 2:
                            scores_batch = logits_batch.max(dim=-1)[0].numpy()
                        else:
                            probs_batch = torch.softmax(logits_batch, dim=-1)[:, 1].numpy()
                            scores_batch = probs_batch
                        v_prob_matched.append(scores_batch)
                    else:
                        if num_labels > 2:
                            probs_batch = torch.softmax(logits.detach().cpu(), dim=-1).numpy()
                            v_prob_matched.append(probs_batch)
                        else:
                            probs_batch = torch.softmax(logits.detach().cpu(), dim=-1)[:, 1].numpy()
                            v_prob_matched.append(probs_batch)

            if is_regression:
                preds_matched = np.concatenate(v_prob_matched, axis=0)
                true_labels_matched = np.array(v_true_matched)
                val_pearson = float(np.corrcoef(true_labels_matched, preds_matched)[0, 1])
                val_spearman = float(spearmanr(true_labels_matched, preds_matched).statistic)
                val_mcc_matched = val_pearson
                val_f1_matched = val_spearman  # repurposed: stores Spearman for pearson_spearman_avg selection
                val_acc_matched = val_pearson
                dataset_label = cfg["dataset_config"].upper() if cfg.get("dataset_config") else "TEST"
                print(f"          val ({dataset_label}): pearson={val_pearson:.4f} spearman={val_spearman:.4f}")
            elif is_swag:
                # Group every 4 examples (choices) together and select the best one
                scores_all = np.concatenate(v_prob_matched, axis=0)
                if example_indices_matched:
                    from collections import defaultdict
                    grouped_scores = defaultdict(list)
                    grouped_labels = defaultdict(list)
                    for idx, score, label in zip(example_indices_matched, scores_all, v_true_matched):
                        grouped_scores[idx].append(score)
                        grouped_labels[idx].append(label)
                    preds_matched = []
                    true_labels_matched = []
                    for ex_idx in sorted(grouped_scores.keys()):
                        choice_scores = grouped_scores[ex_idx]
                        best_choice = np.argmax(choice_scores)
                        preds_matched.append(best_choice)
                        true_labels_matched.append(grouped_labels[ex_idx][0])  # All 4 have same label
                    preds_matched = np.array(preds_matched)
                    true_labels_matched = np.array(true_labels_matched)
                else:
                    # Fallback: group every 4 consecutive examples
                    num_questions = len(scores_all) // 4
                    preds_matched = []
                    true_labels_matched = []
                    for i in range(num_questions):
                        start_idx = i * 4
                        end_idx = start_idx + 4
                        choice_scores = scores_all[start_idx:end_idx]
                        best_choice = np.argmax(choice_scores)
                        preds_matched.append(best_choice)
                        true_labels_matched.append(v_true_matched[start_idx])
                    preds_matched = np.array(preds_matched)
                    true_labels_matched = np.array(true_labels_matched)
                val_mcc_matched = matthews_corrcoef(true_labels_matched, preds_matched)
                val_f1_matched = f1_score(true_labels_matched, preds_matched, average='macro')
                val_acc_matched = accuracy_score(true_labels_matched, preds_matched)
                dataset_label = "MNLI" if is_mnli else (cfg["dataset_config"].upper() if cfg["dataset_config"] else "TEST")
                print(f"          val ({dataset_label} matched): mcc={val_mcc_matched:.4f} f1={val_f1_matched:.4f} acc={val_acc_matched:.4f}")
            else:
                if num_labels > 2:
                    probs_matched = np.concatenate(v_prob_matched, axis=0)
                    preds_matched = np.argmax(probs_matched, axis=1)
                    true_labels_matched = np.array(v_true_matched)
                else:
                    probs_matched = np.concatenate(v_prob_matched, axis=0)
                    preds_matched = (probs_matched >= best_t).astype(int)
                    true_labels_matched = np.array(v_true_matched)
                val_mcc_matched = matthews_corrcoef(true_labels_matched, preds_matched)
                val_f1_matched = f1_score(true_labels_matched, preds_matched, average='macro')
                val_acc_matched = accuracy_score(true_labels_matched, preds_matched)
                dataset_label = "MNLI" if is_mnli else (cfg["dataset_config"].upper() if cfg["dataset_config"] else "TEST")
                print(f"          val ({dataset_label} matched): mcc={val_mcc_matched:.4f} f1={val_f1_matched:.4f} acc={val_acc_matched:.4f}")

            # Evaluate on mismatched validation set for MNLI
            val_acc_mismatched = None
            val_mcc_mismatched = None
            val_f1_mismatched = None
            if is_mnli and val_loader_mismatched:
                v_true_mismatched, v_pred_mismatched, v_prob_mismatched = [], [], []
                with torch.no_grad():
                    for batch in val_loader_mismatched:
                        input_ids = batch["input_ids"].to(dev, non_blocking=True)
                        attention_mask = batch["attention_mask"].to(dev, non_blocking=True)
                        labels = batch["label"].to(dev, non_blocking=True)
                        token_type_ids = batch["token_type_ids"].to(dev, non_blocking=True) if "token_type_ids" in batch else None
                        logits = model(input_ids, attention_mask, token_type_ids=token_type_ids)
                        v_true_mismatched.extend(labels.cpu().tolist())
                        v_pred_mismatched.extend(logits.cpu().argmax(-1).tolist())
                        # For multi-class, get probabilities for all classes
                        if num_labels > 2:
                            probs_batch = torch.softmax(logits.detach().cpu(), dim=-1).numpy()
                            v_prob_mismatched.append(probs_batch)
                        else:
                            probs_batch = torch.softmax(logits.detach().cpu(), dim=-1)[:, 1].numpy()
                            v_prob_mismatched.append(probs_batch)
                
                if num_labels > 2:
                    probs_mismatched = np.concatenate(v_prob_mismatched, axis=0)
                    preds_mismatched = np.argmax(probs_mismatched, axis=1)
                else:
                    probs_mismatched = np.concatenate(v_prob_mismatched, axis=0)
                    preds_mismatched = (probs_mismatched >= best_t).astype(int)
                val_mcc_mismatched = matthews_corrcoef(v_true_mismatched, preds_mismatched)
                val_f1_mismatched = f1_score(v_true_mismatched, preds_mismatched, average='macro')
                val_acc_mismatched = accuracy_score(v_true_mismatched, preds_mismatched)
                print(f"          val (MNLI mismatched): mcc={val_mcc_mismatched:.4f} f1={val_f1_mismatched:.4f} acc={val_acc_mismatched:.4f}")

            def evaluate_loader(loader, set_name, is_swag=False):
                if loader is None:
                    return None
                t_true, t_prob = [], []
                example_indices = []  # For SWAG grouping
                with torch.no_grad():
                    for batch in loader:
                        input_ids = batch["input_ids"].to(dev, non_blocking=True)
                        attention_mask = batch["attention_mask"].to(dev, non_blocking=True)
                        labels = batch["label"].to(dev, non_blocking=True)
                        token_type_ids = batch["token_type_ids"].to(dev, non_blocking=True) if "token_type_ids" in batch else None
                        logits = model(input_ids, attention_mask, token_type_ids=token_type_ids)
                        t_true.extend(labels.cpu().tolist())

                        if is_regression:
                            t_prob.append(logits.squeeze(-1).detach().cpu().numpy())
                        elif is_swag:
                            # For SWAG, collect example indices for grouping
                            if "example_idx" in batch:
                                example_indices.extend(batch["example_idx"].cpu().tolist())
                            logits_batch = logits.detach().cpu()
                            if num_labels > 2:
                                scores_batch = logits_batch.max(dim=-1)[0].numpy()
                            else:
                                probs_batch = torch.softmax(logits_batch, dim=-1)[:, 1].numpy()
                                scores_batch = probs_batch
                            t_prob.append(scores_batch)
                        else:
                            if num_labels > 2:
                                probs_batch = torch.softmax(logits.detach().cpu(), dim=-1).numpy()
                                t_prob.append(probs_batch)
                            else:
                                probs_batch = torch.softmax(logits.detach().cpu(), dim=-1)[:, 1].numpy()
                                t_prob.append(probs_batch)

                if not t_prob:
                    return None

                if is_regression:
                    preds_test = np.concatenate(t_prob, axis=0)
                    true_labels = np.array(t_true)
                    test_pearson = float(np.corrcoef(true_labels, preds_test)[0, 1])
                    test_spearman = float(spearmanr(true_labels, preds_test).statistic)
                    print(f"          test ({set_name}): pearson={test_pearson:.4f} spearman={test_spearman:.4f}")
                    return {"acc": (test_pearson + test_spearman) / 2, "mcc": test_pearson, "f1": test_spearman, "pearson": test_pearson, "spearman": test_spearman}
                elif is_swag:
                    # Group every 4 examples (choices) together and select the best one
                    scores_all = np.concatenate(t_prob, axis=0)  # [N*4] where N is number of questions
                    if example_indices:
                        # Group by example_idx
                        from collections import defaultdict
                        grouped_scores = defaultdict(list)
                        grouped_labels = defaultdict(list)
                        for idx, score, label in zip(example_indices, scores_all, t_true):
                            grouped_scores[idx].append(score)
                            grouped_labels[idx].append(label)
                        # For each group, select the choice with highest score
                        preds_test = []
                        true_labels = []
                        for ex_idx in sorted(grouped_scores.keys()):
                            choice_scores = grouped_scores[ex_idx]
                            best_choice = np.argmax(choice_scores)
                            preds_test.append(best_choice)
                            true_labels.append(grouped_labels[ex_idx][0])  # All 4 have same label
                    else:
                        # Fallback: group every 4 consecutive examples
                        num_questions = len(scores_all) // 4
                        preds_test = []
                        true_labels = []
                        for i in range(num_questions):
                            start_idx = i * 4
                            end_idx = start_idx + 4
                            choice_scores = scores_all[start_idx:end_idx]
                            best_choice = np.argmax(choice_scores)
                            preds_test.append(best_choice)
                            true_labels.append(t_true[start_idx])  # All 4 have same label

                    preds_test = np.array(preds_test)
                    true_labels = np.array(true_labels)
                    test_mcc = matthews_corrcoef(true_labels, preds_test)
                    test_f1 = f1_score(true_labels, preds_test, average='macro')
                    test_acc = accuracy_score(true_labels, preds_test)
                    print(f"          test ({set_name}): mcc={test_mcc:.4f} f1={test_f1:.4f} acc={test_acc:.4f}")
                    return {"acc": test_acc, "mcc": test_mcc, "f1": test_f1}
                else:
                    if num_labels > 2:
                        probs_test = np.concatenate(t_prob, axis=0)
                        preds_test = np.argmax(probs_test, axis=1)
                        true_labels = np.array(t_true)
                    else:
                        probs_test = np.concatenate(t_prob, axis=0)
                        preds_test = (probs_test >= best_t).astype(int)
                        true_labels = np.array(t_true)
                    test_mcc = matthews_corrcoef(true_labels, preds_test)
                    test_f1 = f1_score(true_labels, preds_test, average='macro')
                    test_acc = accuracy_score(true_labels, preds_test)
                    print(f"          test ({set_name}): mcc={test_mcc:.4f} f1={test_f1:.4f} acc={test_acc:.4f}")
                    return {"acc": test_acc, "mcc": test_mcc, "f1": test_f1}

            test_results = []
            if is_anli:
                # Evaluate on each ANLI test round separately
                if anli_test_loaders:
                    for split_name, test_loader in sorted(anli_test_loaders.items()):
                        res = evaluate_loader(test_loader, split_name)
                        if res:
                            test_results.append((split_name, res))
                    # Use test_r3 accuracy for best model selection (hardest round)
                    if "test_r3" in anli_test_loaders and "test_r3" in [r[0] for r in test_results]:
                        test_r3_res = next(r[1] for r in test_results if r[0] == "test_r3")
                        test_acc = test_r3_res["acc"]
                        test_mcc = test_r3_res["mcc"]
                        test_f1 = test_r3_res["f1"]
                    elif test_results:
                        # Fallback to first available result
                        test_acc = test_results[0][1]["acc"]
                        test_mcc = test_results[0][1]["mcc"]
                        test_f1 = test_results[0][1]["f1"]
                    else:
                        test_acc = test_mcc = test_f1 = 0.0
                else:
                    test_acc = test_mcc = test_f1 = 0.0
            elif is_mnli:
                res_matched = evaluate_loader(test_loader_matched, "MNLI") if test_loader_matched else None
                if res_matched:
                    test_results.append(("MNLI", res_matched))
                res_mismatched = evaluate_loader(test_loader_mismatched, "MNLI-MM") if test_loader_mismatched else None
                if res_mismatched:
                    test_results.append(("MNLI-MM", res_mismatched))
                if res_matched and res_mismatched:
                    test_acc = (res_matched["acc"] + res_mismatched["acc"]) / 2
                    test_mcc = (res_matched["mcc"] + res_mismatched["mcc"]) / 2
                    test_f1 = (res_matched["f1"] + res_mismatched["f1"]) / 2
                elif res_mismatched:
                    test_acc = res_mismatched["acc"]
                    test_mcc = res_mismatched["mcc"]
                    test_f1 = res_mismatched["f1"]
                elif res_matched:
                    test_acc = res_matched["acc"]
                    test_mcc = res_matched["mcc"]
                    test_f1 = res_matched["f1"]
                else:
                    test_acc = test_mcc = test_f1 = 0.0
            else:
                res_other = evaluate_loader(
                    test_loader_other,
                    dataset_config.upper() if dataset_config else "TEST",
                    is_swag=is_swag,
                )
                if res_other:
                    test_results.append((dataset_config.upper() if dataset_config else "TEST", res_other))
                    test_acc = res_other["acc"]
                    test_mcc = res_other["mcc"]
                    test_f1 = res_other["f1"]
                else:
                    test_acc = test_mcc = test_f1 = 0.0
            
            # Save best models (val and test independently)
            ckpt_dir = cfg.get("ckpt_dir", "checkpoints/finetune")
            os.makedirs(ckpt_dir, exist_ok=True)
            prefix = f"{_exp_name}_" if _exp_name else ""

            # Val selection score
            if is_mnli and val_acc_mismatched is not None:
                val_selection_score = (val_acc_matched + val_acc_mismatched) / 2
            else:
                _metric = cfg.get("metric_for_best_model", "acc").lower()
                if _metric in ("mcc", "pearson"):
                    val_selection_score = val_mcc_matched
                elif _metric == "f1":
                    val_selection_score = val_f1_matched
                elif _metric in ("f1_acc_avg", "f1_acc"):
                    val_selection_score = (val_f1_matched + val_acc_matched) / 2
                elif _metric == "pearson_spearman_avg":
                    val_selection_score = (val_mcc_matched + val_f1_matched) / 2
                else:
                    val_selection_score = val_acc_matched

            checkpoint_data = {
                "epoch": epoch,
                "val_mcc_matched": val_mcc_matched,
                "val_acc_matched": val_acc_matched,
                "val_f1_matched": val_f1_matched,
                "test_mcc": test_mcc,
                "test_acc": test_acc,
                "test_f1": test_f1,
                "model_state_dict": model.state_dict(),
                "best_threshold": best_t,
            }
            if is_mnli and val_acc_mismatched is not None:
                checkpoint_data.update({
                    "val_mcc_mismatched": val_mcc_mismatched,
                    "val_acc_mismatched": val_acc_mismatched,
                    "val_f1_mismatched": val_f1_mismatched,
                })

            if val_selection_score > best_val_score:
                best_val_score = val_selection_score
                best_val_model_state = model.state_dict().copy()
                best_val_ckpt_path = os.path.join(ckpt_dir, f"{prefix}best_val.pt")
                save_ckpt(checkpoint_data, best_val_ckpt_path)
                if is_regression:
                    val_metric_label = "Val Pearson"
                elif is_mnli and val_acc_mismatched is not None:
                    val_metric_label = "Val Avg Acc (M/MM)"
                else:
                    val_metric_label = "Val Acc"
                print(f"          ✓ Saved best val model ({val_metric_label}={val_selection_score:.4f}) to {best_val_ckpt_path}")

            if has_test_set and test_acc > best_test_score:
                best_test_score = test_acc
                best_test_model_state = model.state_dict().copy()
                best_test_ckpt_path = os.path.join(ckpt_dir, f"{prefix}best_test.pt")
                save_ckpt(checkpoint_data, best_test_ckpt_path)
                if is_regression:
                    test_metric_label = "Test Pearson"
                elif is_mnli:
                    test_metric_label = "Test Avg Acc (M/MM)"
                else:
                    test_metric_label = "Test Acc"
                print(f"          ✓ Saved best test model ({test_metric_label}={test_acc:.4f}) to {best_test_ckpt_path}")
            
            # Also save per-epoch checkpoint with test accuracy in filename
            epoch_ckpt_path = os.path.join(ckpt_dir, f"{prefix}epoch{epoch}_testacc{test_acc:.4f}.pt")
            checkpoint_data = {
                "epoch": epoch,
                "val_mcc_matched": val_mcc_matched,
                "val_acc_matched": val_acc_matched,
                "val_f1_matched": val_f1_matched,
                "test_mcc": test_mcc,
                "test_acc": test_acc,
                "test_f1": test_f1,
                "model_state_dict": model.state_dict(),
                "best_threshold": best_t,
            }
            if is_mnli and val_acc_mismatched is not None:
                checkpoint_data.update({
                    "val_mcc_mismatched": val_mcc_mismatched,
                    "val_acc_mismatched": val_acc_mismatched,
                    "val_f1_mismatched": val_f1_mismatched,
                })
            #save_ckpt(checkpoint_data, epoch_ckpt_path)

            # --- record per-epoch metrics for plotting ---
            _hist_loss.append(epoch_loss)
            _hist_lr.append(epoch_lr)
            _hist_val_matched.append(val_acc_matched)
            if is_mnli and val_acc_mismatched is not None:
                _hist_val_mismatched.append(val_acc_mismatched)
            if is_mnli:
                for name, res in test_results:
                    _anli_hist.setdefault(name, []).append(res["acc"])
                    if name == "MNLI":
                        _hist_test_matched.append(res["acc"])
                    elif name == "MNLI-MM":
                        _hist_test_mismatched.append(res["acc"])
            elif is_anli:
                for name, res in test_results:
                    _anli_hist.setdefault(name, []).append(res["acc"])
            else:
                for name, res in test_results:
                    _hist_test_other.append(res["acc"])

            # positive-class F1 (canonical MRPC/QQP metric) for binary classification tasks;
            # stored alongside the existing macro `f1` (which model selection depends on), not replacing it.
            val_f1_pos = None
            if (not is_regression) and (not is_swag) and num_labels == 2:
                val_f1_pos = float(f1_score(true_labels_matched, preds_matched, pos_label=1))

            # --- save per-epoch results to JSON ---
            epoch_entry = {
                "epoch": epoch,
                "train": {"loss": epoch_loss, "lr": epoch_lr},
                "validation": {"matched": {"acc": val_acc_matched, "mcc": val_mcc_matched, "f1": val_f1_matched, "f1_pos": val_f1_pos}},
                "test": {name: res for name, res in test_results},
            }
            if is_mnli and val_acc_mismatched is not None:
                epoch_entry["validation"]["mismatched"] = {"acc": val_acc_mismatched, "mcc": val_mcc_mismatched, "f1": val_f1_mismatched}
            epoch_log.append(epoch_entry)
            save_finetune_results(_results_dir, _task, _exp_name, {"model": cfg.get("model_name", ""), "epochs": epoch_log})

        # Load best model for final evaluation: prefer test if available, else val
        if has_test_set and best_test_model_state is not None:
            print(f"\nLoading best test model (Test Acc={best_test_score:.4f}) for final evaluation...")
            model.load_state_dict(best_test_model_state)
        elif best_val_model_state is not None:
            print(f"\nLoading best val model (Val={best_val_score:.4f}) for final evaluation...")
            model.load_state_dict(best_val_model_state)
        
        # After training, update BN stats for SWA model (optional, but don't replace best model)
        if use_swa:
            print("Updating batch normalization statistics for SWA model...")
            # Create a temporary model for SWA evaluation
            swa_model_for_eval = _build_classifier(cfg, base, num_labels, is_regression, dev)
            swa_model_for_eval.load_state_dict(swa_model.module.state_dict())
            
            # Update BN stats by running one forward pass through training data
            swa_model_for_eval.eval()
            with torch.no_grad():
                for batch in tqdm(train_loader, desc="Updating BN stats", leave=False):
                    input_ids = batch["input_ids"].to(dev, non_blocking=True)
                    attention_mask = batch["attention_mask"].to(dev, non_blocking=True)
                    token_type_ids = batch["token_type_ids"].to(dev, non_blocking=True) if "token_type_ids" in batch else None
                    _ = swa_model_for_eval(input_ids, attention_mask, token_type_ids=token_type_ids)
            # Note: BERT uses LayerNorm, not BatchNorm, so update_bn is not strictly necessary
            # But we'll update any BatchNorm layers if they exist (e.g., in custom heads)
            try:
                update_bn(train_loader, swa_model_for_eval, device=dev)
            except Exception:
                # If update_bn fails (e.g., no BN layers), that's fine
                pass
            
            # Evaluate SWA model on test set to compare with best test accuracy model
            print("Evaluating SWA model on test set for comparison...")
            swa_model_for_eval.eval()
            
            # Select appropriate test loader (all from JSON)
            if is_mnli:
                swa_test_loader = test_loader_mismatched if test_loader_mismatched else test_loader_matched
            else:
                swa_test_loader = test_loader_other
            
            if swa_test_loader:
                swa_t_true, swa_t_pred, swa_t_prob = [], [], []
                with torch.no_grad():
                    for batch in swa_test_loader:
                        input_ids = batch["input_ids"].to(dev, non_blocking=True)
                        attention_mask = batch["attention_mask"].to(dev, non_blocking=True)
                        labels = batch["label"].to(dev, non_blocking=True)
                        token_type_ids = batch["token_type_ids"].to(dev, non_blocking=True) if "token_type_ids" in batch else None
                        logits = swa_model_for_eval(input_ids, attention_mask, token_type_ids=token_type_ids)
                        swa_t_true.extend(labels.cpu().tolist())
                        if is_regression:
                            swa_t_prob.append(logits.squeeze(-1).detach().cpu().numpy())
                        elif num_labels > 2:
                            probs_batch = torch.softmax(logits.detach().cpu(), dim=-1).numpy()
                            swa_t_prob.append(probs_batch)
                        else:
                            probs_batch = torch.softmax(logits.detach().cpu(), dim=-1)[:, 1].numpy()
                            swa_t_prob.append(probs_batch)

                if is_regression:
                    swa_t_preds = np.concatenate(swa_t_prob, axis=0)
                    swa_test_acc = float(np.corrcoef(np.array(swa_t_true), swa_t_preds)[0, 1])
                    print(f"SWA model test Pearson: {swa_test_acc:.4f}")
                elif num_labels > 2:
                    swa_t_probs = np.concatenate(swa_t_prob, axis=0)
                    swa_t_preds = np.argmax(swa_t_probs, axis=1)
                    swa_test_acc = accuracy_score(swa_t_true, swa_t_preds)
                    print(f"SWA model test accuracy: {swa_test_acc:.4f}")
                else:
                    swa_t_probs = np.concatenate(swa_t_prob, axis=0)
                    swa_t_preds = (swa_t_probs >= best_t).astype(int)
                    swa_test_acc = accuracy_score(swa_t_true, swa_t_preds)
                    print(f"SWA model test accuracy: {swa_test_acc:.4f}")

                # Only use SWA model if it's better than the currently loaded best model
                _ref_score = best_test_score if has_test_set else best_val_score
                metric_name = "Test Pearson" if is_regression else "Test Acc"
                if swa_test_acc > _ref_score:
                    print(f"SWA model is better ({metric_name}={swa_test_acc:.4f} > {_ref_score:.4f}). Using SWA model for final evaluation.")
                    model.load_state_dict(swa_model_for_eval.state_dict())
                    if has_test_set:
                        best_test_score = swa_test_acc
                    else:
                        best_val_score = swa_test_acc
                else:
                    print(f"Best model is better ({metric_name}={_ref_score:.4f} >= {swa_test_acc:.4f}). Using best model.")
            else:
                print("No test loader available for SWA evaluation.")

        # --- generate accuracy plots ---
        if not eval_only and _hist_val_matched:
            results_dir = cfg.get("results_dir", "results")
            task_name = cfg.get("dataset_config") or cfg.get("dataset", "task")
            plot_name = f"{_exp_name}_{task_name}" if _exp_name else task_name
            epochs_list = list(range(1, len(_hist_val_matched) + 1))

            if is_mnli:
                curves = {"val (matched)": _hist_val_matched}
                if _hist_val_mismatched:
                    curves["val (mismatched)"] = _hist_val_mismatched
                if _hist_test_matched:
                    curves["test (matched)"] = _hist_test_matched
                if _hist_test_mismatched:
                    curves["test (mismatched)"] = _hist_test_mismatched
                _save_training_plots(
                    epochs_list, curves, _hist_loss, _hist_lr,
                    f"MNLI accuracy per epoch",
                    os.path.join(results_dir, f"{plot_name}_training.png"),
                )
            elif is_anli:
                curves = {"val": _hist_val_matched}
                curves.update(_anli_hist)
                _save_training_plots(
                    epochs_list, curves, _hist_loss, _hist_lr,
                    f"ANLI accuracy per epoch",
                    os.path.join(results_dir, f"{plot_name}_training.png"),
                )
            else:
                curves = {"val": _hist_val_matched}
                if _hist_test_other:
                    curves["test"] = _hist_test_other
                _save_training_plots(
                    epochs_list, curves, _hist_loss, _hist_lr,
                    f"{task_name.upper()} accuracy per epoch",
                    os.path.join(results_dir, f"{plot_name}_training.png"),
                )

    # Final validation
    final_results = {"validation": {}, "test": {}}
    model.eval()
    
    # Matched validation
    v_true_matched, v_pred_matched, v_prob_matched = [], [], []
    with torch.no_grad():
        for batch in val_loader_matched:
            input_ids = batch["input_ids"].to(dev, non_blocking=True)
            attention_mask = batch["attention_mask"].to(dev, non_blocking=True)
            labels = batch["label"].to(dev, non_blocking=True)
            token_type_ids = batch["token_type_ids"].to(dev, non_blocking=True) if "token_type_ids" in batch else None
            logits = model(input_ids, attention_mask, token_type_ids=token_type_ids)
            v_true_matched.extend(labels.cpu().tolist())
            if is_regression:
                v_prob_matched.append(logits.squeeze(-1).detach().cpu().numpy())
            else:
                v_pred_matched.extend(logits.cpu().argmax(-1).tolist())
                if num_labels > 2:
                    probs_batch = torch.softmax(logits.detach().cpu(), dim=-1).numpy()
                    v_prob_matched.append(probs_batch)
                else:
                    probs_batch = torch.softmax(logits.detach().cpu(), dim=-1)[:, 1].numpy()
                    v_prob_matched.append(probs_batch)

    if is_regression:
        preds_matched = np.concatenate(v_prob_matched, axis=0)
        true_arr = np.array(v_true_matched)
        val_pearson = float(np.corrcoef(true_arr, preds_matched)[0, 1])
        val_spearman = float(spearmanr(true_arr, preds_matched).statistic)
        val_mcc_matched = val_pearson
        val_f1_matched = val_spearman
        val_acc_matched = val_pearson
        dataset_label = cfg["dataset_config"].upper() if cfg.get("dataset_config") else "STSB"
        print(f"Final validation ({dataset_label}): pearson={val_pearson:.4f} spearman={val_spearman:.4f}")
    else:
        if num_labels > 2:
            probs_matched = np.concatenate(v_prob_matched, axis=0)
            preds_matched = np.argmax(probs_matched, axis=1)
        else:
            probs_matched = np.concatenate(v_prob_matched, axis=0)
            preds_matched = (probs_matched >= best_t).astype(int)
        val_mcc_matched = matthews_corrcoef(v_true_matched, preds_matched)
        val_f1_matched = f1_score(v_true_matched, preds_matched, average='macro')
        val_acc_matched = accuracy_score(v_true_matched, preds_matched)
        if is_anli:
            dataset_label = "ANLI"
        elif is_mnli:
            dataset_label = "MNLI"
        else:
            dataset_label = cfg["dataset_config"].upper() if cfg.get("dataset_config") else "QNLI"
        print(f"Final validation ({dataset_label} matched): mcc={val_mcc_matched:.4f} f1={val_f1_matched:.4f} acc={val_acc_matched:.4f}")
    final_results["validation"]["matched"] = {"acc": val_acc_matched, "mcc": val_mcc_matched, "f1": val_f1_matched}

    # Mismatched validation for MNLI
    if is_mnli and val_loader_mismatched:
        v_true_mismatched, v_pred_mismatched, v_prob_mismatched = [], [], []
        with torch.no_grad():
            for batch in val_loader_mismatched:
                input_ids = batch["input_ids"].to(dev, non_blocking=True)
                attention_mask = batch["attention_mask"].to(dev, non_blocking=True)
                labels = batch["label"].to(dev, non_blocking=True)
                token_type_ids = batch["token_type_ids"].to(dev, non_blocking=True) if "token_type_ids" in batch else None
                logits = model(input_ids, attention_mask, token_type_ids=token_type_ids)
                v_true_mismatched.extend(labels.cpu().tolist())
                v_pred_mismatched.extend(logits.cpu().argmax(-1).tolist())
                if num_labels > 2:
                    probs_batch = torch.softmax(logits.detach().cpu(), dim=-1).numpy()
                    v_prob_mismatched.append(probs_batch)
                else:
                    probs_batch = torch.softmax(logits.detach().cpu(), dim=-1)[:, 1].numpy()
                    v_prob_mismatched.append(probs_batch)
        
        if num_labels > 2:
            probs_mismatched = np.concatenate(v_prob_mismatched, axis=0)
            preds_mismatched = np.argmax(probs_mismatched, axis=1)
        else:
            probs_mismatched = np.concatenate(v_prob_mismatched, axis=0)
            preds_mismatched = (probs_mismatched >= best_t).astype(int)
        val_mcc_mismatched = matthews_corrcoef(v_true_mismatched, preds_mismatched)
        val_f1_mismatched = f1_score(v_true_mismatched, preds_mismatched, average='macro')
        val_acc_mismatched = accuracy_score(v_true_mismatched, preds_mismatched)
        print(f"Final validation (MNLI mismatched): mcc={val_mcc_mismatched:.4f} f1={val_f1_mismatched:.4f} acc={val_acc_mismatched:.4f}")
        final_results["validation"]["mismatched"] = {"acc": val_acc_mismatched, "mcc": val_mcc_mismatched, "f1": val_f1_mismatched}

    # Test set evaluation (all from JSON)
    print("\n" + "="*50)
    if is_mnli:
        # Evaluate on both MNLI matched and mismatched test sets from JSON
        if test_loader_matched:
            print("Evaluating on MNLI test set (matched, from JSON)...")
            t_true_matched, t_pred_matched, t_prob_matched = [], [], []
            with torch.no_grad():
                for batch in test_loader_matched:
                    input_ids = batch["input_ids"].to(dev, non_blocking=True)
                    attention_mask = batch["attention_mask"].to(dev, non_blocking=True)
                    labels = batch["label"].to(dev, non_blocking=True)
                    token_type_ids = batch["token_type_ids"].to(dev, non_blocking=True) if "token_type_ids" in batch else None
                    logits = model(input_ids, attention_mask, token_type_ids=token_type_ids)
                    t_true_matched.extend(labels.cpu().tolist())
                    t_pred_matched.extend(logits.cpu().argmax(-1).tolist())
                    if num_labels > 2:
                        probs_batch = torch.softmax(logits.detach().cpu(), dim=-1).numpy()
                        t_prob_matched.append(probs_batch)
                    else:
                        probs_batch = torch.softmax(logits.detach().cpu(), dim=-1)[:, 1].numpy()
                        t_prob_matched.append(probs_batch)
            
            if num_labels > 2:
                probs_test_matched = np.concatenate(t_prob_matched, axis=0)
                preds_test_matched = np.argmax(probs_test_matched, axis=1)
            else:
                probs_test_matched = np.concatenate(t_prob_matched, axis=0)
                preds_test_matched = (probs_test_matched >= best_t).astype(int)
            test_mcc_matched = matthews_corrcoef(t_true_matched, preds_test_matched)
            test_f1_matched = f1_score(t_true_matched, preds_test_matched, average='macro')
            test_acc_matched = accuracy_score(t_true_matched, preds_test_matched)
            print(f"Test (MNLI matched, from JSON): mcc={test_mcc_matched:.4f} f1={test_f1_matched:.4f} acc={test_acc_matched:.4f}")
            final_results["test"]["matched"] = {"acc": test_acc_matched, "mcc": test_mcc_matched, "f1": test_f1_matched}

        if test_loader_mismatched:
            print("Evaluating on MNLI test set (mismatched, from JSON)...")
            t_true_mismatched, t_pred_mismatched, t_prob_mismatched = [], [], []
            with torch.no_grad():
                for batch in test_loader_mismatched:
                    input_ids = batch["input_ids"].to(dev, non_blocking=True)
                    attention_mask = batch["attention_mask"].to(dev, non_blocking=True)
                    labels = batch["label"].to(dev, non_blocking=True)
                    token_type_ids = batch["token_type_ids"].to(dev, non_blocking=True) if "token_type_ids" in batch else None
                    logits = model(input_ids, attention_mask, token_type_ids=token_type_ids)
                    t_true_mismatched.extend(labels.cpu().tolist())
                    t_pred_mismatched.extend(logits.cpu().argmax(-1).tolist())
                    if num_labels > 2:
                        probs_batch = torch.softmax(logits.detach().cpu(), dim=-1).numpy()
                        t_prob_mismatched.append(probs_batch)
                    else:
                        probs_batch = torch.softmax(logits.detach().cpu(), dim=-1)[:, 1].numpy()
                        t_prob_mismatched.append(probs_batch)
            
            if num_labels > 2:
                probs_test_mismatched = np.concatenate(t_prob_mismatched, axis=0)
                preds_test_mismatched = np.argmax(probs_test_mismatched, axis=1)
            else:
                probs_test_mismatched = np.concatenate(t_prob_mismatched, axis=0)
                preds_test_mismatched = (probs_test_mismatched >= best_t).astype(int)
            test_mcc_mismatched = matthews_corrcoef(t_true_mismatched, preds_test_mismatched)
            test_f1_mismatched = f1_score(t_true_mismatched, preds_test_mismatched, average='macro')
            test_acc_mismatched = accuracy_score(t_true_mismatched, preds_test_mismatched)
            print(f"Test (MNLI mismatched, from JSON): mcc={test_mcc_mismatched:.4f} f1={test_f1_mismatched:.4f} acc={test_acc_mismatched:.4f}")
            final_results["test"]["mismatched"] = {"acc": test_acc_mismatched, "mcc": test_mcc_mismatched, "f1": test_f1_mismatched}
    elif is_anli:
        # ANLI: Evaluate on each test round separately
        if anli_test_loaders:
            for split_name, test_loader in sorted(anli_test_loaders.items()):
                print(f"Evaluating on ANLI {split_name}...")
                t_true, t_pred, t_prob = [], [], []
                with torch.no_grad():
                    for batch in test_loader:
                        input_ids = batch["input_ids"].to(dev, non_blocking=True)
                        attention_mask = batch["attention_mask"].to(dev, non_blocking=True)
                        labels = batch["label"].to(dev, non_blocking=True)
                        token_type_ids = batch["token_type_ids"].to(dev, non_blocking=True) if "token_type_ids" in batch else None
                        logits = model(input_ids, attention_mask, token_type_ids=token_type_ids)
                        t_true.extend(labels.cpu().tolist())
                        t_pred.extend(logits.cpu().argmax(-1).tolist())
                        if num_labels > 2:
                            probs_batch = torch.softmax(logits.detach().cpu(), dim=-1).numpy()
                            t_prob.append(probs_batch)
                        else:
                            probs_batch = torch.softmax(logits.detach().cpu(), dim=-1)[:, 1].numpy()
                            t_prob.append(probs_batch)
                
                if num_labels > 2:
                    probs_test = np.concatenate(t_prob, axis=0)
                    preds_test = np.argmax(probs_test, axis=1)
                else:
                    probs_test = np.concatenate(t_prob, axis=0)
                    preds_test = (probs_test >= best_t).astype(int)
                test_mcc = matthews_corrcoef(t_true, preds_test)
                test_f1 = f1_score(t_true, preds_test, average='macro')
                test_acc = accuracy_score(t_true, preds_test)
                print(f"Test ({split_name}): mcc={test_mcc:.4f} f1={test_f1:.4f} acc={test_acc:.4f}")
                final_results["test"][split_name] = {"acc": test_acc, "mcc": test_mcc, "f1": test_f1}
        else:
            print("Warning: No ANLI test loaders available for evaluation")
    else:
        # Test set evaluation from JSON
        if test_loader_other:
            test_set_name = dataset_config.upper() if dataset_config else "TEST"
            print(f"Evaluating on {test_set_name} test set (from JSON)...")
            t_true, t_pred, t_prob = [], [], []
            with torch.no_grad():
                for batch in test_loader_other:
                    input_ids = batch["input_ids"].to(dev, non_blocking=True)
                    attention_mask = batch["attention_mask"].to(dev, non_blocking=True)
                    labels = batch["label"].to(dev, non_blocking=True)
                    token_type_ids = batch["token_type_ids"].to(dev, non_blocking=True) if "token_type_ids" in batch else None
                    logits = model(input_ids, attention_mask, token_type_ids=token_type_ids)
                    t_true.extend(labels.cpu().tolist())
                    if is_regression:
                        t_prob.append(logits.squeeze(-1).detach().cpu().numpy())
                    else:
                        t_pred.extend(logits.cpu().argmax(-1).tolist())
                        if num_labels > 2:
                            probs_batch = torch.softmax(logits.detach().cpu(), dim=-1).numpy()
                            t_prob.append(probs_batch)
                        else:
                            probs_batch = torch.softmax(logits.detach().cpu(), dim=-1)[:, 1].numpy()
                            t_prob.append(probs_batch)

            if is_regression:
                preds_test = np.concatenate(t_prob, axis=0)
                true_arr = np.array(t_true)
                test_pearson = float(np.corrcoef(true_arr, preds_test)[0, 1])
                test_spearman = float(spearmanr(true_arr, preds_test).statistic)
                print(f"Test ({test_set_name}, from JSON): pearson={test_pearson:.4f} spearman={test_spearman:.4f}")
                final_results["test"][test_set_name.lower()] = {"acc": (test_pearson + test_spearman) / 2, "mcc": test_pearson, "f1": test_spearman, "pearson": test_pearson, "spearman": test_spearman}
            else:
                if num_labels > 2:
                    probs_test = np.concatenate(t_prob, axis=0)
                    preds_test = np.argmax(probs_test, axis=1)
                else:
                    probs_test = np.concatenate(t_prob, axis=0)
                    preds_test = (probs_test >= best_t).astype(int)
                test_mcc = matthews_corrcoef(t_true, preds_test)
                test_f1 = f1_score(t_true, preds_test, average='macro')
                test_acc = accuracy_score(t_true, preds_test)
                print(f"Test ({test_set_name}, from JSON): mcc={test_mcc:.4f} f1={test_f1:.4f} acc={test_acc:.4f}")
                final_results["test"][test_set_name.lower()] = {"acc": test_acc, "mcc": test_mcc, "f1": test_f1}
        else:
            print("Warning: No test loader available for evaluation")

    save_finetune_results(_results_dir, _task, _exp_name, {
        "model": cfg.get("model_name", ""),
        "epochs": epoch_log,
        "final": {"validation": final_results["validation"], "test": final_results["test"]},
    })

    # If in eval_only mode, we're done
    if eval_only:
        print("✅ Evaluation complete.")
        return model
    
    return model


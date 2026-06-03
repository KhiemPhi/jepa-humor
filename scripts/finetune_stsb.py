import argparse
import datetime
import json
import math
import os
import random

import numpy as np
import torch
import torch.nn as nn
import yaml
from datasets import load_dataset
from scipy.stats import pearsonr, spearmanr
from torch import amp
from torch.amp import GradScaler
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.environ["TOKENIZERS_PARALLELISM"] = "false"

from jepa_humor.utils import device, load_encoder_state, save_ckpt, set_seed
from jepa_humor.data import preprocess_dataset


class STSBRegressor(nn.Module):
    def __init__(self, encoder, hidden_size, dropout=0.1):
        super().__init__()
        self.encoder = encoder
        self.dropout = nn.Dropout(dropout)
        self.regressor = nn.Linear(hidden_size, 1)

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        pooled = out.last_hidden_state[:, 0, :]
        return self.regressor(self.dropout(pooled)).squeeze(-1)


def _save_training_plots(epochs, metric_curves, loss_hist, lr_hist, title, save_path):
    panels = [("metric", metric_curves), ("loss", loss_hist), ("lr", lr_hist)]
    active = [(k, v) for k, v in panels if v]
    fig, axes = plt.subplots(len(active), 1, figsize=(8, 4 * len(active)), squeeze=False)
    for i, (kind, data) in enumerate(active):
        ax = axes[i][0]
        if kind == "metric":
            for label, vals in data.items():
                ax.plot(epochs[:len(vals)], vals, marker="o", markersize=3, label=label)
            ax.set_ylabel("Pearson r")
            ax.set_title(title)
            ax.legend()
        elif kind == "loss":
            ax.plot(epochs[:len(data)], data, marker="o", markersize=3, color="tab:orange")
            ax.set_ylabel("Train Loss (MSE)")
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


def _save_results(results_dir, task, experiment, data):
    os.makedirs(results_dir, exist_ok=True)
    fname = f"{experiment}_{task}.json" if experiment else f"{task}.json"
    path = os.path.join(results_dir, fname)
    data = {**data, "task": task, "experiment": experiment, "timestamp": datetime.datetime.now().isoformat()}
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Saved results to {path}")


def _build_lr_scheduler(optimizer, num_steps, warmup_ratio, schedule_type):
    warmup_steps = max(1, int(num_steps * warmup_ratio))

    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / max(1, num_steps - warmup_steps)
        if (schedule_type or "cosine").lower() == "linear":
            return max(0.0, 1.0 - progress)
        return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def finetune_stsb(cfg, encoder_ckpt):
    set_seed(cfg["seed"])
    dev = device(cfg.get("gpu_id", None))
    print(f"Using device: {dev}")

    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"])
    base = AutoModel.from_pretrained(cfg["model_name"])
    if encoder_ckpt != "fresh":
        print(f"Loading pretrained encoder from {encoder_ckpt}")
        base = load_encoder_state(base, encoder_ckpt, map_location="cpu")
    else:
        print("Using fresh encoder")

    # STS-B has no public test labels — use train + validation only
    ds_train = load_dataset("nyu-mll/glue", "stsb", split="train")
    ds_val = load_dataset("nyu-mll/glue", "stsb", split="validation")

    text_keys = cfg.get("text_keys", ["sentence1", "sentence2"])
    max_length = cfg.get("max_length", 128)

    print("Pre-tokenizing train dataset...")
    ds_train_tok = preprocess_dataset(ds_train, tokenizer, max_length=max_length, text_keys=text_keys)
    print("Pre-tokenizing validation dataset...")
    ds_val_tok = preprocess_dataset(ds_val, tokenizer, max_length=max_length, text_keys=text_keys)

    def collate_fn(batch):
        result = {
            "input_ids": torch.tensor([b["input_ids"] for b in batch], dtype=torch.long),
            "attention_mask": torch.tensor([b["attention_mask"] for b in batch], dtype=torch.long),
            "label": torch.tensor([b["label"] for b in batch], dtype=torch.float),
        }
        if "token_type_ids" in batch[0]:
            result["token_type_ids"] = torch.tensor([b["token_type_ids"] for b in batch], dtype=torch.long)
        return result

    seed = cfg["seed"]
    g = torch.Generator()
    g.manual_seed(seed)

    def seed_worker(worker_id):
        np.random.seed(seed + worker_id)
        random.seed(seed + worker_id)
        torch.manual_seed(seed + worker_id)

    batch_size = cfg.get("batch_size", 32)
    num_workers = cfg.get("num_workers", 4)

    train_loader = DataLoader(
        ds_train_tok, batch_size=batch_size, shuffle=True, num_workers=num_workers,
        collate_fn=collate_fn, pin_memory=True, generator=g, worker_init_fn=seed_worker,
    )
    val_loader = DataLoader(
        ds_val_tok, batch_size=batch_size, shuffle=False, num_workers=num_workers,
        collate_fn=collate_fn, pin_memory=True,
    )
    print(f"Train size: {len(ds_train_tok)}, Val size: {len(ds_val_tok)}")

    model = STSBRegressor(base, hidden_size=base.config.hidden_size).to(dev)
    criterion = nn.MSELoss()
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg.get("weight_decay", 0.01))
    scaler = GradScaler("cuda", enabled=cfg.get("mixed_precision", True))

    num_steps = len(train_loader) * cfg["epochs"]
    lr_scheduler = _build_lr_scheduler(opt, num_steps, cfg.get("lr_warmup_ratio", 0.06), cfg.get("lr_scheduler", "cosine"))

    _results_dir = cfg.get("results_dir", "results")
    _exp_name = cfg.get("experiment_name", "finetune_stsb_exp1")
    _model_slug = cfg.get("model_name", "").replace("-", "_").replace("/", "_")
    _exp_name_full = f"{_exp_name}_{_model_slug}" if _exp_name else _model_slug
    if encoder_ckpt == "fresh":
        _exp_name_full = _exp_name_full + "_fresh"

    ckpt_dir = cfg.get("ckpt_dir", "checkpoints/finetune")
    os.makedirs(ckpt_dir, exist_ok=True)

    best_corr_avg = -1.0
    best_model_state = None
    epoch_log = []
    _hist_val_pearson = []
    _hist_loss = []
    _hist_lr = []

    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        running = 0.0
        for batch in tqdm(train_loader, desc=f"Finetune {epoch}/{cfg['epochs']}", leave=False):
            input_ids = batch["input_ids"].to(dev, non_blocking=True)
            attention_mask = batch["attention_mask"].to(dev, non_blocking=True)
            labels = batch["label"].to(dev, non_blocking=True)
            token_type_ids = batch.get("token_type_ids")
            if token_type_ids is not None:
                token_type_ids = token_type_ids.to(dev, non_blocking=True)

            opt.zero_grad(set_to_none=True)
            with amp.autocast("cuda", enabled=cfg.get("mixed_precision", True)):
                preds = model(input_ids, attention_mask, token_type_ids=token_type_ids)
                loss = criterion(preds, labels)

            scaler.scale(loss).backward()
            max_grad_norm = cfg.get("max_grad_norm", 1.0)
            if max_grad_norm > 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            scaler.step(opt)
            scaler.update()
            lr_scheduler.step()
            running += float(loss.item())

        epoch_loss = running / len(train_loader)
        epoch_lr = opt.param_groups[0]["lr"]
        print(f"[Finetune] epoch {epoch}: loss={epoch_loss:.4f} lr={epoch_lr:.2e}")

        model.eval()
        v_preds, v_labels = [], []
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(dev, non_blocking=True)
                attention_mask = batch["attention_mask"].to(dev, non_blocking=True)
                token_type_ids = batch.get("token_type_ids")
                if token_type_ids is not None:
                    token_type_ids = token_type_ids.to(dev, non_blocking=True)
                preds = model(input_ids, attention_mask, token_type_ids=token_type_ids)
                v_preds.extend(preds.cpu().tolist())
                v_labels.extend(batch["label"].tolist())

        val_pearson, _ = pearsonr(v_labels, v_preds)
        val_spearman, _ = spearmanr(v_labels, v_preds)
        print(f"          val: pearson={val_pearson:.4f} spearman={val_spearman:.4f}")

        val_corr_avg = (val_pearson + val_spearman) / 2
        if val_corr_avg > best_corr_avg:
            best_corr_avg = val_corr_avg
            best_model_state = {k: v.clone() for k, v in model.state_dict().items()}
            best_ckpt_path = os.path.join(ckpt_dir, f"{_exp_name_full}_best.pt")
            save_ckpt({
                "epoch": epoch,
                "val_pearson": val_pearson,
                "val_spearman": val_spearman,
                "val_corr_avg": val_corr_avg,
                "model_state_dict": model.state_dict(),
            }, best_ckpt_path)
            print(f"          ✓ Saved best model (avg={val_corr_avg:.4f} pearson={val_pearson:.4f} spearman={val_spearman:.4f}) to {best_ckpt_path}")

        _hist_loss.append(epoch_loss)
        _hist_lr.append(epoch_lr)
        _hist_val_pearson.append(float(val_corr_avg))

        epoch_log.append({
            "epoch": epoch,
            "train": {"loss": epoch_loss, "lr": epoch_lr},
            "validation": {"pearson": float(val_pearson), "spearman": float(val_spearman)},
        })
        _save_results(_results_dir, "stsb", _exp_name_full, {"model": cfg.get("model_name", ""), "epochs": epoch_log})

    # Training plots
    if _hist_val_pearson:
        plot_name = f"{_exp_name_full}_stsb"
        _save_training_plots(
            list(range(1, len(_hist_val_pearson) + 1)),
            {"val (avg Pearson+Spearman)": _hist_val_pearson},
            _hist_loss, _hist_lr,
            "STS-B (Pearson+Spearman)/2 per epoch",
            os.path.join(_results_dir, f"{plot_name}_training.png"),
        )

    # Final evaluation with best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    model.eval()
    v_preds, v_labels = [], []
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch["input_ids"].to(dev, non_blocking=True)
            attention_mask = batch["attention_mask"].to(dev, non_blocking=True)
            token_type_ids = batch.get("token_type_ids")
            if token_type_ids is not None:
                token_type_ids = token_type_ids.to(dev, non_blocking=True)
            preds = model(input_ids, attention_mask, token_type_ids=token_type_ids)
            v_preds.extend(preds.cpu().tolist())
            v_labels.extend(batch["label"].tolist())

    final_pearson, _ = pearsonr(v_labels, v_preds)
    final_spearman, _ = spearmanr(v_labels, v_preds)
    final_corr_avg = (final_pearson + final_spearman) / 2
    print(f"\nFinal validation (best model): avg={final_corr_avg:.4f} pearson={final_pearson:.4f} spearman={final_spearman:.4f}")

    _save_results(_results_dir, "stsb", _exp_name_full, {
        "model": cfg.get("model_name", ""),
        "epochs": epoch_log,
        "final": {"validation": {"pearson": float(final_pearson), "spearman": float(final_spearman),
                                 "corr_avg": float(final_corr_avg)}},
    })

    return model


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, default="configs/finetune_stsb.yml")
    p.add_argument("--encoder_ckpt", type=str, default="fresh")
    p.add_argument("--gpu_id", type=int, default=None)
    args, overrides = p.parse_known_args()
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
    for kv in overrides:
        k, _, v = kv.partition("=")
        cfg[k] = yaml.safe_load(v)
    if args.gpu_id is not None:
        cfg["gpu_id"] = args.gpu_id
    finetune_stsb(cfg, encoder_ckpt=args.encoder_ckpt)
    print("✅ Finetune done.")

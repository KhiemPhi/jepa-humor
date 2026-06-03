"""
eval_superglue.py — Evaluate a model on SuperGLUE diagnostic tasks (AX-b, AX-g).

The MNLI checkpoint outputs 3 classes (entailment / neutral / contradiction).
For the 2-class AX-b / AX-g labels (entailment / not_entailment) we collapse:
  MNLI class 0 (entailment)              → AX label 0 (entailment)
  MNLI class 1 (neutral) or 2 (contradiction) → AX label 1 (not_entailment)

Usage:
  python -m scripts.eval_superglue --config configs/eval_axb.yml \
      --ckpt billionare_file/checkpoints_roberta_large/finetune/finetune_mnli_exp1_FacebookAI_roberta_large_best.pt

  python -m scripts.eval_superglue --config configs/eval_axg.yml \
      --ckpt billionare_file/checkpoints_roberta_large/finetune/finetune_mnli_exp1_FacebookAI_roberta_large_best.pt
"""
import argparse
import datetime
import json
import os

import numpy as np
import torch
import yaml
from datasets import load_dataset
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from jepa_humor.finetune import CorrectEnglishSentenceClassifier
from jepa_humor.utils import device, set_seed

os.environ["TOKENIZERS_PARALLELISM"] = "false"

SUPPORTED = {"axb", "axg"}


def _collate(batch):
    result = {
        "input_ids":      torch.tensor([b["input_ids"]      for b in batch], dtype=torch.long),
        "attention_mask": torch.tensor([b["attention_mask"]  for b in batch], dtype=torch.long),
        "label":          torch.tensor([b["label"]           for b in batch], dtype=torch.long),
    }
    if "token_type_ids" in batch[0]:
        result["token_type_ids"] = torch.tensor(
            [b["token_type_ids"] for b in batch], dtype=torch.long
        )
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config",     required=True,  help="Path to eval config YAML (eval_axb.yml / eval_axg.yml)")
    p.add_argument("--ckpt",       required=True,  help="Path to MNLI fine-tuned checkpoint (.pt)")
    p.add_argument("--gpu_id",     type=int, default=None)
    p.add_argument("--output_dir", default="superglue_results")
    args, overrides = p.parse_known_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    for kv in overrides:
        k, _, v = kv.partition("=")
        cfg[k] = yaml.safe_load(v)
    if args.gpu_id is not None:
        cfg["gpu_id"] = args.gpu_id

    set_seed(cfg.get("seed", 42))
    dev  = device(cfg.get("gpu_id", None))
    task = cfg.get("dataset_config", "").lower()

    if task not in SUPPORTED:
        raise ValueError(f"Unknown task '{task}'. Supported: {sorted(SUPPORTED)}")

    num_labels  = cfg.get("num_labels", 3)   # MNLI model outputs 3 classes
    max_length  = cfg.get("max_length", 256)
    batch_size  = cfg.get("batch_size", 64)
    num_workers = cfg.get("num_workers", 4)

    print(f"Loading model '{cfg['model_name']}' ...")
    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"])
    base      = AutoModel.from_pretrained(cfg["model_name"])
    model     = CorrectEnglishSentenceClassifier(base, base.config.hidden_size, num_labels).to(dev)

    ckpt = torch.load(args.ckpt, map_location=dev, weights_only=False)
    state_dict = ckpt.get("model_state_dict") or ckpt.get("state_dict") or ckpt
    model.load_state_dict(state_dict)
    model.eval()
    print(f"Loaded checkpoint: {args.ckpt}")

    print(f"Loading super_glue/{task} test split ...")
    ds = load_dataset("super_glue", task, split="test")

    tok_ds = ds.map(
        lambda ex: tokenizer(
            ex["premise"], ex["hypothesis"],
            truncation=True, padding="max_length", max_length=max_length,
        ),
        batched=True,
        remove_columns=[c for c in ds.column_names
                        if c not in ("input_ids", "attention_mask", "token_type_ids", "label")],
        desc="Tokenizing",
    )
    loader = DataLoader(tok_ds, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, collate_fn=_collate, pin_memory=True)

    all_true, all_pred = [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc=f"Evaluating {task.upper()}"):
            logits = model(batch["input_ids"].to(dev), batch["attention_mask"].to(dev))
            probs  = torch.softmax(logits, dim=-1)  # [B, 3]
            # Collapse MNLI 3-class → 2-class: entailment wins if P(entailment) ≥ 0.5
            preds  = (probs[:, 0] < 0.5).long()    # 0=entailment, 1=not_entailment
            all_pred.extend(preds.cpu().tolist())
            all_true.extend(batch["label"].tolist())

    valid = np.array(all_true) != -1
    results = {
        "task":       task,
        "model":      cfg["model_name"],
        "checkpoint": args.ckpt,
        "timestamp":  datetime.datetime.now().isoformat(),
        "n_examples": int(valid.sum()),
        "accuracy":   None,
    }

    if valid.any():
        acc = accuracy_score(np.array(all_true)[valid], np.array(all_pred)[valid])
        results["accuracy"] = float(acc)
        print(f"\n{task.upper()} accuracy: {acc:.4f}  ({int(valid.sum())} labeled examples)")
    else:
        print(f"\n{task.upper()}: test set has no labeled examples (expected for leaderboard tasks)")

    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, f"{task}_eval.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved → {out_path}")


if __name__ == "__main__":
    main()

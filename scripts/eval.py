#!/usr/bin/env python3
"""
eval.py — Evaluate a fine-tuned GLUE model and produce GLUE leaderboard submission TSVs.

Usage:
  python scripts/eval.py --config configs/finetune_cola.yml --ckpt checkpoints/finetune/best.pt
  python scripts/eval.py --config configs/finetune_mnli.yml --ckpt checkpoints/finetune/best.pt --gpu_id 1
"""
import argparse
import datetime
import json
import os

import numpy as np
import torch
import yaml
from datasets import load_dataset as _hf_load_dataset
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

from jepa_humor.data import load_test_set_from_json, preprocess_dataset
from jepa_humor.finetune import CorrectEnglishSentenceClassifier
from jepa_humor.finetune_wnli import (
    _eval as _wnli_wsc_eval,
    _extract_negatives,
    _score_candidate,
)
from jepa_humor.utils import device, set_seed
from transformers import AutoModelForMaskedLM

os.environ["TOKENIZERS_PARALLELISM"] = "false"

# ── Smoke-test limiting ─────────────────────────────────────────────────────────
# When --limit N (N>0) is passed, every loaded split is capped to N examples so the
# whole eval runs in seconds. Default 0 = no limit (full sets).
_LIMIT = 0


def _apply_limit(ds):
    return ds.select(range(min(_LIMIT, len(ds)))) if _LIMIT and _LIMIT > 0 else ds


def load_dataset(*args, **kwargs):
    return _apply_limit(_hf_load_dataset(*args, **kwargs))


# ── GLUE task metadata ─────────────────────────────────────────────────────────
# label_map: None → submit integer predictions; dict → map int → string label
TASKS = {
    "cola": dict(text_keys=["sentence"],              num_labels=2, tsv="CoLA.tsv",   label_map=None,                                          metric="mcc"),
    "sst2": dict(text_keys=["sentence"],              num_labels=2, tsv="SST-2.tsv",  label_map=None,                                          metric="acc"),
    "mrpc": dict(text_keys=["sentence1", "sentence2"],num_labels=2, tsv="MRPC.tsv",   label_map=None,                                          metric="f1"),
    "stsb": dict(text_keys=["sentence1", "sentence2"],num_labels=1, tsv="STS-B.tsv",  label_map=None,                                          metric="pearson"),
    "qqp":  dict(text_keys=["question1", "question2"],num_labels=2, tsv="QQP.tsv",    label_map=None,                                          metric="f1"),
    "mnli": dict(text_keys=["premise", "hypothesis"], num_labels=3, tsv="MNLI-m.tsv", label_map={0:"entailment",1:"neutral",2:"contradiction"}, metric="acc"),
    "qnli": dict(text_keys=["question", "sentence"],  num_labels=2, tsv="QNLI.tsv",   label_map={0:"entailment",1:"not_entailment"},            metric="acc"),
    "rte":  dict(text_keys=["sentence1", "sentence2"],num_labels=2, tsv="RTE.tsv",    label_map={0:"entailment",1:"not_entailment"},            metric="acc"),
    "wnli": dict(text_keys=["sentence1", "sentence2"],num_labels=2, tsv="WNLI.tsv",   label_map=None,                                          metric="acc"),
    "ax":   dict(text_keys=["premise", "hypothesis"], num_labels=3, tsv="AX.tsv",    label_map={0:"entailment",1:"neutral",2:"contradiction"}, metric="acc"),
}




def _clf_collate(batch):
    result = {
        "input_ids": torch.tensor([b["input_ids"] for b in batch], dtype=torch.long),
        "attention_mask": torch.tensor([b["attention_mask"] for b in batch], dtype=torch.long),
        "label": torch.tensor([b["label"] for b in batch], dtype=torch.long),
    }
    if "token_type_ids" in batch[0]:
        result["token_type_ids"] = torch.tensor([b["token_type_ids"] for b in batch], dtype=torch.long)
    return result


def _reg_collate(batch):
    result = {
        "input_ids": torch.tensor([b["input_ids"] for b in batch], dtype=torch.long),
        "attention_mask": torch.tensor([b["attention_mask"] for b in batch], dtype=torch.long),
        "label": torch.tensor([b["label"] for b in batch], dtype=torch.float),
    }
    if "token_type_ids" in batch[0]:
        result["token_type_ids"] = torch.tensor([b["token_type_ids"] for b in batch], dtype=torch.long)
    return result


def _make_loader(ds_tok, batch_size, num_workers, collate_fn):
    return DataLoader(ds_tok, batch_size=batch_size, shuffle=False,
                      num_workers=num_workers, collate_fn=collate_fn, pin_memory=True)


def _clf_inference(model, loader, dev, num_labels, threshold):
    """Run classification inference. Returns (true_labels, predictions, probs)."""
    model.eval()
    all_true, all_probs = [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating", leave=False):
            input_ids = batch["input_ids"].to(dev, non_blocking=True)
            attention_mask = batch["attention_mask"].to(dev, non_blocking=True)
            token_type_ids = batch.get("token_type_ids")
            if token_type_ids is not None:
                token_type_ids = token_type_ids.to(dev, non_blocking=True)
            logits = model(input_ids, attention_mask, token_type_ids=token_type_ids)
            all_true.extend(batch["label"].tolist())
            if num_labels > 2:
                probs = torch.softmax(logits.detach().cpu(), dim=-1).numpy()
            else:
                probs = torch.softmax(logits.detach().cpu(), dim=-1)[:, 1].numpy()
            all_probs.append(probs)

    all_probs = np.concatenate(all_probs, axis=0)
    preds = np.argmax(all_probs, axis=1) if num_labels > 2 else (all_probs >= threshold).astype(int)
    return np.array(all_true), preds, all_probs


def _reg_inference(model, loader, dev):
    """Run regression inference. Returns (true_labels, predictions)."""
    model.eval()
    all_true, all_preds = [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc="Evaluating", leave=False):
            input_ids = batch["input_ids"].to(dev, non_blocking=True)
            attention_mask = batch["attention_mask"].to(dev, non_blocking=True)
            token_type_ids = batch.get("token_type_ids")
            if token_type_ids is not None:
                token_type_ids = token_type_ids.to(dev, non_blocking=True)
            preds = model(input_ids, attention_mask, token_type_ids=token_type_ids).squeeze(-1)
            all_true.extend(batch["label"].tolist())
            all_preds.extend(preds.cpu().tolist())
    return np.array(all_true), np.array(all_preds)


def _clf_metrics(true, preds):
    valid = np.array(true) != -1
    if not valid.any():
        return {}
    t, p = np.array(true)[valid], np.array(preds)[valid]
    return {
        "acc": float(accuracy_score(t, p)),
        "mcc": float(matthews_corrcoef(t, p)),
        "f1":  float(f1_score(t, p, average="macro")),
    }


def _reg_metrics(true, preds):
    valid = np.array(true) != -1
    if not valid.any():
        return {}
    t, p = np.array(true)[valid], np.array(preds)[valid]
    pear, _ = pearsonr(t, p)
    spear, _ = spearmanr(t, p)
    return {"pearson": float(pear), "spearman": float(spear)}


def _save_tsv(indices, preds, label_map, path):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        f.write("index\tprediction\n")
        for idx, pred in zip(indices, preds):
            label_str = label_map[int(pred)] if label_map else str(int(pred))
            f.write(f"{idx}\t{label_str}\n")
    print(f"Saved {path} ({len(preds)} rows)")


def _save_stsb_tsv(indices, preds, path):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        f.write("index\tprediction\n")
        for idx, pred in zip(indices, preds):
            f.write(f"{idx}\t{float(pred):.4f}\n")
    print(f"Saved {path} ({len(preds)} rows)")


def _wnli_to_wsc(sentence1, sentence2):
    """
    Convert a WNLI (sentence1, sentence2) pair to WSC format by finding word
    differences, as described in the RoBERTa paper.
    Returns (span_idx, span_text, referent_text) or None if no clear replacement.
    span_idx: word index of the changed span in sentence1 (the pronoun)
    span_text: the words that were replaced in sentence1
    referent_text: the replacement words in sentence2
    """
    from difflib import SequenceMatcher
    words1 = sentence1.split()
    words2 = sentence2.split()
    matcher = SequenceMatcher(None, [w.lower() for w in words1], [w.lower() for w in words2])
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "replace":
            return i1, " ".join(words1[i1:i2]), " ".join(words2[j1:j2])
    return None


def _wnli_score_pair(model, tokenizer, sentence1, sentence2, nlp, max_length, dev, mixed_precision):
    """
    Predict WNLI label for a (sentence1, sentence2) pair.
    Converts to WSC format via word-level diff (RoBERTa paper approach), then
    uses MLM scoring: query (referent) wins if its score >= all other NP candidates.
    """
    result = _wnli_to_wsc(sentence1, sentence2)
    if result is None:
        return 0
    span_idx, span_text, referent_text = result
    if not referent_text or not span_text:
        return 0

    query_score = _score_candidate(
        model, tokenizer, sentence1, span_idx, span_text,
        referent_text, max_length, dev, mixed_precision,
    )
    negatives = _extract_negatives(sentence1, referent_text, nlp)
    cand_scores = [
        _score_candidate(
            model, tokenizer, sentence1, span_idx, span_text,
            neg, max_length, dev, mixed_precision,
        )
        for neg in negatives
    ]
    if all(query_score >= cs for cs in cand_scores):
        return 1
    return 0


def _handle_wnli(cfg, args, dev, results):
    import spacy
    from datasets import load_dataset as _load

    max_length      = cfg.get("max_length", 128)
    mixed_precision = cfg.get("mixed_precision", True)

    print(f"Loading AutoModelForMaskedLM '{cfg['model_name']}' ...")
    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"])
    model     = AutoModelForMaskedLM.from_pretrained(cfg["model_name"])
    ckpt      = torch.load(args.ckpt, map_location=dev, weights_only=False)
    model.load_state_dict(ckpt.get("model_state_dict", ckpt))
    model     = model.to(dev)

    nlp = spacy.load("en_core_web_sm")

    # ── Validation on SuperGLUE WSC ─────────────────────────────────────────
    print("\n── Validation (SuperGLUE WSC) ──")
    wsc     = _load("super_glue", "wsc")
    val_acc = _wnli_wsc_eval(model, tokenizer, wsc["validation"], nlp, dev, max_length, mixed_precision)
    print(f"  WSC val acc: {val_acc:.4f}")
    results["validation"] = {"acc": float(val_acc)}

    # ── GLUE WNLI test TSV ───────────────────────────────────────────────────
    print("\n── GLUE WNLI submission TSV ──")
    glue_test = _load("nyu-mll/glue", "wnli", split="test")
    preds = [
        _wnli_score_pair(
            model, tokenizer,
            ex["sentence1"], ex["sentence2"],
            nlp, max_length, dev, mixed_precision,
        )
        for ex in tqdm(glue_test, desc="WNLI test")
    ]
    _save_tsv(glue_test["idx"], preds, None, os.path.join(args.output_dir, "WNLI.tsv"))
    results["test_predictions"] = len(preds)
    results["test_pred_positive_rate"] = float(sum(preds) / len(preds))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config",     required=True,  help="Path to finetune config YAML")
    p.add_argument("--ckpt",       required=True,  help="Path to finetuned model checkpoint (.pt)")
    p.add_argument("--gpu_id",     type=int, default=None)
    p.add_argument("--output_dir", default="glue_submission", help="Directory for TSV files and eval JSON")
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--limit", type=int, default=0, help="If >0, cap each dataset split to N examples (smoke testing)")
    args, overrides = p.parse_known_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    for kv in overrides:
        k, _, v = kv.partition("=")
        cfg[k] = yaml.safe_load(v)
    if args.gpu_id is not None:
        cfg["gpu_id"] = args.gpu_id

    global _LIMIT
    _LIMIT = args.limit
    if _LIMIT > 0:
        print(f"[smoke] limiting every dataset split to {_LIMIT} examples")

    set_seed(cfg.get("seed", 42))
    dev = device(cfg.get("gpu_id", None))
    task = cfg.get("dataset_config", "").lower()

    if task not in TASKS:
        raise ValueError(f"Unknown task '{task}'. Supported: {sorted(TASKS)}")

    os.makedirs(args.output_dir, exist_ok=True)
    results = {
        "task": task,
        "model": cfg["model_name"],
        "checkpoint": args.ckpt,
        "timestamp": datetime.datetime.now().isoformat(),
    }

    # WNLI uses AutoModelForMaskedLM + MLM scoring — handled separately
    if task == "wnli":
        _handle_wnli(cfg, args, dev, results)
        json_path = os.path.join(args.output_dir, "wnli_eval.json")
        with open(json_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nSaved eval results → {json_path}")
        print("\nDone.")
        return

    meta        = TASKS[task]
    num_labels  = meta["num_labels"]
    is_reg      = (task == "stsb")
    text_keys   = cfg.get("text_keys", meta["text_keys"])
    max_length  = cfg.get("max_length", 256)
    batch_size  = args.batch_size or cfg.get("batch_size", 64)
    num_workers = cfg.get("num_workers", 4)
    label_map   = meta["label_map"]
    collate_fn  = _reg_collate if is_reg else _clf_collate

    # ── Load model ────────────────────────────────────────────────────────────
    print(f"Loading model '{cfg['model_name']}' ...")
    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"])
    base      = AutoModel.from_pretrained(cfg["model_name"])
    model     = CorrectEnglishSentenceClassifier(base, base.config.hidden_size, num_labels).to(dev)

    ckpt = torch.load(args.ckpt, map_location=dev, weights_only=False)
    state_dict = ckpt.get("model_state_dict") or ckpt.get("state_dict") or ckpt
    model.load_state_dict(state_dict)
    threshold = ckpt.get("best_threshold", 0.5) if not is_reg else None
    print(f"Loaded {args.ckpt}" + (f"  threshold={threshold:.3f}" if threshold is not None else ""))

    # ── Validation set (official GLUE val, labelled) ──────────────────────────
    print("\n── Validation ──")
    if task == "mnli":
        for split, name in [("validation_matched", "matched"), ("validation_mismatched", "mismatched")]:
            raw = load_dataset("nyu-mll/glue", "mnli", split=split)
            tok = preprocess_dataset(raw, tokenizer, max_length, text_keys)
            loader = _make_loader(tok, batch_size, num_workers, collate_fn)
            true, preds, _ = _clf_inference(model, loader, dev, num_labels, threshold)
            m = _clf_metrics(true, preds)
            print(f"  val ({name}): {m}")
            results[f"validation_{name}"] = m
    elif task == "ax":
        # AX has no validation split; evaluate on its labelled test split
        raw = load_dataset("nyu-mll/glue", "ax", split="test")
        tok = preprocess_dataset(raw, tokenizer, max_length, text_keys)
        loader = _make_loader(tok, batch_size, num_workers, collate_fn)
        true, preds, _ = _clf_inference(model, loader, dev, num_labels, threshold)
        m = _clf_metrics(true, preds)
        print(f"  test (ax diagnostic): {m}")
        results["test_ax"] = m
    elif is_reg:
        raw = load_dataset("nyu-mll/glue", "stsb", split="validation")
        tok = preprocess_dataset(raw, tokenizer, max_length, text_keys)
        loader = _make_loader(tok, batch_size, num_workers, collate_fn)
        true, preds = _reg_inference(model, loader, dev)
        m = _reg_metrics(true, preds)
        print(f"  val: {m}")
        results["validation"] = m
    else:
        raw = load_dataset("nyu-mll/glue", task, split="validation")
        tok = preprocess_dataset(raw, tokenizer, max_length, text_keys)
        loader = _make_loader(tok, batch_size, num_workers, collate_fn)
        true, preds, _ = _clf_inference(model, loader, dev, num_labels, threshold)
        m = _clf_metrics(true, preds)
        print(f"  val: {m}")
        results["validation"] = m

    # ── Internal test set from test_ann.json (labelled, if available) ─────────
    print("\n── Internal test set (test_ann.json) ──")
    if not is_reg:
        test_keys = [("mnli", "matched"), ("mnli-mm", "mismatched")] if task == "mnli" else [(task, task)]
        for json_key, name in test_keys:
            ds_test = load_test_set_from_json(json_key)
            if ds_test is None:
                print(f"  No internal test data for '{json_key}' — skipping")
                continue
            tok = preprocess_dataset(_apply_limit(ds_test["test"]), tokenizer, max_length, text_keys)
            loader = _make_loader(tok, batch_size, num_workers, collate_fn)
            true, preds, _ = _clf_inference(model, loader, dev, num_labels, threshold)
            m = _clf_metrics(true, preds)
            print(f"  test ({name}): {m}")
            results[f"test_{name}" if task == "mnli" else "test"] = m
    else:
        ds_test = load_test_set_from_json("stsb")
        if ds_test is not None:
            tok = preprocess_dataset(_apply_limit(ds_test["test"]), tokenizer, max_length, text_keys)
            loader = _make_loader(tok, batch_size, num_workers, collate_fn)
            true, preds = _reg_inference(model, loader, dev)
            m = _reg_metrics(true, preds)
            print(f"  test (stsb): {m}")
            results["test"] = m
        else:
            print("  No internal test data for 'stsb' — skipping")

    # ── Save eval JSON ────────────────────────────────────────────────────────
    json_path = os.path.join(args.output_dir, f"{task}_eval.json")
    with open(json_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved eval results → {json_path}")

    # ── GLUE official test set → submission TSV ───────────────────────────────
    print("\n── GLUE submission TSV ──")
    if task == "mnli":
        for split, tsv_name in [("test_matched", "MNLI-m.tsv"), ("test_mismatched", "MNLI-mm.tsv")]:
            raw = load_dataset("nyu-mll/glue", "mnli", split=split)
            indices = raw["idx"]
            tok = preprocess_dataset(raw, tokenizer, max_length, text_keys)
            loader = _make_loader(tok, batch_size, num_workers, collate_fn)
            _, preds, _ = _clf_inference(model, loader, dev, num_labels, threshold)
            _save_tsv(indices, preds, label_map, os.path.join(args.output_dir, tsv_name))
    elif is_reg:
        raw = load_dataset("nyu-mll/glue", "stsb", split="test")
        indices = raw["idx"]
        tok = preprocess_dataset(raw, tokenizer, max_length, text_keys)
        loader = _make_loader(tok, batch_size, num_workers, collate_fn)
        _, preds = _reg_inference(model, loader, dev)
        preds = np.clip(preds, 0.0, 5.0)
        _save_stsb_tsv(indices, preds, os.path.join(args.output_dir, meta["tsv"]))
    else:
        raw = load_dataset("nyu-mll/glue", task, split="test")
        indices = raw["idx"]
        tok = preprocess_dataset(raw, tokenizer, max_length, text_keys)
        loader = _make_loader(tok, batch_size, num_workers, collate_fn)
        _, preds, _ = _clf_inference(model, loader, dev, num_labels, threshold)
        _save_tsv(indices, preds, label_map, os.path.join(args.output_dir, meta["tsv"]))

    print("\nDone.")


if __name__ == "__main__":
    main()

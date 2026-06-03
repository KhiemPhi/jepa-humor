"""
SuperGLUE fine-tuning for COPA, MultiRC, WSC (RTE), and ReCoRD.
BoolQ, CB, and WiC use the standard finetune() from finetune.py unchanged.
"""

import os
import random
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
from torch.amp import GradScaler
from torch import amp
from torch.utils.data import DataLoader
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics import f1_score, accuracy_score
from tqdm import tqdm

from .utils import set_seed, device, load_encoder_state, save_ckpt
from .finetune import (
    CorrectEnglishSentenceClassifier,
    _build_finetune_lr_scheduler,
    _save_training_plots,
    save_finetune_results,
)


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _load_model(cfg, encoder_ckpt, num_labels, dev):
    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"])
    base = AutoModel.from_pretrained(cfg["model_name"])
    if encoder_ckpt != "fresh":
        print(f"Loading pretrained encoder from {encoder_ckpt}")
        base = load_encoder_state(base, encoder_ckpt, map_location="cpu")
    else:
        print("Using fresh encoder")
    model = CorrectEnglishSentenceClassifier(
        base, hidden_size=base.config.hidden_size, num_labels=num_labels
    ).to(dev)
    return tokenizer, model


def _build_scheduler(model, cfg, num_steps):
    _betas = tuple(cfg["adam_betas"]) if "adam_betas" in cfg else (0.9, 0.999)
    _eps   = cfg.get("adam_eps", 1e-8)
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["lr"],
        weight_decay=cfg.get("weight_decay", 0.01),
        betas=_betas,
        eps=_eps,
    )
    sched_type = cfg.get("lr_scheduler", "cosine")
    warmup_ratio = cfg.get("lr_warmup_ratio", 0.06)
    scheduler = None
    if sched_type and str(sched_type).lower() not in ("none", "null", ""):
        scheduler = _build_finetune_lr_scheduler(opt, num_steps, warmup_ratio, sched_type)
    return opt, scheduler


def _exp_slug(cfg):
    model_slug = cfg.get("model_name", "").replace("-", "_").replace("/", "_")
    exp = cfg.get("experiment_name", "")
    return f"{exp}_{model_slug}" if exp else model_slug


# ---------------------------------------------------------------------------
# COPA  (2-way multiple choice, 400 train / 100 val)
# ---------------------------------------------------------------------------

def _copa_text(premise, question, choice):
    """Concatenate premise + connector + choice as a single sequence (no [SEP] split)."""
    connector = "because" if question == "cause" else "so"
    return premise + " " + connector + " " + choice


def _expand_copa(split):
    """Expand each COPA example into 2 single-sequence binary classification rows."""
    records = []
    for ex in split:
        for ci, choice in enumerate([ex["choice1"], ex["choice2"]]):
            records.append({
                "text":        _copa_text(ex["premise"], ex["question"], choice),
                "label":       1 if ci == ex["label"] else 0,
                "example_idx": ex["idx"],
                "choice_idx":  ci,
            })
    return Dataset.from_list(records)


def _tok_copa(split, tokenizer, max_length):
    keep = {"input_ids", "attention_mask", "label", "example_idx", "choice_idx"}
    return split.map(
        lambda ex: tokenizer(
            ex["text"],
            truncation=True, padding="max_length", max_length=max_length,
        ),
        batched=True,
        remove_columns=[c for c in split.column_names if c not in keep],
        desc="Tokenizing COPA",
    )


def _eval_copa(model, ds_raw, tokenizer, dev, max_length, mixed_precision):
    model.eval()
    ncorrect = 0
    with torch.no_grad():
        for ex in ds_raw:
            scores = []
            for choice in [ex["choice1"], ex["choice2"]]:
                text = _copa_text(ex["premise"], ex["question"], choice)
                enc  = tokenizer(text, return_tensors="pt", truncation=True,
                                 padding="max_length", max_length=max_length)
                with amp.autocast("cuda", enabled=mixed_precision):
                    logits = model(enc["input_ids"].to(dev), enc["attention_mask"].to(dev))
                scores.append(torch.softmax(logits, dim=-1)[0, 1].item())
            pred = int(scores[1] > scores[0])
            ncorrect += pred == ex["label"]
    return ncorrect / len(ds_raw)


def finetune_copa(cfg, encoder_ckpt: str):
    set_seed(cfg["seed"])
    dev = device(cfg.get("gpu_id", None))
    print(f"Using device: {dev}")

    max_length      = cfg.get("max_length", 128)
    mixed_precision = cfg.get("mixed_precision", True)
    epochs          = cfg["epochs"]

    tokenizer, model = _load_model(cfg, encoder_ckpt, num_labels=2, dev=dev)

    ds = load_dataset("super_glue", "copa")
    print(f"COPA train: {len(ds['train'])}, val: {len(ds['validation'])}")

    train_tok = _tok_copa(_expand_copa(ds["train"]), tokenizer, max_length)
    train_tok.set_format(type="torch")
    train_loader = DataLoader(
        train_tok, batch_size=cfg["batch_size"], shuffle=True,
        num_workers=cfg.get("num_workers", 2),
    )

    opt, scheduler = _build_scheduler(model, cfg, epochs * len(train_loader))
    scaler    = GradScaler("cuda", enabled=mixed_precision)
    criterion = nn.CrossEntropyLoss()

    exp      = _exp_slug(cfg)
    ckpt_dir = cfg.get("ckpt_dir", "checkpoints/finetune")
    res_dir  = cfg.get("results_dir", "results")
    os.makedirs(ckpt_dir, exist_ok=True)

    best_val, best_state = -1.0, None
    epoch_log, _hist_val, _hist_loss, _hist_lr = [], [], [], []

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for batch in tqdm(train_loader, desc=f"COPA {epoch}/{epochs}", leave=False):
            input_ids      = batch["input_ids"].to(dev)
            attention_mask = batch["attention_mask"].to(dev)
            labels         = batch["label"].to(dev)
            opt.zero_grad(set_to_none=True)
            with amp.autocast("cuda", enabled=mixed_precision):
                loss = criterion(model(input_ids, attention_mask), labels)
            scaler.scale(loss).backward()
            max_norm = cfg.get("max_grad_norm", 1.0)
            if max_norm > 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            scaler.step(opt)
            scaler.update()
            if scheduler is not None:
                scheduler.step()
            total_loss += loss.item()

        epoch_loss = total_loss / len(train_loader)
        epoch_lr   = opt.param_groups[0]["lr"]
        val_acc    = _eval_copa(model, ds["validation"], tokenizer, dev, max_length, mixed_precision)
        print(f"[COPA] epoch {epoch}: loss={epoch_loss:.4f} val_acc={val_acc:.4f} lr={epoch_lr:.2e}")

        if val_acc > best_val:
            best_val   = val_acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            ckpt_path  = os.path.join(ckpt_dir, f"{exp}_copa_best.pt")
            save_ckpt({"epoch": epoch, "val_acc": val_acc, "model_state_dict": model.state_dict()}, ckpt_path)
            print(f"          ✓ Saved best model (val_acc={val_acc:.4f}) to {ckpt_path}")

        _hist_val.append(val_acc)
        _hist_loss.append(epoch_loss)
        _hist_lr.append(epoch_lr)
        epoch_log.append({
            "epoch": epoch,
            "train": {"loss": epoch_loss, "lr": epoch_lr},
            "validation": {"acc": val_acc},
        })
        save_finetune_results(res_dir, "copa", exp,
                              {"model": cfg.get("model_name", ""), "epochs": epoch_log})

    if best_state:
        model.load_state_dict(best_state)
    final_acc = _eval_copa(model, ds["validation"], tokenizer, dev, max_length, mixed_precision)
    print(f"Final Val Acc: {final_acc:.4f}")
    save_finetune_results(res_dir, "copa", exp, {
        "model": cfg.get("model_name", ""), "epochs": epoch_log,
        "final": {"validation": {"acc": final_acc}},
    })
    if _hist_val:
        _save_training_plots(
            list(range(1, len(_hist_val) + 1)),
            {"val": _hist_val}, _hist_loss, _hist_lr,
            "COPA accuracy per epoch",
            os.path.join(res_dir, f"{exp}_copa_training.png"),
        )
    return model


# ---------------------------------------------------------------------------
# MultiRC  (binary per answer-candidate, ~27K train / ~4.8K val)
# ---------------------------------------------------------------------------

def _tok_multirc(split, tokenizer, max_length):
    keep = {"input_ids", "attention_mask", "label"}
    def fn(examples):
        # [CLS] paragraph [SEP] question answer [SEP]
        text_b = [q + " " + a for q, a in zip(examples["question"], examples["answer"])]
        return tokenizer(
            examples["paragraph"], text_b,
            truncation=True, padding="max_length", max_length=max_length,
        )
    return split.map(
        fn, batched=True,
        remove_columns=[c for c in split.column_names if c not in keep],
        desc="Tokenizing MultiRC",
    )


def _eval_multirc(model, val_loader, dev, mixed_precision):
    model.eval()
    all_pred, all_true = [], []
    with torch.no_grad():
        for batch in val_loader:
            logits = model(batch["input_ids"].to(dev), batch["attention_mask"].to(dev))
            all_pred.extend(logits.argmax(-1).cpu().tolist())
            all_true.extend(batch["label"].tolist())
    return f1_score(all_true, all_pred, average="binary")


def finetune_multirc(cfg, encoder_ckpt: str):
    set_seed(cfg["seed"])
    dev = device(cfg.get("gpu_id", None))
    print(f"Using device: {dev}")

    max_length      = cfg.get("max_length", 512)
    mixed_precision = cfg.get("mixed_precision", True)
    batch_size      = cfg["batch_size"]
    epochs          = cfg["epochs"]
    num_workers     = cfg.get("num_workers", 2)

    tokenizer, model = _load_model(cfg, encoder_ckpt, num_labels=2, dev=dev)

    ds = load_dataset("super_glue", "multirc")
    print(f"MultiRC train: {len(ds['train'])}, val: {len(ds['validation'])}")

    print("Tokenizing MultiRC train...")
    train_tok = _tok_multirc(ds["train"], tokenizer, max_length)
    train_tok.set_format(type="torch")
    print("Tokenizing MultiRC validation...")
    val_tok = _tok_multirc(ds["validation"], tokenizer, max_length)
    val_tok.set_format(type="torch")

    train_loader = DataLoader(train_tok, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader   = DataLoader(val_tok,   batch_size=batch_size, shuffle=False, num_workers=num_workers)

    opt, scheduler = _build_scheduler(model, cfg, epochs * len(train_loader))
    scaler    = GradScaler("cuda", enabled=mixed_precision)
    criterion = nn.CrossEntropyLoss()

    exp      = _exp_slug(cfg)
    ckpt_dir = cfg.get("ckpt_dir", "checkpoints/finetune")
    res_dir  = cfg.get("results_dir", "results")
    os.makedirs(ckpt_dir, exist_ok=True)

    best_val, best_state = -1.0, None
    epoch_log, _hist_val, _hist_loss, _hist_lr = [], [], [], []

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for batch in tqdm(train_loader, desc=f"MultiRC {epoch}/{epochs}", leave=False):
            input_ids      = batch["input_ids"].to(dev)
            attention_mask = batch["attention_mask"].to(dev)
            labels         = batch["label"].to(dev)
            opt.zero_grad(set_to_none=True)
            with amp.autocast("cuda", enabled=mixed_precision):
                loss = criterion(model(input_ids, attention_mask), labels)
            scaler.scale(loss).backward()
            max_norm = cfg.get("max_grad_norm", 1.0)
            if max_norm > 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            scaler.step(opt)
            scaler.update()
            if scheduler is not None:
                scheduler.step()
            total_loss += loss.item()

        epoch_loss = total_loss / len(train_loader)
        epoch_lr   = opt.param_groups[0]["lr"]
        val_f1     = _eval_multirc(model, val_loader, dev, mixed_precision)
        print(f"[MultiRC] epoch {epoch}: loss={epoch_loss:.4f} val_f1={val_f1:.4f} lr={epoch_lr:.2e}")

        if val_f1 > best_val:
            best_val   = val_f1
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            ckpt_path  = os.path.join(ckpt_dir, f"{exp}_multirc_best.pt")
            save_ckpt({"epoch": epoch, "val_f1": val_f1, "model_state_dict": model.state_dict()}, ckpt_path)
            print(f"          ✓ Saved best model (val_f1={val_f1:.4f}) to {ckpt_path}")

        _hist_val.append(val_f1)
        _hist_loss.append(epoch_loss)
        _hist_lr.append(epoch_lr)
        epoch_log.append({
            "epoch": epoch,
            "train": {"loss": epoch_loss, "lr": epoch_lr},
            "validation": {"f1": val_f1},
        })
        save_finetune_results(res_dir, "multirc", exp,
                              {"model": cfg.get("model_name", ""), "epochs": epoch_log})

    if best_state:
        model.load_state_dict(best_state)
    final_f1 = _eval_multirc(model, val_loader, dev, mixed_precision)
    print(f"Final Val F1: {final_f1:.4f}")
    save_finetune_results(res_dir, "multirc", exp, {
        "model": cfg.get("model_name", ""), "epochs": epoch_log,
        "final": {"validation": {"f1": final_f1}},
    })
    if _hist_val:
        _save_training_plots(
            list(range(1, len(_hist_val) + 1)),
            {"val": _hist_val}, _hist_loss, _hist_lr,
            "MultiRC F1 per epoch",
            os.path.join(res_dir, f"{exp}_multirc_training.png"),
        )
    return model


# ---------------------------------------------------------------------------
# SuperGLUE WiC  (word-in-context: binary classification)
# Standard approach: prepend "{word}: " to sentence1 so the model sees the
# target word explicitly. This matches the common SuperGLUE practice and
# typically gives ~2-3pp over plain sentence-pair classification.
# ---------------------------------------------------------------------------

def _tok_wic(split, tokenizer, max_length, use_word_context=True):
    keep = {"input_ids", "attention_mask", "label"}

    def fn(examples):
        if use_word_context:
            s1 = [f"{w}: {s}" for w, s in zip(examples["word"], examples["sentence1"])]
        else:
            s1 = examples["sentence1"]
        return tokenizer(
            s1, examples["sentence2"],
            truncation=True, padding="max_length", max_length=max_length,
        )

    return split.map(
        fn, batched=True,
        remove_columns=[c for c in split.column_names if c not in keep],
        desc="Tokenizing WiC",
    )


def finetune_superglue_wic(cfg, encoder_ckpt: str):
    """
    WiC: word-in-context binary classification.
    If cfg['use_word_context'] is true (default), prepends '{word}: ' to sentence1
    so the model can use the target word as additional signal.
    """
    set_seed(cfg["seed"])
    dev = device(cfg.get("gpu_id", None))
    print(f"Using device: {dev}")

    max_length       = cfg.get("max_length", 256)
    mixed_precision  = cfg.get("mixed_precision", True)
    epochs           = cfg["epochs"]
    batch_size       = cfg["batch_size"]
    num_workers      = cfg.get("num_workers", 4)
    use_word_context = cfg.get("use_word_context", True)

    tokenizer, model = _load_model(cfg, encoder_ckpt, num_labels=2, dev=dev)

    ds = load_dataset("super_glue", "wic")
    print(f"SuperGLUE WiC train: {len(ds['train'])}, val: {len(ds['validation'])}")
    print(f"  use_word_context={use_word_context}")

    train_tok = _tok_wic(ds["train"],      tokenizer, max_length, use_word_context)
    val_tok   = _tok_wic(ds["validation"], tokenizer, max_length, use_word_context)
    train_tok.set_format(type="torch")
    val_tok.set_format(type="torch")

    train_loader = DataLoader(train_tok, batch_size=batch_size, shuffle=True,  num_workers=num_workers)
    val_loader   = DataLoader(val_tok,   batch_size=batch_size, shuffle=False, num_workers=num_workers)

    opt, scheduler = _build_scheduler(model, cfg, epochs * len(train_loader))
    scaler    = GradScaler("cuda", enabled=mixed_precision)
    criterion = nn.CrossEntropyLoss()

    exp      = _exp_slug(cfg)
    ckpt_dir = cfg.get("ckpt_dir", "checkpoints/finetune")
    res_dir  = cfg.get("results_dir", "results")
    os.makedirs(ckpt_dir, exist_ok=True)

    best_val, best_state = -1.0, None
    epoch_log, _hist_val, _hist_loss, _hist_lr = [], [], [], []

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for batch in tqdm(train_loader, desc=f"WiC {epoch}/{epochs}", leave=False):
            input_ids      = batch["input_ids"].to(dev)
            attention_mask = batch["attention_mask"].to(dev)
            labels         = batch["label"].to(dev)
            opt.zero_grad(set_to_none=True)
            with amp.autocast("cuda", enabled=mixed_precision):
                loss = criterion(model(input_ids, attention_mask), labels)
            scaler.scale(loss).backward()
            max_norm = cfg.get("max_grad_norm", 1.0)
            if max_norm > 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            scaler.step(opt)
            scaler.update()
            if scheduler is not None:
                scheduler.step()
            total_loss += loss.item()

        epoch_loss = total_loss / len(train_loader)
        epoch_lr   = opt.param_groups[0]["lr"]
        val_acc    = _eval_nli_binary(model, val_loader, dev, mixed_precision)
        print(f"[WiC] epoch {epoch}: loss={epoch_loss:.4f} val_acc={val_acc:.4f} lr={epoch_lr:.2e}")

        if val_acc > best_val:
            best_val   = val_acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            ckpt_path  = os.path.join(ckpt_dir, f"{exp}_best.pt")
            save_ckpt({"epoch": epoch, "val_acc": val_acc, "model_state_dict": model.state_dict()}, ckpt_path)
            print(f"          ✓ Saved best model (val_acc={val_acc:.4f}) to {ckpt_path}")

        _hist_val.append(val_acc)
        _hist_loss.append(epoch_loss)
        _hist_lr.append(epoch_lr)
        epoch_log.append({
            "epoch": epoch,
            "train": {"loss": epoch_loss, "lr": epoch_lr},
            "validation": {"acc": val_acc},
        })
        save_finetune_results(res_dir, "wic", exp,
                              {"model": cfg.get("model_name", ""), "epochs": epoch_log})

    if best_state:
        model.load_state_dict(best_state)
    final_acc = _eval_nli_binary(model, val_loader, dev, mixed_precision)
    print(f"Final Val Acc: {final_acc:.4f}")
    save_finetune_results(res_dir, "wic", exp, {
        "model": cfg.get("model_name", ""),
        "epochs": epoch_log,
        "final": {"validation": {"acc": final_acc}},
    })
    if _hist_val:
        _save_training_plots(
            list(range(1, len(_hist_val) + 1)),
            {"val": _hist_val}, _hist_loss, _hist_lr,
            "SuperGLUE WiC accuracy per epoch",
            os.path.join(res_dir, f"{exp}_wic_training.png"),
        )
    return model


# ---------------------------------------------------------------------------
# SuperGLUE RTE  (binary NLI: entailment vs not_entailment)
# ---------------------------------------------------------------------------

def _tok_nli(split, tokenizer, max_length, key1, key2):
    keep = {"input_ids", "attention_mask", "label"}
    return split.map(
        lambda examples: tokenizer(
            examples[key1], examples[key2],
            truncation=True, padding="max_length", max_length=max_length,
        ),
        batched=True,
        remove_columns=[c for c in split.column_names if c not in keep],
        desc=f"Tokenizing {key1}/{key2}",
    )


def _eval_nli_binary(model, loader, dev, mixed_precision):
    model.eval()
    all_pred, all_true = [], []
    with torch.no_grad():
        for batch in loader:
            with amp.autocast("cuda", enabled=mixed_precision):
                logits = model(batch["input_ids"].to(dev), batch["attention_mask"].to(dev))
            all_pred.extend(logits.argmax(-1).cpu().tolist())
            all_true.extend(batch["label"].tolist())
    valid = np.array(all_true) != -1
    if not valid.any():
        return 0.0
    return accuracy_score(np.array(all_true)[valid], np.array(all_pred)[valid])


def finetune_superglue_rte(cfg, encoder_ckpt: str):
    """SuperGLUE RTE: binary NLI (entailment vs not_entailment) on super_glue/rte."""
    set_seed(cfg["seed"])
    dev = device(cfg.get("gpu_id", None))
    print(f"Using device: {dev}")

    max_length      = cfg.get("max_length", 256)
    mixed_precision = cfg.get("mixed_precision", True)
    epochs          = cfg["epochs"]
    batch_size      = cfg["batch_size"]
    num_workers     = cfg.get("num_workers", 4)

    tokenizer, model = _load_model(cfg, encoder_ckpt, num_labels=2, dev=dev)

    ds = load_dataset("super_glue", "rte")
    print(f"SuperGLUE RTE train: {len(ds['train'])}, val: {len(ds['validation'])}")

    train_tok = _tok_nli(ds["train"],      tokenizer, max_length, "premise", "hypothesis")
    val_tok   = _tok_nli(ds["validation"], tokenizer, max_length, "premise", "hypothesis")
    train_tok.set_format(type="torch")
    val_tok.set_format(type="torch")

    train_loader = DataLoader(train_tok, batch_size=batch_size, shuffle=True,  num_workers=num_workers)
    val_loader   = DataLoader(val_tok,   batch_size=batch_size, shuffle=False, num_workers=num_workers)

    opt, scheduler = _build_scheduler(model, cfg, epochs * len(train_loader))
    scaler    = GradScaler("cuda", enabled=mixed_precision)
    criterion = nn.CrossEntropyLoss()

    exp      = _exp_slug(cfg)
    ckpt_dir = cfg.get("ckpt_dir", "checkpoints/finetune")
    res_dir  = cfg.get("results_dir", "results")
    os.makedirs(ckpt_dir, exist_ok=True)

    best_val, best_state = -1.0, None
    epoch_log, _hist_val, _hist_loss, _hist_lr = [], [], [], []

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for batch in tqdm(train_loader, desc=f"RTE {epoch}/{epochs}", leave=False):
            input_ids      = batch["input_ids"].to(dev)
            attention_mask = batch["attention_mask"].to(dev)
            labels         = batch["label"].to(dev)
            opt.zero_grad(set_to_none=True)
            with amp.autocast("cuda", enabled=mixed_precision):
                loss = criterion(model(input_ids, attention_mask), labels)
            scaler.scale(loss).backward()
            max_norm = cfg.get("max_grad_norm", 1.0)
            if max_norm > 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            scaler.step(opt)
            scaler.update()
            if scheduler is not None:
                scheduler.step()
            total_loss += loss.item()

        epoch_loss = total_loss / len(train_loader)
        epoch_lr   = opt.param_groups[0]["lr"]
        val_acc    = _eval_nli_binary(model, val_loader, dev, mixed_precision)
        print(f"[RTE] epoch {epoch}: loss={epoch_loss:.4f} val_acc={val_acc:.4f} lr={epoch_lr:.2e}")

        if val_acc > best_val:
            best_val   = val_acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            ckpt_path  = os.path.join(ckpt_dir, f"{exp}_rte_best.pt")
            save_ckpt({"epoch": epoch, "val_acc": val_acc, "model_state_dict": model.state_dict()}, ckpt_path)
            print(f"          ✓ Saved best model (val_acc={val_acc:.4f}) to {ckpt_path}")

        _hist_val.append(val_acc)
        _hist_loss.append(epoch_loss)
        _hist_lr.append(epoch_lr)
        epoch_log.append({
            "epoch": epoch,
            "train": {"loss": epoch_loss, "lr": epoch_lr},
            "validation": {"acc": val_acc},
        })
        save_finetune_results(res_dir, "rte", exp,
                              {"model": cfg.get("model_name", ""), "epochs": epoch_log})

    if best_state:
        model.load_state_dict(best_state)
    final_acc = _eval_nli_binary(model, val_loader, dev, mixed_precision)
    print(f"Final Val Acc: {final_acc:.4f}")
    save_finetune_results(res_dir, "rte", exp, {
        "model": cfg.get("model_name", ""),
        "epochs": epoch_log,
        "final": {"validation": {"acc": final_acc}},
    })
    if _hist_val:
        _save_training_plots(
            list(range(1, len(_hist_val) + 1)),
            {"val": _hist_val}, _hist_loss, _hist_lr,
            "SuperGLUE RTE accuracy per epoch",
            os.path.join(res_dir, f"{exp}_rte_training.png"),
        )
    return model


# ---------------------------------------------------------------------------
# ReCoRD  (entity-selection reading comprehension)
# ---------------------------------------------------------------------------

def _f1_score_record(prediction: str, ground_truth: str) -> float:
    pred_tokens = prediction.lower().split()
    gold_tokens = ground_truth.lower().split()
    common = Counter(pred_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall    = num_same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def _expand_record(split):
    """One row per (passage, question, entity) triple — binary label."""
    rows = []
    for q_idx, ex in enumerate(split):
        answer_set = set(ex["answers"]) if ex["answers"] else set()
        for entity in ex["entities"]:
            rows.append({
                "passage":      ex["passage"],
                "query":        ex["query"].replace("@placeholder", entity),
                "question_idx": q_idx,
                "entity_str":   entity,
                "label":        1 if entity in answer_set else 0,
            })
    return Dataset.from_list(rows)


def _tok_record(split, tokenizer, max_length):
    keep = {"input_ids", "attention_mask", "label", "question_idx", "entity_str"}
    return split.map(
        lambda examples: tokenizer(
            examples["passage"], examples["query"],
            truncation=True, padding="max_length", max_length=max_length,
        ),
        batched=True,
        remove_columns=[c for c in split.column_names if c not in keep],
        desc="Tokenizing ReCoRD",
    )


def _eval_record(model, val_tok, answers_map, dev, mixed_precision, batch_size):
    """
    Runs the model over flat val_tok, groups by question_idx, picks argmax entity.
    answers_map: sequential_idx -> set of gold answer strings.
    """
    model.eval()
    val_tok.set_format(type="torch", columns=["input_ids", "attention_mask"], output_all_columns=True)
    best_by_q = {}  # question_idx -> (best_score, entity_str)
    with torch.no_grad():
        for i in range(0, len(val_tok), batch_size):
            batch = val_tok[i : i + batch_size]
            with amp.autocast("cuda", enabled=mixed_precision):
                logits = model(batch["input_ids"].to(dev), batch["attention_mask"].to(dev))
            scores = logits[:, 1].cpu().tolist()
            for q_idx, entity, score in zip(
                batch["question_idx"], batch["entity_str"], scores
            ):
                if q_idx not in best_by_q or score > best_by_q[q_idx][0]:
                    best_by_q[q_idx] = (score, entity)
    total_em, total_f1, n = 0, 0.0, 0
    for q_idx, (_, pred) in best_by_q.items():
        ans = answers_map.get(q_idx)
        if not ans:
            continue
        total_em += int(pred in ans)
        total_f1 += max(_f1_score_record(pred, a) for a in ans)
        n += 1
    return (total_em / n, total_f1 / n) if n > 0 else (0.0, 0.0)


def finetune_record(cfg, encoder_ckpt: str):
    set_seed(cfg["seed"])
    dev = device(cfg.get("gpu_id", None))
    print(f"Using device: {dev}")

    max_length      = cfg.get("max_length", 512)
    mixed_precision = cfg.get("mixed_precision", True)
    epochs          = cfg["epochs"]
    batch_size      = cfg["batch_size"]
    num_workers     = cfg.get("num_workers", 2)

    tokenizer, model = _load_model(cfg, encoder_ckpt, num_labels=2, dev=dev)

    ds = load_dataset("super_glue", "record")
    print(f"ReCoRD train: {len(ds['train'])}, val: {len(ds['validation'])}")

    answers_map = {i: set(ex["answers"]) for i, ex in enumerate(ds["validation"])}

    print("Expanding ReCoRD train (one row per entity)...")
    train_expanded = _expand_record(ds["train"])
    print(f"  train rows: {len(train_expanded)}")
    print("Expanding ReCoRD val...")
    val_expanded = _expand_record(ds["validation"])
    print(f"  val rows:   {len(val_expanded)}")

    print("Tokenizing ReCoRD train...")
    train_tok = _tok_record(train_expanded, tokenizer, max_length)
    print("Tokenizing ReCoRD val...")
    val_tok = _tok_record(val_expanded, tokenizer, max_length)

    train_only = train_tok.remove_columns(["entity_str", "question_idx"])
    train_only.set_format(type="torch")
    train_loader = DataLoader(
        train_only, batch_size=batch_size, shuffle=True, num_workers=num_workers,
    )

    opt, scheduler = _build_scheduler(model, cfg, epochs * len(train_loader))
    scaler    = GradScaler("cuda", enabled=mixed_precision)
    criterion = nn.CrossEntropyLoss()

    exp      = _exp_slug(cfg)
    ckpt_dir = cfg.get("ckpt_dir", "checkpoints/finetune")
    res_dir  = cfg.get("results_dir", "results")
    os.makedirs(ckpt_dir, exist_ok=True)

    best_val, best_state = -1.0, None
    epoch_log, _hist_val, _hist_loss, _hist_lr = [], [], [], []

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for batch in tqdm(train_loader, desc=f"ReCoRD {epoch}/{epochs}", leave=False):
            input_ids      = batch["input_ids"].to(dev)
            attention_mask = batch["attention_mask"].to(dev)
            labels         = batch["label"].to(dev)
            opt.zero_grad(set_to_none=True)
            with amp.autocast("cuda", enabled=mixed_precision):
                loss = criterion(model(input_ids, attention_mask), labels)
            scaler.scale(loss).backward()
            max_norm = cfg.get("max_grad_norm", 1.0)
            if max_norm > 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
            scaler.step(opt)
            scaler.update()
            if scheduler is not None:
                scheduler.step()
            total_loss += loss.item()

        epoch_loss = total_loss / len(train_loader)
        epoch_lr   = opt.param_groups[0]["lr"]
        val_em, val_f1 = _eval_record(
            model, val_tok, answers_map, dev, mixed_precision, batch_size * 4,
        )
        print(f"[ReCoRD] epoch {epoch}: loss={epoch_loss:.4f} val_em={val_em:.4f} val_f1={val_f1:.4f} lr={epoch_lr:.2e}")

        if val_f1 > best_val:
            best_val   = val_f1
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            ckpt_path  = os.path.join(ckpt_dir, f"{exp}_record_best.pt")
            save_ckpt({"epoch": epoch, "val_em": val_em, "val_f1": val_f1,
                       "model_state_dict": model.state_dict()}, ckpt_path)
            print(f"          ✓ Saved best model (val_f1={val_f1:.4f}) to {ckpt_path}")

        _hist_val.append(val_f1)
        _hist_loss.append(epoch_loss)
        _hist_lr.append(epoch_lr)
        epoch_log.append({
            "epoch": epoch,
            "train": {"loss": epoch_loss, "lr": epoch_lr},
            "validation": {"em": val_em, "f1": val_f1},
        })
        save_finetune_results(res_dir, "record", exp,
                              {"model": cfg.get("model_name", ""), "epochs": epoch_log})

    if best_state:
        model.load_state_dict(best_state)
    final_em, final_f1 = _eval_record(
        model, val_tok, answers_map, dev, mixed_precision, batch_size * 4,
    )
    print(f"Final Val EM: {final_em:.4f}  F1: {final_f1:.4f}")
    save_finetune_results(res_dir, "record", exp, {
        "model": cfg.get("model_name", ""),
        "epochs": epoch_log,
        "final": {"validation": {"em": final_em, "f1": final_f1}},
    })
    if _hist_val:
        _save_training_plots(
            list(range(1, len(_hist_val) + 1)),
            {"val_f1": _hist_val}, _hist_loss, _hist_lr,
            "ReCoRD F1 per epoch",
            os.path.join(res_dir, f"{exp}_record_training.png"),
        )
    return model

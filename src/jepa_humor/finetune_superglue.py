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


def _build_scheduler(model, cfg, total_num_update):
    """AdamW + polynomial(linear) decay over `total_num_update` optimizer updates.

    fairseq drives every SuperGLUE schedule by updates: warmup over `warmup_updates`
    (or `lr_warmup_ratio` of the budget as a fallback), then linear decay to ~0 at
    `total_num_update`. `_build_finetune_lr_scheduler` (finetune.py) takes a ratio, so
    we convert warmup_updates → ratio here.
    """
    _betas = tuple(cfg["adam_betas"]) if "adam_betas" in cfg else (0.9, 0.999)
    _eps   = cfg.get("adam_eps", 1e-8)
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["lr"],
        weight_decay=cfg.get("weight_decay", 0.01),
        betas=_betas,
        eps=_eps,
    )
    sched_type = cfg.get("lr_scheduler", "linear")
    warmup_updates = cfg.get(
        "warmup_updates", int(total_num_update * cfg.get("lr_warmup_ratio", 0.06))
    )
    scheduler = None
    if sched_type and str(sched_type).lower() not in ("none", "null", ""):
        warmup_ratio = warmup_updates / max(1, total_num_update)
        scheduler = _build_finetune_lr_scheduler(opt, total_num_update, warmup_ratio, sched_type)
    return opt, scheduler


def _total_updates(cfg, train_loader):
    """Resolve the update budget: explicit `total_num_update`, else `epochs`*steps."""
    if cfg.get("total_num_update"):
        return int(cfg["total_num_update"])
    return int(cfg.get("epochs", 10)) * len(train_loader)


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


def _tok_copa_ranking(split, tokenizer, max_length):
    """One row per example, each choice tokenized as a separate sequence.

    Mirrors fairseq `sentence_ranking` (commonsense_qa): each alternative is encoded
    independently; the model produces one score per choice and the loss is a single
    cross-entropy over the choices.
    """
    keep = {"input_ids_0", "attention_mask_0", "input_ids_1", "attention_mask_1", "label"}

    def fn(ex):
        enc0 = tokenizer(_copa_text(ex["premise"], ex["question"], ex["choice1"]),
                         truncation=True, padding="max_length", max_length=max_length)
        enc1 = tokenizer(_copa_text(ex["premise"], ex["question"], ex["choice2"]),
                         truncation=True, padding="max_length", max_length=max_length)
        return {
            "input_ids_0": enc0["input_ids"], "attention_mask_0": enc0["attention_mask"],
            "input_ids_1": enc1["input_ids"], "attention_mask_1": enc1["attention_mask"],
            "label": ex["label"],
        }

    return split.map(
        fn,
        remove_columns=[c for c in split.column_names if c not in keep],
        desc="Tokenizing COPA",
    )


def _copa_logits(model, batch, dev):
    """Forward both choices → [B, 2] ranking logits (one scalar score per choice)."""
    s0 = model(batch["input_ids_0"].to(dev), batch["attention_mask_0"].to(dev))  # [B, 1]
    s1 = model(batch["input_ids_1"].to(dev), batch["attention_mask_1"].to(dev))  # [B, 1]
    return torch.cat([s0, s1], dim=1)  # [B, 2]


def _eval_copa(model, val_loader, dev, mixed_precision):
    model.eval()
    ncorrect, n = 0, 0
    with torch.no_grad():
        for batch in val_loader:
            with amp.autocast("cuda", enabled=mixed_precision):
                logits = _copa_logits(model, batch, dev)
            preds  = logits.argmax(-1).cpu()
            labels = batch["label"]
            ncorrect += (preds == labels).sum().item()
            n += len(labels)
    return ncorrect / max(1, n)


def finetune_copa(cfg, encoder_ckpt: str):
    set_seed(cfg["seed"])
    dev = device(cfg.get("gpu_id", None))
    print(f"Using device: {dev}")

    max_length      = cfg.get("max_length", 128)
    mixed_precision = cfg.get("mixed_precision", True)
    batch_size      = cfg["batch_size"]
    num_workers     = cfg.get("num_workers", 2)
    max_grad_norm   = cfg.get("max_grad_norm", 1.0)

    # fairseq commonsense_qa recipe: sentence_ranking head → one scalar score per choice.
    tokenizer, model = _load_model(cfg, encoder_ckpt, num_labels=1, dev=dev)

    ds = load_dataset("super_glue", "copa")
    print(f"COPA train: {len(ds['train'])}, val: {len(ds['validation'])}")

    train_tok = _tok_copa_ranking(ds["train"],      tokenizer, max_length)
    val_tok   = _tok_copa_ranking(ds["validation"], tokenizer, max_length)
    train_tok.set_format(type="torch")
    val_tok.set_format(type="torch")

    train_loader = DataLoader(train_tok, batch_size=batch_size, shuffle=True,  num_workers=num_workers)
    val_loader   = DataLoader(val_tok,   batch_size=batch_size, shuffle=False, num_workers=num_workers)

    total_num_update = _total_updates(cfg, train_loader)
    opt, scheduler = _build_scheduler(model, cfg, total_num_update)
    scaler    = GradScaler("cuda", enabled=mixed_precision)
    criterion = nn.CrossEntropyLoss()

    exp      = _exp_slug(cfg)
    ckpt_dir = cfg.get("ckpt_dir", "checkpoints/finetune")
    res_dir  = cfg.get("results_dir", "results")
    os.makedirs(ckpt_dir, exist_ok=True)

    best_val, best_state = -1.0, None
    epoch_log, _hist_val, _hist_loss, _hist_lr = [], [], [], []

    update, epoch = 0, 0
    print(f"Training COPA for {total_num_update} updates (batch_size={batch_size})")
    while update < total_num_update:
        epoch += 1
        model.train()
        total_loss, n_batches = 0.0, 0
        pbar = tqdm(train_loader, desc=f"COPA ep{epoch} (upd {update}/{total_num_update})", leave=False)
        for batch in pbar:
            if update >= total_num_update:
                break
            labels = batch["label"].to(dev)
            opt.zero_grad(set_to_none=True)
            with amp.autocast("cuda", enabled=mixed_precision):
                loss = criterion(_copa_logits(model, batch, dev), labels)
            scaler.scale(loss).backward()
            if max_grad_norm > 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            scaler.step(opt)
            scaler.update()
            if scheduler is not None:
                scheduler.step()
            update += 1
            total_loss += loss.item()
            n_batches += 1

        epoch_loss = total_loss / max(1, n_batches)
        epoch_lr   = opt.param_groups[0]["lr"]
        val_acc    = _eval_copa(model, val_loader, dev, mixed_precision)
        print(f"[COPA] epoch {epoch} (update {update}/{total_num_update}): "
              f"loss={epoch_loss:.4f} val_acc={val_acc:.4f} lr={epoch_lr:.2e}")

        if val_acc > best_val:
            best_val   = val_acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            ckpt_path  = os.path.join(ckpt_dir, f"{exp}_copa_best.pt")
            save_ckpt({"epoch": epoch, "update": update, "val_acc": val_acc,
                       "model_state_dict": model.state_dict()}, ckpt_path)
            print(f"          ✓ Saved best model (val_acc={val_acc:.4f}) to {ckpt_path}")

        _hist_val.append(val_acc)
        _hist_loss.append(epoch_loss)
        _hist_lr.append(epoch_lr)
        epoch_log.append({
            "epoch": epoch, "update": update,
            "train": {"loss": epoch_loss, "lr": epoch_lr},
            "validation": {"acc": val_acc},
        })
        save_finetune_results(res_dir, "copa", exp,
                              {"model": cfg.get("model_name", ""), "epochs": epoch_log})

    if best_state:
        model.load_state_dict(best_state)
    final_acc = _eval_copa(model, val_loader, dev, mixed_precision)
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
    num_workers     = cfg.get("num_workers", 2)
    max_grad_norm   = cfg.get("max_grad_norm", 1.0)

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

    total_num_update = _total_updates(cfg, train_loader)
    opt, scheduler = _build_scheduler(model, cfg, total_num_update)
    scaler    = GradScaler("cuda", enabled=mixed_precision)
    criterion = nn.CrossEntropyLoss()

    exp      = _exp_slug(cfg)
    ckpt_dir = cfg.get("ckpt_dir", "checkpoints/finetune")
    res_dir  = cfg.get("results_dir", "results")
    os.makedirs(ckpt_dir, exist_ok=True)

    best_val, best_state = -1.0, None
    epoch_log, _hist_val, _hist_loss, _hist_lr = [], [], [], []

    update, epoch = 0, 0
    print(f"Training MultiRC for {total_num_update} updates (batch_size={batch_size})")
    while update < total_num_update:
        epoch += 1
        model.train()
        total_loss, n_batches = 0.0, 0
        pbar = tqdm(train_loader, desc=f"MultiRC ep{epoch} (upd {update}/{total_num_update})", leave=False)
        for batch in pbar:
            if update >= total_num_update:
                break
            input_ids      = batch["input_ids"].to(dev)
            attention_mask = batch["attention_mask"].to(dev)
            labels         = batch["label"].to(dev)
            opt.zero_grad(set_to_none=True)
            with amp.autocast("cuda", enabled=mixed_precision):
                loss = criterion(model(input_ids, attention_mask), labels)
            scaler.scale(loss).backward()
            if max_grad_norm > 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            scaler.step(opt)
            scaler.update()
            if scheduler is not None:
                scheduler.step()
            update += 1
            total_loss += loss.item()
            n_batches += 1

        epoch_loss = total_loss / max(1, n_batches)
        epoch_lr   = opt.param_groups[0]["lr"]
        val_f1     = _eval_multirc(model, val_loader, dev, mixed_precision)
        print(f"[MultiRC] epoch {epoch} (update {update}/{total_num_update}): "
              f"loss={epoch_loss:.4f} val_f1={val_f1:.4f} lr={epoch_lr:.2e}")

        if val_f1 > best_val:
            best_val   = val_f1
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            ckpt_path  = os.path.join(ckpt_dir, f"{exp}_multirc_best.pt")
            save_ckpt({"epoch": epoch, "update": update, "val_f1": val_f1,
                       "model_state_dict": model.state_dict()}, ckpt_path)
            print(f"          ✓ Saved best model (val_f1={val_f1:.4f}) to {ckpt_path}")

        _hist_val.append(val_f1)
        _hist_loss.append(epoch_loss)
        _hist_lr.append(epoch_lr)
        epoch_log.append({
            "epoch": epoch, "update": update,
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
    batch_size       = cfg["batch_size"]
    num_workers      = cfg.get("num_workers", 4)
    max_grad_norm    = cfg.get("max_grad_norm", 1.0)
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

    total_num_update = _total_updates(cfg, train_loader)
    opt, scheduler = _build_scheduler(model, cfg, total_num_update)
    scaler    = GradScaler("cuda", enabled=mixed_precision)
    criterion = nn.CrossEntropyLoss()

    exp      = _exp_slug(cfg)
    ckpt_dir = cfg.get("ckpt_dir", "checkpoints/finetune")
    res_dir  = cfg.get("results_dir", "results")
    os.makedirs(ckpt_dir, exist_ok=True)

    best_val, best_state = -1.0, None
    epoch_log, _hist_val, _hist_loss, _hist_lr = [], [], [], []

    update, epoch = 0, 0
    print(f"Training WiC for {total_num_update} updates (batch_size={batch_size})")
    while update < total_num_update:
        epoch += 1
        model.train()
        total_loss, n_batches = 0.0, 0
        pbar = tqdm(train_loader, desc=f"WiC ep{epoch} (upd {update}/{total_num_update})", leave=False)
        for batch in pbar:
            if update >= total_num_update:
                break
            input_ids      = batch["input_ids"].to(dev)
            attention_mask = batch["attention_mask"].to(dev)
            labels         = batch["label"].to(dev)
            opt.zero_grad(set_to_none=True)
            with amp.autocast("cuda", enabled=mixed_precision):
                loss = criterion(model(input_ids, attention_mask), labels)
            scaler.scale(loss).backward()
            if max_grad_norm > 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            scaler.step(opt)
            scaler.update()
            if scheduler is not None:
                scheduler.step()
            update += 1
            total_loss += loss.item()
            n_batches += 1

        epoch_loss = total_loss / max(1, n_batches)
        epoch_lr   = opt.param_groups[0]["lr"]
        val_acc    = _eval_nli_binary(model, val_loader, dev, mixed_precision)
        print(f"[WiC] epoch {epoch} (update {update}/{total_num_update}): "
              f"loss={epoch_loss:.4f} val_acc={val_acc:.4f} lr={epoch_lr:.2e}")

        if val_acc > best_val:
            best_val   = val_acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            ckpt_path  = os.path.join(ckpt_dir, f"{exp}_best.pt")
            save_ckpt({"epoch": epoch, "update": update, "val_acc": val_acc,
                       "model_state_dict": model.state_dict()}, ckpt_path)
            print(f"          ✓ Saved best model (val_acc={val_acc:.4f}) to {ckpt_path}")

        _hist_val.append(val_acc)
        _hist_loss.append(epoch_loss)
        _hist_lr.append(epoch_lr)
        epoch_log.append({
            "epoch": epoch, "update": update,
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
    batch_size      = cfg["batch_size"]
    num_workers     = cfg.get("num_workers", 4)
    max_grad_norm   = cfg.get("max_grad_norm", 1.0)

    tokenizer, model = _load_model(cfg, encoder_ckpt, num_labels=2, dev=dev)

    ds = load_dataset("super_glue", "rte")
    print(f"SuperGLUE RTE train: {len(ds['train'])}, val: {len(ds['validation'])}")

    train_tok = _tok_nli(ds["train"],      tokenizer, max_length, "premise", "hypothesis")
    val_tok   = _tok_nli(ds["validation"], tokenizer, max_length, "premise", "hypothesis")
    train_tok.set_format(type="torch")
    val_tok.set_format(type="torch")

    train_loader = DataLoader(train_tok, batch_size=batch_size, shuffle=True,  num_workers=num_workers)
    val_loader   = DataLoader(val_tok,   batch_size=batch_size, shuffle=False, num_workers=num_workers)

    total_num_update = _total_updates(cfg, train_loader)
    opt, scheduler = _build_scheduler(model, cfg, total_num_update)
    scaler    = GradScaler("cuda", enabled=mixed_precision)
    criterion = nn.CrossEntropyLoss()

    exp      = _exp_slug(cfg)
    ckpt_dir = cfg.get("ckpt_dir", "checkpoints/finetune")
    res_dir  = cfg.get("results_dir", "results")
    os.makedirs(ckpt_dir, exist_ok=True)

    best_val, best_state = -1.0, None
    epoch_log, _hist_val, _hist_loss, _hist_lr = [], [], [], []

    update, epoch = 0, 0
    print(f"Training RTE for {total_num_update} updates (batch_size={batch_size})")
    while update < total_num_update:
        epoch += 1
        model.train()
        total_loss, n_batches = 0.0, 0
        pbar = tqdm(train_loader, desc=f"RTE ep{epoch} (upd {update}/{total_num_update})", leave=False)
        for batch in pbar:
            if update >= total_num_update:
                break
            input_ids      = batch["input_ids"].to(dev)
            attention_mask = batch["attention_mask"].to(dev)
            labels         = batch["label"].to(dev)
            opt.zero_grad(set_to_none=True)
            with amp.autocast("cuda", enabled=mixed_precision):
                loss = criterion(model(input_ids, attention_mask), labels)
            scaler.scale(loss).backward()
            if max_grad_norm > 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            scaler.step(opt)
            scaler.update()
            if scheduler is not None:
                scheduler.step()
            update += 1
            total_loss += loss.item()
            n_batches += 1

        epoch_loss = total_loss / max(1, n_batches)
        epoch_lr   = opt.param_groups[0]["lr"]
        val_acc    = _eval_nli_binary(model, val_loader, dev, mixed_precision)
        print(f"[RTE] epoch {epoch} (update {update}/{total_num_update}): "
              f"loss={epoch_loss:.4f} val_acc={val_acc:.4f} lr={epoch_lr:.2e}")

        if val_acc > best_val:
            best_val   = val_acc
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            ckpt_path  = os.path.join(ckpt_dir, f"{exp}_rte_best.pt")
            save_ckpt({"epoch": epoch, "update": update, "val_acc": val_acc,
                       "model_state_dict": model.state_dict()}, ckpt_path)
            print(f"          ✓ Saved best model (val_acc={val_acc:.4f}) to {ckpt_path}")

        _hist_val.append(val_acc)
        _hist_loss.append(epoch_loss)
        _hist_lr.append(epoch_lr)
        epoch_log.append({
            "epoch": epoch, "update": update,
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
    batch_size      = cfg["batch_size"]
    num_workers     = cfg.get("num_workers", 2)
    max_grad_norm   = cfg.get("max_grad_norm", 1.0)

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

    # ── Per-question ranking objective (mirrors finetune_copa's sentence-ranking) ──
    # Independent binary CE collapses here: candidate sets are ~88% negative, so the
    # model just predicts "negative" everywhere (loss stuck at the class prior, dev
    # EM/F1 ~ random). Instead, score every candidate of a question, softmax OVER the
    # candidates, and apply (marginal) cross-entropy toward the gold entity — balanced
    # by construction. The model stays num_labels=2 and we rank on logits[:,1], so the
    # existing argmax eval (_eval_record / eval_superglue_tasks) is unchanged.
    q_of_row     = train_tok["question_idx"]
    label_of_row = train_tok["label"]
    groups = {}
    for i, qi in enumerate(q_of_row):
        groups.setdefault(qi, []).append(i)
    # Train only on questions that have at least one gold candidate among their entities.
    train_qs = [rows for rows in groups.values()
                if any(label_of_row[i] == 1 for i in rows)]
    print(f"  ReCoRD train questions (with gold): {len(train_qs)} (from {len(groups)} total)")

    train_tok.set_format(type="torch", columns=["input_ids", "attention_mask", "label"])

    accum     = batch_size                          # questions per optimizer update
    max_cands = cfg.get("max_candidates", 10)       # cap candidates/question (golds always kept)
    n_upd_epoch = (len(train_qs) + accum - 1) // accum
    epochs_cfg  = int(cfg.get("epochs", 10))
    total_num_update = (int(cfg["total_num_update"]) if cfg.get("total_num_update")
                        else epochs_cfg * n_upd_epoch)
    opt, scheduler = _build_scheduler(model, cfg, total_num_update)
    scaler = GradScaler("cuda", enabled=mixed_precision)

    exp      = _exp_slug(cfg)
    ckpt_dir = cfg.get("ckpt_dir", "checkpoints/finetune")
    res_dir  = cfg.get("results_dir", "results")
    os.makedirs(ckpt_dir, exist_ok=True)

    best_val, best_state = -1.0, None
    epoch_log, _hist_val, _hist_loss, _hist_lr = [], [], [], []

    rng = random.Random(cfg["seed"])
    update, epoch = 0, 0
    print(f"Training ReCoRD (per-question ranking) for {total_num_update} updates "
          f"({accum} questions/update, <={max_cands} cands/question)")
    while update < total_num_update:
        epoch += 1
        model.train()
        order = list(range(len(train_qs)))
        rng.shuffle(order)
        total_loss, n_batches = 0.0, 0
        pbar = tqdm(range(0, len(order), accum),
                    desc=f"ReCoRD ep{epoch} (upd {update}/{total_num_update})", leave=False)
        for start in pbar:
            if update >= total_num_update:
                break
            chunk = order[start:start + accum]
            opt.zero_grad(set_to_none=True)
            # Accumulate over `accum` questions; forward one question at a time so peak
            # memory stays at a single candidate set (RoBERTa-large @512 is O(seqs*L^2)).
            step_loss = 0.0
            for qpos in chunk:
                rows  = train_qs[qpos]
                golds = [i for i in rows if label_of_row[i] == 1]
                negs  = [i for i in rows if label_of_row[i] == 0]
                keep_neg = max(0, max_cands - len(golds))
                if len(negs) > keep_neg:
                    negs = rng.sample(negs, keep_neg)
                sub  = train_tok[golds + negs]
                ii   = sub["input_ids"].to(dev)
                am   = sub["attention_mask"].to(dev)
                gold = (sub["label"] == 1).to(dev)
                with amp.autocast("cuda", enabled=mixed_precision):
                    s = model(ii, am)[:, 1].float()       # positive-class score per candidate
                    loss_q = (torch.logsumexp(s, 0) - torch.logsumexp(s[gold], 0)) / len(chunk)
                scaler.scale(loss_q).backward()
                step_loss += loss_q.item()
            if max_grad_norm > 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            scaler.step(opt)
            scaler.update()
            if scheduler is not None:
                scheduler.step()
            update += 1
            total_loss += step_loss
            n_batches += 1

        epoch_loss = total_loss / max(1, n_batches)
        epoch_lr   = opt.param_groups[0]["lr"]
        val_em, val_f1 = _eval_record(
            model, val_tok, answers_map, dev, mixed_precision, batch_size * 4,
        )
        print(f"[ReCoRD] epoch {epoch} (update {update}/{total_num_update}): "
              f"loss={epoch_loss:.4f} val_em={val_em:.4f} val_f1={val_f1:.4f} lr={epoch_lr:.2e}")

        if val_f1 > best_val:
            best_val   = val_f1
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            ckpt_path  = os.path.join(ckpt_dir, f"{exp}_record_best.pt")
            save_ckpt({"epoch": epoch, "update": update, "val_em": val_em, "val_f1": val_f1,
                       "model_state_dict": model.state_dict()}, ckpt_path)
            print(f"          ✓ Saved best model (val_f1={val_f1:.4f}) to {ckpt_path}")

        _hist_val.append(val_f1)
        _hist_loss.append(epoch_loss)
        _hist_lr.append(epoch_lr)
        epoch_log.append({
            "epoch": epoch, "update": update,
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


# ---------------------------------------------------------------------------
# BoolQ & CB  (sentence-pair classification — RoBERTa sentence_prediction)
# No official fairseq config exists; uses RoBERTa's general SuperGLUE recipe.
# BoolQ selection metric: accuracy. CB (3-class) selection metric: the
# SuperGLUE CB score = mean(macro-F1, accuracy).
# ---------------------------------------------------------------------------

def _eval_clf_metrics(model, loader, dev, mixed_precision):
    """Accuracy + macro-F1 over a classification loader (ignores label == -1)."""
    model.eval()
    all_pred, all_true = [], []
    with torch.no_grad():
        for batch in loader:
            with amp.autocast("cuda", enabled=mixed_precision):
                logits = model(batch["input_ids"].to(dev), batch["attention_mask"].to(dev))
            all_pred.extend(logits.argmax(-1).cpu().tolist())
            all_true.extend(batch["label"].tolist())
    true, pred = np.array(all_true), np.array(all_pred)
    valid = true != -1
    if not valid.any():
        return {"acc": 0.0, "f1_macro": 0.0}
    return {
        "acc":      float(accuracy_score(true[valid], pred[valid])),
        "f1_macro": float(f1_score(true[valid], pred[valid], average="macro")),
    }


def finetune_superglue_clf(cfg, encoder_ckpt: str):
    """Generic update-based sentence-pair classifier for BoolQ and CB.

    Mirrors RoBERTa's `sentence_prediction` head (the shared CorrectEnglishSentence-
    Classifier). Tokenization uses cfg['text_keys']; the selection metric is accuracy
    (BoolQ) or mean(macro-F1, accuracy) for 3-class CB (the SuperGLUE CB metric).
    """
    set_seed(cfg["seed"])
    dev = device(cfg.get("gpu_id", None))
    print(f"Using device: {dev}")

    task            = cfg.get("dataset_config", "")
    num_labels      = cfg.get("num_labels", 2)
    text_keys       = cfg.get("text_keys", ["premise", "hypothesis"])
    key1, key2      = text_keys[0], text_keys[1]
    max_length      = cfg.get("max_length", 256)
    mixed_precision = cfg.get("mixed_precision", True)
    batch_size      = cfg["batch_size"]
    num_workers     = cfg.get("num_workers", 4)
    max_grad_norm   = cfg.get("max_grad_norm", 1.0)
    use_cb_metric   = num_labels == 3   # CB: average macro-F1 and accuracy

    tokenizer, model = _load_model(cfg, encoder_ckpt, num_labels=num_labels, dev=dev)

    ds = load_dataset("super_glue", task)
    print(f"SuperGLUE {task} train: {len(ds['train'])}, val: {len(ds['validation'])}")

    train_tok = _tok_nli(ds["train"],      tokenizer, max_length, key1, key2)
    val_tok   = _tok_nli(ds["validation"], tokenizer, max_length, key1, key2)
    train_tok.set_format(type="torch")
    val_tok.set_format(type="torch")

    train_loader = DataLoader(train_tok, batch_size=batch_size, shuffle=True,  num_workers=num_workers)
    val_loader   = DataLoader(val_tok,   batch_size=batch_size, shuffle=False, num_workers=num_workers)

    total_num_update = _total_updates(cfg, train_loader)
    opt, scheduler = _build_scheduler(model, cfg, total_num_update)
    scaler    = GradScaler("cuda", enabled=mixed_precision)
    criterion = nn.CrossEntropyLoss()

    exp      = _exp_slug(cfg)
    ckpt_dir = cfg.get("ckpt_dir", "checkpoints/finetune")
    res_dir  = cfg.get("results_dir", "results")
    os.makedirs(ckpt_dir, exist_ok=True)

    def _score(m):
        return (m["acc"] + m["f1_macro"]) / 2 if use_cb_metric else m["acc"]

    best_val, best_state = -1.0, None
    epoch_log, _hist_val, _hist_loss, _hist_lr = [], [], [], []

    update, epoch = 0, 0
    print(f"Training {task} for {total_num_update} updates (batch_size={batch_size})")
    while update < total_num_update:
        epoch += 1
        model.train()
        total_loss, n_batches = 0.0, 0
        pbar = tqdm(train_loader, desc=f"{task} ep{epoch} (upd {update}/{total_num_update})", leave=False)
        for batch in pbar:
            if update >= total_num_update:
                break
            input_ids      = batch["input_ids"].to(dev)
            attention_mask = batch["attention_mask"].to(dev)
            labels         = batch["label"].to(dev)
            opt.zero_grad(set_to_none=True)
            with amp.autocast("cuda", enabled=mixed_precision):
                loss = criterion(model(input_ids, attention_mask), labels)
            scaler.scale(loss).backward()
            if max_grad_norm > 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            scaler.step(opt)
            scaler.update()
            if scheduler is not None:
                scheduler.step()
            update += 1
            total_loss += loss.item()
            n_batches += 1

        epoch_loss = total_loss / max(1, n_batches)
        epoch_lr   = opt.param_groups[0]["lr"]
        metrics    = _eval_clf_metrics(model, val_loader, dev, mixed_precision)
        score      = _score(metrics)
        print(f"[{task}] epoch {epoch} (update {update}/{total_num_update}): "
              f"loss={epoch_loss:.4f} acc={metrics['acc']:.4f} f1m={metrics['f1_macro']:.4f} lr={epoch_lr:.2e}")

        if score > best_val:
            best_val   = score
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
            ckpt_path  = os.path.join(ckpt_dir, f"{exp}_best.pt")
            save_ckpt({"epoch": epoch, "update": update, **metrics,
                       "model_state_dict": model.state_dict()}, ckpt_path)
            print(f"          ✓ Saved best model (score={score:.4f}) to {ckpt_path}")

        _hist_val.append(score)
        _hist_loss.append(epoch_loss)
        _hist_lr.append(epoch_lr)
        epoch_log.append({
            "epoch": epoch, "update": update,
            "train": {"loss": epoch_loss, "lr": epoch_lr},
            "validation": metrics,
        })
        save_finetune_results(res_dir, task, exp,
                              {"model": cfg.get("model_name", ""), "epochs": epoch_log})

    if best_state:
        model.load_state_dict(best_state)
    final = _eval_clf_metrics(model, val_loader, dev, mixed_precision)
    print(f"Final Val: acc={final['acc']:.4f} f1m={final['f1_macro']:.4f}")
    save_finetune_results(res_dir, task, exp, {
        "model": cfg.get("model_name", ""),
        "epochs": epoch_log,
        "final": {"validation": final},
    })
    if _hist_val:
        _save_training_plots(
            list(range(1, len(_hist_val) + 1)),
            {"val": _hist_val}, _hist_loss, _hist_lr,
            f"SuperGLUE {task} score per epoch",
            os.path.join(res_dir, f"{exp}_{task}_training.png"),
        )
    return model

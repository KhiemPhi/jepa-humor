"""
WNLI fine-tuning via WSC (SuperGLUE) using RoBERTa MLM scoring.

Follows the fairseq RoBERTa implementation:
  examples/roberta/wsc/wsc_criterion.py + wsc_utils.py

Scoring: given a sentence with the pronoun replaced by a candidate,
mask the candidate tokens and compute avg log P(candidate | context).
Training loss (cross-entropy): query score must beat all negative scores.
Evaluation: query wins if its score >= every noun-chunk candidate's score.
"""

import os
import random
import json
import datetime

import numpy as np
import spacy
import torch
from tqdm import tqdm
import torch.nn.functional as F
from torch.amp import GradScaler
from torch import amp
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForMaskedLM

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .utils import set_seed, device, load_encoder_state, save_ckpt
from .finetune import _build_finetune_lr_scheduler, _save_training_plots, save_finetune_results


# ---------------------------------------------------------------------------
# NP candidate extraction (mirrors fairseq extended_noun_chunks)
# ---------------------------------------------------------------------------

def _extended_noun_chunks(doc):
    """Standard noun chunks + runs of adjacent NOUN/PROPN tokens."""
    spans = {(np.start, np.end) for np in doc.noun_chunks}
    np_start, cur_type = 0, "NONE"
    for i, tok in enumerate(doc):
        t = tok.pos_ if tok.pos_ in {"NOUN", "PROPN"} else "NONE"
        if t != cur_type:
            if cur_type != "NONE":
                spans.add((np_start, i))
            if t != "NONE":
                np_start = i
            cur_type = t
    if cur_type != "NONE":
        spans.add((np_start, len(doc)))
    return [doc[s:e] for s, e in sorted(spans)]


def _extract_negatives(text: str, span1_text: str, nlp) -> list[str]:
    """Return candidate NPs from text that don't overlap with span1_text."""
    doc = nlp(text)
    pos_lower = span1_text.lower().strip()
    seen, negatives = set(), []
    for chunk in _extended_noun_chunks(doc):
        cand = chunk.text.strip()
        key = cand.lower()
        # exclude pronouns — mirrors fairseq filter_noun_chunks(exclude_pronouns=True)
        is_pron = (
            all(t.pos_ == "PRON" for t in chunk)
            or chunk.lemma_ == "-PRON-"  # legacy spaCy < 3 compat
        )
        if is_pron:
            continue
        if key in pos_lower or pos_lower in key:
            continue
        if key not in seen:
            seen.add(key)
            negatives.append(cand)
    return negatives


# ---------------------------------------------------------------------------
# Text manipulation
# ---------------------------------------------------------------------------

def _replace_span(text: str, span_idx: int, span_text: str, replacement: str) -> str:
    words = text.split()
    span_len = len(span_text.split())
    words[span_idx : span_idx + span_len] = [replacement]
    return " ".join(words)


def _split_around_span(text: str, span_idx: int, span_text: str):
    """Return (before, span, after) as plain strings split at the word level."""
    words = text.split()
    span_len = len(span_text.split())
    before = " ".join(words[:span_idx])
    after  = " ".join(words[span_idx + span_len:])
    return before, after


# ---------------------------------------------------------------------------
# MLM scoring
# ---------------------------------------------------------------------------

def _build_mlm_inputs(tokenizer, before: str, candidate: str, after: str,
                      max_length: int, device):
    """
    Build (original_ids, masked_ids, attention_mask, cand_mask) tensors.
    masked_ids has candidate tokens replaced with [MASK].
    cand_mask marks positions of the candidate tokens.
    """
    cls_id  = tokenizer.cls_token_id
    sep_id  = tokenizer.sep_token_id
    mask_id = tokenizer.mask_token_id
    pad_id  = tokenizer.pad_token_id

    # RoBERTa uses GPT-2 BPE: tokens that follow a space get a leading "Ġ" prefix,
    # so we must add the space to get correct token IDs (mirrors fairseq leading_space / trailing_space).
    enc_before = tokenizer(before, add_special_tokens=False)["input_ids"] if before.strip() else []
    cand_str   = (" " + candidate) if before.strip() else candidate
    enc_cand   = tokenizer(cand_str, add_special_tokens=False)["input_ids"]
    after_str  = (" " + after) if after.strip() else after
    enc_after  = tokenizer(after_str, add_special_tokens=False)["input_ids"] if after.strip() else []

    # [CLS] before cand after [SEP]
    full_ids = [cls_id] + enc_before + enc_cand + enc_after + [sep_id]
    cand_start = 1 + len(enc_before)
    cand_end   = cand_start + len(enc_cand)

    # Truncate: keep [CLS]...[SEP] within max_length
    if len(full_ids) > max_length:
        full_ids = full_ids[: max_length - 1] + [sep_id]
        cand_end = min(cand_end, len(full_ids) - 1)

    seq_len = len(full_ids)
    cand_mask = [False] * seq_len
    masked_ids = list(full_ids)
    for i in range(cand_start, cand_end):
        cand_mask[i]  = True
        masked_ids[i] = mask_id

    # Pad
    pad_len = max_length - seq_len
    attn = [1] * seq_len + [0] * pad_len
    full_ids   = full_ids   + [pad_id] * pad_len
    masked_ids = masked_ids + [pad_id] * pad_len
    cand_mask  = cand_mask  + [False]  * pad_len

    to_t = lambda x, dt: torch.tensor([x], dtype=dt, device=device)
    return (
        to_t(full_ids,   torch.long),
        to_t(masked_ids, torch.long),
        to_t(attn,       torch.long),
        to_t(cand_mask,  torch.bool),
    )


def _get_lprobs(model, original_ids, masked_ids, attention_mask, cand_mask, mixed_precision):
    """Average log P of candidate tokens given the masked context."""
    with amp.autocast("cuda", enabled=mixed_precision):
        logits = model(input_ids=masked_ids, attention_mask=attention_mask).logits
    lprobs = F.log_softmax(logits, dim=-1, dtype=torch.float)        # [1, L, V] — float32 for numerical stability
    token_scores = lprobs.gather(2, original_ids.unsqueeze(-1)).squeeze(-1)  # [1, L]
    mask_f = cand_mask.float()
    return (token_scores * mask_f).sum(-1) / mask_f.sum(-1).clamp(min=1)    # [1]


def _score_candidate(model, tokenizer, text, span2_idx, span2_text, candidate,
                     max_length, dev, mixed_precision):
    before, after = _split_around_span(text, span2_idx, span2_text)
    orig, masked, attn, cmask = _build_mlm_inputs(
        tokenizer, before, candidate, after, max_length, dev
    )
    return _get_lprobs(model, orig, masked, attn, cmask, mixed_precision).item()


# ---------------------------------------------------------------------------
# Training pair construction
# ---------------------------------------------------------------------------

def _build_train_pairs(pos_train, nlp):
    """(before, span2_text, after, pos_candidate, [neg_candidates]) per positive example."""
    pairs = []
    skipped = 0
    for ex in pos_train:
        text       = ex["text"]
        span1_text = ex["span1_text"].rstrip('.,')  # mirrors fairseq query cleanup
        span2_idx  = ex["span2_index"]
        span2_text = ex["span2_text"].rstrip('.,')

        negatives = _extract_negatives(text, span1_text, nlp)
        if not negatives:
            skipped += 1
            continue

        before, after = _split_around_span(text, span2_idx, span2_text)
        pairs.append((before, span2_text, after, span1_text, negatives))

    if skipped:
        print(f"Skipped {skipped} examples with no negative candidates")
    return pairs


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def _eval(model, tokenizer, val_ds, nlp, dev, max_length, mixed_precision):
    """
    Predict label=1 if query (span1) score >= all noun-chunk candidate scores.
    Mirrors fairseq: pred = (query_lprobs >= cand_lprobs).all()
    """
    model.eval()
    ncorrect = 0

    with torch.no_grad():
        for ex in val_ds:
            text       = ex["text"]
            span1_text = ex["span1_text"].rstrip('.,')
            span2_idx  = ex["span2_index"]
            span2_text = ex["span2_text"].rstrip('.,')
            label      = int(ex["label"])

            query_score = _score_candidate(
                model, tokenizer, text, span2_idx, span2_text,
                span1_text, max_length, dev, mixed_precision
            )
            candidates = _extract_negatives(text, span1_text, nlp)

            if candidates:
                cand_scores = [
                    _score_candidate(model, tokenizer, text, span2_idx, span2_text,
                                     c, max_length, dev, mixed_precision)
                    for c in candidates
                ]
                pred = int(all(query_score >= cs for cs in cand_scores))
            else:
                pred = 1  # no competitors → predict True

            ncorrect += pred == label

    return ncorrect / len(val_ds)


# ---------------------------------------------------------------------------
# Main fine-tune function
# ---------------------------------------------------------------------------

def finetune_wnli(cfg, encoder_ckpt: str):
    set_seed(cfg["seed"])
    dev = device(cfg.get("gpu_id", None))
    print(f"Using device: {dev}")

    nlp = spacy.load("en_core_web_lg")

    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"])
    model = AutoModelForMaskedLM.from_pretrained(cfg["model_name"])

    if encoder_ckpt != "fresh":
        print(f"Loading pretrained encoder from {encoder_ckpt}")
        state = torch.load(encoder_ckpt, map_location="cpu", weights_only=False)

        if "encoder_state_dict" in state and "model_state_dict" not in state:
            # Pretrain checkpoint: encoder_state_dict holds native RobertaModel keys
            # (embeddings.*, encoder.layer.*, pooler.*). Map directly to roberta.*.
            src = state["encoder_state_dict"]
            roberta_state = {"roberta." + k: v for k, v in src.items()}
        else:
            # Finetune checkpoint: model_state_dict from CorrectEnglishSentenceClassifier
            # whose self.encoder wraps RobertaModel → keys are encoder.embeddings.*,
            # encoder.encoder.layer.*, encoder.pooler.* → strip "encoder." prefix.
            src = state.get("model_state_dict", state)
            encoder_keys = {k: v for k, v in src.items() if k.startswith("encoder.")}
            roberta_state = {
                "roberta." + k[len("encoder."):]: v for k, v in encoder_keys.items()
            }

        missing, unexpected = model.load_state_dict(roberta_state, strict=False)
        print(f"Loaded encoder weights. Missing: {len(missing)}, Unexpected: {len(unexpected)}")
    else:
        print("Using fresh encoder")

    model = model.to(dev)

    ds       = load_dataset("super_glue", "wsc")
    pos_train = ds["train"].filter(lambda x: x["label"] == 1)
    val_ds   = ds["validation"]
    print(f"Positive train: {len(pos_train)}, Validation: {len(val_ds)}")

    print("Building training pairs (spaCy NP extraction)...")
    train_pairs = _build_train_pairs(pos_train, nlp)
    print(f"Training pairs: {len(train_pairs)}")

    max_length     = cfg.get("max_length", 128)
    margin_alpha   = cfg.get("margin_alpha", 1.0)
    margin_beta    = cfg.get("margin_beta", 0.0)
    use_cross_entropy = cfg.get("cross_entropy_loss", True)
    epochs         = cfg["epochs"]
    mixed_precision = cfg.get("mixed_precision", True)

    _betas = tuple(cfg["adam_betas"]) if "adam_betas" in cfg else (0.9, 0.999)
    _eps   = cfg.get("adam_eps", 1e-8)
    opt    = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg.get("weight_decay", 0.01),
                               betas=_betas, eps=_eps)
    scaler = GradScaler("cuda", enabled=mixed_precision)

    lr_sched_cfg   = cfg.get("lr_scheduler", "cosine")
    lr_warmup_ratio = cfg.get("lr_warmup_ratio", 0.06)
    num_steps      = epochs * len(train_pairs)
    lr_scheduler   = None
    if lr_sched_cfg and str(lr_sched_cfg).lower() not in ("none", "null", ""):
        lr_scheduler = _build_finetune_lr_scheduler(opt, num_steps, lr_warmup_ratio, lr_sched_cfg)

    _results_dir = cfg.get("results_dir", "results")
    _model_slug  = cfg.get("model_name", "").replace("-", "_").replace("/", "_")
    _exp_name    = cfg.get("experiment_name", "")
    _exp_name    = f"{_exp_name}_{_model_slug}" if _exp_name else _model_slug

    ckpt_dir = cfg.get("ckpt_dir", "checkpoints/finetune")
    os.makedirs(ckpt_dir, exist_ok=True)
    prefix = f"{_exp_name}_" if _exp_name else ""

    best_val_acc    = -1.0
    best_model_state = None
    epoch_log        = []
    _hist_val, _hist_loss, _hist_lr = [], [], []

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        pairs = list(train_pairs)
        random.shuffle(pairs)

        for before, span2_text, after, pos_cand, neg_cands in tqdm(pairs, desc=f"WNLI {epoch}/{epochs}", leave=False):
            opt.zero_grad(set_to_none=True)

            # Score positive candidate
            orig_q, masked_q, attn_q, cmask_q = _build_mlm_inputs(
                tokenizer, before, pos_cand, after, max_length, dev
            )
            query_lprob = _get_lprobs(model, orig_q, masked_q, attn_q, cmask_q, mixed_precision)

            # Score all negative candidates in one batched forward pass (mirrors fairseq wsc_criterion)
            all_orig, all_masked, all_attn, all_cmask = [], [], [], []
            for neg in neg_cands:
                orig_c, masked_c, attn_c, cmask_c = _build_mlm_inputs(
                    tokenizer, before, neg, after, max_length, dev
                )
                all_orig.append(orig_c)
                all_masked.append(masked_c)
                all_attn.append(attn_c)
                all_cmask.append(cmask_c)
            cand_lprobs_t = _get_lprobs(
                model,
                torch.cat(all_orig),
                torch.cat(all_masked),
                torch.cat(all_attn),
                torch.cat(all_cmask),
                mixed_precision,
            )  # [N]

            # Loss (cross-entropy: query is class 0 among [query, neg1, neg2, ...])
            if use_cross_entropy:
                all_scores = torch.cat([query_lprob, cand_lprobs_t]).unsqueeze(0)  # [1, 1+N]
                loss = F.cross_entropy(all_scores, torch.zeros(1, dtype=torch.long, device=dev))
            else:
                # Margin loss from Kocijan et al. (fairseq default)
                loss = (
                    -query_lprob
                    + margin_alpha * (cand_lprobs_t - query_lprob + margin_beta).clamp(min=0)
                ).sum()

            scaler.scale(loss).backward()
            max_grad_norm = cfg.get("max_grad_norm", 1.0)
            if max_grad_norm > 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            scaler.step(opt)
            scaler.update()
            if lr_scheduler is not None:
                lr_scheduler.step()

            total_loss += loss.item()

        epoch_loss = total_loss / len(pairs)
        epoch_lr   = opt.param_groups[0]["lr"]

        val_acc = _eval(model, tokenizer, val_ds, nlp, dev, max_length, mixed_precision)
        print(f"[WNLI] epoch {epoch}: loss={epoch_loss:.4f} val_acc={val_acc:.4f} lr={epoch_lr:.2e}")

        if val_acc > best_val_acc:
            best_val_acc  = val_acc
            best_model_state = {k: v.clone() for k, v in model.state_dict().items()}
            best_ckpt = os.path.join(ckpt_dir, f"{prefix}best.pt")
            save_ckpt({"epoch": epoch, "val_acc": val_acc, "model_state_dict": model.state_dict()}, best_ckpt)
            print(f"          ✓ Saved best model (Val Acc={val_acc:.4f}) to {best_ckpt}")

        _hist_val.append(val_acc)
        _hist_loss.append(epoch_loss)
        _hist_lr.append(epoch_lr)
        epoch_log.append({
            "epoch": epoch,
            "train": {"loss": epoch_loss, "lr": epoch_lr},
            "validation": {"acc": val_acc},
        })
        save_finetune_results(
            _results_dir, "wnli", _exp_name,
            {"model": cfg.get("model_name", ""), "epochs": epoch_log},
        )

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    final_val_acc = _eval(model, tokenizer, val_ds, nlp, dev, max_length, mixed_precision)
    print(f"Final Val Acc: {final_val_acc:.4f}")
    save_finetune_results(
        _results_dir, "wnli", _exp_name, {
            "model": cfg.get("model_name", ""),
            "epochs": epoch_log,
            "final": {"validation": {"acc": final_val_acc}},
        }
    )

    if _hist_val:
        _save_training_plots(
            list(range(1, len(_hist_val) + 1)),
            {"val": _hist_val},
            _hist_loss,
            _hist_lr,
            "WNLI accuracy per epoch",
            os.path.join(_results_dir, f"{_exp_name}_wnli_training.png"),
        )

    return model

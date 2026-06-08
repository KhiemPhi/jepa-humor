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
from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoConfig

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

def _build_train_examples(train_ds, nlp):
    """
    One entry per WSC train example, preserving order and count so batch
    denominators match fairseq's sample_size = nqueries (# labeled examples).

    Mirrors fairseq wsc_criterion.forward: a loss term is computed ONLY for
    positive (label==1) examples, but every labeled example occupies a batch
    slot and counts toward the gradient normalization. So:
      - positive with >=1 mined negative → (before, span2_text, after, query, [negs])
      - negative, or positive with no candidates → None (slot only, no loss)
    """
    examples = []
    n_pos = 0
    for ex in train_ds:
        if int(ex["label"]) != 1:
            examples.append(None)
            continue
        text       = ex["text"]
        span1_text = ex["span1_text"].rstrip('.,')  # mirrors fairseq query cleanup
        span2_idx  = ex["span2_index"]
        span2_text = ex["span2_text"].rstrip('.,')
        negatives  = _extract_negatives(text, span1_text, nlp)
        if not negatives:
            examples.append(None)
            continue
        before, after = _split_around_span(text, span2_idx, span2_text)
        examples.append((before, span2_text, after, span1_text, negatives))
        n_pos += 1
    print(f"Built {len(examples)} train slots; {n_pos} positive examples contribute loss")
    return examples


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
    enc_config = AutoConfig.from_pretrained(cfg["model_name"])
    if "hidden_dropout" in cfg:
        enc_config.hidden_dropout_prob = cfg["hidden_dropout"]
    if "attention_dropout" in cfg:
        enc_config.attention_probs_dropout_prob = cfg["attention_dropout"]
    model = AutoModelForMaskedLM.from_pretrained(cfg["model_name"], config=enc_config)

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
    val_ds   = ds["validation"]
    print(f"Train: {len(ds['train'])}, Validation: {len(val_ds)}")

    print("Building training examples (spaCy NP extraction)...")
    train_examples = _build_train_examples(ds["train"], nlp)

    max_length     = cfg.get("max_length", 128)
    margin_alpha   = cfg.get("margin_alpha", 1.0)
    margin_beta    = cfg.get("margin_beta", 0.0)
    use_cross_entropy = cfg.get("cross_entropy_loss", True)
    mixed_precision = cfg.get("mixed_precision", True)
    # fairseq WSC: --batch-size 16, --total-num-updates 2000, --warmup-updates 250 (ratio 0.125)
    batch_size        = cfg.get("batch_size", 16)
    total_num_updates = cfg.get("total_num_updates", 2000)

    _betas = tuple(cfg["adam_betas"]) if "adam_betas" in cfg else (0.9, 0.98)
    _eps   = cfg.get("adam_eps", 1e-6)
    opt    = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg.get("weight_decay", 0.01),
                               betas=_betas, eps=_eps)
    scaler = GradScaler("cuda", enabled=mixed_precision)

    lr_sched_cfg   = cfg.get("lr_scheduler", "linear")
    lr_warmup_ratio = cfg.get("lr_warmup_ratio", 0.125)
    lr_scheduler   = None
    if lr_sched_cfg and str(lr_sched_cfg).lower() not in ("none", "null", ""):
        lr_scheduler = _build_finetune_lr_scheduler(opt, total_num_updates, lr_warmup_ratio, lr_sched_cfg)

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

    def _example_loss(before, span2_text, after, pos_cand, neg_cands):
        """Per-example WSC loss (mirrors fairseq get_loss): query vs mined candidates."""
        orig_q, masked_q, attn_q, cmask_q = _build_mlm_inputs(
            tokenizer, before, pos_cand, after, max_length, dev)
        query_lprob = _get_lprobs(model, orig_q, masked_q, attn_q, cmask_q, mixed_precision)

        # Score all negative candidates in one batched forward pass (mirrors fairseq).
        all_orig, all_masked, all_attn, all_cmask = [], [], [], []
        for neg in neg_cands:
            o, m, a, c = _build_mlm_inputs(tokenizer, before, neg, after, max_length, dev)
            all_orig.append(o); all_masked.append(m); all_attn.append(a); all_cmask.append(c)
        cand_lprobs_t = _get_lprobs(
            model, torch.cat(all_orig), torch.cat(all_masked),
            torch.cat(all_attn), torch.cat(all_cmask), mixed_precision)  # [N]

        if use_cross_entropy:
            # query is class 0 among [query, neg1, neg2, ...]
            all_scores = torch.cat([query_lprob, cand_lprobs_t]).unsqueeze(0)  # [1, 1+N]
            return F.cross_entropy(all_scores, torch.zeros(1, dtype=torch.long, device=dev))
        # Margin loss from Kocijan et al. (fairseq default)
        return (-query_lprob
                + margin_alpha * (cand_lprobs_t - query_lprob + margin_beta).clamp(min=0)).sum()

    # fairseq-faithful training: batches of `batch_size` examples form one optimizer
    # update; loss is summed over the positive examples in the batch and the gradient
    # is normalized by sample_size (= #labeled examples in the batch = len(batch)).
    # Training runs for `total_num_updates` updates, cycling the data across epochs.
    max_grad_norm = cfg.get("max_grad_norm", 0.0)
    update, epoch = 0, 0
    print(f"Training for {total_num_updates} updates (batch_size={batch_size})")

    while update < total_num_updates:
        epoch += 1
        random.shuffle(train_examples)
        model.train()
        running, n_batches = 0.0, 0

        n_steps = (len(train_examples) + batch_size - 1) // batch_size
        pbar = tqdm(range(0, len(train_examples), batch_size), total=n_steps,
                    desc=f"WNLI ep{epoch} (upd {update}/{total_num_updates})", leave=False)
        for start in pbar:
            if update >= total_num_updates:
                break
            batch = train_examples[start:start + batch_size]
            denom = len(batch)  # = sample_size (nqueries): every train row is labeled
            opt.zero_grad(set_to_none=True)

            n_terms, batch_loss = 0, 0.0
            for fields in batch:
                if fields is None:
                    continue  # negative / no-candidate: occupies a slot, contributes no loss
                loss_i = _example_loss(*fields)
                scaler.scale(loss_i / denom).backward()  # accumulate; /denom = normalize by sample_size
                batch_loss += loss_i.item() / denom
                n_terms += 1

            if n_terms == 0:
                continue  # all-negative batch → nothing to step
            if max_grad_norm > 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            scaler.step(opt)
            scaler.update()
            if lr_scheduler is not None:
                lr_scheduler.step()
            update += 1
            running += batch_loss
            n_batches += 1

        epoch_loss = running / max(1, n_batches)
        epoch_lr   = opt.param_groups[0]["lr"]

        val_acc = _eval(model, tokenizer, val_ds, nlp, dev, max_length, mixed_precision)
        print(f"[WNLI] epoch {epoch} (update {update}/{total_num_updates}): "
              f"loss={epoch_loss:.4f} val_acc={val_acc:.4f} lr={epoch_lr:.2e}")

        if val_acc > best_val_acc:
            best_val_acc  = val_acc
            best_model_state = {k: v.clone() for k, v in model.state_dict().items()}
            best_ckpt = os.path.join(ckpt_dir, f"{prefix}best.pt")
            save_ckpt({"epoch": epoch, "update": update, "val_acc": val_acc,
                       "model_state_dict": model.state_dict()}, best_ckpt)
            print(f"          ✓ Saved best model (Val Acc={val_acc:.4f}) to {best_ckpt}")

        _hist_val.append(val_acc)
        _hist_loss.append(epoch_loss)
        _hist_lr.append(epoch_lr)
        epoch_log.append({
            "epoch": epoch, "update": update,
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

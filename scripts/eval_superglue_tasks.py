"""
eval_superglue_tasks.py — Eval all 10 SuperGLUE tasks and write leaderboard TSVs.

Outputs to roberta_large_superglue/:
  {Task}.tsv          — test predictions (SuperGLUE leaderboard format)
  {task}_eval.json    — validation metrics

Usage:
  python -m scripts.eval_superglue_tasks               # all tasks
  python -m scripts.eval_superglue_tasks --tasks boolq wic rte
  python -m scripts.eval_superglue_tasks --gpu_id 0
"""
import argparse
import datetime
import json
import os

import numpy as np
import spacy
import torch
import yaml
from datasets import load_dataset
from sklearn.metrics import accuracy_score, f1_score
from torch import amp
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModel, AutoModelForMaskedLM, AutoTokenizer

from jepa_humor.finetune import CorrectEnglishSentenceClassifier
from jepa_humor.finetune_superglue import _copa_text
from jepa_humor.finetune_wnli import _extract_negatives, _score_candidate
from jepa_humor.utils import device, set_seed

os.environ["TOKENIZERS_PARALLELISM"] = "false"

_CKPT = "checkpoints/finetune"
_MNLI = ("billionare_file/checkpoints_roberta_large/finetune/"
         "finetune_mnli_exp1_FacebookAI_roberta_large_best.pt")

# ── Task catalogue ────────────────────────────────────────────────────────────
TASKS = {
    "boolq": dict(
        ckpt=f"{_CKPT}/finetune_boolq_exp1_FacebookAI_roberta_large_best.pt",
        model_name="FacebookAI/roberta-large", num_labels=2,
        text_keys=["passage", "question"], max_length=512,
        label_names={0: "False", 1: "True"},
        eval_type="clf", tsv_name="BoolQ.tsv",
    ),
    "cb": dict(
        ckpt=f"{_CKPT}/finetune_cb_exp1_FacebookAI_roberta_large_best.pt",
        model_name="FacebookAI/roberta-large", num_labels=3,
        text_keys=["premise", "hypothesis"], max_length=256,
        label_names={0: "entailment", 1: "contradiction", 2: "neutral"},
        eval_type="clf", tsv_name="CB.tsv",
    ),
    "copa": dict(
        ckpt=f"{_CKPT}/finetune_copa_exp1_FacebookAI_roberta_large_copa_best.pt",
        model_name="FacebookAI/roberta-large", num_labels=1, max_length=128,
        label_names={0: "0", 1: "1"},
        eval_type="copa", tsv_name="COPA.tsv",
    ),
    "multirc": dict(
        ckpt=f"{_CKPT}/finetune_multirc_exp1_FacebookAI_roberta_large_multirc_best.pt",
        model_name="FacebookAI/roberta-large", num_labels=2, max_length=512,
        label_names={0: "False", 1: "True"},
        eval_type="multirc", tsv_name="MultiRC.tsv",
    ),
    "wic": dict(
        ckpt=f"{_CKPT}/finetune_wic_exp1_FacebookAI_roberta_large_best.pt",
        model_name="FacebookAI/roberta-large", num_labels=2, max_length=256,
        use_word_context=True,          # prepends "{word}: " to sentence1
        label_names={0: "False", 1: "True"},
        eval_type="wic", tsv_name="WiC.tsv",
    ),
    "wsc": dict(
        ckpt=f"{_CKPT}/finetune_wsc_exp1_FacebookAI_roberta_large_best.pt",
        model_name="FacebookAI/roberta-large", max_length=128,
        label_names={0: "False", 1: "True"},
        eval_type="wsc", tsv_name="WSC.tsv",
    ),
    "rte": dict(
        ckpt=f"{_CKPT}/finetune_superglue_rte_exp1_FacebookAI_roberta_large_rte_best.pt",
        model_name="FacebookAI/roberta-large", num_labels=2,
        text_keys=["premise", "hypothesis"], max_length=256,
        label_names={0: "entailment", 1: "not_entailment"},
        eval_type="clf", tsv_name="RTE.tsv",
    ),
    "record": dict(
        ckpt=f"{_CKPT}/finetune_rccr_exp1_FacebookAI_roberta_large_record_best.pt",
        model_name="FacebookAI/roberta-large", num_labels=2, max_length=512,
        cand_batch=8,
        eval_type="record", tsv_name="ReCoRD.tsv",
    ),
    "axb": dict(
        ckpt=_MNLI, model_name="FacebookAI/roberta-large", num_labels=3,
        text_keys=["premise", "hypothesis"], max_length=256,
        label_names={0: "entailment", 1: "not_entailment"},
        hf_name="axb", eval_type="ax", tsv_name="AX-b.tsv",
    ),
    "axg": dict(
        ckpt=_MNLI, model_name="FacebookAI/roberta-large", num_labels=3,
        text_keys=["premise", "hypothesis"], max_length=256,
        label_names={0: "entailment", 1: "not_entailment"},
        hf_name="axg", eval_type="ax", tsv_name="AX-g.tsv",
    ),
}


# ── Shared helpers ────────────────────────────────────────────────────────────

def _load_clf(cfg, dev):
    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"])
    base      = AutoModel.from_pretrained(cfg["model_name"])
    model     = CorrectEnglishSentenceClassifier(
        base, base.config.hidden_size, cfg["num_labels"]
    ).to(dev)
    ckpt      = torch.load(cfg["ckpt"], map_location=dev, weights_only=False)
    state     = ckpt.get("model_state_dict") or ckpt.get("state_dict") or ckpt
    model.load_state_dict(state)
    model.eval()
    print(f"  Loaded {cfg['ckpt']}")
    return tokenizer, model


def _load_mlm(cfg, dev):
    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"])
    model     = AutoModelForMaskedLM.from_pretrained(cfg["model_name"])
    ckpt      = torch.load(cfg["ckpt"], map_location=dev, weights_only=False)
    state     = ckpt.get("model_state_dict") or ckpt.get("state_dict") or ckpt
    model.load_state_dict(state)
    model     = model.to(dev)
    model.eval()
    print(f"  Loaded MLM {cfg['ckpt']}")
    return tokenizer, model


def _tokenize(split, tokenizer, max_length, text_keys):
    keep = {"input_ids", "attention_mask", "label", "idx"}
    def fn(examples):
        return tokenizer(
            *[examples[k] for k in text_keys],
            truncation=True, padding="max_length", max_length=max_length,
        )
    return split.map(fn, batched=True,
                     remove_columns=[c for c in split.column_names if c not in keep],
                     desc="Tokenizing")


def _clf_collate(batch):
    return {
        "input_ids":      torch.tensor([b["input_ids"]      for b in batch], dtype=torch.long),
        "attention_mask": torch.tensor([b["attention_mask"]  for b in batch], dtype=torch.long),
        "label":          torch.tensor([b["label"]           for b in batch], dtype=torch.long),
        "idx":            [b["idx"] for b in batch],
    }


def _run_clf(model, loader, dev, mixed_precision=True):
    all_true, all_pred, all_idx = [], [], []
    with torch.no_grad():
        for batch in tqdm(loader, leave=False):
            logits = model(batch["input_ids"].to(dev), batch["attention_mask"].to(dev))
            all_pred.extend(logits.argmax(-1).cpu().tolist())
            all_true.extend(batch["label"].tolist())
            all_idx.extend(batch["idx"])
    return all_true, all_pred, all_idx


def _save_tsv(path, indices, preds, label_names):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        f.write("idx\tprediction\n")
        for idx, pred in zip(indices, preds):
            if label_names and not isinstance(pred, str):
                label = label_names[int(pred)]
            else:
                label = str(pred)
            f.write(f"{idx}\t{label}\n")
    print(f"  Saved {path}  ({len(preds)} rows)")


def _save_json(path, data):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  Saved {path}")


def _val_metrics(task, true, pred):
    valid = np.array(true) != -1
    if not valid.any():
        return {}
    t, p = np.array(true)[valid], np.array(pred)[valid]
    return {
        "acc":      float(accuracy_score(t, p)),
        "macro_f1": float(f1_score(t, p, average="macro")),
    }


# ── Per-task eval functions ───────────────────────────────────────────────────

def eval_clf(task, cfg, dev, out_dir):
    """Standard CLS-head classification: BoolQ, CB, RTE."""
    print(f"\n{'='*55}\n[{task.upper()}]  {cfg['tsv_name']}\n{'='*55}")
    tokenizer, model = _load_clf(cfg, dev)
    ds         = load_dataset("super_glue", task)
    text_keys  = cfg["text_keys"]
    max_len    = cfg["max_length"]
    lmap       = cfg["label_names"]

    val_tok = _tokenize(ds["validation"], tokenizer, max_len, text_keys)
    val_ldr = DataLoader(val_tok, batch_size=64, shuffle=False, num_workers=2,
                         collate_fn=_clf_collate)
    val_true, val_pred, _ = _run_clf(model, val_ldr, dev)
    val_m = _val_metrics(task, val_true, val_pred)
    print(f"  Val  acc={val_m.get('acc',0):.4f}  f1={val_m.get('macro_f1',0):.4f}")

    test_tok = _tokenize(ds["test"], tokenizer, max_len, text_keys)
    test_ldr = DataLoader(test_tok, batch_size=64, shuffle=False, num_workers=2,
                          collate_fn=_clf_collate)
    _, test_pred, test_idx = _run_clf(model, test_ldr, dev)
    _save_tsv(os.path.join(out_dir, cfg["tsv_name"]), test_idx, test_pred, lmap)

    res = {"task": task, "model": cfg["model_name"], "checkpoint": cfg["ckpt"],
           "timestamp": datetime.datetime.now().isoformat(),
           "validation": val_m, "n_test": len(test_pred)}
    _save_json(os.path.join(out_dir, f"{task}_eval.json"), res)
    return res


def eval_wic(task, cfg, dev, out_dir):
    """WiC: optionally prepend '{word}: ' to sentence1."""
    print(f"\n{'='*55}\n[WIC]  {cfg['tsv_name']}\n{'='*55}")
    tokenizer, model = _load_clf(cfg, dev)
    ds              = load_dataset("super_glue", "wic")
    max_len         = cfg["max_length"]
    use_word        = cfg.get("use_word_context", True)
    lmap            = cfg["label_names"]

    def _tok_wic(split):
        keep = {"input_ids", "attention_mask", "label", "idx"}
        def fn(examples):
            if use_word:
                s1 = [f"{w}: {s}" for w, s in zip(examples["word"], examples["sentence1"])]
            else:
                s1 = examples["sentence1"]
            return tokenizer(s1, examples["sentence2"],
                             truncation=True, padding="max_length", max_length=max_len)
        return split.map(fn, batched=True,
                         remove_columns=[c for c in split.column_names if c not in keep],
                         desc="Tokenizing WiC")

    val_tok = _tok_wic(ds["validation"])
    val_ldr = DataLoader(val_tok, batch_size=64, shuffle=False, num_workers=2,
                         collate_fn=_clf_collate)
    val_true, val_pred, _ = _run_clf(model, val_ldr, dev)
    val_m = _val_metrics(task, val_true, val_pred)
    print(f"  Val  acc={val_m.get('acc',0):.4f}  (use_word_context={use_word})")

    test_tok = _tok_wic(ds["test"])
    test_ldr = DataLoader(test_tok, batch_size=64, shuffle=False, num_workers=2,
                          collate_fn=_clf_collate)
    _, test_pred, test_idx = _run_clf(model, test_ldr, dev)
    _save_tsv(os.path.join(out_dir, cfg["tsv_name"]), test_idx, test_pred, lmap)

    res = {"task": task, "model": cfg["model_name"], "checkpoint": cfg["ckpt"],
           "timestamp": datetime.datetime.now().isoformat(),
           "validation": val_m, "n_test": len(test_pred)}
    _save_json(os.path.join(out_dir, f"{task}_eval.json"), res)
    return res


def eval_copa(task, cfg, dev, out_dir):
    """COPA: ranking head — score each choice (one scalar), pick the higher-scoring one."""
    print(f"\n{'='*55}\n[COPA]  {cfg['tsv_name']}\n{'='*55}")
    tokenizer, model = _load_clf(cfg, dev)
    ds      = load_dataset("super_glue", "copa")
    max_len = cfg["max_length"]
    lmap    = cfg["label_names"]

    def _predict(split):
        preds, trues, idxs = [], [], []
        model.eval()
        with torch.no_grad():
            for ex in tqdm(split, leave=False):
                scores = []
                for choice in [ex["choice1"], ex["choice2"]]:
                    text = _copa_text(ex["premise"], ex["question"], choice)
                    enc  = tokenizer(text, return_tensors="pt", truncation=True,
                                     padding="max_length", max_length=max_len)
                    with amp.autocast("cuda", enabled=True):
                        logits = model(enc["input_ids"].to(dev), enc["attention_mask"].to(dev))
                    scores.append(logits[0, 0].item())  # single ranking score per choice
                pred = int(scores[1] > scores[0])
                preds.append(pred); trues.append(ex["label"]); idxs.append(ex["idx"])
        return trues, preds, idxs

    val_true, val_pred, _ = _predict(ds["validation"])
    valid = np.array(val_true) != -1
    val_acc = float(accuracy_score(np.array(val_true)[valid], np.array(val_pred)[valid]))
    print(f"  Val  acc={val_acc:.4f}")

    _, test_pred, test_idx = _predict(ds["test"])
    _save_tsv(os.path.join(out_dir, cfg["tsv_name"]), test_idx, test_pred, lmap)

    res = {"task": task, "model": cfg["model_name"], "checkpoint": cfg["ckpt"],
           "timestamp": datetime.datetime.now().isoformat(),
           "validation": {"acc": val_acc}, "n_test": len(test_pred)}
    _save_json(os.path.join(out_dir, f"{task}_eval.json"), res)
    return res


def eval_multirc(task, cfg, dev, out_dir):
    """MultiRC: binary per (paragraph, question, answer). Metrics: F1-a and F1-m."""
    print(f"\n{'='*55}\n[MULTIRC]  {cfg['tsv_name']}\n{'='*55}")
    tokenizer, model = _load_clf(cfg, dev)
    ds      = load_dataset("super_glue", "multirc")
    max_len = cfg["max_length"]
    lmap    = cfg["label_names"]

    def _tok(split):
        keep = {"input_ids", "attention_mask", "label", "idx"}
        def fn(examples):
            text_b = [q + " " + a for q, a in zip(examples["question"], examples["answer"])]
            return tokenizer(examples["paragraph"], text_b,
                             truncation=True, padding="max_length", max_length=max_len)
        return split.map(fn, batched=True,
                         remove_columns=[c for c in split.column_names if c not in keep],
                         desc="Tokenizing MultiRC")

    def _collate(batch):
        return {
            "input_ids":      torch.tensor([b["input_ids"]     for b in batch], dtype=torch.long),
            "attention_mask": torch.tensor([b["attention_mask"] for b in batch], dtype=torch.long),
            "label":          torch.tensor([b["label"]          for b in batch], dtype=torch.long),
            "idx":            [b["idx"] for b in batch],
        }

    def _predict(split):
        tok = _tok(split)
        ldr = DataLoader(tok, batch_size=32, shuffle=False, num_workers=2, collate_fn=_collate)
        all_true, all_pred, all_idx = [], [], []
        model.eval()
        with torch.no_grad():
            for batch in tqdm(ldr, leave=False):
                logits = model(batch["input_ids"].to(dev), batch["attention_mask"].to(dev))
                all_pred.extend(logits.argmax(-1).cpu().tolist())
                all_true.extend(batch["label"].tolist())
                all_idx.extend(batch["idx"])
        return all_true, all_pred, all_idx

    val_true, val_pred, val_idx = _predict(ds["validation"])
    valid = np.array(val_true) != -1
    t, p  = np.array(val_true)[valid], np.array(val_pred)[valid]
    f1_a  = float(f1_score(t, p, average="binary")) if valid.any() else 0.0

    # F1-m: macro-average of per-question F1
    q_true, q_pred = {}, {}
    for d, gt, pr in zip(val_idx, val_true, val_pred):
        if gt == -1:
            continue
        qk = (d["paragraph"], d["question"])
        q_true.setdefault(qk, []).append(gt)
        q_pred.setdefault(qk, []).append(pr)

    def _qf1(gts, prs):
        tp = sum(g == 1 and p == 1 for g, p in zip(gts, prs))
        fp = sum(g == 0 and p == 1 for g, p in zip(gts, prs))
        fn = sum(g == 1 and p == 0 for g, p in zip(gts, prs))
        if tp == 0:
            return 0.0
        pr_ = tp / (tp + fp); re_ = tp / (tp + fn)
        return 2 * pr_ * re_ / (pr_ + re_)

    f1_m = float(np.mean([_qf1(q_true[k], q_pred[k]) for k in q_true])) if q_true else 0.0
    print(f"  Val  F1-a={f1_a:.4f}  F1-m={f1_m:.4f}")

    _, test_pred, test_idx = _predict(ds["test"])
    compound_idx = [f"{d['paragraph']}-{d['question']}-{d['answer']}" for d in test_idx]
    _save_tsv(os.path.join(out_dir, cfg["tsv_name"]), compound_idx, test_pred, lmap)

    res = {"task": task, "model": cfg["model_name"], "checkpoint": cfg["ckpt"],
           "timestamp": datetime.datetime.now().isoformat(),
           "validation": {"f1_a": f1_a, "f1_m": f1_m}, "n_test": len(test_pred)}
    _save_json(os.path.join(out_dir, f"{task}_eval.json"), res)
    return res


def eval_wsc(task, cfg, dev, out_dir):
    """WSC: MLM scoring — query (span1_text) must beat all NP-chunk competitors."""
    print(f"\n{'='*55}\n[WSC]  {cfg['tsv_name']}\n{'='*55}")
    tokenizer, model = _load_mlm(cfg, dev)
    ds              = load_dataset("super_glue", "wsc")
    max_len         = cfg["max_length"]
    mixed_precision = True
    lmap            = cfg["label_names"]
    nlp             = spacy.load("en_core_web_lg")

    def _predict_example(ex):
        span1_text = ex["span1_text"].rstrip(".,")
        span2_idx  = ex["span2_index"]
        span2_text = ex["span2_text"].rstrip(".,")
        query_score = _score_candidate(
            model, tokenizer, ex["text"], span2_idx, span2_text,
            span1_text, max_len, dev, mixed_precision,
        )
        candidates = _extract_negatives(ex["text"], span1_text, nlp)
        if candidates:
            cand_scores = [
                _score_candidate(model, tokenizer, ex["text"], span2_idx, span2_text,
                                 c, max_len, dev, mixed_precision)
                for c in candidates
            ]
            return int(all(query_score >= cs for cs in cand_scores))
        return 1  # no competitors → predict True

    # Validation (labelled)
    model.eval()
    with torch.no_grad():
        val_pred = [_predict_example(ex) for ex in tqdm(ds["validation"], desc="WSC val", leave=False)]
    val_true = ds["validation"]["label"]
    valid    = np.array(val_true) != -1
    val_acc  = float(accuracy_score(np.array(val_true)[valid], np.array(val_pred)[valid]))
    print(f"  Val  acc={val_acc:.4f}")

    # Test (unlabelled)
    with torch.no_grad():
        test_pred = [_predict_example(ex) for ex in tqdm(ds["test"], desc="WSC test", leave=False)]
    test_idx = ds["test"]["idx"]
    _save_tsv(os.path.join(out_dir, cfg["tsv_name"]), test_idx, test_pred, lmap)

    res = {"task": task, "model": cfg["model_name"], "checkpoint": cfg["ckpt"],
           "timestamp": datetime.datetime.now().isoformat(),
           "validation": {"acc": val_acc}, "n_test": len(test_pred)}
    _save_json(os.path.join(out_dir, f"{task}_eval.json"), res)
    return res


def eval_record(task, cfg, dev, out_dir):
    """ReCoRD: entity-scoring — pick the candidate entity with the highest score."""
    print(f"\n{'='*55}\n[RECORD]  {cfg['tsv_name']}\n{'='*55}")
    tokenizer, model = _load_clf(cfg, dev)
    ds         = load_dataset("super_glue", "record")
    max_len    = cfg["max_length"]
    cand_batch = cfg.get("cand_batch", 8)
    mixed      = True

    def _score_entities(ex):
        entities = ex["entities"]
        if not entities:
            return 0, ""
        scores = []
        for i in range(0, len(entities), cand_batch):
            batch_ents  = entities[i : i + cand_batch]
            queries_out = [ex["query"].replace("@placeholder", e) for e in batch_ents]
            enc = tokenizer(
                [ex["passage"]] * len(batch_ents), queries_out,
                truncation=True, padding="max_length", max_length=max_len,
                return_tensors="pt",
            )
            with amp.autocast("cuda", enabled=mixed):
                logits = model(enc["input_ids"].to(dev), enc["attention_mask"].to(dev))
            scores.extend(logits[:, 1].tolist())
        best = int(np.argmax(scores))
        return best, entities[best]

    # Validation: compute EM and F1
    from collections import Counter

    def _f1(pred, gold):
        pt = pred.lower().split(); gt = gold.lower().split()
        common = Counter(pt) & Counter(gt)
        ns = sum(common.values())
        if ns == 0: return 0.0
        pr = ns / len(pt); re = ns / len(gt)
        return 2 * pr * re / (pr + re)

    total_em, total_f1, n = 0, 0.0, 0
    model.eval()
    with torch.no_grad():
        for ex in tqdm(ds["validation"], desc="ReCoRD val", leave=False):
            answers = set(ex["answers"])
            if not answers: continue
            _, pred_ent = _score_entities(ex)
            total_em += int(pred_ent in answers)
            total_f1 += max(_f1(pred_ent, a) for a in answers)
            n += 1
    val_em = total_em / max(n, 1)
    val_f1 = total_f1 / max(n, 1)
    print(f"  Val  EM={val_em:.4f}  F1={val_f1:.4f}")

    # Test: generate predictions
    test_idx_strs, test_pred_strs = [], []
    with torch.no_grad():
        for ex in tqdm(ds["test"], desc="ReCoRD test", leave=False):
            _, pred_ent = _score_entities(ex)
            test_idx_strs.append(f"{ex['idx']['passage']}-{ex['idx']['query']}")
            test_pred_strs.append(pred_ent)
    _save_tsv(os.path.join(out_dir, cfg["tsv_name"]),
              test_idx_strs, test_pred_strs, label_names=None)

    res = {"task": task, "model": cfg["model_name"], "checkpoint": cfg["ckpt"],
           "timestamp": datetime.datetime.now().isoformat(),
           "validation": {"em": val_em, "f1": val_f1}, "n_test": len(test_pred_strs)}
    _save_json(os.path.join(out_dir, f"{task}_eval.json"), res)
    return res


def eval_ax(task, cfg, dev, out_dir):
    """AX-b / AX-g: use MNLI model (3-class) collapsed to 2-class entailment."""
    hf_name = cfg["hf_name"]
    print(f"\n{'='*55}\n[{task.upper()}]  {cfg['tsv_name']}\n{'='*55}")
    tokenizer, model = _load_clf(cfg, dev)
    ds      = load_dataset("super_glue", hf_name, split="test")
    max_len = cfg["max_length"]
    lmap    = cfg["label_names"]

    tok_ds = ds.map(
        lambda ex: tokenizer(ex["premise"], ex["hypothesis"],
                             truncation=True, padding="max_length", max_length=max_len),
        batched=True,
        remove_columns=[c for c in ds.column_names
                        if c not in ("input_ids", "attention_mask", "token_type_ids", "label", "idx")],
        desc=f"Tokenizing {hf_name}",
    )

    def _collate(batch):
        return {
            "input_ids":      torch.tensor([b["input_ids"]      for b in batch], dtype=torch.long),
            "attention_mask": torch.tensor([b["attention_mask"]  for b in batch], dtype=torch.long),
            "label":          torch.tensor([b["label"]           for b in batch], dtype=torch.long),
            "idx":            [b["idx"] for b in batch],
        }

    ldr = DataLoader(tok_ds, batch_size=64, shuffle=False, num_workers=2, collate_fn=_collate)
    all_true, all_pred, all_idx = [], [], []
    model.eval()
    with torch.no_grad():
        for batch in tqdm(ldr, leave=False):
            logits = model(batch["input_ids"].to(dev), batch["attention_mask"].to(dev))
            probs  = torch.softmax(logits, dim=-1)
            # MNLI class 0 = entailment → AX label 0; neutral/contradiction → 1
            preds  = (probs[:, 0] < 0.5).long()
            all_pred.extend(preds.cpu().tolist())
            all_true.extend(batch["label"].tolist())
            all_idx.extend(batch["idx"])

    valid = np.array(all_true) != -1
    val_m = {}
    if valid.any():
        acc = float(accuracy_score(np.array(all_true)[valid], np.array(all_pred)[valid]))
        val_m["acc"] = acc
        print(f"  Test acc (labeled)={acc:.4f}  ({int(valid.sum())} examples)")
    else:
        print(f"  Test labels hidden (leaderboard task)")

    _save_tsv(os.path.join(out_dir, cfg["tsv_name"]), all_idx, all_pred, lmap)

    res = {"task": task, "model": cfg["model_name"], "checkpoint": cfg["ckpt"],
           "timestamp": datetime.datetime.now().isoformat(),
           "test_metrics": val_m, "n_test": len(all_pred)}
    _save_json(os.path.join(out_dir, f"{task}_eval.json"), res)
    return res


# ── Dispatch ──────────────────────────────────────────────────────────────────

_DISPATCH = {
    "clf":     eval_clf,
    "wic":     eval_wic,
    "copa":    eval_copa,
    "multirc": eval_multirc,
    "wsc":     eval_wsc,
    "record":  eval_record,
    "ax":      eval_ax,
}


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tasks",      nargs="+", default=list(TASKS.keys()),
                   choices=list(TASKS.keys()),
                   help="Tasks to evaluate (default: all 10)")
    p.add_argument("--gpu_id",     type=int, default=None)
    p.add_argument("--output_dir", default="roberta_large_superglue")
    p.add_argument("--ckpt_dir", default=_CKPT,
                   help=f"Directory of fine-tuned task checkpoints (default: {_CKPT}). "
                        "Point this at a hyperparameter-sweep run's checkpoint dir; "
                        "AX-b/AX-g keep using the MNLI checkpoint.")
    args = p.parse_args()

    # Redirect the per-task checkpoints to --ckpt_dir (their basenames are identical
    # across sweep runs — only the directory changes). AX tasks use the absolute MNLI
    # path, which is not under _CKPT, so they are left untouched.
    if args.ckpt_dir != _CKPT:
        for tcfg in TASKS.values():
            if tcfg["ckpt"].startswith(_CKPT):
                tcfg["ckpt"] = os.path.join(args.ckpt_dir, os.path.basename(tcfg["ckpt"]))

    set_seed(42)
    dev = device(args.gpu_id)
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Device    : {dev}")
    print(f"Output    : {args.output_dir}/")
    print(f"Checkpts  : {args.ckpt_dir}/")
    print(f"Tasks     : {args.tasks}")

    all_results = {}
    for task in args.tasks:
        cfg    = TASKS[task]
        fn     = _DISPATCH[cfg["eval_type"]]
        result = fn(task, cfg, dev, args.output_dir)
        all_results[task] = result

    print(f"\n{'='*55}\nSUMMARY\n{'='*55}")
    for task, res in all_results.items():
        val   = res.get("validation") or res.get("test_metrics") or {}
        mline = "  ".join(f"{k}={v:.4f}" for k, v in val.items() if isinstance(v, float))
        tsv   = TASKS[task]["tsv_name"]
        print(f"  {task:<10} {mline:<35}  → {args.output_dir}/{tsv}")

    print(f"\n✅ All TSVs ready in '{args.output_dir}/'")


if __name__ == "__main__":
    main()

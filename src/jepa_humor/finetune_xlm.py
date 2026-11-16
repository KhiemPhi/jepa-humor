"""
finetune_xlm.py — Extractive-QA finetune of XLM-R XL on English SQuAD v1.1, then
zero-shot XQuAD evaluation (11 languages).

STANDALONE by design: this module does NOT import or modify finetune.py, so every
GLUE/SuperGLUE task path is untouched. It only reuses utils.set_seed/device.

Pipeline
--------
Part 2 (encoder loading): load `facebook/xlm-roberta-xl`; if a JEPA LoRA adapter dir
  is given, apply it with PEFT and `merge_and_unload()` into the QA model's encoder
  before training — so the QA finetune starts from the JEPA-adapted weights.
Part 3 (QA): AutoModelForQuestionAnswering + (LoRA) finetune on SQuAD-en (3.5B won't
  full-finetune on 40GB, so LoRA the QA finetune too), span postprocessing adapted
  from HF `utils_qa`, EM/F1 per language via the XTREME/MLQA language-aware scorer.

Protocol note: XQuAD has NO training split — it is eval-only. The canonical XLM-R
setup is finetune-on-SQuAD-en -> zero-shot eval on XQuAD's 11 languages.
"""
import os
import json
import collections
import datetime

import numpy as np
import torch
from torch import amp
from tqdm import tqdm
from datasets import load_dataset
import evaluate
from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModel,
    AutoModelForQuestionAnswering,
    default_data_collator,
    get_linear_schedule_with_warmup,
)

from .utils import set_seed, device

# TF32 for the FINETUNE only (this module is imported solely by the XLM QA entry;
# the pretrain path scripts/pretrain.py -> train_ssl.py never imports it, so its
# matmuls stay pure fp32). "fast fp32": tensor-core matmul (10-bit mantissa) with
# fp32 accumulation, everything else full fp32 -> ~3-4x faster, ~fp32 numerics.
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

XQUAD_LANGS = ["en", "es", "de", "el", "ru", "tr", "ar", "vi", "th", "zh", "hi"]


# ---------------------------------------------------------------------------
# Part 2: build the QA model, optionally merging a JEPA LoRA adapter first
# ---------------------------------------------------------------------------
def build_qa_model(cfg, jepa_adapter, dev, apply_lora=True, move=True):
    model_name = cfg["model_name"]
    qa_model = AutoModelForQuestionAnswering.from_pretrained(model_name)

    # Merge the JEPA-pretrained LoRA adapter into the encoder (Part 2).
    if jepa_adapter and jepa_adapter != "fresh":
        from peft import PeftModel
        print(f"[finetune_xlm] merging JEPA adapter from {jepa_adapter} into encoder")
        enc = AutoModel.from_pretrained(model_name)
        enc = PeftModel.from_pretrained(enc, jepa_adapter).merge_and_unload()
        encoder = getattr(qa_model, qa_model.base_model_prefix)  # qa_model.roberta
        missing, unexpected = encoder.load_state_dict(enc.state_dict(), strict=False)
        unexpected = [k for k in unexpected if not k.startswith("pooler.")]
        print(f"[finetune_xlm] encoder load: {len(missing)} missing, {len(unexpected)} unexpected")
        del enc
    else:
        print("[finetune_xlm] fresh encoder (no JEPA adapter)")

    # QA finetune with LoRA (gated; default on for the 3.5B model). The newly
    # initialised span head (qa_outputs) is fully trained via modules_to_save.
    # apply_lora=False when reloading a saved adapter in eval_only (caller attaches it).
    if apply_lora and cfg.get("use_lora", True):
        from peft import get_peft_model, LoraConfig, TaskType
        lcfg = LoraConfig(
            task_type=TaskType.QUESTION_ANS,
            r=cfg.get("lora_r", 16),
            lora_alpha=cfg.get("lora_alpha", 32),
            lora_dropout=cfg.get("lora_dropout", 0.05),
            target_modules=cfg.get("lora_target_modules", ["query", "value"]),
            modules_to_save=["qa_outputs"],
            bias="none",
        )
        qa_model = get_peft_model(qa_model, lcfg)
        qa_model.print_trainable_parameters()

    return qa_model.to(dev) if move else qa_model


# ---------------------------------------------------------------------------
# SQuAD/XQuAD feature preparation (adapted from HF run_qa.py)
# ---------------------------------------------------------------------------
def _prepare_train_features(examples, tokenizer, max_length, doc_stride):
    q_key, c_key = "question", "context"
    examples[q_key] = [q.lstrip() for q in examples[q_key]]
    tok = tokenizer(
        examples[q_key], examples[c_key],
        truncation="only_second", max_length=max_length, stride=doc_stride,
        return_overflowing_tokens=True, return_offsets_mapping=True, padding="max_length",
    )
    sample_map = tok.pop("overflow_to_sample_mapping")
    offset_map = tok.pop("offset_mapping")
    tok["start_positions"], tok["end_positions"] = [], []

    for i, offsets in enumerate(offset_map):
        input_ids = tok["input_ids"][i]
        cls_index = input_ids.index(tokenizer.cls_token_id)
        seq_ids = tok.sequence_ids(i)
        sample_idx = sample_map[i]
        answers = examples["answers"][sample_idx]
        if len(answers["answer_start"]) == 0:
            tok["start_positions"].append(cls_index)
            tok["end_positions"].append(cls_index)
            continue
        start_char = answers["answer_start"][0]
        end_char = start_char + len(answers["text"][0])
        # span of the context in the tokenized pair
        t = 0
        while seq_ids[t] != 1:
            t += 1
        ctx_start = t
        while t < len(seq_ids) and seq_ids[t] == 1:
            t += 1
        ctx_end = t - 1
        if not (offsets[ctx_start][0] <= start_char and offsets[ctx_end][1] >= end_char):
            tok["start_positions"].append(cls_index)
            tok["end_positions"].append(cls_index)
        else:
            s = ctx_start
            while s <= ctx_end and offsets[s][0] <= start_char:
                s += 1
            tok["start_positions"].append(s - 1)
            e = ctx_end
            while e >= ctx_start and offsets[e][1] >= end_char:
                e -= 1
            tok["end_positions"].append(e + 1)
    return tok


def _prepare_val_features(examples, tokenizer, max_length, doc_stride):
    q_key, c_key = "question", "context"
    examples[q_key] = [q.lstrip() for q in examples[q_key]]
    tok = tokenizer(
        examples[q_key], examples[c_key],
        truncation="only_second", max_length=max_length, stride=doc_stride,
        return_overflowing_tokens=True, return_offsets_mapping=True, padding="max_length",
    )
    sample_map = tok.pop("overflow_to_sample_mapping")
    tok["example_id"] = []
    for i in range(len(tok["input_ids"])):
        seq_ids = tok.sequence_ids(i)
        sample_idx = sample_map[i]
        tok["example_id"].append(examples["id"][sample_idx])
        # keep offsets only for context tokens (None elsewhere) for postprocessing
        tok["offset_mapping"][i] = [
            o if seq_ids[k] == 1 else None
            for k, o in enumerate(tok["offset_mapping"][i])
        ]
    return tok


def _postprocess(examples, features, start_logits, end_logits, n_best=20, max_answer_len=30):
    """Map start/end logits -> best text span per example (SQuAD v1, no null)."""
    ex_id_to_idx = {ex_id: i for i, ex_id in enumerate(examples["id"])}
    features_per_example = collections.defaultdict(list)
    for i, ex_id in enumerate(features["example_id"]):
        features_per_example[ex_id_to_idx[ex_id]].append(i)

    predictions = {}
    for ex_idx in range(len(examples["id"])):
        ex_id = examples["id"][ex_idx]
        context = examples["context"][ex_idx]
        candidates = []
        for feat_idx in features_per_example[ex_idx]:
            s_logit = start_logits[feat_idx]
            e_logit = end_logits[feat_idx]
            offsets = features["offset_mapping"][feat_idx]
            s_idx = np.argsort(s_logit)[-1: -n_best - 1: -1].tolist()
            e_idx = np.argsort(e_logit)[-1: -n_best - 1: -1].tolist()
            for s in s_idx:
                for e in e_idx:
                    if s >= len(offsets) or e >= len(offsets):
                        continue
                    if offsets[s] is None or offsets[e] is None:
                        continue
                    if e < s or e - s + 1 > max_answer_len:
                        continue
                    candidates.append({
                        "score": float(s_logit[s] + e_logit[e]),
                        "text": context[offsets[s][0]: offsets[e][1]],
                    })
        if candidates:
            best = max(candidates, key=lambda x: x["score"])["text"]
        else:
            best = ""
        predictions[ex_id] = best
    return predictions


# ---------------------------------------------------------------------------
# Training (SQuAD-en) and evaluation (SQuAD-en dev + XQuAD langs)
# ---------------------------------------------------------------------------
def _run_inference(model, features, dev, batch_size, mixed_precision):
    model.eval()
    n = len(features["input_ids"])
    input_ids = torch.tensor(features["input_ids"])
    attn = torch.tensor(features["attention_mask"])
    all_start, all_end = [], []
    with torch.no_grad():
        for i in range(0, n, batch_size):
            ids = input_ids[i:i + batch_size].to(dev)
            am = attn[i:i + batch_size].to(dev)
            with amp.autocast("cuda", enabled=mixed_precision):
                out = model(input_ids=ids, attention_mask=am)
            all_start.append(out.start_logits.float().cpu().numpy())
            all_end.append(out.end_logits.float().cpu().numpy())
    return np.concatenate(all_start), np.concatenate(all_end)


# ---------------------------------------------------------------------------
# XTREME / MLQA-v1 answer scoring (language-aware). Faithful port of the official
# MLQA evaluation script (Lewis et al. 2019), which XTREME uses for XQuAD:
#   - Chinese (zh): mixed_segmentation -> character-level tokens
#   - language-specific article removal (en/es/de/vi/ar); zh/hi none
#   - full-Unicode punctuation stripping (not just ASCII)
# This replaces evaluate.load("squad") whose English-whitespace tokenisation
# mis-scores zh (and, via ASCII-only punct/articles, ar/es/de/vi/hi).
# ---------------------------------------------------------------------------
import re as _re
import string as _string
import unicodedata as _ud

_PUNCT = {chr(i) for i in range(0x110000)
          if _ud.category(chr(i)).startswith('P')} | set(_string.punctuation)
_WHITESPACE_LANGS = ['en', 'es', 'hi', 'vi', 'de', 'ar', 'el', 'ru', 'tr', 'th']
_MIXED_SEG_LANGS = ['zh']


def _xtreme_whitespace_tokenize(text):
    return text.split()


def _xtreme_mixed_segmentation(text):
    segs, tmp = [], ""
    for ch in text:
        if _re.search(r'[一-龥]', ch) or ch in _PUNCT:
            if tmp:
                segs.extend(_xtreme_whitespace_tokenize(tmp)); tmp = ""
            segs.append(ch)
        else:
            tmp += ch
    if tmp:
        segs.extend(_xtreme_whitespace_tokenize(tmp))
    return segs


def _xtreme_normalize(s, lang):
    def remove_articles(text, lang):
        if lang == 'en':
            return _re.sub(r'\b(a|an|the)\b', ' ', text)
        if lang == 'es':
            return _re.sub(r'\b(un|una|unos|unas|el|la|los|las)\b', ' ', text)
        if lang == 'vi':
            return _re.sub(r'\b(của|là|cái|chiếc|những)\b', ' ', text)
        if lang == 'de':
            return _re.sub(r'\b(ein|eine|einen|einem|eines|einer|der|die|das|den|dem|des)\b', ' ', text)
        if lang == 'ar':
            return _re.sub(r'\sال^|ال', ' ', text)
        return text  # hi, zh, el, ru, tr, th -> no article list
    def white_space_fix(text, lang):
        if lang in _WHITESPACE_LANGS:
            toks = _xtreme_whitespace_tokenize(text)
        elif lang in _MIXED_SEG_LANGS:
            toks = _xtreme_mixed_segmentation(text)
        else:
            toks = _xtreme_whitespace_tokenize(text)
        return ' '.join(t for t in toks if t.strip())
    def remove_punc(text):
        return ''.join(ch for ch in text if ch not in _PUNCT)
    return white_space_fix(remove_articles(remove_punc(s.lower()), lang), lang)


def _xtreme_tokens(s, lang):
    if lang in _MIXED_SEG_LANGS:
        return _xtreme_mixed_segmentation(_xtreme_normalize(s, lang))
    return _xtreme_normalize(s, lang).split()


def _xtreme_f1(pred, golds, lang):
    best = 0.0
    p_toks = _xtreme_tokens(pred, lang)
    for g in golds:
        g_toks = _xtreme_tokens(g, lang)
        common = collections.Counter(p_toks) & collections.Counter(g_toks)
        n_same = sum(common.values())
        if not p_toks or not g_toks:
            f1 = float(p_toks == g_toks)
        elif n_same == 0:
            f1 = 0.0
        else:
            prec = n_same / len(p_toks); rec = n_same / len(g_toks)
            f1 = 2 * prec * rec / (prec + rec)
        best = max(best, f1)
    return best


def _xtreme_em(pred, golds, lang):
    pn = _xtreme_normalize(pred, lang)
    return float(any(pn == _xtreme_normalize(g, lang) for g in golds))


def _xtreme_score(preds, references, lang):
    """preds: {id: text}; references: [{id, answers:{text:[...]}}] -> (EM, F1) in %."""
    em = f1 = 0.0
    for ref in references:
        golds = ref["answers"]["text"] or [""]
        p = preds.get(ref["id"], "")
        em += _xtreme_em(p, golds, lang)
        f1 += _xtreme_f1(p, golds, lang)
    n = max(1, len(references))
    return 100.0 * em / n, 100.0 * f1 / n


def _evaluate(model, tokenizer, dev, examples, cfg, tag):
    max_length = cfg.get("max_length", 384)
    doc_stride = cfg.get("doc_stride", 128)
    features = examples.map(
        lambda x: _prepare_val_features(x, tokenizer, max_length, doc_stride),
        batched=True, remove_columns=examples.column_names,
    )
    feat_dict = features.to_dict()
    start_logits, end_logits = _run_inference(
        model, feat_dict, dev, cfg.get("eval_batch_size", 16), cfg.get("mixed_precision", True))
    preds = _postprocess(examples.to_dict(), feat_dict, start_logits, end_logits,
                         n_best=cfg.get("n_best_size", 20),
                         max_answer_len=cfg.get("max_answer_length", 30))
    references = [{"id": ex["id"], "answers": ex["answers"]} for ex in examples]
    if cfg.get("use_xtreme_metric", True):
        # XTREME/MLQA language-aware scorer: "xquad_zh" -> zh, "squad_en_dev" -> en
        lang = tag.split("_")[1] if tag.startswith("xquad_") else "en"
        em, f1 = _xtreme_score(preds, references, lang)
        print(f"[eval:{tag}] EM={em:.2f}  F1={f1:.2f}  (XTREME/{lang})")
    else:
        # plain SQuAD v1.1 metric (English whitespace tokenisation, no lang handling)
        metric = evaluate.load("squad")
        predictions = [{"id": k, "prediction_text": v} for k, v in preds.items()]
        res = metric.compute(predictions=predictions, references=references)
        em, f1 = res["exact_match"], res["f1"]
        print(f"[eval:{tag}] EM={em:.2f}  F1={f1:.2f}  (squad)")
    return {"exact_match": em, "f1": f1}


def _load_eval_sets(cfg):
    """Load the eval example sets once (SQuAD-en dev + the XQuAD languages)."""
    sets = {}
    if cfg.get("eval_squad_dev", True):
        d = load_dataset(cfg.get("train_dataset", "rajpurkar/squad"), split="validation")
        if cfg.get("max_eval_samples"):
            d = d.select(range(cfg["max_eval_samples"]))
        sets["squad_en_dev"] = d
    for lang in cfg.get("xquad_langs", XQUAD_LANGS):
        sets[f"xquad_{lang}"] = load_dataset(
            cfg.get("xquad_dataset", "google/xquad"), f"xquad.{lang}", split="validation")
    return sets


def _eval_all(model, tokenizer, dev, cfg, eval_sets):
    """Evaluate on every eval set; return {tag: {exact_match, f1}, xquad_avg_f1}."""
    scores = {}
    for tag, ex in eval_sets.items():
        scores[tag] = _evaluate(model, tokenizer, dev, ex, cfg, tag)
    xq = [v["f1"] for k, v in scores.items() if k.startswith("xquad_")]
    if xq:
        scores["xquad_avg_f1"] = sum(xq) / len(xq)
        print(f"[eval] XQuAD mean F1 over {len(xq)} langs = {scores['xquad_avg_f1']:.2f}")
    return scores


def _write_results(results, path):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(results, f, indent=2)


def _train_squad_fsdp(cfg, accelerator, model, tokenizer, eval_sets, results, results_path, eval_base):
    """Full-parameter SQuAD-en finetune under FSDP (via accelerate), with a per-epoch
    XQuAD eval done on the main process from the gathered full model."""
    max_length = cfg.get("max_length", 384)
    doc_stride = cfg.get("doc_stride", 128)
    squad = load_dataset(cfg.get("train_dataset", "rajpurkar/squad"))
    train_examples = squad["train"]
    if cfg.get("max_train_samples"):
        train_examples = train_examples.select(range(cfg["max_train_samples"]))
    with accelerator.main_process_first():   # tokenize once on main; others reuse cache
        train_features = train_examples.map(
            lambda x: _prepare_train_features(x, tokenizer, max_length, doc_stride),
            batched=True, remove_columns=train_examples.column_names,
        )
    train_features.set_format(type="torch",
                              columns=["input_ids", "attention_mask", "start_positions", "end_positions"])
    loader = torch.utils.data.DataLoader(
        train_features, batch_size=cfg.get("batch_size", 4), shuffle=True,
        num_workers=cfg.get("num_workers", 4), collate_fn=default_data_collator)

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=cfg.get("lr", 1.0e-5),
                            weight_decay=cfg.get("weight_decay", 0.01))
    model, opt, loader = accelerator.prepare(model, opt, loader)   # FSDP-shards the model

    epochs = cfg.get("epochs", 10)
    total_steps = epochs * len(loader)
    scheduler = get_linear_schedule_with_warmup(
        opt, int(total_steps * cfg.get("lr_warmup_ratio", 0.06)), total_steps)
    max_grad_norm = cfg.get("max_grad_norm", 1.0)

    # early stopping on XQuAD avg F1 (disabled when patience <= 0). Decision is made on the
    # main process then broadcast so all FSDP ranks break together (avoids a collective deadlock).
    from accelerate.utils import broadcast
    patience = int(cfg.get("early_stop_patience", 0) or 0)
    best_f1, epochs_no_improve = -1.0, 0
    for epoch in range(1, epochs + 1):
        model.train()
        running, steps = 0.0, 0
        pbar = tqdm(loader, desc=f"SQuAD-ft {epoch}/{epochs}", leave=False,
                    disable=not accelerator.is_main_process)
        for batch in pbar:
            opt.zero_grad(set_to_none=True)
            out = model(**batch)                       # accelerate autocasts under mixed_precision
            loss = out.loss
            accelerator.backward(loss)
            if max_grad_norm and max_grad_norm > 0:
                accelerator.clip_grad_norm_(model.parameters(), max_grad_norm)
            opt.step()
            scheduler.step()
            running += float(loss.item()); steps += 1
            if accelerator.is_main_process:
                pbar.set_postfix(loss=f"{running/steps:.4f}")
        train_loss = running / max(1, steps)
        accelerator.print(f"[SQuAD-ft] epoch {epoch}: loss={train_loss:.4f} -> gather + eval XQuAD")

        # gather the full (unsharded) model -> evaluate on the main process only
        full_state = accelerator.get_state_dict(model)   # collective; full state on main
        accelerator.wait_for_everyone()
        if accelerator.is_main_process:
            torch.cuda.empty_cache()
            eval_base.load_state_dict(full_state, strict=False)
            eval_base.to(accelerator.device); eval_base.eval()  # fp32 weights; autocast in _run_inference gives true mixed precision
            scores = _eval_all(eval_base, tokenizer, accelerator.device, cfg, eval_sets)
            eval_base.to("cpu"); torch.cuda.empty_cache()
            results["epochs"].append({"epoch": epoch, "train_loss": train_loss, "scores": scores})
            _write_results(results, results_path)
            accelerator.print(f"[SQuAD-ft] epoch {epoch}: results -> {results_path}")

        # early-stop decision on the main process, then broadcast to all ranks
        stop = torch.zeros(1, device=accelerator.device)
        if accelerator.is_main_process and patience > 0:
            cur = scores.get("xquad_avg_f1", -1.0)
            if cur > best_f1:
                best_f1, epochs_no_improve = cur, 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    stop.fill_(1.0)
                    accelerator.print(f"[SQuAD-ft] early stop: XQuAD avg F1 did not improve for "
                                      f"{patience} epoch(s) (best={best_f1:.2f}); stopping at epoch {epoch}.")
        accelerator.wait_for_everyone()
        if broadcast(stop).item() > 0:
            break
    return model


def finetune_xlm(cfg, jepa_adapter="fresh", eval_only=False, model_ckpt=None):
    set_seed(cfg.get("seed", 42))
    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"])
    results_dir = cfg.get("results_dir", "results")
    results_path = os.path.join(results_dir, f"{cfg.get('experiment_name', 'xlm_qa')}_xquad.json")
    results = {"model": cfg["model_name"], "jepa_adapter": jepa_adapter,
               "config": {k: cfg.get(k) for k in ("lr", "batch_size", "epochs", "max_length", "seed")},
               "epochs": []}

    if eval_only:
        # single-process eval (no FSDP): rebuild the base (+JEPA merge), load a saved
        # full-param state dict if given, then evaluate.
        dev = device(cfg.get("gpu_id", None))
        eval_sets = _load_eval_sets(cfg)
        model = build_qa_model(cfg, jepa_adapter, dev, apply_lora=False, move=True)
        if model_ckpt:
            print(f"[finetune_xlm] loading QA weights from {model_ckpt}")
            model.load_state_dict(torch.load(model_ckpt, map_location="cpu"), strict=False)
        model.eval()
        scores = _eval_all(model, tokenizer, dev, cfg, eval_sets)
        results["epochs"].append({"epoch": "eval_only", "scores": scores})
        results["timestamp"] = datetime.datetime.now().isoformat()
        _write_results(results, results_path)
        return results

    # --- full-parameter FSDP finetune (launched via `accelerate launch`) ---
    from accelerate import Accelerator
    accelerator = Accelerator(mixed_precision="fp16" if cfg.get("mixed_precision", True) else "no")
    if accelerator.is_main_process:
        print(f"[finetune_xlm] FSDP world_size={accelerator.num_processes}")
    eval_sets = _load_eval_sets(cfg)
    # eval reference model (main process only), built from config so no re-download
    eval_base = None
    if accelerator.is_main_process:
        eval_base = AutoModelForQuestionAnswering.from_config(AutoConfig.from_pretrained(cfg["model_name"]))
    # full-param training model on CPU (accelerate shards to GPUs); merges the JEPA adapter
    model = build_qa_model(cfg, jepa_adapter, dev=None,
                           apply_lora=cfg.get("use_lora", False), move=False)
    model = _train_squad_fsdp(cfg, accelerator, model, tokenizer, eval_sets, results, results_path, eval_base)

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        results["timestamp"] = datetime.datetime.now().isoformat()
        _write_results(results, results_path)
        print(f"[finetune_xlm] final results at {results_path}")
    return results

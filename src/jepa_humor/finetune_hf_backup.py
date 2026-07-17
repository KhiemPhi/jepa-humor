"""HuggingFace-Trainer GLUE finetune (AutoModelForSequenceClassification).

Drop-in replacement for the previous custom trainer (backed up in
finetune_backup_custom.py). Same entry point `finetune(cfg, encoder_ckpt, ...)`.

Design: applies ALL fields from the finetune config yml —
  epochs, max_length, lr, batch_size (lr/bs set per sweep combo),
  weight_decay, lr_warmup_ratio, lr_scheduler, adam_betas, adam_eps,
  max_grad_norm (0.0 => clipping off), mixed_precision (fp16),
  hidden_dropout, attention_dropout, head_dropout. HF defaults for the rest.
For rte/qnli/mnli/sst2/qqp a LOCAL labeled test set (src/jepa_humor/test_ann.json)
is also evaluated each epoch, and we save best_val.pt AND best_test.pt separately.
Other tasks (cola/mrpc/stsb) save only best_val.pt. WNLI -> finetune_wnli.
Tested against transformers 4.46.
"""
import os
import json
import math
import datetime

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from datasets import load_dataset
from transformers import (
    AutoTokenizer, AutoConfig, AutoModelForSequenceClassification,
    TrainingArguments, Trainer, DataCollatorWithPadding, TrainerCallback,
)
from sklearn.metrics import matthews_corrcoef, f1_score, accuracy_score
from scipy.stats import pearsonr, spearmanr

from .utils import set_seed, device, load_encoder_state, save_ckpt
from .data import load_test_set_from_json

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

TASKS = {
    "cola": dict(text_keys=["sentence"],               num_labels=2, metric="mcc"),
    "sst2": dict(text_keys=["sentence"],               num_labels=2, metric="acc"),
    "mrpc": dict(text_keys=["sentence1", "sentence2"], num_labels=2, metric="f1"),
    "stsb": dict(text_keys=["sentence1", "sentence2"], num_labels=1, metric="spearmanr"),
    "qqp":  dict(text_keys=["question1", "question2"], num_labels=2, metric="f1"),
    "mnli": dict(text_keys=["premise", "hypothesis"],  num_labels=3, metric="acc"),
    "qnli": dict(text_keys=["question", "sentence"],   num_labels=2, metric="acc"),
    "rte":  dict(text_keys=["sentence1", "sentence2"], num_labels=2, metric="acc"),
}
# tasks with a local labeled test set -> also track best_test
LOCAL_TEST = {"rte", "qnli", "mnli", "sst2", "qqp"}


# ── helpers retained for finetune_wnli.py (keeps its own custom trainer) ────────
def _save_training_plots(epochs, acc_curves, loss_hist, lr_hist, title, save_path):
    panels = [("acc", acc_curves), ("loss", loss_hist), ("lr", lr_hist)]
    active = [(k, v) for k, v in panels if v]
    fig, axes = plt.subplots(len(active), 1, figsize=(8, 4 * len(active)), squeeze=False)
    for i, (kind, data) in enumerate(active):
        ax = axes[i][0]
        if kind == "acc":
            for label, accs in data.items():
                ax.plot(epochs[:len(accs)], accs, marker="o", markersize=3, label=label)
            ax.set_ylabel("Accuracy"); ax.set_title(title); ax.legend()
        elif kind == "loss":
            ax.plot(epochs[:len(data)], data, marker="o", markersize=3, color="tab:orange")
            ax.set_ylabel("Train Loss"); ax.set_title("Train Loss")
        elif kind == "lr":
            ax.plot(epochs[:len(data)], data, marker="o", markersize=3, color="tab:green")
            ax.set_ylabel("Learning Rate"); ax.set_title("Learning Rate")
        ax.set_xlabel("Epoch"); ax.grid(True, alpha=0.3)
    fig.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def _build_finetune_lr_scheduler(optimizer, num_training_steps, warmup_ratio, schedule_type):
    schedule_type = (schedule_type or "cosine").lower()
    warmup_steps = max(1, int(num_training_steps * warmup_ratio))
    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / max(1, num_training_steps - warmup_steps)
        if schedule_type == "linear":
            return max(0.0, 1.0 - progress)
        return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def save_finetune_results(results_dir, task, experiment, data):
    os.makedirs(results_dir, exist_ok=True)
    fname = f"{experiment}_{task}.json" if experiment else f"{task}.json"
    path = os.path.join(results_dir, fname)
    data = {**data, "task": task, "experiment": experiment,
            "timestamp": datetime.datetime.now().isoformat()}
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Saved results to {path}")


def _load_encoder(model, ckpt_path):
    """Load just the transformer encoder (base_model) from any of our ckpt formats:
    JEPA pretrain (encoder_state_dict), HF finetune ('hf' / base-prefixed model_state_dict),
    or the old custom-head checkpoint (encoder.* in model_state_dict)."""
    st = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    base = model.base_model
    prefix = model.base_model_prefix + "."  # e.g. "roberta."
    if "encoder_state_dict" in st:                       # JEPA pretrain
        sd = st["encoder_state_dict"]
    elif "model_state_dict" in st:
        msd = st["model_state_dict"]
        if any(k.startswith(prefix) for k in msd):       # HF finetune checkpoint
            sd = {k[len(prefix):]: v for k, v in msd.items() if k.startswith(prefix)}
        elif any(k.startswith("encoder.") for k in msd): # old custom-head checkpoint
            sd = {k[len("encoder."):]: v for k, v in msd.items() if k.startswith("encoder.")}
        else:
            sd = msd
    else:
        sd = st
    missing, unexpected = base.load_state_dict(sd, strict=False)
    print(f"  encoder loaded (missing={len(missing)}, unexpected={len(unexpected)})")


def _exp_prefix(cfg):
    exp = cfg.get("experiment_name", "")
    slug = cfg.get("model_name", "").replace("-", "_").replace("/", "_")
    return f"{exp}_{slug}" if exp else slug


def _metrics_fn(task, is_reg):
    def compute(ep):
        logits, labels = ep
        labels = np.array(labels)
        if is_reg:
            preds = np.array(logits).reshape(-1)
            pear = float(pearsonr(preds, labels)[0]); spear = float(spearmanr(preds, labels)[0])
            return {"acc": pear, "mcc": pear, "f1": spear, "pearson": pear, "spearman": spear, "score": pear}
        preds = np.argmax(logits, axis=1)
        acc = float(accuracy_score(labels, preds)); mcc = float(matthews_corrcoef(labels, preds))
        f1 = float(f1_score(labels, preds, average="binary" if TASKS[task]["num_labels"] == 2 else "macro"))
        return {"acc": acc, "mcc": mcc, "f1": f1, "score": {"acc": acc, "mcc": mcc, "f1": f1}.get(TASKS[task]["metric"], acc)}
    return compute


class _BestSaver(TrainerCallback):
    """Saves best_<split>.pt for each eval split as its 'score' metric improves."""
    def __init__(self, prefix, ckpt_dir, splits, num_labels, task):
        self.prefix, self.ckpt_dir, self.task, self.num_labels = prefix, ckpt_dir, task, num_labels
        self.best = {s: -1e9 for s in splits}
        self.best_epoch = {s: -1 for s in splits}
    def on_evaluate(self, args, state, control, metrics=None, **kw):
        model = kw["model"]
        for s in self.best:
            sc = (metrics or {}).get(f"eval_{s}_score")
            if sc is not None and sc > self.best[s]:
                self.best[s] = sc
                self.best_epoch[s] = int(round(state.epoch or 0))
                path = os.path.join(self.ckpt_dir, f"{self.prefix}_best_{s}.pt")
                save_ckpt({"model_state_dict": model.state_dict(), "hf": True,
                           "num_labels": self.num_labels, "task": self.task, "split": s, "score": sc}, path)
                print(f"  ✓ best_{s} ({sc:.4f}) -> {path}")


def finetune(cfg, encoder_ckpt: str, eval_only: bool = False, model_ckpt: str = None):
    gpu_id = cfg.get("gpu_id", None)
    if gpu_id is not None and "CUDA_VISIBLE_DEVICES" not in os.environ:
        # Fallback when not launched with a shell-level CUDA_VISIBLE_DEVICES (e.g. standalone).
        # In the sweep the shell sets it before Python starts (reliable); don't override that.
        os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    set_seed(cfg["seed"])
    print(f"CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES', 'all')}")

    task = cfg.get("dataset_config")
    if task not in TASKS:
        raise ValueError(f"HF finetune supports {list(TASKS)}; got dataset_config={task!r}")
    meta = TASKS[task]
    num_labels = meta["num_labels"]
    is_reg = num_labels == 1
    max_length = cfg.get("max_length", 512)
    text_keys = cfg.get("text_keys", meta["text_keys"])

    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"])
    mcfg = AutoConfig.from_pretrained(cfg["model_name"], num_labels=num_labels)
    if is_reg:
        mcfg.problem_type = "regression"
    # dropout settings from config (BERT/RoBERTa attribute names)
    if "hidden_dropout" in cfg:
        mcfg.hidden_dropout_prob = cfg["hidden_dropout"]
    if "attention_dropout" in cfg:
        mcfg.attention_probs_dropout_prob = cfg["attention_dropout"]
    if "head_dropout" in cfg:
        mcfg.classifier_dropout = cfg["head_dropout"]  # head; falls back to hidden_dropout if None
    model = AutoModelForSequenceClassification.from_pretrained(cfg["model_name"], config=mcfg)
    if not eval_only and encoder_ckpt and encoder_ckpt != "fresh":
        print(f"Loading pretrained encoder from {encoder_ckpt}")
        _load_encoder(model, encoder_ckpt)
    if eval_only and model_ckpt:
        st = torch.load(model_ckpt, map_location="cpu")
        model.load_state_dict(st.get("model_state_dict", st))

    # ── data ────────────────────────────────────────────────────────────────────
    def tok(batch):
        a = [batch[text_keys[0]]] + ([batch[text_keys[1]]] if len(text_keys) > 1 else [])
        return tokenizer(*a, truncation=True, max_length=max_length)
    def prep(ds):
        return ds.map(tok, batched=True, remove_columns=[c for c in text_keys if c in ds.column_names])

    raw = load_dataset(cfg["dataset"], task)
    val_key = "validation_matched" if task == "mnli" else "validation"
    eval_sets = {"val": prep(raw[val_key])}
    if task in LOCAL_TEST:
        lt = load_test_set_from_json(task)
        if lt and lt.get("test") is not None:
            eval_sets["test"] = prep(lt["test"])
            print(f"Local test set: {len(eval_sets['test'])} examples")

    prefix = _exp_prefix(cfg)
    ckpt_dir = cfg.get("ckpt_dir", "checkpoints/finetune")
    results_dir = cfg.get("results_dir", "results")
    saver = _BestSaver(prefix, ckpt_dir, list(eval_sets), num_labels, task)

    # ── apply all config fields (epochs/bs/lr/max_len/dropouts above) ───────────
    adam_betas = cfg.get("adam_betas", [0.9, 0.999])
    args = TrainingArguments(
        output_dir=os.path.join(ckpt_dir, "_hf_tmp", task),
        num_train_epochs=cfg["epochs"],
        per_device_train_batch_size=cfg["batch_size"],
        per_device_eval_batch_size=cfg["batch_size"],
        learning_rate=cfg["lr"],
        weight_decay=cfg.get("weight_decay", 0.1),
        warmup_ratio=cfg.get("lr_warmup_ratio", 0.06),
        lr_scheduler_type=cfg.get("lr_scheduler", "linear"),
        adam_beta1=adam_betas[0],
        adam_beta2=adam_betas[1],
        adam_epsilon=cfg.get("adam_eps", 1e-8),
        max_grad_norm=cfg.get("max_grad_norm", 1.0),   # 0.0 => clipping disabled (HF: >0 guard)
        fp16=cfg.get("mixed_precision", False),
        dataloader_num_workers=cfg.get("num_workers", 0),
        eval_strategy="epoch", save_strategy="no", logging_strategy="epoch",
        report_to=[], seed=cfg["seed"],
    )
    trainer = Trainer(model=model, args=args, train_dataset=prep(raw["train"]),
                      eval_dataset=eval_sets, processing_class=tokenizer,
                      data_collator=DataCollatorWithPadding(tokenizer),
                      compute_metrics=_metrics_fn(task, is_reg), callbacks=[saver])

    if eval_only:
        return model

    trainer.train()

    # ── per-epoch log (val + local test) + best scores -> results JSON ──────────
    epochs_log = {}
    for r in trainer.state.log_history:
        e = int(round(r.get("epoch", 0)))
        for s in eval_sets:
            if f"eval_{s}_score" in r:
                m = {k: r.get(f"eval_{s}_{k}") for k in ("acc", "mcc", "f1") if f"eval_{s}_{k}" in r}
                slot = "matched" if s == "val" else s
                epochs_log.setdefault(e, {}).setdefault("validation" if s == "val" else "test", {})
                if s == "val":
                    epochs_log[e]["validation"] = {"matched": m}
                else:
                    epochs_log[e]["test"] = m
    save_finetune_results(results_dir, task, prefix, {
        "model": cfg["model_name"],
        "epochs": [{"epoch": e, **v} for e, v in sorted(epochs_log.items())],
        "best_val": {"score": saver.best.get("val"), "epoch": saver.best_epoch.get("val")},
        "best_test": ({"score": saver.best.get("test"), "epoch": saver.best_epoch.get("test")}
                      if "test" in eval_sets else None),
    })
    print(f"Done. best_val={saver.best.get('val')}" +
          (f" best_test={saver.best.get('test')}" if "test" in eval_sets else ""))
    return model

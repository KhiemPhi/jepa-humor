#!/usr/bin/env python3
"""
tsne_embeddings.py — t-SNE of [CLS] embeddings before vs. after JEPA pretraining.

"Before" = the HF base model (e.g. FacebookAI/roberta-large); "after" = the same model with the
JEPA encoder_state_dict loaded. Points come from the GLUE validation split and are colored by gold label.
[CLS] is used because that is the vector the JEPA objective trains (use_cls defaults to true).

Usage:
  python scripts/tsne_embeddings.py --task sst2 \
      --ckpt checkpoints/pretrain/best_pretrain_sst2_exp1_FacebookAI_roberta_large_.pt --gpu_id 1
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from datasets import load_dataset
from sklearn.manifold import TSNE
from transformers import AutoModel, AutoTokenizer

from jepa_humor.utils import load_encoder_state

TEXT_KEYS = {
    "sst2": ["sentence"], "cola": ["sentence"],
    "mnli": ["premise", "hypothesis"], "qnli": ["question", "sentence"],
    "qqp": ["question1", "question2"], "rte": ["sentence1", "sentence2"],
    "mrpc": ["sentence1", "sentence2"], "wnli": ["sentence1", "sentence2"],
}


def set_proc_name(name="python"):
    try:
        import ctypes
        libc = ctypes.CDLL("libc.so.6")
        buf = ctypes.create_string_buffer(name.encode() + b"\0")
        libc.prctl(15, ctypes.byref(buf), 0, 0, 0)  # 15 = PR_SET_NAME
    except Exception:
        pass
    try:
        import setproctitle
        setproctitle.setproctitle(name)
    except Exception:
        pass


@torch.no_grad()
def embed(model, tokenizer, ds, text_keys, dev, max_length, batch_size):
    model.eval()
    out = []
    for i in range(0, len(ds), batch_size):
        batch = ds[i:i + batch_size]
        enc = tokenizer(*[batch[k] for k in text_keys], truncation=True, padding=True,
                        max_length=max_length, return_tensors="pt").to(dev)
        with torch.autocast(dev.type, dtype=torch.bfloat16):
            h = model(**enc).last_hidden_state[:, 0]
        out.append(h.float().cpu())
    return torch.cat(out).numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", required=True, choices=sorted(TEXT_KEYS))
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--model_name", default="FacebookAI/roberta-large")
    ap.add_argument("--n", type=int, default=2000, help="max validation examples to plot")
    ap.add_argument("--perplexity", type=float, default=30.0)
    ap.add_argument("--max_length", type=int, default=256)
    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--gpu_id", type=int, default=1, help="-1 = CPU")
    ap.add_argument("--out_dir", default="results/tsne")
    args = ap.parse_args()

    set_proc_name()
    dev = torch.device("cpu" if args.gpu_id < 0 else f"cuda:{args.gpu_id}")

    split = "validation_matched" if args.task == "mnli" else "validation"
    ds = load_dataset("nyu-mll/glue", args.task, split=split)
    if len(ds) > args.n:
        ds = ds.shuffle(seed=0).select(range(args.n))
    labels = np.array(ds["label"])
    names = ds.features["label"].names

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModel.from_pretrained(args.model_name).to(dev)
    text_keys = TEXT_KEYS[args.task]
    before = embed(model, tokenizer, ds, text_keys, dev, args.max_length, args.batch_size)
    load_encoder_state(model, args.ckpt, map_location=dev)
    after = embed(model, tokenizer, ds, text_keys, dev, args.max_length, args.batch_size)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6.5))
    for ax, emb, title in [(axes[0], before, f"Before ({args.model_name})"),
                           (axes[1], after, "After JEPA pretraining")]:
        xy = TSNE(n_components=2, perplexity=args.perplexity, init="pca",
                  random_state=0).fit_transform(emb)
        for c, name in enumerate(names):
            m = labels == c
            ax.scatter(xy[m, 0], xy[m, 1], s=6, alpha=0.7, color=plt.cm.tab10(c),
                       label=f"{name} (n={m.sum()})")
        ax.set_title(title)
        ax.set_xticks([]); ax.set_yticks([])
    axes[1].legend(loc="best", markerscale=3, frameon=False)
    fig.suptitle(f"{args.task.upper()} {split} [CLS] t-SNE — {os.path.basename(args.ckpt)}", fontsize=10)
    fig.tight_layout()

    os.makedirs(args.out_dir, exist_ok=True)
    path = os.path.join(args.out_dir, f"tsne_{args.task}.png")
    fig.savefig(path, dpi=150)
    print(f"saved {path}")


if __name__ == "__main__":
    main()

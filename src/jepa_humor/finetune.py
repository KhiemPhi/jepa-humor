from dataclasses import dataclass
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.cuda.amp import GradScaler
from torch.optim.swa_utils import AveragedModel, SWALR, update_bn
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score
from .utils import set_seed, device, load_encoder_state
from .data import load_hf_splits, make_dataloaders, load_hf_splits_test
import os
import torch.nn.utils as nn_utils
import numpy as np

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class TemperatureClassifier(nn.Module):
    def __init__(self, in_dim, num_labels):
        super().__init__()
        self.linear = nn.Linear(in_dim, num_labels)
        self.T = nn.Parameter(torch.tensor(1.0))  # learnable temperature

    def forward(self, x):
        return self.linear(x) / self.T


class EntailmentSentenceClassifier(nn.Module):
    def __init__(self, encoder, hidden_size=768, num_labels=2, dropout=0.1):
        super().__init__()
        self.encoder = encoder
        self.dropout = nn.Dropout(dropout)
        self.classifier = TemperatureClassifier(hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = out.pooler_output if out.pooler_output is not None else out.last_hidden_state[:, 0, :]
        return self.classifier(self.dropout(pooled))


def finetune(cfg, encoder_ckpt: str):
    dev = device()

    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"])
    base = AutoModel.from_pretrained(cfg["model_name"]) if encoder_ckpt is None else AutoModel.from_pretrained(cfg["model_name"])

    if encoder_ckpt != "fresh":
        print(f"Loading pretrained encoder from {encoder_ckpt}")
        base = load_encoder_state(base, encoder_ckpt, map_location="cpu")

    ds = load_hf_splits(cfg["dataset"])
    test_ds = load_hf_splits_test(cfg["test_dataset"])
    train_loader, val_loader = make_dataloaders(ds["train"], test_ds["validation"], cfg["batch_size"], cfg["num_workers"])

    model = EntailmentSentenceClassifier(base, hidden_size=base.config.hidden_size).to(dev)

    lr = float(cfg["lr"])
    weight_decay = float(cfg.get("weight_decay", 0.01))
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    total_steps = len(train_loader) * cfg["epochs"]
    warmup_steps = int(cfg.get("warmup_ratio", 0.1) * total_steps)
    scheduler = get_linear_schedule_with_warmup(opt, warmup_steps, total_steps)

    criterion = nn.CrossEntropyLoss()
    best_run_val_acc = 0.0
    best_model_state = None

    set_seed(cfg["seed"])

    for run in range(1, 10):
        print(f"\n{'='*60}")
        print(f"Run {run}/10")
        print(f"{'='*60}")

        model = EntailmentSentenceClassifier(base, hidden_size=base.config.hidden_size).to(dev)
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = get_linear_schedule_with_warmup(opt, warmup_steps, total_steps)
        scaler = GradScaler(enabled=cfg.get("mixed_precision", True))
        best_val_acc = 0.0

        set_seed(cfg["seed"] + run)

        for epoch in range(1, cfg["epochs"] + 1):
            model.train()
            running, y_true, y_pred = 0.0, [], []

            for batch in tqdm(train_loader, desc=f"Finetune {epoch}/{cfg['epochs']}", leave=False):
                sent1 = batch["sentence1"]
                sent2 = batch["sentence2"]
                labels = torch.tensor(batch["label"], dtype=torch.long).to(dev, non_blocking=True)

                enc = tokenizer(
                    list(sent1), list(sent2),
                    return_tensors="pt",
                    padding="max_length",
                    truncation=True,
                    max_length=cfg["max_length"]
                )
                enc = {k: v.to(dev, non_blocking=True) for k, v in enc.items()}

                opt.zero_grad(set_to_none=True)
                with torch.cuda.amp.autocast(enabled=cfg.get("mixed_precision", True)):
                    logits = model(enc["input_ids"], enc["attention_mask"])
                    loss = criterion(logits, labels)

                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                nn_utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt)
                scaler.update()
                scheduler.step()

                running += loss.item()
                y_true.extend(labels.cpu().tolist())
                y_pred.extend(logits.detach().cpu().argmax(-1).tolist())

            print(f"[Finetune] epoch {epoch}: loss={running/len(train_loader):.4f} | acc={accuracy_score(y_true,y_pred):.4f}")

            model.eval()
            v_true, v_pred = [], []
            with torch.no_grad():
                for batch in val_loader:
                    sent1 = batch["sentence1"]
                    sent2 = batch["sentence2"]
                    labels = torch.tensor(batch["label"], dtype=torch.long).to(dev, non_blocking=True)

                    enc = tokenizer(
                        list(sent1), list(sent2),
                        return_tensors="pt",
                        padding="max_length",
                        truncation=True,
                        max_length=cfg["max_length"]
                    )
                    enc = {k: v.to(dev, non_blocking=True) for k, v in enc.items()}

                    logits = model(enc["input_ids"], enc["attention_mask"])
                    v_true.extend(labels.cpu().tolist())
                    v_pred.extend(logits.cpu().argmax(-1).tolist())

            val_acc = accuracy_score(v_true, v_pred)
            best_val_acc = max(best_val_acc, val_acc)
            print(f"          val: acc={val_acc:.4f} (best={best_val_acc:.4f})")

        if best_val_acc > best_run_val_acc:
            best_run_val_acc = best_val_acc
            best_model_state = model.state_dict().copy()
            print(f"  ✓ New best run! Validation acc: {best_run_val_acc:.4f}")
        else:
            print(f"  Run {run} best val acc: {best_val_acc:.4f} (overall best: {best_run_val_acc:.4f})")

    print(f"\n{'='*60}")
    print(f"Best run validation accuracy: {best_run_val_acc:.4f}")
    print(f"{'='*60}\n")

    model = EntailmentSentenceClassifier(base, hidden_size=base.config.hidden_size).to(dev)
    model.load_state_dict(best_model_state)

    return model

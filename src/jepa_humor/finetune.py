from dataclasses import dataclass
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from torch.optim.swa_utils import AveragedModel, SWALR, update_bn
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics import accuracy_score, f1_score
from scipy.stats import pearsonr, spearmanr
from .utils import set_seed, device, load_encoder_state
from .data import load_hf_splits, make_dataloaders
from .train_ssl import pretrain
import os
from torch import amp
import numpy as np
os.environ["TOKENIZERS_PARALLELISM"] = "false"

class NextSentencePredictor(nn.Module):
    def __init__(self, encoder, hidden_size=768):
        super().__init__()
        self.encoder = encoder
        self.hidden_size = hidden_size
        self.dropout = nn.Dropout(0.1)
        # MLP scoring head
        self.classifier = nn.Linear(hidden_size, 1)

    def forward(self, input_ids, attention_mask ):
        """
        input_ids: (B*4, L)
        attention_mask: (B*4, L)
        Returns:
            logits: (B, 4)
        """
        B, C, L = input_ids.shape  # B=batch size, C=num_choices
        input_ids = input_ids.view(B*C, L)
        outputs = self.encoder(
            input_ids,
            attention_mask = attention_mask.view(B*C, L),
        )
        last_hidden = outputs.last_hidden_state  # (B*4, seq_len, hidden_size)
        pooled_output = last_hidden[:, 0, :]    # Take CLS token
        logits = self.classifier(self.dropout(pooled_output))  # (B*C, 1)
        logits = logits.view(B, C)  # (B, num_choices)
        return logits

choice = ['choice1', 'choice2']

def process_data( x ):
    return {
        "true_sentence": x[choice[x["label"]]], 
    }

def finetune(cfg, encoder_ckpt: str):
    set_seed(cfg["seed"])
    dev = device()

    # 1. Load Encoder
    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"])
    base = AutoModel.from_pretrained(cfg["model_name"]) if encoder_ckpt is None else AutoModel.from_pretrained(cfg["model_name"])
    if encoder_ckpt != "fresh":
        print(f"Loading pretrained encoder from {encoder_ckpt}")
        base = load_encoder_state(base, encoder_ckpt, map_location="cpu")
    else: 
        print(f"Loading fresh encoder")

    ds = load_hf_splits(cfg["dataset"])
   
    ds = ds.map(lambda x: process_data(x))
    train_loader, val_loader = make_dataloaders(ds["train"], ds["validation"], cfg["batch_size"], cfg["num_workers"])
    print(f"Train dataset size: {len(ds['train'])}")
    print(f"Test dataset size: {len(ds['validation'])}")

    model = NextSentencePredictor(base, hidden_size=base.config.hidden_size).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=0.09)
    criterion =  nn.CrossEntropyLoss()
    scaler = GradScaler(enabled=cfg.get("mixed_precision", True))

    best_val = 0.0
    for epoch in range(1, cfg["epochs"] + 1):
        
        # train
        model.train()
        running, y_true, y_pred = 0.0, [], []
        for batch in tqdm(train_loader, desc=f"Finetune {epoch}/{cfg['epochs']}", leave=False):
            premise = batch["premise"]
            sentence1, sentence2 = batch["choice1"], batch["choice2"]
            sentences = [sentence1, sentence2]
            num_choices = len(sentences)

            # Flatten the pairs (so tokenizer can handle them all at once)
            premise = [[s] * num_choices for s in premise]
            sentences = list(zip(*sentences))  # [(ending0[i], ending1[i], ...), ...]

            # Flatten properly
            premise_flat= sum(premise, [])  # still okay, already lists
            sentences_flat= [s for tup in sentences for s in tup]

            # Tokenize
            enc = tokenizer(
                premise_flat,
                sentences_flat,
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors="pt"
            )
            enc = {k: v.to(dev, non_blocking=True) for k, v in enc.items()}

            labels = torch.tensor(batch["label"], dtype=torch.long)
            labels = labels.to(dev, non_blocking=True)

            opt.zero_grad(set_to_none=True)
            with amp.autocast("cuda", enabled=cfg.get("mixed_precision", True)):
                logits = model(enc["input_ids"].view(len(batch["label"]), 2, -1),
                               enc["attention_mask"].view(len(batch["label"]), 2, -1))
                loss = criterion(logits, labels)

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            running += loss.item() * labels.size(0)
            y_pred.extend(logits.detach().cpu().argmax(-1).tolist())
            y_true.extend(labels.detach().cpu().tolist())

        epoch_loss = running / len(train_loader.dataset)
        train_f1 = f1_score(y_pred, y_true)
        train_val = accuracy_score(y_pred, y_true)
        print(f"Epoch {epoch}: Train Loss={epoch_loss:.4f}, acc={train_val:.4f}, f1={train_f1:.4f}")

        # validate
        model.eval()
        v_true, v_pred, v_prob = [], [], []
        with torch.no_grad():
            for batch in val_loader:
                premise = batch["premise"]
                sentence1, sentence2 = batch["choice1"], batch["choice2"]
                sentences = [sentence1, sentence2]
                num_choices = len(sentences)

                # Flatten the pairs (so tokenizer can handle them all at once)
                premise = [[s] * num_choices for s in premise]
                sentences = list(zip(*sentences))  # [(ending0[i], ending1[i], ...), ...]

                # Flatten properly
                premise_flat= sum(premise, [])  # still okay, already lists
                sentences_flat= [s for tup in sentences for s in tup]

                # Tokenize
                enc = tokenizer(
                    premise_flat,
                    sentences_flat,
                    padding=True,
                    truncation=True,
                    max_length=128,
                    return_tensors="pt"
                )
                enc = {k: v.to(dev, non_blocking=True) for k, v in enc.items()}

                labels = torch.tensor(batch["label"], dtype=torch.long)
                labels = labels.to(dev, non_blocking=True)
                logits = model(enc["input_ids"].view(len(batch["label"]), 2, -1),
                               enc["attention_mask"].view(len(batch["label"]), 2, -1))
                #logits = logits.squeeze(-1)
                v_true.extend(labels.cpu().tolist())
                v_pred.extend(logits.detach().cpu().argmax(-1).tolist())

        val_f1 = f1_score(v_pred, v_true)
        val_acc = accuracy_score(v_pred, v_true)
        best_val = max(best_val, val_acc)
        print(f"          val: acc={val_acc:.4f}, f1={val_f1:.4f}")
        print(f"          best acc so far: {best_val:.4f}")
    
     
    return model

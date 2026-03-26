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

class STSBRegressor(nn.Module):
    def __init__(self, encoder, hidden_size=768, dropout=0.3):
        super().__init__()
        self.encoder = encoder
        self.dropout = nn.Dropout(dropout)
        self.reg_head = nn.Linear(hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = out.last_hidden_state[:, 0, :]
        pooled = self.dropout(pooled)
        return self.reg_head(pooled).squeeze(-1) 



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
   
    # split train/val from train_ds
    train_loader, val_loader = make_dataloaders(ds["train"], ds["validation"], cfg["batch_size"], cfg["num_workers"])
    print(f"Train dataset size: {len(ds['train'])}")
    print(f"Test dataset size: {len(ds['validation'])}")

    model = STSBRegressor(base, hidden_size=base.config.hidden_size).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=0.9)
    criterion = nn.MSELoss()
    scaler = GradScaler(enabled=cfg.get("mixed_precision", True))

    best_val = 0.0
    for epoch in range(1, cfg["epochs"] + 1):
        
        # train
        model.train()
        running, y_true, y_pred = 0.0, [], []
        for batch in tqdm(train_loader, desc=f"Finetune {epoch}/{cfg['epochs']}", leave=False):
            sent1 = batch["sentence1"]
            sent2 = batch["sentence2"]
            labels = torch.tensor(batch["label"], dtype=torch.float32)

            enc = tokenizer(list(sent1), list(sent2), return_tensors="pt", padding="longest",
                            truncation=True, max_length=cfg["max_length"])
            enc = {k: v.to(dev, non_blocking=True) for k, v in enc.items()}
            labels = labels.to(dev, non_blocking=True)

            opt.zero_grad(set_to_none=True)
            with amp.autocast("cuda", enabled=cfg.get("mixed_precision", True)):
                logits = model(enc["input_ids"], enc["attention_mask"])  # outputs: [B, 1]
                #logits = logits.squeeze(-1)  # shape: [B]
                loss = criterion(logits, labels)

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            running += loss.item() * labels.size(0)
            y_pred.extend(logits.detach().cpu().tolist())
            y_true.extend(labels.detach().cpu().tolist())

        epoch_loss = running / len(train_loader.dataset)
        pearson = pearsonr(y_pred, y_true)[0]
        spearman = spearmanr(y_pred, y_true)[0]
        final_score = (pearson + spearman) / 2
        print(f"Epoch {epoch}: Train Loss={epoch_loss:.4f}, Pearson={pearson:.4f}, Spearman={spearman:.4f}, Final={final_score:.4f}")

        # validate
        model.eval()
        v_true, v_pred, v_prob = [], [], []
        with torch.no_grad():
            for batch in val_loader:
                sent1 = batch["sentence1"]
                sent2 = batch["sentence2"]
                labels = torch.tensor(batch["label"], dtype=torch.float32)

                enc = tokenizer(list(sent1), list(sent2), return_tensors="pt", padding="longest",
                            truncation=True, max_length=cfg["max_length"])
                enc = {k: v.to(dev, non_blocking=True) for k, v in enc.items()}
                labels = labels.to(dev, non_blocking=True)
                logits = model(enc["input_ids"], enc["attention_mask"])
                #logits = logits.squeeze(-1)
                v_true.extend(labels.cpu().tolist())
                v_pred.extend(logits.cpu().tolist())

        pearson = pearsonr(v_pred, v_true)[0]
        spearman = spearmanr(v_pred, v_true)[0]
        final_score = (pearson + spearman) / 2
        best_val = max(best_val, final_score)
        #print(f"STS-B score: {final_score * 100:.2f}")
        print(f"          val: Pearson={pearson:.4f}, Spearman={spearman:.4f}, Final={final_score:.4f}")
        print(f"          best STS-B score so far: {best_val:.4f}")
    
     
    return model

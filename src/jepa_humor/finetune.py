from dataclasses import dataclass
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from torch.optim.swa_utils import AveragedModel, SWALR, update_bn
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics import accuracy_score, f1_score
from .utils import set_seed, device, load_encoder_state
from .data import load_hf_splits, make_dataloaders
from .train_ssl import pretrain
import os
from torch import amp
import numpy as np
import copy
os.environ["TOKENIZERS_PARALLELISM"] = "false"

class CorrectEnglishSentenceClassifier(nn.Module):
    def __init__(self, encoder, hidden_size=768, num_labels=2, dropout=0.5):
        super().__init__()
        self.encoder = encoder
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_labels)
    def forward(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = out.last_hidden_state[:, 0, :]
        pooled = self.dropout(pooled)
        return self.classifier(pooled)


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

    model = CorrectEnglishSentenceClassifier(base, hidden_size=base.config.hidden_size).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=0.072)
    
    # Setup SWA (Stochastic Weight Averaging)
    swa_model = AveragedModel(model)
    swa_scheduler = SWALR(opt, swa_lr=cfg["lr"] * 0.35)
    use_swa = cfg.get("use_swa", True)
    swa_started = False
    
    # Compute class weights from training labels (inverse frequency)
    try:
        train_labels = np.asarray(ds["train"]["label"])
        counts = np.bincount(train_labels, minlength=2).astype(np.float32)
        counts[counts == 0] = 1.0
        inv = 1.0 / counts
        weights = (inv / inv.sum()) * 2.0
        class_weights = torch.tensor(weights, dtype=torch.float, device=dev)
        print(f"Class counts: {counts.astype(int).tolist()} -> weights: {weights.round(4).tolist()}")
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    except Exception as e:
        print(f"Warning: falling back to unweighted CE (reason: {e})")
        criterion = nn.CrossEntropyLoss(label_smoothing=0.01)
    scaler = GradScaler(enabled=cfg.get("mixed_precision", True))

    best_val_acc = -1.0
    best_epoch = 0
    best_model_state = None

    for epoch in range(1, cfg["epochs"] + 1):
        # train
        model.train()
        running, y_true, y_pred = 0.0, [], []
        for batch in tqdm(train_loader, desc=f"Finetune {epoch}/{cfg['epochs']}", leave=False):
            sentences = batch["sentence"]
            labels = torch.tensor(batch["label"], dtype=torch.long)

            enc = tokenizer(list(sentences), return_tensors="pt", padding="longest",
                            truncation=True, max_length=cfg["max_length"])
            enc = {k: v.to(dev, non_blocking=True) for k, v in enc.items()}
            labels = labels.to(dev, non_blocking=True)

            opt.zero_grad(set_to_none=True)
            with amp.autocast("cuda", enabled=cfg.get("mixed_precision", True)):
                logits = model(enc["input_ids"], enc["attention_mask"])
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            
            if use_swa and swa_started:
                swa_model.update_parameters(model)
                swa_scheduler.step()

            running += float(loss.item())
            y_true.extend(labels.detach().cpu().tolist())
            y_pred.extend(logits.detach().cpu().argmax(-1).tolist())

        train_acc = accuracy_score(y_true, y_pred)
        train_f1 = f1_score(y_true, y_pred)
        print(f"[Finetune] epoch {epoch}: loss={running/len(train_loader):.4f} acc={train_acc:.4f} f1={train_f1:.4f}")

        # validate
        model.eval()
        v_true, v_pred = [], []
        with torch.no_grad():
            for batch in val_loader:
                sentences = batch["sentence"]
                labels = torch.tensor(batch["label"], dtype=torch.long)
                enc = tokenizer(list(sentences), return_tensors="pt", padding="longest",
                                truncation=True, max_length=cfg["max_length"])
                enc = {k: v.to(dev, non_blocking=True) for k, v in enc.items()}
                labels = labels.to(dev, non_blocking=True)
                logits = model(enc["input_ids"], enc["attention_mask"])
                v_true.extend(labels.cpu().tolist())
                v_pred.extend(logits.cpu().argmax(-1).tolist())

        val_acc = accuracy_score(v_true, v_pred)
        val_f1 = f1_score(v_true, v_pred)
        print(f"          val: acc={val_acc:.4f} f1={val_f1:.4f}")
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            best_model_state = copy.deepcopy(model.state_dict())
            print(f"          *** New best validation Acc: {best_val_acc:.4f} at epoch {epoch} ***")
        
        if use_swa and not swa_started and epoch > best_epoch and best_epoch > 0:
            swa_started = True
            if best_model_state is not None:
                model.load_state_dict(best_model_state)
                swa_model = AveragedModel(model)
            print(f"Starting SWA at epoch {epoch} (best epoch was {best_epoch})")
    
    # After training, evaluate SWA model and compare with best single epoch
    if use_swa and swa_started:
        print("\nEvaluating SWA model vs best single epoch model...")
        print(f"Best single epoch: {best_epoch} with Acc: {best_val_acc:.4f}")
        
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
        
        swa_model.eval()
        swa_v_true, swa_v_pred = [], []
        with torch.no_grad():
            for batch in val_loader:
                sentences = batch["sentence"]
                labels = torch.tensor(batch["label"], dtype=torch.long)
                enc = tokenizer(list(sentences), return_tensors="pt", padding="longest",
                                truncation=True, max_length=cfg["max_length"])
                enc = {k: v.to(dev, non_blocking=True) for k, v in enc.items()}
                labels = labels.to(dev, non_blocking=True)
                logits = swa_model(enc["input_ids"], enc["attention_mask"])
                swa_v_true.extend(labels.cpu().tolist())
                swa_v_pred.extend(logits.cpu().argmax(-1).tolist())
        
        swa_acc = accuracy_score(swa_v_true, swa_v_pred)
        print(f"SWA model Acc: {swa_acc:.4f}")
        
        model.load_state_dict(swa_model.module.state_dict())
    elif best_model_state is not None:
        print(f"Loading best model from epoch {best_epoch} with Acc: {best_val_acc:.4f}")
        model.load_state_dict(best_model_state)
    elif use_swa:
        print("Updating batch normalization statistics for SWA model...")
        model.eval()
        with torch.no_grad():
            for batch in tqdm(train_loader, desc="Updating BN stats", leave=False):
                sentences = batch["sentence"]
                enc = tokenizer(list(sentences), return_tensors="pt", padding="longest",
                                truncation=True, max_length=cfg["max_length"])
                enc = {k: v.to(dev, non_blocking=True) for k, v in enc.items()}
                _ = swa_model(enc["input_ids"], enc["attention_mask"])
        try:
            update_bn(train_loader, swa_model, device=dev)
        except Exception:
            pass
        
        print("Replacing model with SWA model...")
        model.load_state_dict(swa_model.module.state_dict())
        print("SWA model loaded!")
        
    model.eval()
    v_true, v_pred = [], []
    with torch.no_grad():
        for batch in val_loader:
            sentences = batch["sentence"]
            labels = torch.tensor(batch["label"], dtype=torch.long)
            enc = tokenizer(list(sentences), return_tensors="pt", padding="longest",
                            truncation=True, max_length=cfg["max_length"])
            enc = {k: v.to(dev, non_blocking=True) for k, v in enc.items()}
            labels = labels.to(dev, non_blocking=True)
            logits = model(enc["input_ids"], enc["attention_mask"])
            v_true.extend(labels.cpu().tolist())
            v_pred.extend(logits.cpu().argmax(-1).tolist())

    val_acc = accuracy_score(v_true, v_pred)
    val_f1 = f1_score(v_true, v_pred)
    print(f"Final validation: acc={val_acc:.4f} f1={val_f1:.4f}")
     
    return model
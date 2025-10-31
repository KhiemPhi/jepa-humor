from dataclasses import dataclass
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from torch.optim.swa_utils import AveragedModel, SWALR, update_bn
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics import matthews_corrcoef, f1_score, accuracy_score
from .utils import set_seed, device, load_encoder_state
from .data import load_hf_splits, make_dataloaders
from .train_ssl import pretrain
import os
from torch import amp
import numpy as np
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
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=0.15)
    
    # Setup SWA (Stochastic Weight Averaging)
    swa_start_epoch = max(1, int(cfg["epochs"] * 0.9))  # Start SWA after 75% of training
    swa_model = AveragedModel(model)
    swa_scheduler = SWALR(opt, swa_lr=cfg["lr"] * 0.5)  # Lower LR for averaging phase
    use_swa = cfg.get("use_swa", True)
    
    if use_swa:
        print(f"SWA will start at epoch {swa_start_epoch}/{cfg['epochs']}")
    
    # Compute class weights from training labels (inverse frequency)
    try:
        train_labels = np.asarray(ds["train"]["label"])  # expects integer labels 0/1
        counts = np.bincount(train_labels, minlength=2).astype(np.float32)
        counts[counts == 0] = 1.0
        inv = 1.0 / counts
        weights = (inv / inv.sum()) * 2.0  # normalize roughly to sum=2
        class_weights = torch.tensor(weights, dtype=torch.float, device=dev)
        print(f"Class counts: {counts.astype(int).tolist()} -> weights: {weights.round(4).tolist()}")
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    except Exception as e:
        print(f"Warning: falling back to unweighted CE (reason: {e})")
        criterion = nn.CrossEntropyLoss(label_smoothing=0.01)
    scaler = GradScaler(enabled=cfg.get("mixed_precision", True))

    for epoch in range(1, cfg["epochs"] + 1):
        # Check if SWA phase starts this epoch
        if use_swa and epoch == swa_start_epoch:
            print(f"Starting SWA at epoch {epoch}")
        
        # train
        model.train()
        running, y_true, y_pred = 0.0, [], []
        y_prob = []  # store positive-class probabilities across the epoch
        for batch in tqdm(train_loader, desc=f"Finetune {epoch}/{cfg['epochs']}", leave=False):
            sent1 = batch["sentence1"]
            sent2 = batch["sentence2"]
            labels = torch.tensor(batch["label"], dtype=torch.long)

            enc = tokenizer(list(sent1), list(sent2), return_tensors="pt", padding="longest",
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
            
            # Update SWA model and scheduler if in SWA phase (after each batch)
            if use_swa and epoch >= swa_start_epoch:
                swa_model.update_parameters(model)
                swa_scheduler.step()

            running += float(loss.item())
            y_true.extend(labels.detach().cpu().tolist())
            y_pred.extend(logits.detach().cpu().argmax(-1).tolist())
            # collect probabilities for class 1
            probs_batch = torch.softmax(logits.detach().cpu(), dim=-1)[:, 1].numpy()
            y_prob.append(probs_batch)
            

        train_mcc = matthews_corrcoef(y_true, y_pred)
        train_f1 = f1_score(y_true, y_pred)
        # concatenate per-batch probabilities to a single vector aligned with y_true
        probs = np.concatenate(y_prob, axis=0)
        true = np.asarray(y_true)
        
        # Search across thresholds
        best_t, best_mcc = 0.0, -1.0
        for t in np.linspace(0.01, 0.99, 99):
            preds = (probs >= t).astype(int)
            mcc = matthews_corrcoef(true, preds)
            if mcc > best_mcc:
                best_mcc, best_t = mcc, t

        print(f"Best MCC for Train MCC = {best_mcc:.4f} at threshold = {best_t:.3f}")
 
        print(f"[Finetune] epoch {epoch}: loss={running/len(train_loader):.4f} mcc={train_mcc:.4f} f1={train_f1:.4f}")

        # validate
        model.eval()
        v_true, v_pred, v_prob = [], [], []
        with torch.no_grad():
            for batch in val_loader:
                sent1 = batch["sentence1"]
                sent2 = batch["sentence2"]
                labels = torch.tensor(batch["label"], dtype=torch.long)

                enc = tokenizer(list(sent1), list(sent2), return_tensors="pt", padding="longest",
                            truncation=True, max_length=cfg["max_length"])
                enc = {k: v.to(dev, non_blocking=True) for k, v in enc.items()}
                labels = labels.to(dev, non_blocking=True)
                logits = model(enc["input_ids"], enc["attention_mask"])
                v_true.extend(labels.cpu().tolist())
                v_pred.extend(logits.cpu().argmax(-1).tolist())
                probs_batch = torch.softmax(logits.detach().cpu(), dim=-1)[:, 1].numpy()
                v_prob.append(probs_batch)

        probs = np.concatenate(v_prob, axis=0) 
        preds = (probs >= best_t).astype(int)
        val_mcc = matthews_corrcoef(v_true, preds)
        val_f1 = f1_score(v_true, preds)
        val_acc = accuracy_score(v_true, preds)
        print(f"          val: mcc={val_mcc:.4f} f1={val_f1:.4f} acc={val_acc:.4f}")
    
    # After training, update BN stats for SWA model and replace regular model
    if use_swa:
        print("Updating batch normalization statistics for SWA model...")
        # Update BN stats by running one forward pass through training data
        model.eval()
        with torch.no_grad():
            for batch in tqdm(train_loader, desc="Updating BN stats", leave=False):
                sent1 = batch["sentence1"]
                sent2 = batch["sentence2"]
                labels = torch.tensor(batch["label"], dtype=torch.long)

                enc = tokenizer(list(sent1), list(sent2), return_tensors="pt", padding="longest",
                            truncation=True, max_length=cfg["max_length"])
                enc = {k: v.to(dev, non_blocking=True) for k, v in enc.items()}
                labels = labels.to(dev, non_blocking=True)
                
                _ = swa_model(enc["input_ids"], enc["attention_mask"])
        # Note: BERT uses LayerNorm, not BatchNorm, so update_bn is not strictly necessary
        # But we'll update any BatchNorm layers if they exist (e.g., in custom heads)
        try:
            update_bn(train_loader, swa_model, device=dev)
        except Exception:
            # If update_bn fails (e.g., no BN layers), that's fine
            pass
        
        print("Replacing model with SWA model...")
        model.load_state_dict(swa_model.module.state_dict())
        print("SWA model loaded!")
        
    model.eval()
    v_true, v_pred, v_prob = [], [], []
    with torch.no_grad():
        for batch in val_loader:
            sent1 = batch["sentence1"]
            sent2 = batch["sentence2"]
            labels = torch.tensor(batch["label"], dtype=torch.long)

            enc = tokenizer(list(sent1), list(sent2), return_tensors="pt", padding="longest",
                            truncation=True, max_length=cfg["max_length"])
            enc = {k: v.to(dev, non_blocking=True) for k, v in enc.items()}
            labels = labels.to(dev, non_blocking=True)
            logits = model(enc["input_ids"], enc["attention_mask"])
            v_true.extend(labels.cpu().tolist())
            v_pred.extend(logits.cpu().argmax(-1).tolist())
            probs_batch = torch.softmax(logits.detach().cpu(), dim=-1)[:, 1].numpy()
            v_prob.append(probs_batch)

    probs = np.concatenate(v_prob, axis=0)
    preds = (probs >= best_t).astype(int)
    val_mcc = matthews_corrcoef(v_true, preds)
    val_f1 = f1_score(v_true, preds)
    val_acc = accuracy_score(v_true, preds)
    print(f"Final validation: mcc={val_mcc:.4f} f1={val_f1:.4f}, val_acc={val_acc:.4f}")
     
    return model

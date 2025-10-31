from dataclasses import dataclass
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from torch.optim.swa_utils import AveragedModel, SWALR, update_bn
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics import matthews_corrcoef, f1_score
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
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=0.08)
    
    # Setup SWA (Stochastic Weight Averaging)
    # Start SWA after best epoch to avoid averaging in worse weights
    swa_model = AveragedModel(model)
    swa_scheduler = SWALR(opt, swa_lr=cfg["lr"] * 0.35)  # Lower LR for averaging phase
    use_swa = cfg.get("use_swa", True)
    swa_started = False
    
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

    # Track best model for early stopping and SWA comparison
    best_val_mcc = -1.0
    best_epoch = 0
    best_model_state = None
    best_val_threshold = 0.5

    for epoch in range(1, cfg["epochs"] + 1):
        # train
        model.train()
        running, y_true, y_pred = 0.0, [], []
        y_prob = []  # store positive-class probabilities across the epoch
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
            
            # Update SWA model and scheduler if in SWA phase (after each batch)
            if use_swa and swa_started:
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
                sentences = batch["sentence"]
                labels = torch.tensor(batch["label"], dtype=torch.long)
                enc = tokenizer(list(sentences), return_tensors="pt", padding="longest",
                                truncation=True, max_length=cfg["max_length"])
                enc = {k: v.to(dev, non_blocking=True) for k, v in enc.items()}
                labels = labels.to(dev, non_blocking=True)
                logits = model(enc["input_ids"], enc["attention_mask"])
                v_true.extend(labels.cpu().tolist())
                v_pred.extend(logits.cpu().argmax(-1).tolist())
                probs_batch = torch.softmax(logits.detach().cpu(), dim=-1)[:, 1].numpy()
                v_prob.append(probs_batch)

        val_probs = np.concatenate(v_prob, axis=0)
        val_true_arr = np.asarray(v_true)
        
        # Find best threshold on VALIDATION set (not training!)
        best_val_t, best_val_mcc_thresh = 0.5, -1.0
        for t in np.linspace(0.01, 0.99, 99):
            preds_t = (val_probs >= t).astype(int)
            mcc_t = matthews_corrcoef(val_true_arr, preds_t)
            if mcc_t > best_val_mcc_thresh:
                best_val_mcc_thresh, best_val_t = mcc_t, t
        
        val_mcc = best_val_mcc_thresh  # Use threshold-optimized MCC
        val_preds = (val_probs >= best_t).astype(int)
        val_f1 = f1_score(val_true_arr, val_preds)
        argmax_mcc = matthews_corrcoef(v_true, v_pred)
        print(f"          val: mcc={val_mcc:.4f} f1={val_f1:.4f} (threshold={best_val_t:.3f}, argmax_mcc={argmax_mcc:.4f})")
        
        # Track best model and start SWA after finding best epoch
        if val_mcc > best_val_mcc:
            best_val_mcc = val_mcc
            best_epoch = epoch
            best_val_threshold = best_val_t
            best_model_state = copy.deepcopy(model.state_dict())
            print(f"          *** New best validation MCC: {best_val_mcc:.4f} at epoch {epoch} ***")
        
        # Start SWA after best epoch (with 1 epoch delay) to only average good weights
        if use_swa and not swa_started and epoch > best_epoch and best_epoch > 0:
            swa_started = True
            # Initialize SWA model with best checkpoint
            if best_model_state is not None:
                model.load_state_dict(best_model_state)
                swa_model = AveragedModel(model)
            print(f"Starting SWA at epoch {epoch} (best epoch was {best_epoch})")
    
    # After training, evaluate SWA model and compare with best single epoch
    if use_swa and swa_started:
        print("\nEvaluating SWA model vs best single epoch model...")
        print(f"Best single epoch: {best_epoch} with MCC: {best_val_mcc:.4f}")
        
        # Load best single epoch model first
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
        
        # Evaluate SWA-averaged model
        swa_model.eval()
        swa_v_true, swa_v_prob = [], []
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
                probs_batch = torch.softmax(logits.detach().cpu(), dim=-1)[:, 1].numpy()
                swa_v_prob.append(probs_batch)
        
        swa_probs = np.concatenate(swa_v_prob, axis=0)
        swa_true_arr = np.asarray(swa_v_true)
        swa_preds = (swa_probs >= best_t).astype(int)
        swa_mcc = matthews_corrcoef(swa_true_arr, swa_preds)
        print(f"SWA model MCC: {swa_mcc:.4f} (threshold={best_t:.3f})")
        
        model.load_state_dict(swa_model.module.state_dict())
    elif best_model_state is not None:
        #uci No SWA or SWA didn't start, just load best model
        print(f"Loading best model from epoch {best_epoch} with MCC: {best_val_mcc:.4f}")
        model.load_state_dict(best_model_state)
    elif use_swa:
        print("Updating batch normalization statistics for SWA model...")
        # Update BN stats by running one forward pass through training data
        model.eval()
        with torch.no_grad():
            for batch in tqdm(train_loader, desc="Updating BN stats", leave=False):
                sentences = batch["sentence"]
                enc = tokenizer(list(sentences), return_tensors="pt", padding="longest",
                                truncation=True, max_length=cfg["max_length"])
                enc = {k: v.to(dev, non_blocking=True) for k, v in enc.items()}
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
            sentences = batch["sentence"]
            labels = torch.tensor(batch["label"], dtype=torch.long)
            enc = tokenizer(list(sentences), return_tensors="pt", padding="longest",
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
    print(f"Final validation: mcc={val_mcc:.4f} f1={val_f1:.4f}")
     
    return model

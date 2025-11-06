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
    def __init__(self, encoder, hidden_size=768, use_pooler=True):
        super().__init__()
        self.encoder = encoder
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(hidden_size, 1)
        self.use_pooler = use_pooler

    def forward(self, input_ids, attention_mask, token_type_ids=None):
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        if self.use_pooler and getattr(outputs, "pooler_output", None) is not None:
            pooled = outputs.pooler_output                      # (B*C, H)
        else:
            pooled = outputs.last_hidden_state[:, 0, :]         # (B*C, H)
        logits = self.classifier(self.dropout(pooled))          # (B*C, 1)
        B = input_ids.shape[0] // 2
        return logits.view(B, 2)
    
choice = ['choice1', 'choice2']
def process_data( x ):
    return {
        "true_sentence": x[choice[x["label"]]], 
    }

def build_pairs(premise, choice, question):
    if question == "cause":
        # choice is a cause of the premise
        return f"{choice} because {premise}"
    else:  # question == "effect"
        return f"{premise} so {choice}"
    
def finetune(cfg, encoder_ckpt: str):
    """
    Fine-tune model on COPA task.
    COPA format: premise + 2 choices, predict which choice is more plausible.
    """
    set_seed(cfg["seed"])
    dev = device()

    # 1. Load Encoder
    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"])
    base = AutoModel.from_pretrained(cfg["model_name"])
    
    if encoder_ckpt and encoder_ckpt != "fresh":
        print(f"Loading pretrained encoder from {encoder_ckpt}")
        base = load_encoder_state(base, encoder_ckpt, map_location="cpu")
    else: 
        print(f"Loading fresh encoder from {cfg['model_name']}")

    # 2. Load COPA dataset
    ds = load_hf_splits(cfg["dataset"])
    ds = ds.map(lambda x: process_data(x))
    
    # Remove process_data since COPA already has the right structure
    # COPA dataset should have: premise, choice1, choice2, label
    train_loader, val_loader = make_dataloaders(
        ds["train"], 
        ds["validation"], 
        cfg["batch_size"], 
        cfg["num_workers"]
    )
    print(f"Train dataset size: {len(ds['train'])}")
    print(f"Validation dataset size: {len(ds['validation'])}")

    # 3. Initialize model
    model = NextSentencePredictor(base, hidden_size=base.config.hidden_size).to(dev)
    
    # Standard BERT fine-tuning hyperparameters for COPA
    # Learning rate 2e-5 is standard for BERT fine-tuning
    opt = torch.optim.AdamW(
        model.parameters(), 
        lr=cfg.get("lr", 2e-5),  # Default to 2e-5 for BERT fine-tuning
        weight_decay=cfg.get("weight_decay", 0.01)
    )
    criterion = nn.CrossEntropyLoss()
    scaler = GradScaler(enabled=cfg.get("mixed_precision", True))
    # Stochastic Weight Averaging, Learning Scheduler, Gradient Clip, Weight Decay AdamW
    best_val = 0.0
    best_model_state = None
    num_training_steps = len(train_loader) * cfg["epochs"]
    num_warmup_steps = int(num_training_steps * 0.1)

    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
        opt,
        lr_lambda=lambda step: min((step+1) / num_warmup_steps, 1) if step < num_warmup_steps else max(0.0, (num_training_steps - step) / (num_training_steps - num_warmup_steps))
    )
    
    for epoch in range(1, cfg["epochs"] + 1):
        # Training
        model.train()
        running, y_true, y_pred = 0.0, [], []
        
        for batch in tqdm(train_loader, desc=f"Finetune {epoch}/{cfg['epochs']}", leave=False):
            premise = batch["premise"]
            sentence1 = batch["choice1"]
            sentence2 = batch["choice2"]
            labels = torch.tensor(batch["label"], dtype=torch.long).to(dev, non_blocking=True)
            
            inputs = []
            for p, s1, s2, q in zip(premise, sentence1, sentence2, batch["question"]):
                inputs.append(build_pairs(p, s1, q))
                inputs.append(build_pairs(p, s2, q))

            enc = tokenizer(
                inputs,
                padding=True,
                truncation=True,
                max_length=cfg.get("max_length", 128),
                return_tensors="pt"
            )
            enc = {k: v.to(dev, non_blocking=True) for k, v in enc.items()}

            opt.zero_grad(set_to_none=True)
            with amp.autocast("cuda", enabled=cfg.get("mixed_precision", True)):
                # Input is already (B*2, L) format
                logits = model(enc["input_ids"], enc["attention_mask"], enc["token_type_ids"])
                loss = criterion(logits, labels)

            scaler.scale(loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(opt)
            scaler.update()
            lr_scheduler.step()
            running += loss.item() * labels.size(0)
            y_pred.extend(logits.detach().cpu().argmax(-1).tolist())
            y_true.extend(labels.detach().cpu().tolist())

        epoch_loss = running / len(train_loader.dataset)
        train_acc = accuracy_score(y_true, y_pred)
        train_f1 = f1_score(y_true, y_pred, average='binary')
        print(f"Epoch {epoch}: Train Loss={epoch_loss:.4f}, Acc={train_acc:.4f}, F1={train_f1:.4f}")

        # Validation
        model.eval()
        v_true, v_pred = [], []
        with torch.no_grad():
            for batch in val_loader:
                premise = batch["premise"]
                sentence1 = batch["choice1"]
                sentence2 = batch["choice2"]
                labels = torch.tensor(batch["label"], dtype=torch.long).to(dev, non_blocking=True)
                
                # Create pairs
                inputs = []
                for p, s1, s2, q in zip(premise, sentence1, sentence2, batch["question"]):
                    inputs.append(build_pairs(p, s1, q))
                    inputs.append(build_pairs(p, s2, q))

                enc = tokenizer(
                    inputs,
                    padding=True,
                    truncation=True,
                    max_length=cfg.get("max_length", 128),
                    return_tensors="pt"
                )
                enc = {k: v.to(dev, non_blocking=True) for k, v in enc.items()}

                logits = model(enc["input_ids"], enc["attention_mask"], enc["token_type_ids"])
                v_true.extend(labels.cpu().tolist())
                v_pred.extend(logits.detach().cpu().argmax(-1).tolist())

        val_acc = accuracy_score(v_pred, v_true)
        val_f1 = f1_score(v_pred, v_true, average='binary')
        
        if val_acc > best_val:
            best_val = val_acc
            best_model_state = model.state_dict().copy()
        
        print(f"          Val: Acc={val_acc:.4f}, F1={val_f1:.4f}")
        print(f"          Best Acc: {best_val:.4f}")
        
        # Expected baseline: ~70.6% accuracy for BERT-base on COPA
        if best_val >= 0.70:
            print(f"✓ Reached BERT baseline performance!")
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return model


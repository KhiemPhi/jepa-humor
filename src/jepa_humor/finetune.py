from dataclasses import dataclass
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics import accuracy_score, f1_score
from .utils import set_seed, device, load_encoder_state
from .data import load_hf_splits, make_dataloaders
from .train_ssl import pretrain
import os
from torch import amp
os.environ["TOKENIZERS_PARALLELISM"] = "false"

class NextPhrasePredictor(nn.Module):
    def __init__(self, encoder, hidden_dim=768, mlp_hidden=512):
        super().__init__()
        self.encoder = encoder
        self.hidden_dim = hidden_dim
        
        # MLP scoring head
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, mlp_hidden),
            nn.ReLU(),
            nn.Linear(mlp_hidden, 1)   # scalar score for each candidate
        )

    def forward(self, input_ids, attention_mask, batch_size):
        """
        input_ids: (B*4, L)
        attention_mask: (B*4, L)
        batch_size: B
        Returns:
            logits: (B, 4)
        """
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls_vec = outputs.last_hidden_state[:, 0, :]  # (B*4, hidden_dim)
        scores = self.mlp(cls_vec).squeeze(-1)         # (B*4,)

        # reshape back to (B, 4)
        logits = scores.view(batch_size, 4)
        return logits

def finetune(cfg, encoder_ckpt: str):
    set_seed(cfg["seed"])
    dev = device()

    # 1. Load Encoder
    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"])
    base = AutoModel.from_pretrained(cfg["model_name"]) if encoder_ckpt is None else AutoModel.from_pretrained(cfg["model_name"])
    if encoder_ckpt:
        print(f"Loading pretrained encoder from {encoder_ckpt}")
        base = load_encoder_state(base, encoder_ckpt, map_location="cpu")
    else: 
        print(f"Loading fresh encoder")

    ds = load_hf_splits(cfg["dataset"])
   
    # split train/val from train_ds
    split = ds['train'].train_test_split(test_size=cfg["val_split"], seed=cfg["seed"])
   
    train_loader, val_loader = make_dataloaders(split["train"], split["test"], cfg["batch_size"], cfg["num_workers"])
    print(f"Train dataset size: {len(split['train'])}")
    print(f"Test dataset size: {len(split['test'])}")

    model = HumorClassifier(base, hidden_size=base.config.hidden_size).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"])
    criterion = nn.CrossEntropyLoss()
    scaler = GradScaler(enabled=cfg.get("mixed_precision", True))

    for epoch in range(1, cfg["epochs"] + 1):
        # train
        model.train()
        running, y_true, y_pred = 0.0, [], []
        for batch in tqdm(train_loader, desc=f"Finetune {epoch}/{cfg['epochs']}", leave=False):
            texts = batch["text"]
            labels = torch.tensor(batch["humor"], dtype=torch.long)

            enc = tokenizer(list(texts), return_tensors="pt", padding="longest",
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
                texts = batch["text"]
                labels = torch.tensor(batch["humor"], dtype=torch.long)
                enc = tokenizer(list(texts), return_tensors="pt", padding="longest",
                                truncation=True, max_length=cfg["max_length"])
                enc = {k: v.to(dev, non_blocking=True) for k, v in enc.items()}
                labels = labels.to(dev, non_blocking=True)
                logits = model(enc["input_ids"], enc["attention_mask"])
                v_true.extend(labels.cpu().tolist())
                v_pred.extend(logits.cpu().argmax(-1).tolist())

        val_acc = accuracy_score(v_true, v_pred)
        val_f1 = f1_score(v_true, v_pred)
        print(f"          val: acc={val_acc:.4f} f1={val_f1:.4f}")

    return model

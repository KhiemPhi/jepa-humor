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
from transformers import get_linear_schedule_with_warmup

class NextPhrasePredictor(nn.Module):
    def __init__(self, encoder, hidden_dim=768, mlp_hidden=512):
        super().__init__()
        self.encoder = encoder
        self.hidden_dim = hidden_dim
        self.dropout = nn.Dropout(0.1)
        # MLP scoring head
        self.classifier = nn.Linear(hidden_dim, 1)
        nn.init.normal_(self.classifier.weight, std=0.02)
        nn.init.zeros_(self.classifier.bias)

    def forward(self, input_ids, attention_mask):
        B, C, L = input_ids.shape
        input_ids = input_ids.view(B*C, L)
        attention_mask = attention_mask.view(B*C, L)

        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        pooled = outputs.pooler_output if hasattr(outputs, "pooler_output") else outputs.last_hidden_state[:, 0, :]
        logits = self.classifier(self.dropout(pooled))
        return logits.view(B, C)

def encode_multiple_choice(batch, tokenizer, max_length, device):
    """
    batch: dict with keys ["sent1", "next_sentence0"..."next_sentence3", "label"]
    Returns:
      input_ids: (B, 4, L)
      attention_mask: (B, 4, L)
      labels: (B,)
    """

    endings = ["ending0", "ending1", "ending2", "ending3"]

    # Build list of 4 candidate full sentences per example
    all_pairs = []
    for i in range(len(batch["sent1"])):
        context = batch["sent1"][i] + " " + batch["sent2"][i]  # or sent1+sent2 depending on dataset
        candidates = [batch[e][i] for e in endings]
        # Form 4 text pairs: (context, ending)
        pairs = [(context, c) for c in candidates]
        all_pairs.extend(pairs)

    # Tokenize all pairs at once
    enc = tokenizer(
        [p[0] for p in all_pairs],
        [p[1] for p in all_pairs],
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=max_length
    ).to(device)

    B = len(batch["sent1"])
    input_ids = enc["input_ids"].view(B, 4, -1)
    attention_mask = enc["attention_mask"].view(B, 4, -1)

    labels = torch.tensor(batch["label"], dtype=torch.long).to(device)

    return input_ids, attention_mask, labels

ending_names = ["ending0", "ending1", "ending2", "ending3"]
next_sentences = ["next_sentence0", "next_sentence1", "next_sentence2", "next_sentence3"]

def make_next_sentence(example):
    example = dict(example)  # clone input
    example['true_next_sentence'] = example["sent2"] + " " + example[ending_names[example['label']]]
    example['true_start_sentence'] = example["sent1"] + " " + example["sent2"]
    return example

def finetune(cfg, encoder_ckpt: str):
    set_seed(cfg["seed"])
    dev = device()

    # 1. Load Encoder
    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"])
    base = AutoModel.from_pretrained(cfg["model_name"]) if encoder_ckpt is None else AutoModel.from_pretrained(cfg["model_name"])
    
    if encoder_ckpt and encoder_ckpt != "fresh":
        print(f"Loading pretrained encoder from {encoder_ckpt}")
        base = load_encoder_state(base, encoder_ckpt, map_location="cpu")
    else: 
        print(f"Loading fresh encoder")

    ds = load_hf_splits(cfg["dataset"], "regular")
    ds['test'] = ds['validation']
    del ds['validation']
    ds = ds.map(make_next_sentence)
    # split train/val from train_ds
    
    split = ds

    #split = ds['train'].train_test_split(test_size=cfg["val_split"], seed=cfg["seed"])
    
    train_loader, val_loader = make_dataloaders(split["train"], split["test"], cfg["batch_size"], cfg["num_workers"])
    
    print(f"Train dataset size: {len(ds['train'])}")
    print(f"Test dataset size: {len(ds['test'])}")

    model = NextPhrasePredictor(base, hidden_dim=base.config.hidden_size).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg.get("weight_decay", 0.1))
    criterion = nn.CrossEntropyLoss()
    scaler = GradScaler(enabled=cfg.get("mixed_precision", True))
     # ✅ Standard GLUE schedule: linear warmup → linear decay
    total_steps = len(train_loader) * cfg["epochs"]
    warmup_steps = int(0.3 * total_steps)
    scheduler = get_linear_schedule_with_warmup(opt, warmup_steps, total_steps)
    
    scaler = GradScaler(enabled=cfg.get("mixed_precision", True))
    best_val_acc = 0.0

    for epoch in range(1, cfg["epochs"] + 1):
        # train
        model.train()
        running, y_true, y_pred = 0.0, [], []
        for batch in tqdm(train_loader, desc=f"Finetune {epoch}/{cfg['epochs']}", leave=False):
            input_ids, attention_mask, labels = encode_multiple_choice(batch, tokenizer, cfg["max_length"], dev)
            input_ids = input_ids.to(dev, non_blocking=True)
            attention_mask = attention_mask.to(dev, non_blocking=True)
          
            #labels = torch.tensor(batch["label"], dtype=torch.long)
            labels = labels.to(dev, non_blocking=True)

            opt.zero_grad(set_to_none=True)
            with amp.autocast("cuda", enabled=cfg.get("mixed_precision", True)):
                logits = model(input_ids, attention_mask)
                loss = criterion(logits, labels)

            scaler.scale(loss).backward()
            scaler.unscale_(opt) 
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            scheduler.step()
            running += float(loss.item())
            y_true.extend(labels.detach().cpu().tolist())
            y_pred.extend(logits.detach().cpu().argmax(-1).tolist())

        train_acc = accuracy_score(y_true, y_pred)
        train_f1 = f1_score(y_true, y_pred, average='macro')
        print(f"[Finetune] epoch {epoch}: loss={running/len(train_loader):.4f} acc={train_acc:.4f} f1={train_f1:.4f}")

        # validate
        model.eval()
        v_true, v_pred = [], []
        with torch.no_grad():
            for batch in val_loader:
                input_ids, attention_mask, labels = encode_multiple_choice(batch, tokenizer, cfg["max_length"], dev)
                input_ids = input_ids.to(dev, non_blocking=True)
                attention_mask = attention_mask.to(dev, non_blocking=True)
                #labels = labels.to(dev, non_blocking=True)

                logits = model(input_ids, attention_mask)
                v_true.extend(labels.cpu().tolist())
                v_pred.extend(logits.cpu().argmax(-1).tolist())

        val_acc = accuracy_score(v_true, v_pred)
        val_f1 = f1_score(v_true, v_pred, average='macro')
        print(f"          val: acc={val_acc:.4f} f1={val_f1:.4f}")

    return model

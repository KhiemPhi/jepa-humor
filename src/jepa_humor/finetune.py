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
        self.dropout = nn.Dropout(0.1)
        # MLP scoring head
        self.classifier = nn.Linear(hidden_dim, 1)

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

ending_names = ["ending0", "ending1", "ending2", "ending3"]

next_sentences = ["next_sentence0", "next_sentence1", "next_sentence2", "next_sentence3"]

def make_next_sentence(example):
    res = {}
    for i in range(4):
        res[next_sentences[i]] = example["sent2"] + " " + example[ending_names[i]]
    res['true_next_sentence'] = example["sent2"] + " " + example[ending_names[example['label']]]
    return res

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

    ds = load_hf_splits(cfg["dataset"], "regular")
    ds['test'] = ds['validation']
    del ds['validation']
    ds = ds.map( make_next_sentence)
    # split train/val from train_ds
    
    split = ds

    #split = ds['train'].train_test_split(test_size=cfg["val_split"], seed=cfg["seed"])
    
    train_loader, val_loader = make_dataloaders(split["train"], split["test"], cfg["batch_size"], cfg["num_workers"])
    
    print(f"Train dataset size: {len(ds['train'])}")
    print(f"Test dataset size: {len(ds['test'])}")

    model = NextPhrasePredictor(base, hidden_dim=base.config.hidden_size).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"])
    criterion = nn.CrossEntropyLoss()
    scaler = GradScaler(enabled=cfg.get("mixed_precision", True))

    for epoch in range(1, cfg["epochs"] + 1):
        # train
        model.train()
        running, y_true, y_pred = 0.0, [], []
        for batch in tqdm(train_loader, desc=f"Finetune {epoch}/{cfg['epochs']}", leave=False):
            start_phrase = list(batch["sent1"])
            ending0, ending1, ending2, ending3 = (
                list(batch["next_sentence0"]),
                list(batch["next_sentence1"]),
                list(batch["next_sentence2"]),
                list(batch["next_sentence3"]),
            )

            #start_phrase = tokenizer(list(start_phrase), return_tensors="pt", padding="longest", truncation=True, max_length=cfg["max_length"])
            #start_phrase = {k: v.to(dev, non_blocking=True) for k, v in start_phrase.items()}

            endings = [ending0, ending1, ending2, ending3]
            num_choices = len(endings)

            # Flatten the pairs (so tokenizer can handle them all at once)
            first_sentences = [[s] * num_choices for s in start_phrase]
            second_sentences = list(zip(*endings))  # [(ending0[i], ending1[i], ...), ...]

            # Flatten properly
            first_sentences_flat = sum(first_sentences, [])  # still okay, already lists
            second_sentences_flat = [s for tup in second_sentences for s in tup]

            # Tokenize
            enc = tokenizer(
                first_sentences_flat,
                second_sentences_flat,
                padding=True,
                truncation=True,
                max_length=128,
                return_tensors="pt"
            )
            enc = {k: v.to(dev, non_blocking=True) for k, v in enc.items()}
            input_ids = enc["input_ids"].view(len(batch["label"]), 4, -1)
            attention_mask = enc["attention_mask"].view(len(batch["label"]), 4, -1)

            labels = torch.tensor(batch["label"], dtype=torch.long)
            labels = labels.to(dev, non_blocking=True)

            opt.zero_grad(set_to_none=True)
            with amp.autocast("cuda", enabled=cfg.get("mixed_precision", True)):
                logits = model(input_ids, attention_mask)
                loss = criterion(logits, labels)

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

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
                start_phrase = list(batch["sent1"])
                ending0, ending1, ending2, ending3 = (
                    list(batch["next_sentence0"]),
                    list(batch["next_sentence1"]),
                    list(batch["next_sentence2"]),
                    list(batch["next_sentence3"]),
                )

                #start_phrase = tokenizer(list(start_phrase), return_tensors="pt", padding="longest", truncation=True, max_length=cfg["max_length"])
                #start_phrase = {k: v.to(dev, non_blocking=True) for k, v in start_phrase.items()}

                endings = [ending0, ending1, ending2, ending3]
                num_choices = len(endings)

                # Flatten the pairs (so tokenizer can handle them all at once)
                first_sentences = [[s] * num_choices for s in start_phrase]
                second_sentences = list(zip(*endings))  # [(ending0[i], ending1[i], ...), ...]

                # Flatten properly
                first_sentences_flat = sum(first_sentences, [])  # still okay, already lists
                second_sentences_flat = [s for tup in second_sentences for s in tup]

                # Tokenize
                enc = tokenizer(
                    first_sentences_flat,
                    second_sentences_flat,
                    padding=True,
                    truncation=True,
                    max_length=128,
                    return_tensors="pt"
                )
                enc = {k: v.to(dev, non_blocking=True) for k, v in enc.items()}
                input_ids = enc["input_ids"].view(len(batch["label"]), 4, -1)
                attention_mask = enc["attention_mask"].view(len(batch["label"]), 4, -1)

                labels = torch.tensor(batch["label"], dtype=torch.long)
                labels = labels.to(dev, non_blocking=True)
                logits = model(input_ids, attention_mask)
                v_true.extend(labels.cpu().tolist())
                v_pred.extend(logits.cpu().argmax(-1).tolist())

        val_acc = accuracy_score(v_true, v_pred)
        val_f1 = f1_score(v_true, v_pred, average='macro')
        print(f"          val: acc={val_acc:.4f} f1={val_f1:.4f}")

    return model

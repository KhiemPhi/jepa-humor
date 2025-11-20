from dataclasses import dataclass
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from torch.optim.swa_utils import AveragedModel, SWALR, update_bn
from transformers import AutoTokenizer, AutoModel
import os
from torch import amp
import numpy as np
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from .utils import set_seed, device, load_encoder_state
from .data import load_hf_splits, make_dataloaders
import evaluate
from .process_data import preprocess_training_examples, preprocess_validation_examples, compute_metrics

metric = evaluate.load("squad")
max_length = 384
stride = 128
n_best = 20  # Define n_best for compute_metrics
max_answer_length = 30 # Define max_answer_length for compute_metrics


class QAModel(nn.Module):
    def __init__(self, encoder, hidden_size=768):
        super().__init__()
        self.encoder = encoder
        self.hidden_size = hidden_size
        self.qa_outputs = nn.Linear(self.hidden_size, 2)

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        seq = outputs[0]
        logits = self.qa_outputs(seq)
        start_logits, end_logits = logits.split(1, dim=-1)
        return start_logits.squeeze(-1), end_logits.squeeze(-1)



def finetune(cfg, encoder_ckpt: str):
    set_seed(cfg["seed"])
    dev = device()

    # 1. Load Encoder
    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"])
    base = AutoModel.from_pretrained(cfg["model_name"])
    if encoder_ckpt and encoder_ckpt != "fresh":
        print(f"Loading pretrained encoder from {encoder_ckpt}")
        base = load_encoder_state(base, encoder_ckpt, map_location="cpu")
    else: 
        print(f"Loading fresh encoder")

    # Load SQuAD dataset
    raw_datasets = load_hf_splits(cfg["dataset"])
   
    train_dataset = raw_datasets["train"].map(
        lambda batch: preprocess_training_examples(batch, tokenizer),
        batched=True,
        remove_columns=raw_datasets["train"].column_names,
    )

    validation_dataset = raw_datasets["validation"].map(
        lambda batch: preprocess_validation_examples(batch, tokenizer = tokenizer),
        batched=True,
        remove_columns=raw_datasets["validation"].column_names,
    )

    train_loader, val_loader = make_dataloaders(train_dataset, validation_dataset, cfg["batch_size"], cfg["num_workers"])
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(validation_dataset)}")

    # Initialize QA model
    model = QAModel(base, hidden_size=base.config.hidden_size).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"])
    
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # ignore index for invalid positions
    
    scaler = GradScaler(enabled=cfg.get("mixed_precision", True))

    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        running_loss = 0.0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch}"):
            start_positions, end_positions = batch["start_positions"], batch["end_positions"]

            start_positions = torch.tensor(start_positions).to(dev)
            end_positions = torch.tensor(end_positions).to(dev)

            input_ids = batch["input_ids"].to(dev)
            #input_ids = input_ids.t()
            attention_mask = batch["attention_mask"].to(dev)
            #attention_mask = attention_mask.t()

            opt.zero_grad(set_to_none=True)

            with amp.autocast(device_type='cuda'):
                start_logits, end_logits = model(input_ids, attention_mask)
                loss = (criterion(start_logits, start_positions) + criterion(end_logits, end_positions)) / 2

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            running_loss += loss.item()

        model.eval()
        start_logits = []
        end_logits = []
        for batch in tqdm(val_loader, desc=f"[Val] Epoch {epoch}"):
            input_ids = batch["input_ids"].to(dev)
            #input_ids = input_ids.t()
            attention_mask = batch["attention_mask"].to(dev)
            #attention_mask = attention_mask.t()
            with torch.no_grad():
                start,end = model(input_ids,attention_mask)

            start_logits.append((start).cpu().numpy())
            end_logits.append((end).cpu().numpy())

        start_logits = np.concatenate(start_logits)
        end_logits = np.concatenate(end_logits)
        start_logits = start_logits[: len(validation_dataset)]
        end_logits = end_logits[: len(validation_dataset)]

        val_score = compute_metrics(
            start_logits, end_logits, validation_dataset, raw_datasets["validation"]
        )
        val_em = val_score["exact_match"]
        val_f1 = val_score["f1"]
        print(f"          val: EM={val_em:.4f} F1={val_f1:.4f}")
    return model
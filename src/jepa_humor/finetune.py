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

metric = evaluate.load("squad")

class QAModel(nn.Module):
    def __init__(self, encoder, hidden_size=768):
        super().__init__()
        self.encoder = encoder
        self.hidden_size = hidden_size
        self.qa_outputs = nn.Linear(self.hidden_size, 2)
    
    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids = input_ids,
                               attention_mask=attention_mask)

        sequence_output = outputs[0]

        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        
        return start_logits, end_logits

def find_answer_positions(enc, answer_starts, answers, max_length):
    offset_mapping = enc["offset_mapping"]   # [batch, seq, 2]
    start_positions = []
    end_positions = []

    for i in range(len(answers)):
        offsets = offset_mapping[i]
        answer_text = answers[i]
        answer_char_start = answer_starts[i]
        answer_char_end = answer_char_start + len(answer_text)

        start_pos = 0
        end_pos = 0

        # Search through offset mapping
        for idx, (start, end) in enumerate(offsets):
            if start <= answer_char_start < end:
                start_pos = idx
            if start < answer_char_end <= end:
                end_pos = idx

        # Safety: clamp to max_length
        if start_pos >= max_length:
            start_pos = max_length - 1
        if end_pos >= max_length:
            end_pos = max_length - 1

        start_positions.append(start_pos)
        end_positions.append(end_pos)

    return start_positions, end_positions


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
    ds = load_hf_splits(cfg["dataset"])
   
    # Create dataloaders for SQuAD
    train_loader, val_loader = make_dataloaders(ds["train"], ds["validation"], cfg["batch_size"], cfg["num_workers"])
    print(f"Train dataset size: {len(ds['train'])}")
    print(f"Validation dataset size: {len(ds['validation'])}")

    # Initialize QA model
    model = QAModel(base, hidden_size=base.config.hidden_size).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"])
    
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # ignore index for invalid positions
    
    scaler = GradScaler(enabled=cfg.get("mixed_precision", True))

    for epoch in range(1, cfg["epochs"] + 1):

        model.train()
        running_loss = 0.0

        all_predictions = []
        all_references = []

        for batch in tqdm(train_loader, desc=f"Finetune {epoch}/{cfg['epochs']}", leave=False):

            contexts = batch["context"]
            questions = batch["question"]
            answers = batch["answer_text"]
            answer_starts = batch["answer_start"]

            enc = tokenizer(
                questions,
                contexts,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=cfg["max_length"],
                return_offsets_mapping=True
            )

            input_ids = enc["input_ids"].to(dev)
            attention_mask = enc["attention_mask"].to(dev)
            offset_mapping = enc["offset_mapping"]

            start_positions, end_positions = find_answer_positions(
                enc, answer_starts, answers, cfg["max_length"]
            )

            start_positions = torch.tensor(start_positions, device=dev)
            end_positions = torch.tensor(end_positions, device=dev)

            opt.zero_grad(set_to_none=True)

            with amp.autocast("cuda", enabled=cfg.get("mixed_precision", True)):
                start_logits, end_logits = model(input_ids, attention_mask)

                loss_start = criterion(start_logits, start_positions)
                loss_end = criterion(end_logits, end_positions)
                loss = (loss_start + loss_end) / 2

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            running_loss += loss.item()

            # Decode predictions
            start_idx = start_logits.argmax(dim=-1).cpu().tolist()
            end_idx = end_logits.argmax(dim=-1).cpu().tolist()
            offsets = offset_mapping.tolist()

            for i in range(len(contexts)):
                s = start_idx[i]
                e = end_idx[i]
                off = offsets[i]

                if s < len(off) and e < len(off) and s <= e:
                    start_char = off[s][0]
                    end_char = off[e][1]
                    pred = contexts[i][start_char:end_char]
                else:
                    pred = ""

                all_predictions.append({
                    "id": str(len(all_predictions)),
                    "prediction_text": pred
                })

                all_references.append({
                    "id": str(len(all_references)),
                    "answers": {
                        "text": [answers[i]],
                        "answer_start": [answer_starts[i]]
                    }
                })

        # Compute EM & F1 for the entire epoch
        score = metric.compute(
            predictions=all_predictions,
            references=all_references
        )
        train_em = score["exact_match"]
        train_f1 = score["f1"]
        
        print(f"[Finetune] epoch {epoch}: loss={running_loss/len(train_loader):.4f} EM={train_em:.4f} F1={train_f1:.4f}")

        # Validation
        # Eval on training set for EM/F1
        model.eval()

        all_predictions = []
        all_references = []

        with torch.no_grad():
            for batch in data_loader:
                contexts = batch["context"]
                questions = batch["question"]
                answers = batch["answer_text"]
                answer_starts = batch["answer_start"]

                enc = tokenizer(
                    questions,
                    contexts,
                    return_tensors="pt",
                    padding="max_length",
                    truncation=True,
                    max_length=cfg["max_length"],
                    return_offsets_mapping=True
                )

                input_ids = enc["input_ids"].to(dev)
                attention_mask = enc["attention_mask"].to(dev)
                offset_mapping = enc["offset_mapping"].tolist()

                start_logits, end_logits = model(input_ids, attention_mask)

                start_idx = start_logits.argmax(dim=-1).cpu().tolist()
                end_idx = end_logits.argmax(dim=-1).cpu().tolist()

                for i in range(len(contexts)):
                    s = start_idx[i]
                    e = end_idx[i]
                    off = offset_mapping[i]

                    if s < len(off) and e < len(off) and s <= e:
                        start_char = off[s][0]
                        end_char = off[e][1]
                        pred_text = contexts[i][start_char:end_char]
                    else:
                        pred_text = ""

                    all_predictions.append({
                        "id": str(len(all_predictions)),
                        "prediction_text": pred_text
                    })

                    all_references.append({
                        "id": str(len(all_references)),
                        "answers": {
                            "text": [answers[i]],
                            "answer_start": [answer_starts[i]]
                        }
                    })

        score = metric.compute(
            predictions=all_predictions,
            references=all_references
        )
        val_em = score["exact_match"]
        val_f1 = score["f1"]
        
        print(f"          val: EM={val_em:.4f} F1={val_f1:.4f}")
    return model
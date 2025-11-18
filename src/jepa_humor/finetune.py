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

class QAModel(nn.Module):
    def __init__(self, encoder, hidden_size=768, dropout=0.1):
        super().__init__()
        self.encoder = encoder
        self.dropout = nn.Dropout(dropout)
        self.qa_outputs = nn.Linear(hidden_size, 2)  # start and end logits
    
    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state
        sequence_output = self.dropout(sequence_output)
        logits = self.qa_outputs(sequence_output)
        
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)
        
        return start_logits, end_logits

def compute_squad_metrics(start_logits, end_logits, input_ids, tokenizer, 
                         start_positions, end_positions, answers, contexts):
    """Compute EM and F1 scores for SQuAD"""
    pred_start = torch.argmax(start_logits, dim=1)
    pred_end = torch.argmax(end_logits, dim=1)
    
    em_scores = []
    f1_scores = []
    
    def exact_match(pred, truth):
        return pred.strip().lower() == truth.strip().lower()
    
    def f1_score(pred, truth):
        pred_tokens = pred.strip().lower().split()
        truth_tokens = truth.strip().lower().split()
        
        if len(pred_tokens) == 0 or len(truth_tokens) == 0:
            return 0
            
        common = set(pred_tokens) & set(truth_tokens)
        if len(common) == 0:
            return 0
            
        precision = len(common) / len(pred_tokens)
        recall = len(common) / len(truth_tokens)
        
        return 2 * (precision * recall) / (precision + recall)
    
    for i in range(len(pred_start)):
        # Reconstruct predicted answer
        pred_tokens = input_ids[i][pred_start[i]:pred_end[i]+1]
        pred_answer = tokenizer.decode(pred_tokens, skip_special_tokens=True)
        gt_answer = answers[i]
        
        em = exact_match(pred_answer, gt_answer)
        f1 = f1_score(pred_answer, gt_answer)
        
        em_scores.append(em)
        f1_scores.append(f1)
    
    return np.mean(em_scores), np.mean(f1_scores)

def find_answer_positions(encodings, answer_starts, answer_texts, max_length):
    """Convert char positions to token positions for SQuAD"""
    start_positions = []
    end_positions = []
    
    # Get offsets from the encodings - FIXED: use correct attribute access
    offsets = encodings['offset_mapping']
    
    for i in range(len(encodings['input_ids'])):
        start_char = answer_starts[i]
        end_char = start_char + len(answer_texts[i])
        
        start_pos = 0
        end_pos = 0
        found_start = False
        found_end = False
        
        # Find token indices for answer span
        for idx, (start, end) in enumerate(offsets[i]):
            if start == 0 and end == 0:  # Skip padding tokens
                continue
            if not found_start and start <= start_char < end:
                start_pos = idx
                found_start = True
            if not found_end and start < end_char <= end:
                end_pos = idx
                found_end = True
        
        # If answer not found, use fallback (CLS token)
        if not found_start or not found_end:
            start_pos = 0
            end_pos = 0
        
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
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=0.01)
    
    # Setup SWA
    swa_start_epoch = max(1, int(cfg["epochs"] * 0.7))
    swa_model = AveragedModel(model)
    swa_scheduler = SWALR(opt, swa_lr=cfg["lr"] * 0.5)
    use_swa = cfg.get("use_swa", True)
    
    if use_swa:
        print(f"SWA will start at epoch {swa_start_epoch}/{cfg['epochs']}")
    
    # Loss function for QA
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # ignore index for invalid positions
    
    scaler = GradScaler(enabled=cfg.get("mixed_precision", True))

    for epoch in range(1, cfg["epochs"] + 1):
        # Check if SWA phase starts this epoch
        if use_swa and epoch == swa_start_epoch:
            print(f"Starting SWA at epoch {epoch}")
        
        # Training
        model.train()
        running_loss = 0.0
        all_start_logits = []
        all_end_logits = []
        all_start_positions = []
        all_end_positions = []
        all_input_ids = []
        all_answers = []
        all_contexts = []
        
        for batch in tqdm(train_loader, desc=f"Finetune {epoch}/{cfg['epochs']}", leave=False):
            contexts = batch["context"]
            questions = batch["question"]
            answers = batch["answer_text"]  # Direct answer text
            answer_starts = batch["answer_start"]  # Direct answer start positions
            
            # Tokenize question + context - FIXED: ensure offset_mapping is returned
            enc = tokenizer(
                questions,
                contexts,
                return_tensors="pt",
                padding="max_length",
                truncation=True,
                max_length=cfg["max_length"],
                return_offsets_mapping=True  # Ensure this is included
            )
            
            input_ids = enc["input_ids"].to(dev)
            attention_mask = enc["attention_mask"].to(dev)
            offset_mapping = enc["offset_mapping"]  # Get offset mapping
            
            # Compute answer positions - FIXED: pass offset_mapping directly
            start_positions, end_positions = find_answer_positions(
                enc, answer_starts, answers, cfg["max_length"]
            )
            
            start_positions = torch.tensor(start_positions, device=dev)
            end_positions = torch.tensor(end_positions, device=dev)
            
            # Training step
            opt.zero_grad(set_to_none=True)
            with amp.autocast("cuda", enabled=cfg.get("mixed_precision", True)):
                start_logits, end_logits = model(input_ids, attention_mask)
                loss_start = criterion(start_logits, start_positions)
                loss_end = criterion(end_logits, end_positions)
                loss = (loss_start + loss_end) / 2
            
            scaler.scale(loss).backward()
            
            # Gradient clipping
            max_grad_norm = cfg.get("max_grad_norm", 1.0)
            if max_grad_norm > 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            
            scaler.step(opt)
            scaler.update()
            
            # Update SWA if in SWA phase
            if use_swa and epoch >= swa_start_epoch:
                swa_model.update_parameters(model)
                swa_scheduler.step()
            
            running_loss += loss.item()
            
            # Store for metrics computation
            all_start_logits.append(start_logits.detach().cpu())
            all_end_logits.append(end_logits.detach().cpu())
            all_start_positions.append(start_positions.cpu())
            all_end_positions.append(end_positions.cpu())
            all_input_ids.append(input_ids.cpu())
            all_answers.extend(answers)
            all_contexts.extend(contexts)
        
        # Compute training metrics
        if all_start_logits:  # Check if we have any data
            start_logits_all = torch.cat(all_start_logits)
            end_logits_all = torch.cat(all_end_logits)
            start_positions_all = torch.cat(all_start_positions)
            end_positions_all = torch.cat(all_end_positions)
            input_ids_all = torch.cat(all_input_ids)
            
            train_em, train_f1 = compute_squad_metrics(
                start_logits_all, end_logits_all, input_ids_all, tokenizer,
                start_positions_all, end_positions_all, all_answers, all_contexts
            )
        else:
            train_em, train_f1 = 0.0, 0.0
        
        print(f"[Finetune] epoch {epoch}: loss={running_loss/len(train_loader):.4f} EM={train_em:.4f} F1={train_f1:.4f}")

        # Validation
        model.eval()
        val_loss = 0.0
        val_start_logits = []
        val_end_logits = []
        val_start_positions = []
        val_end_positions = []
        val_input_ids = []
        val_answers = []
        val_contexts = []
        
        with torch.no_grad():
            for batch in val_loader:
                contexts = batch["context"]
                questions = batch["question"]
                answers = batch["answer_text"]
                answer_starts = batch["answer_start"]
                
                # Tokenize
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
                
                # Compute answer positions
                start_positions, end_positions = find_answer_positions(
                    enc, answer_starts, answers, cfg["max_length"]
                )
                
                start_positions = torch.tensor(start_positions, device=dev)
                end_positions = torch.tensor(end_positions, device=dev)
                
                # Forward pass
                start_logits, end_logits = model(input_ids, attention_mask)
                loss_start = criterion(start_logits, start_positions)
                loss_end = criterion(end_logits, end_positions)
                loss = (loss_start + loss_end) / 2
                val_loss += loss.item()
                
                # Store for metrics
                val_start_logits.append(start_logits.cpu())
                val_end_logits.append(end_logits.cpu())
                val_start_positions.append(start_positions.cpu())
                val_end_positions.append(end_positions.cpu())
                val_input_ids.append(input_ids.cpu())
                val_answers.extend(answers)
                val_contexts.extend(contexts)
        
        # Compute validation metrics
        if val_start_logits:  # Check if we have any data
            val_start_logits_all = torch.cat(val_start_logits)
            val_end_logits_all = torch.cat(val_end_logits)
            val_start_positions_all = torch.cat(val_start_positions)
            val_end_positions_all = torch.cat(val_end_positions)
            val_input_ids_all = torch.cat(val_input_ids)
            
            val_em, val_f1 = compute_squad_metrics(
                val_start_logits_all, val_end_logits_all, val_input_ids_all, tokenizer,
                val_start_positions_all, val_end_positions_all, val_answers, val_contexts
            )
        else:
            val_em, val_f1 = 0.0, 0.0
        
        print(f"          val: loss={val_loss/len(val_loader):.4f} EM={val_em:.4f} F1={val_f1:.4f}")
    
    # Final SWA processing
    if use_swa:
        print("Updating batch normalization statistics for SWA model...")
        model.eval()
        with torch.no_grad():
            for batch in tqdm(train_loader, desc="Updating BN stats", leave=False):
                contexts = batch["context"]
                questions = batch["question"]
                enc = tokenizer(
                    questions,
                    contexts,
                    return_tensors="pt",
                    padding="max_length",
                    truncation=True,
                    max_length=cfg["max_length"]
                )
                enc = {k: v.to(dev, non_blocking=True) for k, v in enc.items()}
                _ = swa_model(enc["input_ids"], enc["attention_mask"])
        
        print("Replacing model with SWA model...")
        model.load_state_dict(swa_model.module.state_dict())
        print("SWA model loaded!")
    
    # Final evaluation
    model.eval()
    final_start_logits = []
    final_end_logits = []
    final_start_positions = []
    final_end_positions = []
    final_input_ids = []
    final_answers = []
    final_contexts = []
    
    with torch.no_grad():
        for batch in val_loader:
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
            
            start_positions, end_positions = find_answer_positions(
                enc, answer_starts, answers, cfg["max_length"]
            )
            
            start_logits, end_logits = model(input_ids, attention_mask)
            
            final_start_logits.append(start_logits.cpu())
            final_end_logits.append(end_logits.cpu())
            final_start_positions.append(torch.tensor(start_positions))
            final_end_positions.append(torch.tensor(end_positions))
            final_input_ids.append(input_ids.cpu())
            final_answers.extend(answers)
            final_contexts.extend(contexts)
    
    # Final metrics
    if final_start_logits:  # Check if we have any data
        final_start_logits_all = torch.cat(final_start_logits)
        final_end_logits_all = torch.cat(final_end_logits)
        final_start_positions_all = torch.cat(final_start_positions)
        final_end_positions_all = torch.cat(final_end_positions)
        final_input_ids_all = torch.cat(final_input_ids)
        
        final_em, final_f1 = compute_squad_metrics(
            final_start_logits_all, final_end_logits_all, final_input_ids_all, tokenizer,
            final_start_positions_all, final_end_positions_all, final_answers, final_contexts
        )
    else:
        final_em, final_f1 = 0.0, 0.0
    
    print(f"Final validation: EM={final_em:.4f} F1={final_f1:.4f}")
     
    return model
from dataclasses import dataclass
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from sklearn.metrics import accuracy_score
from .utils import set_seed, device, load_encoder_state
from .data import load_hf_splits, make_dataloaders, load_hf_splits_test
import torch.nn.utils as nn_utils
import numpy as np
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

class TemperatureClassifier(nn.Module):
    def __init__(self, in_dim, num_labels):
        super().__init__()
        self.linear = nn.Linear(in_dim, num_labels)
        self.T = nn.Parameter(torch.tensor(1.0))

    def forward(self, x):
        return self.linear(x) / self.T

class EntailmentSentenceClassifier(nn.Module):
    def __init__(self, encoder, hidden_size=768, num_labels=2, dropout=0.1):
        super().__init__()
        self.encoder = encoder
        self.dropout = nn.Dropout(dropout)
        self.classifier = TemperatureClassifier(hidden_size, num_labels)

    def forward(self, input_ids, attention_mask, inputs_embeds=None):
        # allow passing inputs_embeds (used for embedding-space adversary)
        if inputs_embeds is not None:
            out = self.encoder(attention_mask=attention_mask, inputs_embeds=inputs_embeds)
        else:
            out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = out.pooler_output if getattr(out, "pooler_output", None) is not None else out.last_hidden_state[:, 0, :]
        return self.classifier(self.dropout(pooled))

# -------------------------
# Utilities: get embeddings + named param for embed
# -------------------------
def get_word_embedding_module_name(model):
    # common HF BERT-like attribute path to embeddings
    # returns a substring to match named_parameters like 'embeddings.word_embeddings'
    # tries several common patterns.
    candidates = [
        "embeddings.word_embeddings",   # bert / roberta
        "embeddings",                   # fallback
        "encoder.embed_tokens"          # t5-like (not used here but safe)
    ]
    for c in candidates:
        for n, _ in model.named_parameters():
            if c in n:
                return c
    return "embeddings.word_embeddings"

def get_input_embeddings(model):
    # huggingface AutoModel: model.get_input_embeddings() exists
    try:
        return model.get_input_embeddings()
    except Exception:
        # fallback to name scan
        name = get_word_embedding_module_name(model)
        for n, p in model.named_parameters():
            if name in n:
                # try to walk attribute chain
                attrs = name.split('.')
                cur = model
                for a in attrs:
                    if hasattr(cur, a):
                        cur = getattr(cur, a)
                return cur
    raise RuntimeError("Could not find model input embeddings")

# -------------------------
# PGD adversary (embedding-space)
# -------------------------
def pgd_attack(model, batch_enc, labels, criterion, cfg, dev):
    """
    K-step PGD on embeddings. Returns tuple (clean_loss, adv_loss).
    This routine performs inner maximization and returns adversarial loss.
    """
    K = cfg.get("adv_pgd_k", 3)
    alpha = cfg.get("adv_pgd_alpha", 1e-1)      # step size
    epsilon = cfg.get("adv_epsilon", 1.0)      # Frobenius norm bound
    emb_layer = get_input_embeddings(model.encoder)
    input_ids = batch_enc["input_ids"]
    attention_mask = batch_enc["attention_mask"]

    # get original embeddings
    inputs_embeds = emb_layer(input_ids)
    delta = torch.zeros_like(inputs_embeds).uniform_(-epsilon, epsilon).to(dev)
    delta.requires_grad_()

    adv_loss = 0.0
    for t in range(K):
        # forward with perturbed embeddings
        logits = model(None, attention_mask, inputs_embeds + delta)
        loss = criterion(logits, labels)
        # maximize loss wrt delta
        loss.backward(retain_graph=True)
        grad = delta.grad.detach()
        # normalize grad
        grad_norm = torch.norm(grad.view(grad.shape[0], -1), dim=1).view(-1,1,1).clamp(min=1e-8)
        delta.data = delta.data + alpha * grad / grad_norm
        # project back to epsilon ball (Frobenius)
        delta_flat = delta.view(delta.shape[0], -1)
        norms = torch.norm(delta_flat, dim=1).clamp(min=1e-8)
        factor = torch.clamp(epsilon / norms, max=1.0).view(-1,1)
        delta.data = (delta_flat * factor).view_as(delta)
        delta.grad.zero_()
        adv_loss = loss.detach()  # keep last-step loss

    # final adv loss with delta (do not keep graph)
    with torch.no_grad():
        adv_logits = model(None, attention_mask, inputs_embeds + delta.detach())
        adv_loss_final = criterion(adv_logits, labels)
    return adv_loss_final, inputs_embeds, delta.detach()

# -------------------------
# FreeLB adversary (follows Algorithm 1 in FreeLB paper)
# -------------------------
def freelb_attack_and_step(model, batch_enc, labels, criterion, cfg, optimizer, scaler, dev):
    """
    Implements FreeLB-K as described in the FreeLB paper.
    This function performs internal adversarial ascent steps and returns the overall loss
    to backpropagate through model parameters as in FreeLB algorithm.
    """
    K = cfg.get("freelb_k", 3)
    alpha = cfg.get("freelb_alpha", 1e-1)
    epsilon = cfg.get("adv_epsilon", 1.0)

    emb_layer = get_input_embeddings(model.encoder)
    input_ids = batch_enc["input_ids"]
    attention_mask = batch_enc["attention_mask"]
    inputs_embeds_orig = emb_layer(input_ids)

    # initialize delta uniformly small
    delta = torch.zeros_like(inputs_embeds_orig).uniform_(-epsilon, epsilon).to(dev)
    delta.requires_grad_()

    acc_grad = None
    total_loss = 0.0

    for t in range(K):
        with autocast(enabled=cfg.get("mixed_precision", True)):
            logits = model(None, attention_mask, inputs_embeds_orig + delta)
            loss = criterion(logits, labels)
            loss = loss / K  # average over K as in FreeLB (accumulate)
        # compute gradients w.r.t model params and delta
        scaler.scale(loss).backward(retain_graph=True)
        total_loss += loss.detach().item()

        # gradient ascent on delta
        if delta.grad is None:
            # if delta has no grad, create it by computing grad of loss wrt delta
            grad_delta = torch.autograd.grad(loss, delta, retain_graph=True)[0]
        else:
            grad_delta = delta.grad.detach()

        # normalize grad_delta per sample
        grad_flat = grad_delta.view(grad_delta.shape[0], -1)
        grad_norm = torch.norm(grad_flat, dim=1).view(-1,1,1).clamp(min=1e-8)
        delta.data = delta.data + alpha * (grad_delta / grad_norm)
        # projection
        delta_flat = delta.data.view(delta.shape[0], -1)
        norms = torch.norm(delta_flat, dim=1).clamp(min=1e-8)
        factor = torch.clamp(epsilon / norms, max=1.0).view(-1,1)
        delta.data = (delta_flat * factor).view_as(delta).detach()
        delta.requires_grad_()
        # zero delta grad for next step
        if delta.grad is not None:
            delta.grad.zero_()

    # After K steps, perform optimizer step on accumulated gradients on model params
    # Note: model parameter grads have been accumulated across K steps
    return total_loss  # model grads already accumulated via backward calls

# -------------------------
# Main finetune entry
# -------------------------
def finetune(cfg, encoder_ckpt: str):
    dev = device()

    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"])
    base = AutoModel.from_pretrained(cfg["model_name"])
    if encoder_ckpt and encoder_ckpt != "fresh":
        print(f"Loading pretrained encoder from {encoder_ckpt}")
        base = load_encoder_state(base, encoder_ckpt, map_location="cpu")

    ds = load_hf_splits(cfg["dataset"])
    test_ds = load_hf_splits_test(cfg["test_dataset"])
    train_loader, val_loader = make_dataloaders(ds["train"], test_ds["validation"], cfg["batch_size"], cfg["num_workers"])

    model = EntailmentSentenceClassifier(base, hidden_size=base.config.hidden_size, num_labels=cfg.get("num_labels", 2)).to(dev)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg.get("weight_decay", 0.0))
    total_steps = len(train_loader) * cfg["epochs"]
    warmup_steps = int(cfg.get("warmup_ratio", 0.1) * total_steps)
    scheduler = get_linear_schedule_with_warmup(opt, warmup_steps, total_steps)

    criterion = nn.CrossEntropyLoss()
    scaler = GradScaler(enabled=cfg.get("mixed_precision", True))

    best_val_acc = 0.0
    best_model_state = None

    # seeds and runs as in your previous script
    set_seed(cfg.get("seed", 42))
    runs = cfg.get("runs", 1)
    for run in range(1, runs + 1):
        print(f"Run {run}/{runs}")
        # reinit model & optimizer for reproducibility between runs
        model = EntailmentSentenceClassifier(base, hidden_size=base.config.hidden_size, num_labels=cfg.get("num_labels", 2)).to(dev)
        opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg.get("weight_decay", 0.0))
        scheduler = get_linear_schedule_with_warmup(opt, warmup_steps, total_steps)
        scaler = GradScaler(enabled=cfg.get("mixed_precision", True))

        set_seed(cfg.get("seed", 42) + run)

        for epoch in range(1, cfg["epochs"] + 1):
            model.train()
            running_loss = 0.0
            y_true, y_pred = [], []

            for batch in tqdm(train_loader, desc=f"Finetune {epoch}/{cfg['epochs']}", leave=False):
                # build inputs
                # adapt fields for both single sentence and sentence-pair tasks
                # assume dataset yields "sentence1","sentence2" as before
                sent1 = batch.get("sentence1") or batch.get("sentence") or batch.get("question")
                sent2 = batch.get("sentence2") or batch.get("sentence") or batch.get("sentence2") or batch.get("question2")

                labels = torch.tensor(batch["label"], dtype=torch.long).to(dev, non_blocking=True)

                enc = tokenizer(
                    list(sent1), list(sent2),
                    return_tensors="pt",
                    padding="max_length",
                    truncation=True,
                    max_length=cfg["max_length"]
                )
                enc = {k: v.to(dev, non_blocking=True) for k, v in enc.items()}

                opt.zero_grad(set_to_none=True)

                adv_method = cfg.get("adv_method", "none").lower()
                if adv_method == "none":
                    with autocast(enabled=cfg.get("mixed_precision", True)):
                        logits = model(enc["input_ids"], enc["attention_mask"])
                        loss = criterion(logits, labels)
                    scaler.scale(loss).backward()
                    scaler.unscale_(opt)
                    nn_utils.clip_grad_norm_(model.parameters(), cfg.get("max_grad_norm", 1.0))
                    scaler.step(opt)
                    scaler.update()
                    scheduler.step()
                    running_loss += loss.item()
                    y_true.extend(labels.cpu().tolist())
                    y_pred.extend(logits.detach().cpu().argmax(-1).tolist())

                elif adv_method == "pgd":
                    # first compute clean forward/backward then do PGD ascent and add adversarial loss
                    emb_layer = get_input_embeddings(model.encoder)
                    with autocast(enabled=cfg.get("mixed_precision", True)):
                        logits = model(enc["input_ids"], enc["attention_mask"])
                        loss = criterion(logits, labels)
                    scaler.scale(loss).backward()
                    # PGD on embeddings to get adv_loss (we performed backward for param grads on clean)
                    adv_loss_final, _, _ = pgd_attack(model, enc, labels, criterion, cfg, dev)
                    scaler.scale(adv_loss_final).backward()
                    scaler.unscale_(opt)
                    nn_utils.clip_grad_norm_(model.parameters(), cfg.get("max_grad_norm", 1.0))
                    scaler.step(opt)
                    scaler.update()
                    scheduler.step()
                    running_loss += (loss.item() + adv_loss_final.item())
                    y_true.extend(labels.cpu().tolist())
                    y_pred.extend(logits.detach().cpu().argmax(-1).tolist())

                elif adv_method == "freelb":
                    # FreeLB performs K inner steps and accumulates grads over them.
                    # We follow FreeLB algorithm: run K ascent steps, accumulate model grads, then optimizer.step()
                    emb_layer = get_input_embeddings(model.encoder)

                    # Call freelb routine which runs K steps and does backward each step (scaler.scale used)
                    total_loss = freelb_attack_and_step(model, enc, labels, criterion, cfg, opt, scaler, dev)

                    scaler.unscale_(opt)
                    nn_utils.clip_grad_norm_(model.parameters(), cfg.get("max_grad_norm", 1.0))
                    scaler.step(opt)
                    scaler.update()
                    scheduler.step()
                    running_loss += total_loss
                    # For reporting, run a clean forward to get logits for metrics
                    with torch.no_grad():
                        logits = model(enc["input_ids"], enc["attention_mask"])
                    y_true.extend(labels.cpu().tolist())
                    y_pred.extend(logits.detach().cpu().argmax(-1).tolist())

                else:
                    raise ValueError(f"Unknown adv_method: {adv_method}")

            epoch_loss = running_loss / len(train_loader)
            train_acc = accuracy_score(y_true, y_pred)
            print(f"epoch {epoch}: loss={epoch_loss:.4f} acc={train_acc:.4f}")

            # Validation
            model.eval()
            v_true, v_pred = [], []
            with torch.no_grad():
                for batch in val_loader:
                    sent1 = batch.get("sentence1") or batch.get("sentence") or batch.get("question")
                    sent2 = batch.get("sentence2") or batch.get("sentence") or batch.get("question2")
                    labels = torch.tensor(batch["label"], dtype=torch.long).to(dev, non_blocking=True)
                    enc = tokenizer(
                        list(sent1), list(sent2),
                        return_tensors="pt",
                        padding="max_length",
                        truncation=True,
                        max_length=cfg["max_length"]
                    )
                    enc = {k: v.to(dev, non_blocking=True) for k, v in enc.items()}
                    logits = model(enc["input_ids"], enc["attention_mask"])
                    v_true.extend(labels.cpu().tolist())
                    v_pred.extend(logits.cpu().argmax(-1).tolist())
            val_acc = accuracy_score(v_true, v_pred)
            print(f"          val: acc={val_acc:.4f}")

        # check best after run
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = model.state_dict().copy()
            print(f"New best run validation acc {best_val_acc:.4f}")

    print(f"Best validation acc: {best_val_acc:.4f}")
    # load best
    model = EntailmentSentenceClassifier(base, hidden_size=base.config.hidden_size, num_labels=cfg.get("num_labels", 2)).to(dev)
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    return model

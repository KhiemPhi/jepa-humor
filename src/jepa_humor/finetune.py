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
import torch.nn.functional as F
from torch import amp
import nltk
from nltk.tokenize import sent_tokenize
nltk.download('punkt')
os.environ["TOKENIZERS_PARALLELISM"] = "false"

class ClaimEvidenceSelectorClassifier(nn.Module):
    """
    Claim-evidence reasoning model that works with pre-tokenized IDs.
    Inputs:
      - claim_ids: (B, Lc)
      - claim_mask: (B, Lc)
      - evidence_ids: (B, Ne, Le)
      - evidence_mask: (B, Ne, Le)
    Output:
      - logits: (B, num_classes)
    """

    def __init__(
        self,
        encoder,
        hidden_dim=768,
        k=3,
        num_classes=2,
        attn_heads=8,
        mlp_hidden=512,
        dropout=0.1,
        freeze_encoder=False,
    ):
        super().__init__()
        self.k = k
        self.hidden_dim = hidden_dim
        self.encoder = encoder

        if freeze_encoder:
            for p in self.encoder.parameters():
                p.requires_grad = False

        # Cross attention: query = claim CLS, key/value = evidence sentence CLSs
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=attn_heads, dropout=dropout, batch_first=True
        )

        # Classification head
        self.input_dim = (1 + k) * hidden_dim
        self.mlp = nn.Sequential(
            nn.LayerNorm(self.input_dim),
            nn.Linear(self.input_dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, num_classes)
        )

    def forward(self, claim_ids, claim_mask, evidence_ids, evidence_mask):
        """
        claim_ids:     (B, Lc)
        claim_mask:    (B, Lc)
        evidence_ids:  (B, Ne, Le)
        evidence_mask: (B, Ne, Le)
        """

        B, Ne, Le = evidence_ids.size()

        # Encode claims
        claim_outputs = self.encoder(input_ids=claim_ids, attention_mask=claim_mask, return_dict=True)
        claim_vecs = claim_outputs.last_hidden_state[:, 0, :]  # (B, hidden_dim)

        # Flatten evidence batch
        flat_evi_ids = evidence_ids.contiguous().view(B * Ne, Le)
        flat_evi_mask = evidence_mask.contiguous().view(B * Ne, Le)
        evidence_outputs = self.encoder(input_ids=flat_evi_ids, attention_mask=flat_evi_mask, return_dict=True)
        evidence_vecs = evidence_outputs.last_hidden_state[:, 0, :].view(B, Ne, self.hidden_dim)  # (B, Ne, hidden_dim)

        # Cross-attention: claim as query, evidence sentences as key/value
        # We make the claim vector a sequence of length 1 (Lq = 1)
        claim_q = claim_vecs.unsqueeze(1)  # (B, 1, hidden_dim)
        attn_output, attn_weights = self.cross_attn(query=claim_q, key=evidence_vecs, value=evidence_vecs)
        # attn_weights: (B, 1, Ne)
        attn_weights = attn_weights.squeeze(1)  # (B, Ne)

        # Top-k selection
        topk_vals, topk_idx = torch.topk(attn_weights, k=min(self.k, Ne), dim=1)
        # Gather top-k evidence sentence vectors
        batch_indices = torch.arange(B).unsqueeze(1).to(evidence_vecs.device)
        topk_sentences = evidence_vecs[batch_indices, topk_idx]  # (B, k, hidden_dim)

        # Pad if needed
        if topk_sentences.size(1) < self.k:
            pad_len = self.k - topk_sentences.size(1)
            pad = torch.zeros((B, pad_len, self.hidden_dim), device=evidence_vecs.device)
            topk_sentences = torch.cat([topk_sentences, pad], dim=1)

        # Concatenate claim and selected sentences
        combined = torch.cat([claim_q.squeeze(1), topk_sentences.reshape(B, -1)], dim=-1)  # (B, (1+k)*hidden_dim)

        # Classification
        logits = self.mlp(combined)  # (B, num_classes)
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
   
    train_loader, val_loader = make_dataloaders(ds["train"], ds["test"], cfg["batch_size"], cfg["num_workers"])
    print(f"Train dataset size: {len(ds['train'])}")
    print(f"Test dataset size: {len(ds['test'])}")

    model = ClaimEvidenceSelectorClassifier(base, hidden_dim=base.config.hidden_size).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"])
    criterion = nn.CrossEntropyLoss()
    scaler = GradScaler(enabled=cfg.get("mixed_precision", True))

    for epoch in range(1, cfg["epochs"] + 1):
        # train
        model.train()
        running, y_true, y_pred = 0.0, [], []
        for batch in tqdm(train_loader, desc=f"Finetune {epoch}/{cfg['epochs']}", leave=False):
            claims = batch["claim"]
            raw_evidences = batch["evidence"]     # list[str] (paragraphs or long text)
            evidences = [sent_tokenize(evi) for evi in raw_evidences]
            labels = torch.tensor(batch["label"], dtype=torch.long)

            enc_claims = tokenizer(list(claims), return_tensors="pt", padding="longest", truncation=True, max_length=cfg["max_length"])
            enc_claims = {k: v.to(dev, non_blocking=True) for k, v in enc_claims.items()}

            batch_evidence_ids, batch_evidence_mask = [], []
            for evi_list in evidences:
                enc_evi = tokenizer(
                    evi_list,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=cfg["max_length"]
                )
                ids = enc_evi["input_ids"]
                mask = enc_evi["attention_mask"]

                # pad to max sentences (Ne) across batch
                batch_evidence_ids.append(ids)
                batch_evidence_mask.append(mask)

            # pad to same number of evidence sentences
            max_ne = max(e.shape[0] for e in batch_evidence_ids)
            for i in range(len(batch_evidence_ids)):
                pad_sent = max_ne - batch_evidence_ids[i].size(0)
                if pad_sent > 0:
                    pad_ids = torch.zeros((pad_sent, cfg["max_length"]), dtype=torch.long)
                    pad_mask = torch.zeros((pad_sent, cfg["max_length"]), dtype=torch.long)
                    batch_evidence_ids[i] = torch.cat([batch_evidence_ids[i], pad_ids], dim=0)
                    batch_evidence_mask[i] = torch.cat([batch_evidence_mask[i], pad_mask], dim=0)

            evidence_ids = torch.stack(batch_evidence_ids).to(dev)
            evidence_mask = torch.stack(batch_evidence_mask).to(dev)


            labels = labels.to(dev, non_blocking=True)

            opt.zero_grad(set_to_none=True)
            with amp.autocast("cuda", enabled=cfg.get("mixed_precision", True)):
                logits = model(enc_claims["input_ids"], enc_claims["attention_mask"], evidence_ids, evidence_mask )
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
                claims = batch["claim"]
                raw_evidences = batch["evidence"]     # list[str] (paragraphs or long text)
                evidences = [sent_tokenize(evi) for evi in raw_evidences]
                labels = torch.tensor(batch["label"], dtype=torch.long)

                enc_claims = tokenizer(list(claims), return_tensors="pt", padding="longest", truncation=True, max_length=cfg["max_length"])
                enc_claims = {k: v.to(dev, non_blocking=True) for k, v in enc_claims.items()}
                batch_evidence_ids, batch_evidence_mask = [], []
                for evi_list in evidences:
                    enc_evi = tokenizer(
                        evi_list,
                        return_tensors="pt",
                        padding=True,
                        truncation=True,
                        max_length=cfg["max_length"]
                    )
                    ids = enc_evi["input_ids"]
                    mask = enc_evi["attention_mask"]

                    # pad to max sentences (Ne) across batch
                    batch_evidence_ids.append(ids)
                    batch_evidence_mask.append(mask)

                # pad to same number of evidence sentences
                max_ne = max(e.shape[0] for e in batch_evidence_ids)
                for i in range(len(batch_evidence_ids)):
                    pad_sent = max_ne - batch_evidence_ids[i].size(0)
                    if pad_sent > 0:
                        pad_ids = torch.zeros((pad_sent, cfg["max_length"]), dtype=torch.long)
                        pad_mask = torch.zeros((pad_sent, cfg["max_length"]), dtype=torch.long)
                        batch_evidence_ids[i] = torch.cat([batch_evidence_ids[i], pad_ids], dim=0)
                        batch_evidence_mask[i] = torch.cat([batch_evidence_mask[i], pad_mask], dim=0)

                evidence_ids = torch.stack(batch_evidence_ids).to(dev)
                evidence_mask = torch.stack(batch_evidence_mask).to(dev)


                labels = labels.to(dev, non_blocking=True)
                logits = model(enc_claims["input_ids"], enc_claims["attention_mask"], evidence_ids, evidence_mask )
                v_true.extend(labels.cpu().tolist())
                v_pred.extend(logits.cpu().argmax(-1).tolist())

        val_acc = accuracy_score(v_true, v_pred)
        val_f1 = f1_score(v_true, v_pred)
        print(f"          val: acc={val_acc:.4f} f1={val_f1:.4f}")

    return model

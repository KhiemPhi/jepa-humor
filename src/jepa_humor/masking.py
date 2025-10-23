import math, random, re
from typing import List, Optional, Sequence, Tuple, Dict, Any
import torch
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from transformers import AutoModel, AutoTokenizer
import nltk
nltk.download('punkt')

import torch.nn.functional as F

def compute_claim_evidence_entropy(
    model, tokenizer, claim: str, evidence: str, max_length: int = 256, tau: float = 0.5
) -> float:
    """
    Compute Shannon entropy over cosine similarity distribution
    between a claim and each sentence in the evidence.
    """

    # Split evidence into sentences
    sentences = nltk.sent_tokenize(evidence)
    if len(sentences) == 0:
        return 0.0

    # Encode claim
    claim_inputs = tokenizer(
        claim, return_tensors="pt", truncation=True, padding=True, max_length=max_length
    )
    with torch.no_grad():
        claim_emb = model(**claim_inputs).last_hidden_state.mean(dim=1)  # [1, hidden_dim]

    # Encode sentences
    sent_inputs = tokenizer(
        sentences, return_tensors="pt", truncation=True, padding=True, max_length=max_length
    )
    with torch.no_grad():
        sent_embs = model(**sent_inputs).last_hidden_state.mean(dim=1)  # [n_sents, hidden_dim]

    # Cosine similarities
    sims = F.cosine_similarity(claim_emb, sent_embs)  # [n_sents]

    # Convert to probability distribution with temperature scaling
    probs = F.softmax(sims / tau, dim=0)

    # Shannon entropy: H = -Σ p_i log(p_i)
    entropy = -torch.sum(probs * torch.log(probs + 1e-12)).item()

    return entropy


def process_evidence_deletions(
    claims, evidences, model, tokenizer, max_length=256, tau=0.5
):
    processed_evidence = []

    for claim, evidence in zip(claims, evidences):
        sentences = nltk.sent_tokenize(evidence)
        if len(sentences) <= 1:
            processed_evidence.append(evidence)
            continue

        # 1️⃣ Encode once
        with torch.no_grad():
            claim_inputs = tokenizer(claim, return_tensors="pt", truncation=True, max_length=max_length)
            claim_emb = model(**claim_inputs).last_hidden_state.mean(dim=1)  # [1,D]
            
            sent_inputs = tokenizer(sentences, return_tensors="pt", truncation=True, padding=True, max_length=max_length)
            sent_embs = model(**sent_inputs).last_hidden_state.mean(dim=1)    # [N,D]

        # 2️⃣ Compute base entropy once
        sims = F.cosine_similarity(claim_emb, sent_embs)
        probs = F.softmax(sims / tau, dim=0)
        base_entropy = -(probs * probs.log()).sum().item()

        # 3️⃣ Estimate information gain *without re-encoding*
        # When deleting a sentence, remove its entry and recompute entropy directly
        worst_ig = float("inf")
        worst_idx = -1

        for i in range(len(sentences)):
            new_sims = torch.cat([sims[:i], sims[i+1:]])
            new_probs = F.softmax(new_sims / tau, dim=0)
            new_entropy = -(new_probs * new_probs.log()).sum().item()

            info_gain = base_entropy - new_entropy  # positive = good removal
            if info_gain < worst_ig:
                worst_ig = info_gain
                worst_idx = i

        # 4️⃣ Delete least-informative sentence
        evidence_after_deletion = " ".join(
            sent for j, sent in enumerate(sentences) if j != worst_idx
        )
        processed_evidence.append(evidence_after_deletion)

    return processed_evidence


def make_claim_mask_views(
    claims: Sequence[str],
    evidences: Sequence[str],
    model,
    tokenizer,
    max_length: int = 256,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], torch.Tensor, List[List[Tuple[int, int]]]]:
    """
    Given evidence and claim pairs:
    - Tokenize the evidence as the model input (keep_tok)
    - Mask the claim tokens (target_tok)
    - Return a prediction mask and span indices for reconstruction

    Args:
        claims: Sequence of claim strings (to be masked and predicted)
        evidences: Sequence of evidence strings (context inputs)
        tokenizer: HuggingFace tokenizer
        max_length: Maximum tokenization length

    Returns:
        keep_tok: Tokenized evidence (model inputs)
        target_tok: Tokenized claim (model targets)
        predict_mask: Mask tensor for claim tokens to predict
        spans_used: List of token spans for each claim
    """
    assert len(claims) == len(evidences), "Claims and evidences must have the same length"

    evidences = process_evidence_deletions(claims, evidences, model, tokenizer)

    # Tokenize evidence as inputs
    keep_tok = tokenizer(
        list(evidences),
        return_tensors="pt",
        padding="longest",
        truncation=True,
        max_length=max_length,
    )

    # Tokenize claim as target
    target_tok = tokenizer(
        list(claims),
        return_tensors="pt",
        padding="longest",
        truncation=True,
        max_length=max_length,
    )

    B, T = target_tok["input_ids"].shape

    # Mask all claim tokens for prediction
    predict_mask = torch.ones_like(target_tok["attention_mask"], dtype=torch.float)

    # Record the full span of each claim (so model predicts all tokens)
    spans_used = []
    for b in range(B):
        seq_len = int(target_tok["attention_mask"][b].sum().item())
        spans_used.append([(0, seq_len)])

    return keep_tok, target_tok, predict_mask, spans_used
import math, random, re
from typing import List, Optional, Sequence, Tuple, Dict, Any
import torch
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from transformers import AutoModel, AutoTokenizer

def make_next_phrases_mask_views(
    startphrase: Sequence[str],
    nextphrase: Sequence[str],
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
    assert len(startphrase) == len(nextphrase), "Claims and evidences must have the same length"

    # Tokenize evidence as inputs
    keep_tok = tokenizer(
        list(startphrase),
        return_tensors="pt",
        padding="longest",
        truncation=True,
        max_length=max_length,
    )

    # Tokenize claim as target
    target_tok = tokenizer(
        list(nextphrase),
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

import math, random, re
from typing import List, Optional, Sequence, Tuple, Dict, Any
import torch
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
    from transformers import AutoModel, AutoTokenizer

def make_second_sent_mask_views(
    sent1: Sequence[str],
    sent2: Sequence[str],
    tokenizer,
    max_length: int = 256,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], torch.Tensor, List[List[Tuple[int, int]]]]:
    """
    Given sent1 and sent2 pairs:
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
    assert len(sent1) == len(sent2), "Claims and evidences must have the same length"

    # Tokenize evidence as inputs (pad to fixed max_length so time-dim matches target)
    keep_tok = tokenizer(
        list(sent1),
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=max_length,
    )

    # Tokenize claim as target (pad to fixed max_length so time-dim matches context)
    target_tok = tokenizer(
        list(sent2),
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=max_length,
    )

    B, T = target_tok["input_ids"].shape

    # Return a prediction mask aligned with the token time-dimension [B, T]
    # (model hidden states will have shape [B, T, D], the mask is applied along T)
    predict_mask = target_tok["attention_mask"].float()

    # Record the full span of each claim (so model predicts all tokens)
    spans_used = []
    for b in range(B):
        seq_len = int(target_tok["attention_mask"][b].sum().item())
        spans_used.append([(0, seq_len)])

    return keep_tok, target_tok, predict_mask, spans_used

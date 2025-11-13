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

    # Tokenize evidence as inputs (pad to fixed max_length so time-dim matches target)
    keep_tok = tokenizer(
        list(startphrase),
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=max_length,
    )

    # Tokenize claim as target (pad to fixed max_length so time-dim matches context)
    target_tok = tokenizer(
        list(nextphrase),
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


def encode_multiple_choice(batch, tokenizer, max_length, device):
    """
    batch: dict with keys ["sent1", "next_sentence0"..."next_sentence3", "label"]
    Returns:
      input_ids: (B, 4, L)
      attention_mask: (B, 4, L)
      labels: (B,)
      keep_tok: dict with tokenized incorrect pairs (context, incorrect_ending) for all 3 incorrect endings per example
      target_tok: dict with tokenized correct pair (context, correct_ending) for each example
    """

    endings = ["ending0", "ending1", "ending2", "ending3"]

    label=batch["label"]
    # Create keep_tok (all incorrect pairs) and target_tok (correct pair)
    keep_pairs = []  # List of (context, incorrect_ending) pairs
    target_pairs = []  # List of (context, correct_ending) pairs
    
    for i in range(len(batch["sent1"])):
        context = batch["sent1"][i] + " " + batch["sent2"][i]
        label = batch["label"][i]
        
        # target_tok: correct pair (context, correct_ending)
        correct_ending = batch[endings[label]][i]
        target_pairs.append((context, correct_ending))
        
        # keep_tok: all incorrect pairs (context, incorrect_ending)
        for j in range(4):
            if j != label:  # Skip the correct one
                incorrect_ending = batch[endings[j]][i]
                keep_pairs.append((context, incorrect_ending))

    # Tokenize keep_tok (all incorrect pairs)
    keep_tok = tokenizer(
        [p[0] for p in keep_pairs],
        [p[1] for p in keep_pairs],
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=max_length
    )
    keep_tok = {k: v.to(device) for k, v in keep_tok.items()}
    

    # Tokenize target_tok (correct pairs)
    target_tok = tokenizer(
        [p[0] for p in target_pairs],
        [p[1] for p in target_pairs],
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=max_length
    )
    target_tok = {k: v.to(device) for k, v in target_tok.items()} 
    
    

    return keep_tok, target_tok
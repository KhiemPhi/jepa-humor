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


def make_copa_format_mask_views(
    sent1: Sequence[str],
    sent2: Sequence[str],
    tokenizer,
    max_length: int = 256,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], torch.Tensor, List[List[Tuple[int, int]]]]:
    """
    COPA format: premise [SEP] true_sentence
    - Formats input as premise [SEP] true_sentence (consistent with fine-tuning)
    - Masks the true_sentence portion for prediction
    - Returns tokenized inputs with masking and target tokens
    
    Args:
        sent1: Sequence of premise strings
        sent2: Sequence of true_sentence strings (to be predicted)
        tokenizer: HuggingFace tokenizer
        max_length: Maximum tokenization length
        
    Returns:
        keep_tok: Tokenized premise [SEP] [MASK]... (input with true_sentence masked)
        target_tok: Tokenized true_sentence (target to predict)
        predict_mask: Mask tensor for true_sentence tokens (aligned with target_tok)
        spans_used: List of token spans for each true_sentence
    """
    assert len(sent1) == len(sent2), "Premises and true_sentences must have the same length"
    
    # Format as COPA format: premise [SEP] true_sentence
    combined_sentences = []
    for p, s in zip(sent1, sent2):
        combined_sentences.append(f"{p} {tokenizer.sep_token} {s}")
    
    # Tokenize the combined format
    combined_tok = tokenizer(
        combined_sentences,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=max_length,
    )
    
    # Tokenize just the true_sentence for target
    target_tok = tokenizer(
        list(sent2),
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=max_length,
    )
    
    B, T = target_tok["input_ids"].shape
    sep_token_id = tokenizer.sep_token_id
    mask_token_id = tokenizer.mask_token_id
    
    # Create keep_tok with masked true_sentence portion
    keep_input_ids = combined_tok["input_ids"].clone()
    
    # predict_mask should be aligned with target_tok shape [B, T]
    predict_mask = target_tok["attention_mask"].float().clone()
    
    spans_used = []
    for b in range(B):
        # Find [SEP] token position in combined sequence
        sep_positions = (combined_tok["input_ids"][b] == sep_token_id).nonzero(as_tuple=True)[0]
        if len(sep_positions) > 0:
            sep_pos = sep_positions[0].item()
            # Find actual sequence length (excluding padding)
            combined_seq_len = int(combined_tok["attention_mask"][b].sum().item())
            
            # Mask tokens after [SEP] (true_sentence portion) in keep_input_ids
            mask_start = sep_pos + 1
            mask_end = combined_seq_len
            keep_input_ids[b, mask_start:mask_end] = mask_token_id
            
            # Record span for this true_sentence (in target_tok coordinates, all tokens are used)
            target_seq_len = int(target_tok["attention_mask"][b].sum().item())
            spans_used.append([(0, target_seq_len)])
        else:
            # No [SEP] found, fallback: mask everything after [CLS]
            combined_seq_len = int(combined_tok["attention_mask"][b].sum().item())
            if combined_seq_len > 1:
                mask_start = 1  # Skip [CLS]
                mask_end = combined_seq_len
                keep_input_ids[b, mask_start:mask_end] = mask_token_id
                target_seq_len = int(target_tok["attention_mask"][b].sum().item())
                spans_used.append([(0, target_seq_len)])
            else:
                spans_used.append([(0, 0)])
    
    # Create keep_tok dict with masked input_ids
    keep_tok = {
        "input_ids": keep_input_ids,
        "attention_mask": combined_tok["attention_mask"],
        "token_type_ids": combined_tok.get("token_type_ids", None)
    }
    
    return keep_tok, target_tok, predict_mask, spans_used
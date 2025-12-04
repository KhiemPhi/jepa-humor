import math, random, re
from typing import List, Optional, Sequence, Tuple, Dict, Any
import torch
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from transformers import AutoModel, AutoTokenizer

def make_squad_views(contexts, questions, answer_texts, answer_starts, tokenizer, max_length=512):
    masked_texts = []
    for ctx, q, ans, start_char in zip(contexts, questions, answer_texts, answer_starts):
        end_char = start_char + len(ans)
        # full tokenization to get offsets and full input ids
        enc_full = tokenizer(
            ctx,
            return_offsets_mapping=True,
            truncation=False,
            add_special_tokens=False,
        )
        full_ids = enc_full["input_ids"]
        offsets = enc_full["offset_mapping"]  # list of (start,end)
    
        # find token indices that overlap the answer span
        ans_token_idxs = []
        for t_idx, (off_start, off_end) in enumerate(offsets):
            # skip tokens with zero offsets (should not happen with add_special_tokens=False)
            if off_end == 0 and off_start == 0:
                continue
            if not (off_end <= start_char or off_start >= end_char):
                ans_token_idxs.append(t_idx)
    
        if not ans_token_idxs:
            # fallback: use whole context if alignment fails
            window_ids = full_ids
            window_start_idx = 0
            window_end_idx = len(full_ids)
        else:
            ans_start_tok = ans_token_idxs[0]
            ans_end_tok = ans_token_idxs[-1] + 1  # exclusive
    
            # compute question token length (without special tokens)
            q_enc = tokenizer(q, add_special_tokens=False)
            q_len = len(q_enc["input_ids"])
    
            # determine number of special tokens added when pairing question and context
            try:
                special = tokenizer.num_special_tokens_to_add(pair=True)
            except Exception:
                special = 2
    
            max_ctx_tokens = max_length - q_len - special
            if max_ctx_tokens < 10:
                # ensure some context remains
                max_ctx_tokens = max_length - 4
    
            ans_len_tokens = ans_end_tok - ans_start_tok
            # center window on answer
            half = max(0, (max_ctx_tokens - ans_len_tokens) // 2)
            window_start_idx = max(0, ans_start_tok - half)
            window_end_idx = window_start_idx + max_ctx_tokens
            if window_end_idx > len(full_ids):
                window_end_idx = len(full_ids)
                window_start_idx = max(0, window_end_idx - max_ctx_tokens)
    
            window_ids = full_ids[window_start_idx:window_end_idx]
    
        # mask tokens in the window that overlap the answer (adjust token indices)
        mask_id = tokenizer.mask_token_id
        if mask_id is None:
            raise ValueError("Tokenizer has no mask token")
    
        # create a mutable copy
        window_ids_masked = list(window_ids)
        # compute offsets for the window slice
        window_offsets = offsets[window_start_idx:window_end_idx] if ans_token_idxs else offsets
    
        for local_idx, (off_start, off_end) in enumerate(window_offsets):
            if off_end == 0 and off_start == 0:
                continue
            if not (off_end <= start_char or off_start >= end_char):
                window_ids_masked[local_idx] = mask_id
    
        # decode masked window back to text for pairwise tokenization
        masked_text = tokenizer.decode(window_ids_masked, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        masked_texts.append(masked_text)
    
    # tokenize question + masked context pairs in batch
    pair_enc = tokenizer(
        questions,
        masked_texts,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=max_length,
    )
    
    # targets: tokenized answers
    target_tok = tokenizer(
        answer_texts,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=max_length,
    )
    
    predict_mask = target_tok["attention_mask"].float()
    
    spans_used = []
    for b in range(target_tok["input_ids"].shape[0]):
        seq_len = int(predict_mask[b].sum().item())
        spans_used.append([(0, seq_len)])
    
    return pair_enc, target_tok, predict_mask, spans_used

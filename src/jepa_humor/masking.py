import math, random, re
from typing import List, Optional, Sequence, Tuple, Dict, Any
import torch
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from transformers import AutoModel, AutoTokenizer


# ---------------------------------------------------------------------------
# Masking strategies
# ---------------------------------------------------------------------------

def _mask_random_spans(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    tokenizer,
    mask_ratio: float = 0.30,
    max_span_len: int = 3,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    SpanBERT-style random span masking on the TARGET sequence.
    Masks contiguous spans summing to ~mask_ratio of real tokens.

    Inductive bias: forces the encoder to fill gaps using cross-sentence
    context, not just local token statistics.
    """
    masked_ids = input_ids.clone()
    predict_mask = torch.zeros_like(attention_mask, dtype=torch.float)
    mask_token_id = tokenizer.mask_token_id
    B, T = input_ids.shape

    for b in range(B):
        real_len = int(attention_mask[b].sum().item())
        # skip [CLS] at 0 and [SEP] at real_len-1
        budget = max(1, int((real_len - 2) * mask_ratio))
        pos = 1  # start after [CLS]
        while budget > 0 and pos < real_len - 1:
            span_len = min(random.randint(1, max_span_len), budget, real_len - 1 - pos)
            for i in range(span_len):
                masked_ids[b, pos + i] = mask_token_id
                predict_mask[b, pos + i] = 1.0
            pos += span_len + random.randint(1, 3)  # random gap between spans
            budget -= span_len

    return masked_ids, predict_mask


def _mask_content_words(
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    tokenizer,
    mask_ratio: float = 0.40,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Preferentially mask content words (non-subword, non-stopword tokens).
    Falls back to random masking if not enough content tokens are found.

    Inductive bias: similarity is driven by content overlap, not function
    words. Masking content words forces the embedding to capture meaning,
    not syntax.
    """
    # Rough stopword set — extend as needed
    STOPWORDS = {
        "the", "a", "an", "is", "are", "was", "were", "be", "been",
        "being", "have", "has", "had", "do", "does", "did", "will",
        "would", "could", "should", "may", "might", "to", "of", "in",
        "on", "at", "by", "for", "with", "from", "and", "or", "but",
        "not", "no", "it", "its", "this", "that", "he", "she", "they",
    }

    masked_ids = input_ids.clone()
    predict_mask = torch.zeros_like(attention_mask, dtype=torch.float)
    mask_token_id = tokenizer.mask_token_id
    B, T = input_ids.shape

    for b in range(B):
        real_len = int(attention_mask[b].sum().item())
        tokens = tokenizer.convert_ids_to_tokens(input_ids[b, 1:real_len - 1].tolist())

        # Identify content token positions (skip [CLS], [SEP], subword continuations)
        content_positions = [
            i + 1  # offset for [CLS]
            for i, tok in enumerate(tokens)
            if not tok.startswith("##")
            and tok.lower() not in STOPWORDS
            and re.match(r"^[a-zA-Z]", tok)
        ]

        n_mask = max(1, int(len(content_positions) * mask_ratio))
        chosen = random.sample(content_positions, min(n_mask, len(content_positions)))

        for pos in chosen:
            masked_ids[b, pos] = mask_token_id
            predict_mask[b, pos] = 1.0

    return masked_ids, predict_mask


def _mask_overlap_biased(
    input_ids_keep: torch.Tensor,    # context sentence (s1)
    input_ids_target: torch.Tensor,  # target sentence  (s2)
    attention_mask_target: torch.Tensor,
    tokenizer,
    overlap_mask_prob: float = 0.70,
    non_overlap_mask_prob: float = 0.20,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Overlap-biased masking: tokens in s2 that also appear in s1 are masked
    with HIGH probability; tokens unique to s2 are masked with LOW probability.

    Inductive bias: the model must use s1's embedding to recover shared
    content — directly rewarding representations that encode lexical and
    semantic overlap. This is the strongest inductive bias for similarity.
    """
    masked_ids = input_ids_target.clone()
    predict_mask = torch.zeros_like(attention_mask_target, dtype=torch.float)
    mask_token_id = tokenizer.mask_token_id
    B, T = input_ids_target.shape

    for b in range(B):
        real_len = int(attention_mask_target[b].sum().item())

        # Vocabulary set of the context sentence (s1), ignoring special tokens
        special_ids = set(tokenizer.all_special_ids)
        keep_vocab = {
            int(input_ids_keep[b, t].item())
            for t in range(T)
            if int(input_ids_keep[b, t].item()) not in special_ids
        }

        for t in range(1, real_len - 1):  # skip [CLS] and [SEP]
            tok_id = int(input_ids_target[b, t].item())
            if tok_id in special_ids:
                continue
            p = overlap_mask_prob if tok_id in keep_vocab else non_overlap_mask_prob
            if random.random() < p:
                masked_ids[b, t] = mask_token_id
                predict_mask[b, t] = 1.0

    return masked_ids, predict_mask


# ---------------------------------------------------------------------------
# Strategy registry
# ---------------------------------------------------------------------------

MASK_STRATEGIES = ("random_span", "content_word", "overlap_biased", "mixed")


def _apply_masking(
    strategy: str,
    input_ids_keep: torch.Tensor,
    tok_target: Dict[str, torch.Tensor],
    tokenizer,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Dispatch to the chosen masking strategy. Returns (masked_ids, predict_mask)."""
    ids_t = tok_target["input_ids"]
    attn_t = tok_target["attention_mask"]

    if strategy == "random_span":
        return _mask_random_spans(ids_t, attn_t, tokenizer)

    elif strategy == "content_word":
        return _mask_content_words(ids_t, attn_t, tokenizer)

    elif strategy == "overlap_biased":
        return _mask_overlap_biased(input_ids_keep, ids_t, attn_t, tokenizer)

    elif strategy == "mixed":
        # Sample one strategy per call for curriculum diversity
        chosen = random.choice(["random_span", "content_word", "overlap_biased"])
        return _apply_masking(chosen, input_ids_keep, tok_target, tokenizer)

    else:
        raise ValueError(f"Unknown masking strategy: {strategy!r}. Choose from {MASK_STRATEGIES}")


# ---------------------------------------------------------------------------
# Main function
# ---------------------------------------------------------------------------

def make_second_sent_mask_views(
    sent1: Sequence[str],
    sent2: Sequence[str],
    tokenizer,
    max_length: int = 256,
    mask_strategy: str = "overlap_biased",
) -> Tuple[
    List[Dict[str, torch.Tensor]],       # keep_tok_list
    List[Dict[str, torch.Tensor]],       # target_tok_list  (ids replaced with masked ids)
    List[torch.Tensor],                  # predict_mask_list
    List[List[List[Tuple[int, int]]]],   # spans_used_list
]:
    """
    Symmetric masked reconstruction with pluggable masking strategies.

    Views:
      View 0 — context: sent1, target: sent2 (masked)
      View 1 — context: sent2, target: sent1 (masked)

    Masking strategies:
      "random_span"    — SpanBERT-style contiguous span masking
      "content_word"   — preferentially mask content / meaning-bearing words
      "overlap_biased" — mask shared tokens with high prob, unique tokens with low prob
      "mixed"          — sample one of the above randomly (recommended for training)

    Args:
        sent1:         First sentences in each pair
        sent2:         Second sentences in each pair
        tokenizer:     HuggingFace tokenizer
        max_length:    Padding/truncation length
        mask_strategy: One of MASK_STRATEGIES

    Returns:
        keep_tok_list:     [tok_s1, tok_s2]          context encodings (unmasked)
        target_tok_list:   [tok_s2_masked, tok_s1_masked]  targets with masked input_ids
        predict_mask_list: [mask_v0, mask_v1]         1.0 at positions to predict
        spans_used_list:   [spans_v0, spans_v1]       token spans per example
    """
    assert len(sent1) == len(sent2), "sent1 and sent2 must have the same length"
    assert mask_strategy in MASK_STRATEGIES, (
        f"mask_strategy must be one of {MASK_STRATEGIES}, got {mask_strategy!r}"
    )

    def _tokenize(sentences: Sequence[str]) -> Dict[str, torch.Tensor]:
        return tokenizer(
            list(sentences),
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=max_length,
        )

    def _spans_from_mask(predict_mask: torch.Tensor) -> List[List[Tuple[int, int]]]:
        """Convert a binary predict_mask [B, T] into contiguous span lists."""
        spans_used = []
        B, T = predict_mask.shape
        for b in range(B):
            spans = []
            start = None
            for t in range(T):
                if predict_mask[b, t] == 1.0:
                    if start is None:
                        start = t
                else:
                    if start is not None:
                        spans.append((start, t))
                        start = None
            if start is not None:
                spans.append((start, T))
            spans_used.append(spans)
        return spans_used

    tok_s1 = _tokenize(sent1)
    tok_s2 = _tokenize(sent2)

    # View 0: context = s1, reconstruct masked s2
    masked_ids_v0, predict_mask_v0 = _apply_masking(
        mask_strategy, tok_s1["input_ids"], tok_s2, tokenizer
    )
    tok_s2_masked = {**tok_s2, "input_ids": masked_ids_v0}

    # View 1: context = s2, reconstruct masked s1
    masked_ids_v1, predict_mask_v1 = _apply_masking(
        mask_strategy, tok_s2["input_ids"], tok_s1, tokenizer
    )
    tok_s1_masked = {**tok_s1, "input_ids": masked_ids_v1}

    keep_tok_list    = [tok_s1,        tok_s2]
    target_tok_list  = [tok_s2_masked, tok_s1_masked]
    predict_mask_list = [predict_mask_v0, predict_mask_v1]
    spans_used_list   = [
        _spans_from_mask(predict_mask_v0),
        _spans_from_mask(predict_mask_v1),
    ]

    return keep_tok_list, target_tok_list, predict_mask_list, spans_used_list
import math, random, re
from typing import List, Optional, Sequence, Tuple, Dict, Any
import torch
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from transformers import AutoModel, AutoTokenizer

# def make_second_sent_mask_views(
#     sent1: Sequence[str],
#     sent2: Sequence[str],
#     tokenizer,
#     max_length: int = 256,
# ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], torch.Tensor, List[List[Tuple[int, int]]]]:
#     """
#     Given sent1 and sent2 pairs:
#     - Tokenize the evidence as the model input (keep_tok)
#     - Mask the claim tokens (target_tok)
#     - Return a prediction mask and span indices for reconstruction

#     Args:
#         claims: Sequence of claim strings (to be masked and predicted)
#         evidences: Sequence of evidence strings (context inputs)
#         tokenizer: HuggingFace tokenizer
#         max_length: Maximum tokenization length

#     Returns:
#         keep_tok: Tokenized evidence (model inputs)
#         target_tok: Tokenized claim (model targets)
#         predict_mask: Mask tensor for claim tokens to predict
#         spans_used: List of token spans for each claim
#     """
#     assert len(sent1) == len(sent2), "Claims and evidences must have the same length"

#     # Tokenize evidence as inputs (pad to fixed max_length so time-dim matches target)
#     keep_tok = tokenizer(
#         list(sent1),
#         return_tensors="pt",
#         padding="max_length",
#         truncation=True,
#         max_length=max_length,
#     )

#     # Tokenize claim as target (pad to fixed max_length so time-dim matches context)
#     target_tok = tokenizer(
#         list(sent2),
#         return_tensors="pt",
#         padding="max_length",
#         truncation=True,
#         max_length=max_length,
#     )

#     B, T = target_tok["input_ids"].shape

#     # Return a prediction mask aligned with the token time-dimension [B, T]
#     # (model hidden states will have shape [B, T, D], the mask is applied along T)
#     predict_mask = target_tok["attention_mask"].float()

#     # Record the full span of each claim (so model predicts all tokens)
#     spans_used = []
#     for b in range(B):
#         seq_len = int(target_tok["attention_mask"][b].sum().item())
#         spans_used.append([(0, seq_len)])

#     return keep_tok, target_tok, predict_mask, spans_used


# def make_second_sent_mask_views(
#     premises: Sequence[str],
#     hypotheses: Sequence[str],
#     tokenizer,
#     max_length: int = 256,
#     mask_mode: str = "cls",  # "cls" | "content" | "all"
# ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], torch.Tensor]:

#     assert len(premises) == len(hypotheses)

#     context_tok = tokenizer(
#         list(premises),
#         return_tensors="pt",
#         padding="max_length",
#         truncation=True,
#         max_length=max_length,
#     )
#     target_tok = tokenizer(
#         list(hypotheses),
#         return_tensors="pt",
#         padding="max_length",
#         truncation=True,
#         max_length=max_length,
#     )

#     B, T = target_tok["input_ids"].shape
#     predict_mask = torch.zeros(B, T, dtype=torch.float)

#     if mask_mode == "cls":
#         # Predict only the hypothesis CLS vector
#         # Forces sentence-level semantic compression —
#         # directly aligned with how the fine-tuning classifier reads the encoder
#         predict_mask[:, 0] = 1.0

#     elif mask_mode == "content":
#         # Predict all real tokens except CLS and SEP
#         # Token-level reconstruction without boilerplate noise
#         predict_mask = target_tok["attention_mask"].clone().float()
#         predict_mask[:, 0] = 0.0  # zero CLS
#         for b in range(B):
#             seq_len = int(target_tok["attention_mask"][b].sum().item())
#             if seq_len > 1:
#                 predict_mask[b, seq_len - 1] = 0.0  # zero SEP

#     elif mask_mode == "all":
#         # Predict everything including CLS and SEP
#         # Most diffuse signal, least recommended
#         predict_mask = target_tok["attention_mask"].clone().float()

#     return context_tok, target_tok, predict_mask


# def make_second_sent_mask_views(
#     premises: Sequence[str],
#     hypotheses: Sequence[str],
#     tokenizer,
#     max_length: int = 256,
#     mask_mode: str = "content",
# ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], torch.Tensor]:
#     """
#     Context: full concatenated pair [CLS] premise [SEP] hypothesis [SEP]
#              → online encoder sees both sentences together, forcing it to
#                model the relationship rather than each sentence in isolation.

#     Target:  hypothesis only [CLS] hypothesis [SEP]
#              → EMA encoder produces hypothesis embeddings the predictor
#                must reconstruct from the joint representation.

#     This prevents the shortcut where the encoder ignores the premise
#     entirely and fits hypothesis-only features.
#     """
#     assert len(premises) == len(hypotheses)

#     # Context: concatenated pair — RoBERTa handles [CLS] s1 [SEP][SEP] s2 [SEP]
#     context_tok = tokenizer(
#         list(premises),
#         list(hypotheses),          # tokenizer handles pair concatenation
#         return_tensors="pt",
#         padding="max_length",
#         truncation=True,
#         max_length=max_length,
#     )

#     # Target: hypothesis alone — what the predictor must reconstruct
#     target_tok = tokenizer(
#         list(hypotheses),
#         return_tensors="pt",
#         padding="max_length",
#         truncation=True,
#         max_length=max_length,
#     )

#     B, T = target_tok["input_ids"].shape
#     predict_mask = torch.zeros(B, T, dtype=torch.float)

#     if mask_mode == "cls":
#         predict_mask[:, 0] = 1.0

#     elif mask_mode == "content":
#         predict_mask = target_tok["attention_mask"].clone().float()
#         predict_mask[:, 0] = 0.0
#         for b in range(B):
#             seq_len = int(target_tok["attention_mask"][b].sum().item())
#             if seq_len > 1:
#                 predict_mask[b, seq_len - 1] = 0.0

#     elif mask_mode == "all":
#         predict_mask = target_tok["attention_mask"].clone().float()

#     return context_tok, target_tok, predict_mask


def make_second_sent_mask_views(
    premises: Sequence[str],
    hypotheses: Sequence[str],
    tokenizer,
    max_length: int = 256,
    mask_mode: str = "second_half",
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], torch.Tensor]:

    assert len(premises) == len(hypotheses)

    context_tok = tokenizer(
        list(premises),
        list(hypotheses),
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=max_length,
    )
    target_tok = tokenizer(
        list(hypotheses),
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=max_length,
    )

    B, T = target_tok["input_ids"].shape
    predict_mask = torch.zeros(B, T, dtype=torch.float)

    if mask_mode == "cls":
        # Only the CLS token — sentence-level signal only.
        # Most aligned with fine-tuning since classifier reads CLS.
        predict_mask[:, 0] = 1.0

    elif mask_mode == "content":
        # All real tokens except CLS and SEP.
        # Token-level signal without boilerplate.
        predict_mask = target_tok["attention_mask"].clone().float()
        predict_mask[:, 0] = 0.0
        for b in range(B):
            seq_len = int(target_tok["attention_mask"][b].sum().item())
            if seq_len > 1:
                predict_mask[b, seq_len - 1] = 0.0

    elif mask_mode == "all":
        # Everything including CLS and SEP.
        predict_mask = target_tok["attention_mask"].clone().float()

    elif mask_mode == "cls_and_content":
        # CLS + content tokens, exclude SEP only.
        # Combines sentence-level and token-level signal in one objective.
        predict_mask = target_tok["attention_mask"].clone().float()
        for b in range(B):
            seq_len = int(target_tok["attention_mask"][b].sum().item())
            if seq_len > 1:
                predict_mask[b, seq_len - 1] = 0.0  # zero SEP only

    elif mask_mode == "random_span":
        # Random contiguous span of content tokens — closest to original JEPA.
        # Forces the predictor to reconstruct a specific region rather than
        # averaging over all positions, which can produce a stronger signal.
        predict_mask = target_tok["attention_mask"].clone().float()
        predict_mask[:, 0] = 0.0  # zero CLS
        for b in range(B):
            seq_len = int(target_tok["attention_mask"][b].sum().item())
            if seq_len <= 2:
                continue
            predict_mask[b, seq_len - 1] = 0.0  # zero SEP
            content_len = seq_len - 2            # exclude CLS and SEP
            span_len  = max(1, int(content_len * 0.5))
            start = random.randint(1, max(1, content_len - span_len + 1))
            # Zero out everything outside the span
            predict_mask[b, 1:start] = 0.0
            predict_mask[b, start + span_len: seq_len - 1] = 0.0

    elif mask_mode == "second_half":
        # Always mask the second half of content tokens.
        # Deterministic version of random_span — forces the predictor to
        # reconstruct the end of the hypothesis from the beginning,
        # which correlates with how entailment conclusions are structured.
        for b in range(B):
            seq_len = int(target_tok["attention_mask"][b].sum().item())
            if seq_len <= 2:
                continue
            content_len = seq_len - 2
            midpoint = 1 + content_len // 2     # start of second half
            predict_mask[b, midpoint: seq_len - 1] = 1.0

    elif mask_mode == "keyword":
        # Mask only the highest-IDF tokens — content-bearing words only.
        # Nouns, verbs, adjectives carry the entailment signal; function
        # words do not. This focuses the loss on the semantically heavy tokens.
        # Approximated here by masking tokens that are not subword continuations
        # (i.e. tokens that start a new word, indicated by no ## prefix in BERT
        # or a Ġ prefix in RoBERTa).
        ids = target_tok["input_ids"]           # [B, T]
        for b in range(B):
            seq_len = int(target_tok["attention_mask"][b].sum().item())
            for t in range(1, seq_len - 1):     # skip CLS and SEP
                tok_str = tokenizer.convert_ids_to_tokens(ids[b, t].item())
                # RoBERTa: word-starting tokens begin with Ġ
                # BERT: subword continuations begin with ##
                is_word_start = (
                    tok_str.startswith("Ġ") or      # RoBERTa
                    not tok_str.startswith("##")     # BERT fallback
                )
                if is_word_start:
                    predict_mask[b, t] = 1.0

    return context_tok, target_tok, predict_mask
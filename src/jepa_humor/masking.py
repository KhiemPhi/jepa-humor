import torch
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from typing import List, Sequence, Tuple, Dict

def make_second_sent_mask_views(
    sent1: Sequence[str],
    sent2: Sequence[str],
    tokenizer,
    max_length: int = 256,
) -> Tuple[
    Dict[str, torch.Tensor],  # keep_tok_sent1
    Dict[str, torch.Tensor],  # target_tok_sent1
    Dict[str, torch.Tensor],  # keep_tok_sent2
    Dict[str, torch.Tensor],  # target_tok_sent2
    torch.Tensor,             # predict_mask_sent1
    torch.Tensor,             # predict_mask_sent2
    List[List[Tuple[int, int]]],  # spans_used_sent1
    List[List[Tuple[int, int]]],  # spans_used_sent2
]:
    """
    Entailment-aware span masking:
    - If sent1 (premise) and sent2 (hypothesis) are semantically similar,
      mask *differing* spans in the hypothesis (entailment-focused).
    - If they differ a lot, mask *shared* spans (encourages contradiction discrimination).
    - Keeps the same input/output structure as before.
    """

    assert len(sent1) == len(sent2), "sent1 and sent2 must have the same length"

    # Tokenize both sentences
    tok1 = tokenizer(
        list(sent1),
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=max_length,
    )
    tok2 = tokenizer(
        list(sent2),
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=max_length,
    )

    B, T = tok1["input_ids"].shape
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

    # Prepare tensors and lists
    keep_input_ids_sent1 = tok1["input_ids"].clone()
    keep_attention_mask_sent1 = tok1["attention_mask"].clone()
    target_input_ids_sent1 = tok1["input_ids"].clone()
    target_attention_mask_sent1 = tok1["attention_mask"].clone()
    predict_mask_sent1 = torch.zeros((B, T), dtype=torch.float32)
    spans_used_sent1: List[List[Tuple[int, int]]] = []

    keep_input_ids_sent2 = tok2["input_ids"].clone()
    keep_attention_mask_sent2 = tok2["attention_mask"].clone()
    target_input_ids_sent2 = tok2["input_ids"].clone()
    target_attention_mask_sent2 = tok2["attention_mask"].clone()
    predict_mask_sent2 = torch.zeros((B, T), dtype=torch.float32)
    spans_used_sent2: List[List[Tuple[int, int]]] = []

    # Build a quick TF-IDF similarity heuristic
    tfidf = TfidfVectorizer(stop_words="english").fit(sent1 + sent2)
    vec1 = tfidf.transform(sent1).toarray()
    vec2 = tfidf.transform(sent2).toarray()

    for b in range(B):
        seq_len1 = int(tok1["attention_mask"][b].sum().item())
        seq_len2 = int(tok2["attention_mask"][b].sum().item())
        if seq_len1 <= 2 or seq_len2 <= 2:
            spans_used_sent1.append([])
            spans_used_sent2.append([])
            continue

        # Similarity between premise and hypothesis
        sim = np.dot(vec1[b], vec2[b]) / (np.linalg.norm(vec1[b]) * np.linalg.norm(vec2[b]) + 1e-8)

        # Decide split dynamically: high sim → mask differing half; low sim → mask shared half
        split1 = max(1, (seq_len1 - 2) // 2)
        split2 = max(1, (seq_len2 - 2) // 2)

        # --- Sentence 1 (Premise) ---
        # Always mask the last 50% (consistent context side)
        keep_input_ids_sent1[b, split1:seq_len1 - 1] = pad_token_id
        keep_attention_mask_sent1[b, split1:seq_len1 - 1] = 0
        target_input_ids_sent1[b, 1:split1] = pad_token_id
        target_attention_mask_sent1[b, 1:split1] = 0
        predict_mask_sent1[b, split1:seq_len1 - 1] = 1.0
        spans_used_sent1.append([(split1, seq_len1 - 1)])

        # --- Sentence 2 (Hypothesis) ---
        if sim > 0.5:
            # Mask the *differing* part (simulate entailment prediction)
            # Keep the first half as context (shared info)
            keep_input_ids_sent2[b, split2:seq_len2 - 1] = pad_token_id
            keep_attention_mask_sent2[b, split2:seq_len2 - 1] = 0
            target_input_ids_sent2[b, 1:split2] = pad_token_id
            target_attention_mask_sent2[b, 1:split2] = 0
            predict_mask_sent2[b, split2:seq_len2 - 1] = 1.0
            spans_used_sent2.append([(split2, seq_len2 - 1)])
        else:
            # Mask the *shared* half (simulate contrastive reasoning)
            keep_input_ids_sent2[b, 1:split2] = pad_token_id
            keep_attention_mask_sent2[b, 1:split2] = 0
            target_input_ids_sent2[b, split2:seq_len2 - 1] = pad_token_id
            target_attention_mask_sent2[b, split2:seq_len2 - 1] = 0
            predict_mask_sent2[b, 1:split2] = 1.0
            spans_used_sent2.append([(1, split2)])

    # Package outputs (same structure as before)
    keep_tok_sent1 = {"input_ids": keep_input_ids_sent1, "attention_mask": keep_attention_mask_sent1}
    target_tok_sent1 = {"input_ids": target_input_ids_sent1, "attention_mask": target_attention_mask_sent1}
    keep_tok_sent2 = {"input_ids": keep_input_ids_sent2, "attention_mask": keep_attention_mask_sent2}
    target_tok_sent2 = {"input_ids": target_input_ids_sent2, "attention_mask": target_attention_mask_sent2}

    return (
        keep_tok_sent1,
        target_tok_sent1,
        keep_tok_sent2,
        target_tok_sent2,
        predict_mask_sent1,
        predict_mask_sent2,
        spans_used_sent1,
        spans_used_sent2,
    )

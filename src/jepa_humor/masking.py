import math, random, re
from typing import List, Optional, Sequence, Tuple
import torch

def make_token_level_views_punchline(
    texts: Sequence[str],
    tokenizer,
    max_length=256,
    punchline_ratio: float = 0.3,  # proportion of tokens from the end to mask if last sentence detection fails
):
    target_tok = tokenizer(
        list(texts),
        return_tensors="pt",
        padding="longest",
        truncation=True,
        max_length=max_length,
        return_offsets_mapping=True
    )

    input_ids = target_tok["input_ids"].clone()
    attn = target_tok["attention_mask"]
    offsets = target_tok["offset_mapping"]
    B, T = input_ids.shape

    keep_ids = input_ids.clone()
    predict_mask = torch.zeros_like(attn, dtype=torch.float)
    mask_token_id = tokenizer.mask_token_id
    assert mask_token_id is not None, "Tokenizer needs a [MASK] token."
    special_ids = set(tokenizer.all_special_ids)
    spans_used = [[] for _ in range(B)]

    for b in range(B):
        # 1️⃣ Identify valid token positions
        valid = [t for t in range(T)
                 if attn[b, t].item() == 1 and input_ids[b, t].item() not in special_ids]

        # 2️⃣ Find the punchline span
        text = texts[b]
        sent_splits = list(re.finditer(r"[.!?\n]", text))
        if sent_splits:
            # Take last sentence boundary
            last_end = sent_splits[-1].end()
            punchline_start_char = sent_splits[0].start()
        else:
            # No punctuation, fallback to last 30% of text
            punchline_start_char = int(len(text) * (1 - punchline_ratio))
        punchline_end_char = len(text)

        # 3️⃣ Map char spans to token spans using offsets
        off_b = [(int(o[0].item()), int(o[1].item())) for o in offsets[b]]
        start_t, end_t = 0, len(off_b)
        for i, (s, e) in enumerate(off_b):
            if s <= punchline_start_char < e:
                start_t = i
                break
        for j, (s, e) in enumerate(off_b):
            if s < punchline_end_char <= e:
                end_t = j + 1
                break

        # 4️⃣ Mask that punchline span
        for t in range(start_t, end_t):
            if t in valid:
                keep_ids[b, t] = mask_token_id
                predict_mask[b, t] = 1.0
        spans_used[b] = [(start_t, end_t)]

    keep_tok = {k: v.clone() for k, v in target_tok.items() if k != "offset_mapping"}
    keep_tok["input_ids"] = keep_ids
    target_tok = {k: v for k, v in target_tok.items() if k != "offset_mapping"}

    return keep_tok, target_tok, predict_mask, spans_used

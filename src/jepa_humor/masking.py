import math, random, re
from typing import List, Optional, Sequence, Tuple
import torch

def _geometric_span_len(mean_len: float, min_len: int = 1, max_len: Optional[int] = None) -> int:
    mean_len = max(mean_len, 1.0)
    p = 1.0 / mean_len
    u = random.random()
    L = math.ceil(math.log(1 - u) / math.log(1 - p))
    if max_len is not None:
        L = min(L, max_len)
    return max(L, min_len)

def _choose_start_with_tail_bias(candidates: List[int], T: int, tail_bias: float) -> int:
    if len(candidates) == 1 or tail_bias <= 0:
        return random.choice(candidates)
    denom = max(T - 1, 1)
    weights = [(c / denom + 1e-6) ** tail_bias for c in candidates]
    s = sum(weights); r = random.random() * s; acc = 0.0
    for c, w in zip(candidates, weights):
        acc += w
        if r <= acc: return c
    return candidates[-1]

def _mask_spans_auto(valid: List[int], T: int, mask_ratio: float, mean_span_len: float, tail_bias: float):
    if not valid: return []
    valid_set = set(valid); covered = set(); budget = max(1, int(len(valid) * mask_ratio))
    spans, occupied = [], [False] * T
    def _fits(start: int, L: int) -> bool:
        end = min(start + L, T)
        return all((t in valid_set and not occupied[t]) for t in range(start, end))
    while len(covered) < budget:
        L = _geometric_span_len(mean_span_len, 1, min(32, T))
        cands = [i for i in valid if _fits(i, L)]
        if not cands:
            for L_try in range(L-1, 0, -1):
                cands = [i for i in valid if _fits(i, L_try)]
                if cands:
                    L = L_try; break
            if not cands: break
        start = _choose_start_with_tail_bias(cands, T, tail_bias); end = min(start + L, T)
        for t in range(start, end):
            occupied[t] = True; covered.add(t)
        spans.append((start, end))
        if len(covered) >= budget: break
    spans.sort()
    merged = []
    for s, e in spans:
        if not merged or s > merged[-1][1]: merged.append([s, e])
        else: merged[-1][1] = max(merged[-1][1], e)
    return [(s, e) for s, e in merged]

def _char_spans_to_token_spans(offsets: List[Tuple[int, int]], char_spans: List[Tuple[int, int]]):
    token_spans = []
    for cs, ce in char_spans:
        t_start = t_end = None
        for ti, (os, oe) in enumerate(offsets):
            if os == oe: continue
            if not (oe <= cs or os >= ce):
                if t_start is None: t_start = ti
                t_end = ti + 1
        if t_start is not None and t_end is not None and t_start < t_end:
            token_spans.append((t_start, t_end))
    return token_spans

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

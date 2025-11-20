import math, random, re
from typing import List, Optional, Sequence, Tuple, Dict, Any
import torch
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from transformers import AutoModel, AutoTokenizer

def make_squad_views(contexts, questions, answer_texts, answer_starts, tokenizer, max_length=384):
    sent1_list = []
    sent2_list = []

    import re
    sep = tokenizer.sep_token

    for ctx, q, ans, start in zip(contexts, questions, answer_texts, answer_starts):
        end = start + len(ans)

        sents = re.split(r'(?<=[.!?])\s+', ctx)

        chosen = None
        pos = 0
        for s in sents:
            s_start = pos
            s_end = pos + len(s)
            if s_start <= start < s_end:
                chosen = s
                break
            pos = s_end + 1

        if chosen is None:
            chosen = ctx

        sent1_list.append(q)
        sent2_list.append(chosen)

    keep_tok = tokenizer(
        sent1_list,
        sent2_list,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=512,
    )

    target_tok = tokenizer(
        answer_texts,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=512,
    )

    predict_mask = target_tok["attention_mask"].float()

    spans_used = []
    for b in range(target_tok["input_ids"].shape[0]):
        seq_len = int(predict_mask[b].sum().item())
        spans_used.append([(0, seq_len)])

    return keep_tok, target_tok, predict_mask, spans_used

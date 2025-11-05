from typing import Sequence, Tuple, Dict, List
import torch
import re

def make_hypothesis_mask_views(
    premise: Sequence[str],
    hypothesis: Sequence[str],
    tokenizer,
    max_length: int = 256,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], torch.Tensor, List[List[Tuple[int, int]]]]:
    """
    Modified version:
    - keep_tok: only the last sentence from each premise (sent1)
    - target_tok: full hypothesis (sent2)
    - predict_mask: covers all tokens of hypothesis
    - spans_used: list of token spans for each hypothesis sequence
    """

    assert len(premise) == len(hypothesis), "Premises and hypotheses must have the same length"

    def extract_last_sentence(text: str) -> str:
        """Extract the last sentence using punctuation splitting."""
        # Split on ., ?, or ! followed by a space or end of string
        parts = re.split(r'(?<=[.!?])\s+', text.strip())
        if len(parts) == 0:
            return text.strip()
        return parts[-1].strip()

    # Take only the last sentence of each premise
    last_sentences = [extract_last_sentence(s) for s in premise]

    # Tokenize last sentence of premise as model input
    keep_tok = tokenizer(
        last_sentences,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=max_length,
    )

    # Tokenize full hypothesis as target
    target_tok = tokenizer(
        list(hypothesis),
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=max_length,
    )

    B, T = target_tok["input_ids"].shape

    # Predict all tokens of hypothesis
    predict_mask = target_tok["attention_mask"].float()

    # Record spans for each hypothesis (full span)
    spans_used = []
    for b in range(B):
        seq_len = int(target_tok["attention_mask"][b].sum().item())
        spans_used.append([(0, seq_len)])

    return keep_tok, target_tok, predict_mask, spans_used

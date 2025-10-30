import spacy
import torch
from typing import List, Tuple, Dict, Sequence

# Load spaCy English model (for POS + dependency parsing)
nlp = spacy.load("en_core_web_sm")


def make_grammar_mask_views(
    sentences: Sequence[str],
    tokenizer,
    max_length: int = 256,
) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], torch.Tensor, List[List[Tuple[int, int]]]]:
    """
    Create tokenized input/target pairs by masking the grammatically challenging parts of sentences.

    Rules:
      1. For parallel/comparative sentences (containing ','): mask everything *after the comma*.
      2. For simple sentences: mask everything *after the main verb (ROOT)*.

    Args:
        sentences: Sequence of English sentences.
        tokenizer: HuggingFace tokenizer.
        max_length: Max token length for padding/truncation.

    Returns:
        keep_tok: Tokenized masked input sentences.
        target_tok: Tokenized original sentences.
        predict_mask: Mask tensor for predicting masked parts.
        spans_used: List of (start, end) token spans for masked sections.
    """
    masked_sentences = []
    spans_used = []

    for sent in sentences:
        doc = nlp(sent)
        masked_sentence = sent
        mask_start = None

        # Rule 1: mask after comma (often parallel or comparative structure)
        if "," in sent:
            mask_start = sent.index(",") + 1
            masked_sentence = sent[:mask_start] + " [MASK]"
        
        else:
            # Rule 2: mask after main verb
            for token in doc:
                if token.dep_ == "ROOT" and token.pos_ == "VERB":
                    mask_start = token.idx + len(token.text)
                    masked_sentence = sent[:mask_start] + " [MASK]"
                    break

        masked_sentences.append(masked_sentence)

        # Compute token-level span (after tokenization)
        tok = tokenizer(masked_sentence, truncation=True, max_length=max_length)
        orig_tok = tokenizer(sent, truncation=True, max_length=max_length)
        mask_token_id = tokenizer.mask_token_id

        # find where [MASK] appears
        mask_pos = [i for i, t in enumerate(tok["input_ids"]) if t == mask_token_id]
        if mask_pos:
            spans_used.append([(mask_pos[0], len(tok["input_ids"]))])
        else:
            spans_used.append([])

    # Tokenize both masked inputs and original targets
    keep_tok = tokenizer(
        masked_sentences,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=max_length,
    )
    target_tok = tokenizer(
        sentences,
        return_tensors="pt",
        padding="max_length",
        truncation=True,
        max_length=max_length,
    )

    B, T = target_tok["input_ids"].shape
    predict_mask = torch.zeros((B, T), dtype=torch.float32)

    # mark [MASK] region as 1 in predict_mask
    for b, spans in enumerate(spans_used):
        for (start, end) in spans:
            predict_mask[b, start:end] = 1.0

    return keep_tok, target_tok, predict_mask, spans_used

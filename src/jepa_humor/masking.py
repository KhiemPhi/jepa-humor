import math, random, re
from typing import List, Optional, Sequence, Tuple, Dict, Any
import torch
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from transformers import AutoModel, AutoTokenizer

def make_sentence_deletion_views(
    claims: Sequence[str],
    evidence_claims: Sequence[str],
    model,  # DistilBERT model
    tokenizer,  # DistilBERT tokenizer
    max_length: int = 256,
    similarity_threshold: float = 0.0,
):
    """
    For each claim-evidence pair, iteratively delete sentences from evidence_claim
    and keep deletions that increase similarity with the claim.
    """
    
    assert len(claims) == len(evidence_claims), "Claims and evidence_claims must have same length"
    
    # Process the evidence claims to get optimized versions
    processed_results = process_evidence_deletions(
        claims, evidence_claims, model, tokenizer, max_length, similarity_threshold
    )
    
    processed_evidence = processed_results['processed_evidence']
    
    # Tokenize the processed evidence (input)
    keep_tok = tokenizer(
        list(processed_evidence),
        return_tensors="pt",
        padding="longest", 
        truncation=True,
        max_length=max_length,
    )
    
    # Tokenize the claims (targets)
    target_tok = tokenizer(
        list(claims),
        return_tensors="pt",
        padding="longest",
        truncation=True,
        max_length=max_length,
    )
    
    B, T = target_tok["input_ids"].shape
    
    # Create predict_mask - predict all tokens in the target (claims)
    predict_mask = torch.ones_like(target_tok["attention_mask"], dtype=torch.float)
    
    # For spans_used, mark the entire claim as what needs to be predicted
    spans_used = [[] for _ in range(B)]
    for b in range(B):
        seq_len = target_tok["attention_mask"][b].sum().item()
        spans_used[b] = [(0, seq_len)]
    
    return keep_tok, target_tok, predict_mask, spans_used

def process_evidence_deletions(
    claims: Sequence[str],
    evidence_claims: Sequence[str], 
    model,
    tokenizer,
    max_length: int = 256,
    similarity_threshold: float = 0.0,
) -> Dict[str, Any]:
    """Core logic for sentence deletion and similarity comparison."""
    
    results = {
        'processed_evidence': [],
        'deletion_log': [],
        'final_similarities': [],
        'original_similarities': []
    }
    
    for i, (claim, evidence) in enumerate(zip(claims, evidence_claims)):
        # Get original similarity
        original_similarity = compute_text_similarity_distilbert(claim, evidence, model, tokenizer, max_length)
        results['original_similarities'].append(original_similarity)
        
        # Split evidence into sentences
        sentences = split_into_sentences(evidence)
        
        if len(sentences) <= 1:
            # Cannot delete if only one sentence
            results['processed_evidence'].append(evidence)
            results['deletion_log'].append([])
            results['final_similarities'].append(original_similarity)
            continue
        
        current_evidence = evidence
        current_similarity = original_similarity
        deletion_log = []
        current_sentences = sentences.copy()
        
        # Keep trying deletions until no improvements are found
        improved = True
        while improved and len(current_sentences) > 1:
            improved = False
            
            # Try deleting each remaining sentence (in random order)
            sentence_indices = list(range(len(current_sentences)))
            random.shuffle(sentence_indices)
            
            for sent_idx in sentence_indices:
                # Create evidence with this sentence deleted
                modified_sentences = current_sentences.copy()
                deleted_sentence = modified_sentences[sent_idx]
                del modified_sentences[sent_idx]
                modified_evidence = ' '.join(modified_sentences)
                
                # Compute similarity
                new_similarity = compute_text_similarity_distilbert(claim, modified_evidence, model, tokenizer, max_length)
                
                # Keep deletion if similarity increases beyond threshold
                if new_similarity > current_similarity + similarity_threshold:
                    current_evidence = modified_evidence
                    current_similarity = new_similarity
                    current_sentences = modified_sentences
                    deletion_log.append({
                        'deleted_sentence_idx': sent_idx,
                        'deleted_sentence': deleted_sentence,
                        'new_similarity': new_similarity,
                        'improvement': new_similarity - current_similarity
                    })
                    improved = True
                    break
        
        results['processed_evidence'].append(current_evidence)
        results['deletion_log'].append(deletion_log)
        results['final_similarities'].append(current_similarity)
    
    return results

def compute_text_similarity_distilbert(text1: str, text2: str, model, tokenizer, max_length: int) -> float:
    """Compute cosine similarity between two texts using DistilBERT embeddings."""
    
    # Tokenize both texts
    tokens1 = tokenizer(
        text1, 
        return_tensors="pt", 
        padding=True, 
        truncation=True, 
        max_length=max_length
    )
    
    tokens2 = tokenizer(
        text2, 
        return_tensors="pt", 
        padding=True, 
        truncation=True, 
        max_length=max_length
    )
    
    # Get embeddings from DistilBERT
    with torch.no_grad():
        # DistilBERT returns a tuple where the first element is the last_hidden_state
        outputs1 = model(**tokens1)
        outputs2 = model(**tokens2)
        
        # For DistilBERT, we get the [CLS] token embedding or mean pooling
        if isinstance(outputs1, tuple):
            # Use mean pooling over all tokens
            emb1 = outputs1[0].mean(dim=1)  # Shape: (1, hidden_size)
            emb2 = outputs2[0].mean(dim=1)
        else:
            # Fallback to mean pooling
            emb1 = outputs1.last_hidden_state.mean(dim=1)
            emb2 = outputs2.last_hidden_state.mean(dim=1)
    
    # Compute cosine similarity
    similarity = cosine_similarity(emb1.cpu().numpy(), emb2.cpu().numpy())[0][0]
    return float(similarity)

def split_into_sentences(text: str) -> List[str]:
    """Split text into sentences using punctuation marks."""
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    return sentences


import math, random, re
from typing import List, Optional, Sequence, Tuple, Dict, Any
import torch
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

def make_sentence_deletion_views(
    claims: Sequence[str],
    evidence_claims: Sequence[str],
    model,  # model to get embeddings
    tokenizer,
    max_length: int = 256,
    similarity_threshold: float = 0.0,  # similarity must be higher by this margin
):
    """
    For each claim-evidence pair, iteratively delete sentences from evidence_claim
    and keep deletions that increase similarity with the claim.
    Returns format similar to the original punchline function.
    """
    
    assert len(claims) == len(evidence_claims), "Claims and evidence_claims must have same length"
    
    # First, process the evidence claims to get the optimized versions
    processed_results = process_evidence_deletions(
        claims, evidence_claims, model, tokenizer, max_length, similarity_threshold
    )
    
    # Now create the tokenized views
    processed_evidence = processed_results['processed_evidence']
    
    # Tokenize the processed evidence (with deletions) - this is the input
    keep_tok = tokenizer(
        list(processed_evidence),
        return_tensors="pt",
        padding="longest", 
        truncation=True,
        max_length=max_length,
    )
    
    # Tokenize the CLAIMS - these are the targets
    target_tok = tokenizer(
        list(claims),  # This is the key change - targets are the claims
        return_tensors="pt",
        padding="longest",
        truncation=True,
        max_length=max_length,
    )
    
    B, T = target_tok["input_ids"].shape
    
    # Create predict_mask - we need to predict ALL tokens in the target (claims)
    # since we're transforming from processed evidence to claim
    predict_mask = torch.ones_like(target_tok["attention_mask"], dtype=torch.float)
    
    # For spans_used, we'll mark the entire claim as what needs to be predicted
    spans_used = [[] for _ in range(B)]
    for b in range(B):
        # Mark the entire claim sequence as needing prediction
        seq_len = target_tok["attention_mask"][b].sum().item()
        spans_used[b] = [(0, seq_len)]  # Entire sequence needs to be predicted
    
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
        original_similarity = compute_text_similarity(claim, evidence, model, tokenizer, max_length)
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
                new_similarity = compute_text_similarity(claim, modified_evidence, model, tokenizer, max_length)
                
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
                    break  # Restart the process with new sentence list
        
        results['processed_evidence'].append(current_evidence)
        results['deletion_log'].append(deletion_log)
        results['final_similarities'].append(current_similarity)
    
    return results

def split_into_sentences(text: str) -> List[str]:
    """Split text into sentences using punctuation marks."""
    sentences = re.split(r'[.!?]+', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    return sentences

def compute_text_similarity(text1: str, text2: str, model, tokenizer, max_length: int) -> float:
    """Compute cosine similarity between two texts using model embeddings."""
    
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
    
    # Get embeddings
    with torch.no_grad():
        outputs1 = model(**tokens1)
        outputs2 = model(**tokens2)
        
        # Handle different model output formats
        if hasattr(outputs1, 'last_hidden_state'):
            emb1 = outputs1.last_hidden_state.mean(dim=1)
            emb2 = outputs2.last_hidden_state.mean(dim=1)
        elif hasattr(outputs1, 'pooler_output'):
            emb1 = outputs1.pooler_output
            emb2 = outputs2.pooler_output
        elif isinstance(outputs1, tuple):
            emb1 = outputs1[0].mean(dim=1)
            emb2 = outputs2[0].mean(dim=1)
        else:
            emb1 = outputs1
            emb2 = outputs2
    
    # Compute cosine similarity
    similarity = cosine_similarity(emb1.cpu().numpy(), emb2.cpu().numpy())[0][0]
    return float(similarity)

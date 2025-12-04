import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy


def jepa_loss(pred, target, mask=None, temperature=0.1):
    # normalize embeddings to unit length
    pred = F.normalize(pred, dim=-1)              # [B, T, D]
    target = F.normalize(target.detach(), dim=-1) # stop-gradient

    # optionally select only masked tokens
    if mask is not None:
        mask = mask.bool()
        pred = pred[mask]      # [M, D]
        target = target[mask]  # [M, D]

    # compute cosine similarities (dot product since normalized)
    sim_matrix = pred @ target.T / temperature    # [M, M]

    # compute contrastive loss
    labels = torch.arange(sim_matrix.size(0), device=pred.device)
    loss_i2t = F.cross_entropy(sim_matrix, labels)       # pred -> target
    loss_t2i = F.cross_entropy(sim_matrix.T, labels)     # target -> pred
    loss = (loss_i2t + loss_t2i) / 2

    return loss

def freeze_distilbert_layers(encoder, unfreeze_last_n=2):
    # DistilBERT has encoder.transformer.layer[i]
    for p in encoder.parameters():
        p.requires_grad = False
    # unfreeze last N layers
    for i in range(6 - unfreeze_last_n, 6):
        for p in encoder.transformer.layer[i].parameters():
            p.requires_grad = True
    # keep LayerNorms trainable in the whole stack (optional)
    for m in encoder.modules():
        if isinstance(m, torch.nn.LayerNorm):
            for p in m.parameters():
                p.requires_grad = True

class JEPA(nn.Module):
    def __init__(self, initial_encoder, hidden_size=768, pred_hid=128, ema_decay=0.996, use_cls=False):
        super().__init__()
        self.encoder = initial_encoder
        self.target_encoder = deepcopy(initial_encoder)
        for p in self.target_encoder.parameters():
            p.requires_grad = False

        self.predictor = nn.Sequential(
            nn.Linear(hidden_size, pred_hid),
            nn.ReLU(),
            nn.Linear(pred_hid, hidden_size)
        )
        self.ema_decay = ema_decay
        self.use_cls = use_cls
        #freeze_distilbert_layers(self.encoder)

    @torch.no_grad()
    def update_target(self):
        d = self.ema_decay
        for p_o, p_t in zip(self.encoder.parameters(), self.target_encoder.parameters()):
            p_t.data.mul_(d).add_(p_o.data, alpha=1 - d)

    def forward(self, context_inputs, target_inputs, mask=None, temperature=0.1):
        online = self.encoder(**context_inputs).last_hidden_state   # [B,T,D]
        pred   = self.predictor(online)

        with torch.no_grad():
            target = self.target_encoder(**target_inputs).last_hidden_state
      
        loss = jepa_loss(pred, target, mask, temperature)
        return loss, pred, target


class MLMJEPA(nn.Module):
    """
    BERT-style MLM model that predicts token IDs directly from hidden states.
    Matches original BERT architecture: single encoder + vocab projection head.
    No EMA target encoder (unlike standard JEPA).
    """
    def __init__(self, initial_encoder, vocab_size, hidden_size=768, use_cls=False):
        super().__init__()
        self.encoder = initial_encoder
        
        # BERT-style MLM head: direct linear projection from hidden states to vocabulary
        # This matches the original BERT architecture
        self.vocab_projection = nn.Linear(hidden_size, vocab_size)
        self.use_cls = use_cls

    def forward(self, context_inputs, target_inputs, mask=None, ignore_index=-100):
        """
        Args:
            context_inputs: Input dict with input_ids, attention_mask for context (unmasked text)
            target_inputs: Input dict with input_ids, attention_mask for target (original tokens to predict)
            mask: Optional mask tensor [B, T] indicating which tokens to predict (1 = predict, 0 = ignore)
            ignore_index: Token ID to ignore in loss computation (typically pad token or -100)
        
        Returns:
            loss: Cross-entropy loss for token prediction
            logits: Predicted logits [B, T, vocab_size]
            target_ids: Target token IDs [B, T]
        """
        # Encode context - get hidden states directly (BERT-style)
        hidden_states = self.encoder(**context_inputs).last_hidden_state   # [B, T, D]
        
        # Direct projection to vocabulary logits (BERT MLM head)
        logits = self.vocab_projection(hidden_states)  # [B, T, vocab_size]
        
        # Get target token IDs (original tokens to predict)
        target_ids = target_inputs["input_ids"]  # [B, T]
        
        # Compute loss only on masked tokens
        if mask is not None:
            # mask is [B, T] with 1.0 for tokens to predict, 0.0 for ignore
            mask = mask.bool()
            
            # Flatten for loss computation
            logits_flat = logits.view(-1, logits.size(-1))  # [B*T, vocab_size]
            target_ids_flat = target_ids.view(-1)  # [B*T]
            mask_flat = mask.view(-1)  # [B*T]
            
            # Only compute loss on masked positions
            if mask_flat.sum() > 0:
                # Set non-masked positions to ignore_index
                target_ids_flat = target_ids_flat.clone()
                target_ids_flat[~mask_flat] = ignore_index
                
                loss = F.cross_entropy(logits_flat, target_ids_flat, ignore_index=ignore_index)
            else:
                # No tokens to predict
                loss = torch.tensor(0.0, device=logits.device, requires_grad=True)
        else:
            # Predict all tokens (except padding)
            # Set padding tokens to ignore_index
            attention_mask = target_inputs.get("attention_mask", None)
            if attention_mask is not None:
                target_ids_masked = target_ids.clone()
                target_ids_masked[attention_mask == 0] = ignore_index
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), target_ids_masked.view(-1), ignore_index=ignore_index)
            else:
                loss = F.cross_entropy(logits.view(-1, logits.size(-1)), target_ids.view(-1), ignore_index=ignore_index)
        
        return loss, logits, target_ids

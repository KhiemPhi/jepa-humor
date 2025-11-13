import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy


def jepa_loss(pred, target, mask=None, temperature=0.1):
    # pred: [B, T, D], target: [B, T, D]
    # normalize embeddings to unit length
    pred = F.normalize(pred, dim=-1)              # [B, T, D]
    target = F.normalize(target.detach(), dim=-1) # stop-gradient [B, T, D]

    # Get original shape before flattening
    B, T, D = pred.shape

    # optionally select only masked tokens before flattening
    if mask is not None:
        # mask should be [B, T] - select only valid tokens
        mask = mask.bool()  # [B, T]
        pred = pred[mask]      # [M, D] where M is number of True values
        target = target[mask]  # [M, D]
    else:
        # Flatten batch and sequence dimensions: [B, T, D] -> [B*T, D]
        pred = pred.view(B * T, D)      # [B*T, D]
        target = target.view(B * T, D)  # [B*T, D]

    # compute cosine similarities (dot product since normalized)
    sim_matrix = pred @ target.T / temperature    # [M, M] or [B*T, B*T]

    # compute contrastive loss
    labels = torch.arange(sim_matrix.size(0), device=pred.device)
    loss = F.cross_entropy(sim_matrix, labels)       # pred -> target
   
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
       

    @torch.no_grad()
    def update_target(self):
        d = self.ema_decay
        for p_o, p_t in zip(self.encoder.parameters(), self.target_encoder.parameters()):
            p_t.data.mul_(d).add_(p_o.data, alpha=1 - d)

    def forward(self, context_inputs, target_inputs, mask=None, temperature=0.1):
        online = self.encoder(**context_inputs).last_hidden_state   # [B*C, T, D] = [48, 128, 768]
        
        # Process through MLP first (MLP operates on last dimension D)
        pred_flat = self.predictor(online)  # [48, 128, 768]
        
        # Reshape from (B*C, T, D) to (B, C, T, D) where C=3
        B_C, T, D = pred_flat.shape
        B = B_C // 3  # Assuming 3 incorrect pairs per example
        C = 3
        pred = pred_flat.view(B, C, T, D)  # [16, 3, 128, 768]
        
        # Aggregate over C dimension (mean of 3 incorrect pairs) to get (B, T, D)
        pred = pred.mean(dim=1)  # [16, 128, 768]

        with torch.no_grad():
            target = self.target_encoder(**target_inputs).last_hidden_state  # [B, T, D] = [16, 128, 768]
        
        # Extract attention mask from target_inputs to mask out padding tokens
        # Use target's attention_mask since pred and target should have same sequence structure
        if mask is None and 'attention_mask' in target_inputs:
            mask = target_inputs['attention_mask']  # Could be [B, T] or [B*1, T] or [B, 1, T]
            # Ensure mask is 2D with shape [B, T]
            if mask.dim() == 3:
                # Reshape from [B, 1, T] to [B, T]
                mask = mask.squeeze(1) if mask.shape[1] == 1 else mask.view(B, T)
            elif mask.dim() == 2:
                if mask.shape[0] == B and mask.shape[1] == T:
                    # Already correct shape [B, T]
                    pass
                elif mask.shape[0] == B * 1:
                    # Reshape from [B*1, T] to [B, T]
                    mask = mask.view(B, T)
                else:
                    # Shape doesn't match, don't use mask
                    mask = None
            else:
                mask = None
      
        loss = jepa_loss(pred, target, mask, temperature)
        return loss, pred, target

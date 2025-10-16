import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy

def jepa_loss(pred, target, mask=None, temperature=0.1):
    # normalize
    pred = F.normalize(pred, dim=-1)        # [B,T,D]
    target = F.normalize(target.detach(), dim=-1)  # stop-grad

    # flatten only masked tokens
    if mask is not None:
        mask = mask.bool()
        pred = pred[mask]      # [M, D]
        target = target[mask]  # [M, D]

    B = pred.size(0)
    logits = pred @ target.T / temperature   # [M, M]
    labels = torch.arange(B, device=pred.device)
    return F.cross_entropy(logits, labels)

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
        freeze_distilbert_layers(self.encoder)

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

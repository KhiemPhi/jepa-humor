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

def freeze_bert_layers(encoder, unfreeze_last_n=2):
    """
    Freeze all BERT layers except for the last `unfreeze_last_n` encoder layers.
    Optionally keeps LayerNorm parameters trainable throughout.
    
    Args:
        encoder: BERT model (e.g., AutoModel.from_pretrained("bert-base-uncased"))
        unfreeze_last_n (int): Number of last transformer layers to unfreeze.
    """
    # 1️⃣ Freeze everything
    for p in encoder.parameters():
        p.requires_grad = False

    # 2️⃣ Unfreeze last N layers (BERT has 12 layers)
    total_layers = len(encoder.encoder.layer)
    for i in range(total_layers - unfreeze_last_n, total_layers):
        for p in encoder.encoder.layer[i].parameters():
            p.requires_grad = True

    # 3️⃣ (Optional) Keep all LayerNorms trainable for stability
    for m in encoder.modules():
        if isinstance(m, torch.nn.LayerNorm):
            for p in m.parameters():
                p.requires_grad = True

    #print(f"✅ Frozen {total_layers - unfreeze_last_n} layers, unfrozen last {unfreeze_last_n} layers.")


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
        #freeze_bert_layers(self.encoder)

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

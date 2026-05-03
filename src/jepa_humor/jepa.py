import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
import math

import torch
import torch.nn.functional as F

# Loss function currently for run_25513 (best)
def jepa_loss(
    pred, target, mask=None,
    std_weight=0.1, cov_weight=0.04, knn_weight=0.1,
    huber_beta=0.1, knn_k=8,
):
    target = target.detach()
    target = F.normalize(target, dim=-1)
    pred   = F.normalize(pred,   dim=-1)

    if mask is not None and pred.size(1) > 1:
        mask   = mask.bool()
        pred   = pred[mask]    # [M, D]
        target = target[mask]  # [M, D]
    else:
        # Flatten [B, T, D] or [B, 1, D] → [B*T, D] so covariance math works
        B, T, D = pred.shape
        pred   = pred.reshape(B * T, D)
        target = target.reshape(B * T, D)

    N, D = pred.shape

    # ── 1. Invariance: Huber (smooth-L1) ───────────────────────────────────
    # Robust to outliers from the EMA target encoder's drift.
    inv_loss = F.smooth_l1_loss(pred, target, beta=huber_beta)

    # ── 2. Variance ────────────────────────────────────────────────────────
    pred_std = torch.sqrt(pred.var(dim=0) + 1e-4)
    std_loss = F.relu(1.0 - pred_std).mean()

    # ── 3. Covariance ──────────────────────────────────────────────────────
    pred_zm  = pred - pred.mean(dim=0)
    cov      = (pred_zm.T @ pred_zm) / (N - 1)
    off_diag = cov.pow(2).sum() - cov.diagonal().pow(2).sum()
    cov_loss = off_diag / D

    # ── 4. k-NN consistency (replaces full gram) ───────────────────────────
    # Match local neighborhood structure rather than all N×N pairs:
    # for each token, the distribution over its k nearest neighbors in
    # target space should match the distribution in pred space.
    k = min(knn_k, N - 1)  # safety: can't have more neighbors than tokens-1

    sim_t = target @ target.T                           # [N, N]
    sim_p = pred   @ pred.T                             # [N, N]

    # Mask out self-similarity (diagonal) so a token isn't its own neighbor
    eye = torch.eye(N, dtype=torch.bool, device=pred.device)
    sim_t = sim_t.masked_fill(eye, -float('inf'))
    sim_p = sim_p.masked_fill(eye, -float('inf'))

    # Restrict to top-k neighbors per row in TARGET space, then match
    # the predicted distribution over those same indices.
    _, knn_idx = sim_t.topk(k, dim=-1)                  # [N, k]
    sim_t_topk = sim_t.gather(1, knn_idx)               # [N, k]
    sim_p_topk = sim_p.gather(1, knn_idx)               # [N, k]

    log_p = F.log_softmax(sim_p_topk, dim=-1)
    p_t   = F.softmax(sim_t_topk, dim=-1)
    knn_loss = F.kl_div(log_p, p_t, reduction='batchmean')

    return (
        inv_loss
        + std_weight * std_loss
        + cov_weight * cov_loss
        + knn_weight * knn_loss
    )

def get_encoder_inputs(tok_dict: dict) -> dict:
    """
    Safely extract encoder inputs, dropping keys the model does not accept.
    RoBERTa:    accepts input_ids, attention_mask only — no token_type_ids.
    BERT:       accepts input_ids, attention_mask, token_type_ids.
    Any other key (e.g. offset_mapping) is silently dropped.
    """
    allowed = {"input_ids", "attention_mask", "token_type_ids"}
    return {k: v for k, v in tok_dict.items() if k in allowed}


class TransformerPredictor(nn.Module):
    """
    Sequence-aware predictor with explicit position embeddings.

    RoBERTa note: RoBERTa's internal position embeddings start at offset 2
    (padding_idx=1, so positions are 2..max_length+1). The predictor's own
    position embeddings are independent of the encoder's — they live in a
    separate learned space and are simply added to the projected hidden states.
    There is no conflict because the encoder has already consumed its own
    positional information before the hidden states arrive here.
    """
    def __init__(self, hidden_size=768, num_heads=8, num_layers=2,
                 ffn_dim=1024, dropout=0.1, max_length=256):
        super().__init__()
        # Simple 0-indexed positions — independent of encoder position convention
        self.pos_embed  = nn.Embedding(max_length, hidden_size)
        self.input_norm = nn.LayerNorm(hidden_size)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.proj = nn.Linear(hidden_size, hidden_size)

    def forward(self, x, key_padding_mask=None):
        B, T, D = x.shape
        positions = torch.arange(T, device=x.device).unsqueeze(0)  # [1, T]
        x = self.input_norm(x + self.pos_embed(positions))
        x = self.transformer(x, src_key_padding_mask=key_padding_mask)
        return self.proj(x)


class JEPA(nn.Module):
    """
    Joint Embedding Predictive Architecture — compatible with both BERT and RoBERTa.

    RoBERTa differences handled:
      - token_type_ids stripped via get_encoder_inputs() before every encoder call
      - Position embeddings in TransformerPredictor are 0-indexed and independent
        of RoBERTa's offset-2 position convention
      - get_ema_decay() is a method for clean epoch-level scheduling
    """
    def __init__(
        self,
        initial_encoder,
        hidden_size: int = 768,
        ema_decay: float = 0.996,
        use_cls: bool = False,
        predictor_type: str = "transformer",
        # MLP kwargs
        pred_hid: int = 2048,
        # Transformer kwargs
        pred_num_heads: int = 8,
        pred_num_layers: int = 2,
        pred_ffn_dim: int = 1024,
        pred_dropout: float = 0.1,
        pred_max_length: int = 256,
        # Projection head
        use_projector: bool = False,
        proj_dim: int = 256,
    ):
        super().__init__()
        self.encoder = initial_encoder
        self.target_encoder = deepcopy(initial_encoder)
        for p in self.target_encoder.parameters():
            p.requires_grad = False

        self.use_projector = use_projector
        if use_projector:
            self.projector = nn.Sequential(
                nn.Linear(hidden_size, proj_dim),
                nn.GELU(),
                nn.LayerNorm(proj_dim),
            )
            self.expander = nn.Linear(proj_dim, hidden_size)

        if predictor_type == "transformer":
            self.predictor = TransformerPredictor(
                hidden_size=hidden_size,
                num_heads=pred_num_heads,
                num_layers=pred_num_layers,
                ffn_dim=pred_ffn_dim,
                dropout=pred_dropout,
                max_length=pred_max_length,
            )
        else:
            self.predictor = nn.Sequential(
                nn.Linear(hidden_size, pred_hid),
                nn.GELU(),
                nn.LayerNorm(pred_hid),
                nn.Linear(pred_hid, hidden_size),
            )

        self.ema_decay      = ema_decay
        self.use_cls        = use_cls
        self.predictor_type = predictor_type

    def get_ema_decay(self, epoch: int, total_epochs: int,
                      start: float = 0.996, end: float = 0.9999) -> float:
        """
        Cosine EMA decay schedule over epochs.
        epoch 0              → start (0.996): target updates quickly
        epoch total_epochs-1 → end   (0.9999): target nearly frozen
        """
        progress = min(epoch / max(total_epochs - 1, 1), 1.0)
        cosine   = (1 + math.cos(math.pi * progress)) / 2
        return end - (end - start) * cosine

    @torch.no_grad()
    def update_target(self):
        """EMA update. Call after every optimizer step."""
        d = self.ema_decay
        for p_online, p_target in zip(self.encoder.parameters(),
                                       self.target_encoder.parameters()):
            p_target.data.mul_(d).add_(p_online.data, alpha=1 - d)

    def forward(self, context_inputs, target_inputs, mask=None):
        # Strip token_type_ids for RoBERTa; harmless for BERT
        context_inputs = get_encoder_inputs(context_inputs)
        target_inputs  = get_encoder_inputs(target_inputs)

        context_hidden = self.encoder(**context_inputs).last_hidden_state  # [B, T, D]

        if self.use_cls:
            context_hidden = context_hidden[:, :1, :]

        if self.use_projector:
            context_hidden = self.expander(self.projector(context_hidden))

        if self.predictor_type == "transformer":
            key_padding_mask = (
                None if self.use_cls
                else (context_inputs["attention_mask"] == 0)
            )
            pred = self.predictor(context_hidden, key_padding_mask=key_padding_mask)
        else:
            pred = self.predictor(context_hidden)

        with torch.no_grad():
            target = self.target_encoder(**target_inputs).last_hidden_state
            if self.use_cls:
                target = target[:, :1, :]

        loss = jepa_loss(pred, target, mask if not self.use_cls else None)
        return loss, pred, target

    def smart_penalty(
        self,
        context_inputs: dict,
        mask=None,
        epsilon: float = 1e-6,
        num_steps: int = 1,
        step_size: float = 1e-3,
    ) -> torch.Tensor:
        """
        SMART regularization for JEPA: L2 penalty between clean and adversarial
        predictor outputs.

        NOTE: inner backward passes accumulate stale grads in model params.
        Call opt.zero_grad() after this method and before the main backward pass.
        """
        context_inputs = get_encoder_inputs(context_inputs)
        attention_mask = context_inputs.get("attention_mask")

        embedding_layer = self.encoder.get_input_embeddings()
        with torch.no_grad():
            embeds = embedding_layer(context_inputs["input_ids"])  # [B, T, D]

        enc_kw = {"inputs_embeds": embeds, "attention_mask": attention_mask}
        if "token_type_ids" in context_inputs:
            enc_kw["token_type_ids"] = context_inputs["token_type_ids"]
        key_padding_mask = (
            None if self.use_cls
            else (attention_mask == 0 if attention_mask is not None else None)
        )

        def _forward(emb):
            kw = dict(enc_kw)
            kw["inputs_embeds"] = emb
            h = self.encoder(**kw).last_hidden_state
            if self.use_cls:
                h = h[:, :1, :]
            if self.use_projector:
                h = self.expander(self.projector(h))
            if self.predictor_type == "transformer":
                return self.predictor(h, key_padding_mask=key_padding_mask)
            return self.predictor(h)

        # Step 1 — Clean forward pass
        with torch.no_grad():
            clean_pred = _forward(embeds)

        # Step 2 — Gradient ascent to find worst-case δ
        delta = torch.zeros_like(embeds, requires_grad=True)

        for _ in range(num_steps):
            adv_pred = _forward(embeds + delta)
            F.mse_loss(adv_pred, clean_pred).backward()

            with torch.no_grad():
                g = delta.grad
                g_norm = g.norm(p=2, dim=-1, keepdim=True).clamp(min=1e-8)
                delta_new = delta + step_size * (g / g_norm)
                d_norm = delta_new.norm(p=2, dim=-1, keepdim=True)
                delta_new = delta_new * (epsilon / d_norm).clamp(max=1.0)

            delta = delta_new.requires_grad_(True)

        # Steps 3+4 — Adversarial forward pass → L2 penalty
        adv_pred = _forward(embeds + delta.detach())
        return F.mse_loss(adv_pred, clean_pred)


class MLMJEPA(nn.Module):
    """
    BERT-style MLM baseline — also RoBERTa compatible via get_encoder_inputs().
    """
    def __init__(self, initial_encoder, vocab_size: int,
                 hidden_size: int = 768, use_cls: bool = False):
        super().__init__()
        self.encoder         = initial_encoder
        self.vocab_projection = nn.Linear(hidden_size, vocab_size)
        self.use_cls         = use_cls

    def forward(self, context_inputs, target_inputs, mask=None, ignore_index=-100):
        context_inputs = get_encoder_inputs(context_inputs)
        target_inputs  = get_encoder_inputs(target_inputs)

        hidden     = self.encoder(**context_inputs).last_hidden_state
        logits     = self.vocab_projection(hidden)
        target_ids = target_inputs["input_ids"]

        if mask is not None:
            mask = mask.bool()
            target_ids_masked = target_ids.clone()
            target_ids_masked[~mask] = ignore_index
        else:
            attention_mask = target_inputs.get("attention_mask", None)
            target_ids_masked = target_ids.clone()
            if attention_mask is not None:
                target_ids_masked[attention_mask == 0] = ignore_index

        loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            target_ids_masked.view(-1),
            ignore_index=ignore_index,
        )
        return loss, logits, target_ids

    def smart_penalty(
        self,
        context_inputs: dict,
        mask=None,
        epsilon: float = 1e-6,
        num_steps: int = 1,
        step_size: float = 1e-3,
    ) -> torch.Tensor:
        """
        SMART regularization (Liu et al. 2020).

        Finds worst-case embedding perturbation δ (per-token L2 norm ≤ ε) via
        gradient ascent, then returns KL(p_clean || p_adv) at masked positions.

        NOTE: inner backward passes accumulate stale grads in model params.
        Call opt.zero_grad() after this method and before the main backward pass.
        """
        context_inputs = get_encoder_inputs(context_inputs)
        attention_mask = context_inputs.get("attention_mask")

        # Word embeddings only; position/type embeddings are added by the encoder
        embedding_layer = self.encoder.get_input_embeddings()
        with torch.no_grad():
            embeds = embedding_layer(context_inputs["input_ids"])  # [B, T, D]

        encoder_kwargs = {"inputs_embeds": embeds, "attention_mask": attention_mask}
        if "token_type_ids" in context_inputs:
            encoder_kwargs["token_type_ids"] = context_inputs["token_type_ids"]

        # Step 1 — Clean forward pass: baseline vocabulary distributions
        with torch.no_grad():
            clean_hidden = self.encoder(**encoder_kwargs).last_hidden_state
            clean_logits = self.vocab_projection(clean_hidden)      # [B, T, V]
            clean_probs  = F.softmax(clean_logits, dim=-1)          # [B, T, V]

        # Step 2 — Gradient ascent to find worst-case δ
        delta = torch.zeros_like(embeds, requires_grad=True)

        for _ in range(num_steps):
            adv_kwargs = dict(encoder_kwargs)
            adv_kwargs["inputs_embeds"] = embeds + delta
            adv_log_probs = F.log_softmax(
                self.vocab_projection(self.encoder(**adv_kwargs).last_hidden_state),
                dim=-1,
            )

            if mask is not None:
                m = mask.bool()
                kl = F.kl_div(adv_log_probs[m], clean_probs[m], reduction="batchmean")
            else:
                kl = F.kl_div(
                    adv_log_probs.view(-1, adv_log_probs.size(-1)),
                    clean_probs.view(-1, clean_probs.size(-1)),
                    reduction="batchmean",
                )

            kl.backward()

            with torch.no_grad():
                g = delta.grad
                g_norm = g.norm(p=2, dim=-1, keepdim=True).clamp(min=1e-8)
                delta_new = delta + step_size * (g / g_norm)
                # Per-token L2 projection back into ε-ball
                d_norm = delta_new.norm(p=2, dim=-1, keepdim=True)
                delta_new = delta_new * (epsilon / d_norm).clamp(max=1.0)

            delta = delta_new.requires_grad_(True)

        # Step 3 — Adversarial forward pass with the found δ
        adv_kwargs = dict(encoder_kwargs)
        adv_kwargs["inputs_embeds"] = embeds + delta.detach()
        adv_log_probs = F.log_softmax(
            self.vocab_projection(self.encoder(**adv_kwargs).last_hidden_state),
            dim=-1,
        )

        # Step 4 — SMART penalty: KL(p_clean || p_adv) at masked positions
        if mask is not None:
            m = mask.bool()
            return F.kl_div(adv_log_probs[m], clean_probs[m], reduction="batchmean")
        return F.kl_div(
            adv_log_probs.view(-1, adv_log_probs.size(-1)),
            clean_probs.view(-1, clean_probs.size(-1)),
            reduction="batchmean",
        )
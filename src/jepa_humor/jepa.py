import torch
import torch.nn as nn
import torch.nn.functional as F
from copy import deepcopy
import math


# Loss function currently for run_25513 (best)
def jepa_loss(
    pred, target, mask=None,
    std_weight=0.1, cov_weight=0.04, knn_weight=0.1,
    huber_beta=0.1, knn_k=14,
    logvar=None,
):
    # logvar (gated, 2026-11-14 idea 2): per-prediction log-variance [B,T,1]. None = unchanged.
    target = target.detach()
    target = F.normalize(target, dim=-1)
    pred   = F.normalize(pred,   dim=-1)

    if mask is not None and pred.size(1) > 1:
        mask   = mask.bool()
        pred   = pred[mask]    # [M, D]
        target = target[mask]  # [M, D]
        if logvar is not None:
            logvar = logvar[mask]
    else:
        # Flatten [B, T, D] or [B, 1, D] → [B*T, D] so covariance math works
        B, T, D = pred.shape
        pred   = pred.reshape(B * T, D)
        target = target.reshape(B * T, D)
        if logvar is not None:
            logvar = logvar.reshape(B * T, -1)

    N, D = pred.shape

    # ── 1. Invariance: Huber (smooth-L1) ───────────────────────────────────
    # Robust to outliers from the EMA target encoder's drift.
    #inv_loss = F.smooth_l1_loss(pred, target, beta=huber_beta)
    nll_loss = 0.0
    if logvar is None:
        inv_loss = F.smooth_l1_loss(pred, target)
    else:
        # Uncertainty-weighted invariance: per-prediction Huber weighted by precision 1/sigma^2,
        # normalised to mean 1 over the batch (total invariance strength unchanged; ambiguous
        # targets count less). sigma is trained only by the Gaussian NLL below (error detached).
        # Zero-initialised head -> all weights 1 at init -> encoder gradient == unweighted Huber.
        lv = logvar.float().reshape(-1).clamp(-15.0, 10.0)
        h = F.smooth_l1_loss(pred, target, reduction="none").mean(-1)       # [N]
        prec = torch.exp(-lv)
        w = (prec / prec.mean()).detach()
        inv_loss = (w * h).mean()
        nll_loss = 0.5 * (h.detach() * prec + lv).mean()

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

    # sim_t = target @ target.T                           # [N, N]
    # sim_p = pred   @ pred.T                             # [N, N]

    target_zm = target - target.mean(dim=0)
    cov_t     = (target_zm.T @ target_zm) / (N - 1)
    try:
        C_t_inv = torch.linalg.pinv(cov_t + 1e-4 * torch.eye(D, device=target.device))
    except torch._C._LinAlgError:
        try:
            C_t_inv = torch.linalg.pinv(cov_t + 1e-3 * torch.eye(D, device=target.device))
        except torch._C._LinAlgError:
            C_t_inv = torch.eye(D, device=target.device)
    except RuntimeError:
        # cuSOLVER handle init can fail on some GPUs (e.g. H100); pinv on CPU (identical result)
        C_t_inv = torch.linalg.pinv(cov_t.cpu() + 1e-4 * torch.eye(D)).to(target.device)

    diff_t  = target.unsqueeze(1) - target.unsqueeze(0)   # [N, N, D]
    sim_t   = -torch.einsum('ijd,dk,ijk->ij', diff_t, C_t_inv, diff_t)

    # use the pred covariance you already have
    try:
        C_inv = torch.linalg.pinv(cov.detach() + 1e-4 * torch.eye(D, device=pred.device))
    except torch._C._LinAlgError:
        try:
            C_inv = torch.linalg.pinv(cov.detach() + 1e-3 * torch.eye(D, device=pred.device))
        except torch._C._LinAlgError:
            C_inv = torch.eye(D, device=pred.device)
    except RuntimeError:
        # cuSOLVER handle init can fail on some GPUs (e.g. H100); pinv on CPU (identical result)
        C_inv = torch.linalg.pinv(cov.detach().cpu() + 1e-4 * torch.eye(D)).to(pred.device)
    # pairwise Mahalanobis: (xᵢ - xⱼ)ᵀ C⁻¹ (xᵢ - xⱼ)
    diff = pred.unsqueeze(1) - pred.unsqueeze(0)   # [N, N, D]
    sim_p = -torch.einsum('ijd,dk,ijk->ij', diff, C_inv, diff)

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
        + nll_loss
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
        # Loss weights (config-overridable; defaults match jepa_loss)
        knn_weight: float = 0.1,
        # VICReg uses inv:var:cov = 25:25:1. Ours is 1:0.1:0.04 -- the covariance ratio is
        # right (1:0.04 == 25:1) but variance is 10x too small, which makes total collapse the
        # CHEAPER minimum: collapsed loss is 0.1*relu(1-0) = 0.10 against a measured 0.33.
        # Exposed so it can be swept; 0.1 keeps every existing run byte-identical.
        std_weight: float = 0.1,
        # Gated 2026-11-14 (user idea 1, "average several maskings"): the target is the mean of the
        # EMA encoder's output over K maskings of the same sentence -- the data's masked copy plus
        # K-1 fresh ones drawn here with the data's random_token rule. 1 = unchanged (single target).
        target_num_masks: int = 1,
        mask_ratio: float = 0.0,
        mask_token_id: int = -1,
        special_token_ids: tuple = (),
        # Gated 2026-11-14: how the K targets are used. "mean" (default) = one averaged target;
        # "separate" = K separate targets, loss = mean over k of jepa_loss(pred, target_k).
        target_multi_mode: str = "mean",
        # Gated 2026-11-14 (user idea 2): predictor also outputs a log-variance per prediction;
        # invariance becomes uncertainty-weighted (see jepa_loss logvar). False = unchanged.
        pred_uncertainty: bool = False,
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
        self.knn_weight     = knn_weight
        self.std_weight     = std_weight
        self.predictor_type = predictor_type
        self.target_num_masks  = int(target_num_masks)
        self.mask_ratio        = mask_ratio
        self.mask_token_id     = mask_token_id
        self.special_token_ids = tuple(special_token_ids)
        if target_multi_mode not in ("mean", "separate"):
            raise ValueError(f"target_multi_mode must be mean|separate, got {target_multi_mode}")
        self.target_multi_mode = target_multi_mode
        self.pred_uncertainty  = bool(pred_uncertainty)
        if self.pred_uncertainty:
            if predictor_type != "transformer":   # nn.Sequential would run an attached head in forward
                raise ValueError("pred_uncertainty requires predictor_type transformer")
            # registered under the predictor so the optimizer's predictor group trains it;
            # zero init -> logvar 0 for every prediction -> all weights 1 at step 0
            self.predictor.logvar_head = nn.Linear(hidden_size, 1)
            nn.init.zeros_(self.predictor.logvar_head.weight)
            nn.init.zeros_(self.predictor.logvar_head.bias)

    @torch.no_grad()
    def _random_token_mask(self, input_ids, attention_mask):
        """Torch twin of data.py random_token_mask: per row, exactly max(1, int(n * mask_ratio)) of the
        n maskable tokens (attended, not pad/cls/sep) become mask_token_id, uniformly without replacement."""
        maskable = attention_mask.bool()
        for t in self.special_token_ids:
            maskable &= input_ids != t
        n_mask = (maskable.sum(1).float() * self.mask_ratio).long().clamp(min=1)
        n_mask = torch.minimum(n_mask, maskable.sum(1))
        keys = torch.rand(input_ids.shape, device=input_ids.device).masked_fill(~maskable, 2.0)
        rank = keys.argsort(dim=1).argsort(dim=1)           # 0 = smallest key
        sel = (rank < n_mask.unsqueeze(1)) & maskable
        return input_ids.masked_fill(sel, self.mask_token_id)

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
        logvar = self.predictor.logvar_head(pred) if self.pred_uncertainty else None   # [B,T,1]

        with torch.no_grad():
            target = self.target_encoder(**target_inputs).last_hidden_state
            if self.target_num_masks > 1 and self.target_multi_mode == "separate":
                targets = [target]
                for _ in range(self.target_num_masks - 1):
                    extra = dict(target_inputs)
                    extra["input_ids"] = self._random_token_mask(context_inputs["input_ids"],
                                                                 context_inputs["attention_mask"])
                    targets.append(self.target_encoder(**extra).last_hidden_state)
                if self.use_cls:
                    targets = [t[:, :1, :] for t in targets]
        if self.target_num_masks > 1 and self.target_multi_mode == "separate":
            m = mask if not self.use_cls else None
            loss = sum(jepa_loss(pred, t, m, knn_weight=self.knn_weight, std_weight=self.std_weight, logvar=logvar)
                       for t in targets) / len(targets)
            return loss, pred, torch.stack(targets).mean(0)

        with torch.no_grad():
            if self.target_num_masks > 1:
                # extra maskings come from the CLEAN context ids (the sentence the data masked);
                # raw hidden states are averaged, jepa_loss L2-normalises the mean as before
                for _ in range(self.target_num_masks - 1):
                    extra = dict(target_inputs)
                    extra["input_ids"] = self._random_token_mask(context_inputs["input_ids"],
                                                                 context_inputs["attention_mask"])
                    target = target + self.target_encoder(**extra).last_hidden_state
                target = target / self.target_num_masks
            if self.use_cls:
                target = target[:, :1, :]

        loss = jepa_loss(pred, target, mask if not self.use_cls else None,
                         knn_weight=self.knn_weight, std_weight=self.std_weight, logvar=logvar)
        return loss, pred, target


class MLMJEPA(nn.Module):
    """
    BERT-style MLM baseline — also RoBERTa compatible via get_encoder_inputs().
    """
    def __init__(self, initial_encoder, vocab_size: int,
                 hidden_size: int = 768, use_cls: bool = False,
                 bert_head: bool = False, pretrained_head_from: str = None, head_any: bool = False):
        super().__init__()
        self.encoder         = initial_encoder
        self.use_cls         = use_cls
        self.bert_head       = bert_head

        if bert_head:
            # Original BERT MLM head: dense -> GELU -> LayerNorm -> decoder TIED to the input
            # embeddings (+ its own bias). Gated; default False keeps the single untied Linear
            # that every previous mlm run used.
            emb = initial_encoder.get_input_embeddings().weight      # [vocab, hidden]
            self.mlm_transform = nn.Sequential(
                nn.Linear(hidden_size, hidden_size),
                nn.GELU(),
                nn.LayerNorm(hidden_size, eps=1e-12),
            )
            self.mlm_decoder = nn.Linear(hidden_size, emb.size(0), bias=False)  # weight [vocab, hidden]
            self.mlm_decoder.weight = emb                            # shared tensor, not a copy
            self.mlm_bias = nn.Parameter(torch.zeros(emb.size(0)))
            self.vocab_projection = None
            if pretrained_head_from:
                self._load_pretrained_head_any(pretrained_head_from) if head_any else self._load_pretrained_head(pretrained_head_from)
        else:
            self.vocab_projection = nn.Linear(hidden_size, vocab_size)

    def _load_pretrained_head(self, name: str):
        """Restore the MLM head the public checkpoint already ships.

        AutoModel/BertModel drops `cls.predictions.*`, so the transform and bias would otherwise
        be random while the tied decoder is pretrained -- a broken composition whose initial loss
        is ln(V). Measured on bert-base-uncased: fresh transform 10.87 vs pretrained 4.97. With a
        pretrained ENCODER, relearning the head wastes the step budget and pushes large
        off-objective gradients into weights we want to nudge, so continued MLM pretraining should
        resume from BERT's own head. Gated by `bert_mlm_head_pretrained`.
        """
        from transformers import AutoModelForMaskedLM
        sd = AutoModelForMaskedLM.from_pretrained(name).state_dict()
        p = "cls.predictions."
        if p + "transform.dense.weight" not in sd:       # e.g. a model with no MLM head shipped
            print(f"[MLMJEPA] WARNING: {name} ships no {p}transform.* — head stays random")
            return
        with torch.no_grad():
            self.mlm_transform[0].weight.copy_(sd[p + "transform.dense.weight"])
            self.mlm_transform[0].bias.copy_(sd[p + "transform.dense.bias"])
            self.mlm_transform[2].weight.copy_(sd[p + "transform.LayerNorm.weight"])
            self.mlm_transform[2].bias.copy_(sd[p + "transform.LayerNorm.bias"])
            self.mlm_bias.copy_(sd[p + "bias"])
        # decoder is deliberately NOT copied: it is already the tied input-embedding tensor, which
        # is what cls.predictions.decoder.weight is tied to in the checkpoint.
        print(f"[MLMJEPA] loaded pretrained MLM head from {name} "
              f"(transform dense+LayerNorm, bias; decoder stays tied)")

    def _load_pretrained_head_any(self, name: str):
        """2026-10-26, gated (mlm_head_pretrained_any): restore the shipped MLM head for BERT (cls.predictions.*), RoBERTa
        (lm_head.*) or DeBERTa v1, whose head HF's DebertaForMaskedLM leaves RANDOM (the raw checkpoint names it
        lm_predictions.lm_head.*). Same composition everywhere: dense -> GELU -> LayerNorm -> tied decoder + bias."""
        from huggingface_hub import hf_hub_download
        sd = None
        for fn in ("pytorch_model.bin", "model.safetensors"):
            try:
                f = hf_hub_download(name, fn)
                if fn.endswith(".bin"):
                    sd = torch.load(f, map_location="cpu", weights_only=False)
                else:
                    from safetensors.torch import load_file
                    sd = load_file(f)
                break
            except Exception:
                continue
        if sd is None:
            raise RuntimeError(f"[MLMJEPA] cannot read the raw checkpoint of {name}")
        for d, ln, b in (("cls.predictions.transform.dense", "cls.predictions.transform.LayerNorm", "cls.predictions.bias"),
                         ("lm_head.dense", "lm_head.layer_norm", "lm_head.bias"),
                         ("lm_predictions.lm_head.dense", "lm_predictions.lm_head.LayerNorm", "lm_predictions.lm_head.bias")):
            if d + ".weight" in sd:
                with torch.no_grad():
                    self.mlm_transform[0].weight.copy_(sd[d + ".weight"]); self.mlm_transform[0].bias.copy_(sd[d + ".bias"])
                    g = sd.get(ln + ".weight", sd.get(ln + ".gamma")); bt = sd.get(ln + ".bias", sd.get(ln + ".beta"))   # old BERT ckpts: gamma/beta
                    self.mlm_transform[2].weight.copy_(g); self.mlm_transform[2].bias.copy_(bt)
                    self.mlm_bias.copy_(sd[b])
                print(f"[MLMJEPA] loaded pretrained MLM head from {name} ({d.rsplit('.', 1)[0]}.*; decoder stays tied)")
                return
        raise RuntimeError(f"[MLMJEPA] {name}: no MLM head found in the raw checkpoint")

    def project(self, h):
        """hidden states -> vocab logits"""
        if self.bert_head:
            return F.linear(self.mlm_transform(h), self.mlm_decoder.weight, self.mlm_bias)
        return self.vocab_projection(h)

    def forward(self, context_inputs, target_inputs, mask=None, ignore_index=-100,
                sparse_loss=False):
        context_inputs = get_encoder_inputs(context_inputs)
        target_inputs  = get_encoder_inputs(target_inputs)

        hidden = self.encoder(**context_inputs).last_hidden_state

        if sparse_loss and mask is not None:
            # Project ONLY the selected positions, as BERT's MLM head does. The dense path
            # below materialises [B, T, vocab] (128x512x30522 ~ 8GB at batch 128/seq 512) and
            # OOMs a 40GB card; gathering first cuts that by 1/mask_ratio.
            sel        = mask.bool()
            logits_sel = self.project(hidden[sel])               # [n_selected, vocab]
            target_sel = target_inputs["input_ids"][sel]
            loss = F.cross_entropy(logits_sel, target_sel, ignore_index=ignore_index)
            return loss, logits_sel, target_sel

        logits     = self.project(hidden)
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

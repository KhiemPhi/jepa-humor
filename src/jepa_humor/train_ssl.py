import os
from tqdm import tqdm
import torch
from torch.cuda.amp import autocast
from torch.amp import GradScaler
from transformers import AutoTokenizer, AutoModel
from .utils import set_seed, device, save_ckpt
from .jepa import JEPA, MLMJEPA
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import os
from torch import amp
import torch.nn.functional as F
import math
from .data import (
    load_hf_splits,
    make_dataloaders,
    preprocess_dataset_ssl,
    tokenize_only_ssl,
    remask_ssl_dataset,
    sample_dataset,
    load_anli_splits,
    load_snli_mnli_for_anli,
    load_composite_nli_dataset,
    load_swag,
    expand_swag_for_pretraining,
    expand_copa_for_pretraining,
    preprocess_multirc_for_pretraining,
    expand_record_for_pretraining,
)

def build_optimizer_and_scheduler(jepa, lr_pred=1e-4, lr_enc=5e-5, weight_decay=0.01, total_steps=1000, warmup_ratio=0.06, model_type="jepa", enc_delay_steps=0):
    # param groups: predictor/vocab_projection (higher LR), encoder (lower LR), target is frozen
    if model_type == "mlm":
        # MLMJEPA: has vocab_projection instead of predictor
        if getattr(jepa, "bert_head", False):
            # Tied head: the decoder weight IS the input-embedding matrix, so it is already in
            # enc_params — adding it here too would make AdamW see a duplicate parameter. Only
            # the transform and the decoder bias are head-specific. (Consequence: the tied
            # decoder trains at the encoder LR, which is what BERT does — one LR throughout.)
            pred_params = list(jepa.mlm_transform.parameters()) + [jepa.mlm_bias]
        else:
            pred_params = list(jepa.vocab_projection.parameters())
    else:
        # Standard JEPA: has predictor
        pred_params = list(jepa.predictor.parameters())
    
    enc_params  = [p for p in jepa.encoder.parameters() if p.requires_grad]

    optimizer = torch.optim.AdamW(
        [
            {"params": pred_params, "lr": lr_pred},
            {"params": enc_params,  "lr": lr_enc}
        ],
        lr=lr_pred, weight_decay=weight_decay, betas=(0.9, 0.98), eps=1e-8
    )

    warmup_steps = max(1, int(total_steps * warmup_ratio))
    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        # cosine decay to 10% of base
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress))

    if enc_delay_steps <= 0:
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
        return optimizer, scheduler
    # gated (pred_warmup_steps > 0, 2026-11-14): the encoder group sits at lr 0 for the first
    # enc_delay_steps optimizer steps, then runs its own warmup + cosine over the remaining
    # steps. The predictor group keeps the unchanged schedule. Grads are also dropped in the loop.
    enc_total = max(1, total_steps - enc_delay_steps)
    enc_warm = max(1, int(enc_total * warmup_ratio))
    def enc_lambda(step):
        s = step - enc_delay_steps
        if s < 0:
            return 0.0
        if s < enc_warm:
            return s / enc_warm
        progress = (s - enc_warm) / max(1, enc_total - enc_warm)
        return 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, [lr_lambda, enc_lambda])
    return optimizer, scheduler

def pretrain(cfg):
    set_seed(cfg["seed"])
    # Get GPU ID from config, default to None (which will use cuda:0 or cpu)
    gpu_id = cfg.get("gpu_id", None)
    dev = device(gpu_id)
    if gpu_id is not None:
        print(f"Using device: {dev}")
    else:
        print(f"Using device: {dev} (default)")

    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"])
    base = AutoModel.from_pretrained(cfg["model_name"])
    if tokenizer.mask_token_id is None:
        raise ValueError("Tokenizer must have a [MASK] token.")

    # --- LoRA adaptation (GATED, default OFF) ---
    # Existing configs have no `use_lora` key -> this block is skipped and `base`
    # stays a plain AutoModel, so the BERT/RoBERTa/DeBERTa path is byte-identical.
    # When use_lora=true (e.g. adapting XLM-R XL), the encoder is wrapped in PEFT
    # LoRA adapters: the base weights freeze and only the adapters train. A PeftModel
    # transparently proxies .config/.get_input_embeddings()/forward/deepcopy/parameters(),
    # so JEPA/MLMJEPA and the EMA target work unchanged.
    use_lora = cfg.get("use_lora", False)
    if use_lora:
        from peft import get_peft_model, LoraConfig
        lora_cfg = LoraConfig(
            r=cfg.get("lora_r", 16),
            lora_alpha=cfg.get("lora_alpha", 32),
            lora_dropout=cfg.get("lora_dropout", 0.05),
            target_modules=cfg.get("lora_target_modules", ["query", "value"]),
            bias="none",
        )
        base = get_peft_model(base, lora_cfg)
        base.print_trainable_parameters()

    # Choose model type: "jepa" (embedding reconstruction) or "mlm" (token ID prediction)
    model_type = cfg.get("model_type", "jepa").lower()
    
    if model_type == "mlm":
        # BERT-style MLM: single encoder + vocab projection (no EMA target encoder)
        jepa = MLMJEPA(
            base,
            vocab_size=tokenizer.vocab_size,
            hidden_size=base.config.hidden_size,
            use_cls=True,
            # gated: original BERT's dense->GELU->LN->tied-decoder head. Default False keeps the
            # single untied Linear used by every previous mlm run.
            bert_head=bool(cfg.get("bert_mlm_head", False)),
            # gated: restore the checkpoint's own cls.predictions head instead of a random one.
            # Only meaningful with bert_mlm_head=True (that is the branch that builds the head).
            pretrained_head_from=(cfg["model_name"]
                                  if cfg.get("bert_mlm_head_pretrained", False) else None),
            head_any=bool(cfg.get("mlm_head_pretrained_any", False)),   # gated 2026-10-26: roberta / deberta heads too
        ).to(dev)
        print("Using BERT-style MLM (single encoder, no EMA target encoder)"
              + (" — original BERT head, decoder tied to input embeddings"
                 if cfg.get("bert_mlm_head", False) else ""))
    else:
        # Original JEPA that reconstructs embeddings
        embedding_matrix = base.get_input_embeddings().weight.detach().cpu().numpy()
        jepa = JEPA(
            base,
            hidden_size=base.config.hidden_size,
            pred_hid=512,
            ema_decay=cfg["ema_decay"],
            use_cls=cfg.get("use_cls", True),   # default True preserves all existing tasks; set False for token-level JEPA
            knn_weight=cfg.get("knn_weight", 0.1),
            std_weight=cfg.get("std_weight", 0.1),   # gated; 0.1 = unchanged
            # token-level JEPA feeds the predictor the full T-length sequence, so its pos-embed must cover max_length.
            # CLS runs only ever see T=1 -> keep 256 (unchanged) so no other task's model is altered.
            pred_max_length=(cfg["max_length"] if not cfg.get("use_cls", True) else 256),
            # gated 2026-11-14: multi-mask averaged target; 1 (default) = unchanged
            target_num_masks=int(cfg.get("target_num_masks", 1)),
            mask_ratio=cfg.get("mask_ratio", 0.0),
            mask_token_id=tokenizer.mask_token_id,
            special_token_ids=tuple(t for t in (tokenizer.pad_token_id, tokenizer.cls_token_id, tokenizer.sep_token_id) if t is not None),
            target_multi_mode=cfg.get("target_multi_mode", "mean"),   # gated: "separate" = K targets, mean of K losses
            pred_uncertainty=bool(cfg.get("pred_uncertainty", False)),  # gated: idea 2, uncertainty-weighted invariance
        ).to(dev)
        if int(cfg.get("target_num_masks", 1)) > 1 and cfg.get("mask_type", "random_token") != "random_token":
            raise ValueError("target_num_masks > 1 only implements mask_type random_token")
        print("Using standard JEPA (reconstructs embeddings)")
    
    jepa.train()

  
    dataset_config = cfg.get("dataset_config", None)
    is_anli = dataset_config == "anli" or (dataset_config is None and "anli" in cfg["dataset"].lower())
    is_swag = dataset_config == "swag" or (dataset_config is None and "swag" in cfg["dataset"].lower())
    
    if dataset_config == "composite_nli":
        # Use the flexible composite loader
        snli_subset = cfg.get("snli_subset", None)
        mnli_splits = cfg.get("mnli_splits", None)
        anli_train_rounds = cfg.get("anli_train_rounds", None)
        anli_val_rounds = cfg.get("anli_val_rounds", [1, 2, 3])
        anli_val_type = cfg.get("anli_val_type", "test")
        ds = load_composite_nli_dataset(
            snli_subset=snli_subset,
            mnli_splits=mnli_splits,
            anli_train_rounds=anli_train_rounds,
            anli_val_rounds=anli_val_rounds,
            anli_val_type=anli_val_type,
        )
    elif dataset_config == "snli_mnli_anli":
        anli_rounds = cfg.get("anli_rounds", [1, 2, 3])
        mnli_splits = cfg.get("mnli_splits", ["validation_matched", "validation_mismatched"])
        snli_subset = cfg.get("snli_split", "train")
        ds = load_snli_mnli_for_anli(
            snli_subset=snli_subset,
            mnli_splits=mnli_splits,
            anli_rounds=anli_rounds,
        )
    elif is_anli:
        # ANLI: aggregate training sets from specified rounds
        train_rounds = cfg.get("train_rounds", [1, 2, 3])  # Default: use all rounds
        if isinstance(train_rounds, int):
            train_rounds = [train_rounds]  # Convert single int to list
        ds = load_anli_splits(cfg["dataset"], train_rounds=train_rounds, seed=cfg["seed"])
    elif is_swag:
        swag_ds = load_swag()
        # Expand SWAG for pretraining: create one example per ending
        ds = {
            "train": expand_swag_for_pretraining(swag_ds["train"]),
            "validation": expand_swag_for_pretraining(swag_ds["validation"]),
        }
    elif dataset_config == "squad_context":
        # JEPA pretrain corpus = unique SQuAD train contexts (dedup'd passages).
        from datasets import load_dataset as _ld, Dataset as _DS
        _raw = _ld(cfg["dataset"], split="train")
        _seen, _rows = set(), []
        for _c in _raw["context"]:
            _c = (_c or "").strip()
            if _c and _c not in _seen:
                _seen.add(_c)
                _rows.append({"text": _c})
        print(f"[squad_context] {len(_rows)} unique contexts from {cfg['dataset']}")
        ds = {"train": _DS.from_list(_rows)}
    elif dataset_config == "dpr_context":
        # Pretrain corpus = DPR multiset passages (NQ+TriviaQA+WebQ+CuratedTREC), prebuilt and
        # dedup'd by experiment/dpr_qapre/scripts/build_corpus.py. cfg["dataset"] is a
        # save_to_disk path, not an HF hub id.
        # Keeps `title` and `text` as separate columns so that text_keys: ["title", "text"]
        # tokenizes them as a PAIR -> [CLS] title [SEP] passage [SEP], matching DPR's context
        # encoder. Concatenating them here would collapse token_type_ids to all-zero.
        from datasets import load_from_disk as _lfd
        _d = _lfd(cfg["dataset"])
        print(f"[dpr_context] {len(_d)} unique passages from {cfg['dataset']}")
        ds = {"train": _d}
    else:
        ds = load_hf_splits(cfg["dataset"], config=dataset_config,
                            sample_per_config=cfg.get("sample_per_config"))
    
    # Sample datasets if sample_percent is specified
    sample_percent = cfg.get("sample_percent", None)
    if sample_percent is not None and sample_percent < 100.0:
        print(f"Sampling {sample_percent}% of datasets for quick prototyping...")
        ds = sample_dataset(ds, sample_percent=sample_percent, seed=cfg["seed"])
    
    split = {}
    split['train'] = ds['train']
    #split['train'] = split['train'].filter(lambda x: x['label'] == 1)
    if is_anli:
        split['test'] = ds['validation']
    elif dataset_config == 'mnli':
        split['test'] = ds['validation_matched']
    elif is_swag or dataset_config == "swag":
        split['test'] = ds['validation']
    elif 'validation' in ds:
        split['test'] = ds['validation']
    else:
        # Dataset has no validation split; carve one from train
        val_frac = cfg.get("val_split", 0.0001)
        n_val = max(1, int(len(ds['train']) * val_frac))
        shuffled = ds['train'].shuffle(seed=cfg.get("seed", 42))
        split['train'] = shuffled.select(range(n_val, len(shuffled)))
        split['test'] = shuffled.select(range(n_val))
    
    # Apply task-specific input transformations to match finetune format
    if dataset_config == "copa":
        split['train'] = expand_copa_for_pretraining(split['train'])
        split['test'] = expand_copa_for_pretraining(split['test'])
    elif dataset_config == "multirc":
        split['train'] = preprocess_multirc_for_pretraining(split['train'])
        split['test'] = preprocess_multirc_for_pretraining(split['test'])
    elif dataset_config == "record":
        split['train'] = expand_record_for_pretraining(split['train'])
        split['test'] = expand_record_for_pretraining(split['test'])

    # Get text keys from config (defaults to ["question", "sentence"])
    if is_anli or dataset_config == "snli_mnli_anli":
        text_keys = cfg.get("text_keys", ["premise", "hypothesis"])  # ANLI uses premise/hypothesis like MNLI
    elif dataset_config == 'mnli':
        text_keys = cfg.get("text_keys", ["premise", "hypothesis"])
    elif is_swag or dataset_config == "swag":
        text_keys = cfg.get("text_keys", ["question", "sentence"])
    elif dataset_config == "copa":
        text_keys = cfg.get("text_keys", ["text"])
    elif dataset_config == "multirc":
        text_keys = cfg.get("text_keys", ["paragraph", "question_answer"])
    else:
        text_keys = cfg.get("text_keys", ["question", "sentence"])
    print(f"Using text keys: {text_keys}")
    
    # Pre-tokenize datasets once before training
    mask_prob = cfg.get("mask_ratio", 0.15)
    span_len = cfg.get("mean_span_len", 3.0)
    mask_type = cfg.get("mask_type", "random_span")
    dynamic_masking = cfg.get("dynamic_masking", False)
    if dynamic_masking:
        cfg["experiment_name"] = cfg.get("experiment_name", "") + "_dynamic_masking"

    # Original-BERT MLM corruption (80/10/10 + explicit labels). Default False keeps every
    # existing task on the old 100%-[MASK] path. Only meaningful with model_type: mlm.
    bert_mlm = bool(cfg.get("bert_mlm_masking", False))
    # mask_type: spacy_entity (opt-in, 2026-10-10): precomputed spaCy entity spans; None for every other mask type
    ent_spans = cfg.get("ent_spans_file") if mask_type in ("spacy_entity", "spacy_entity_only", "weighted", "cue", "cue30", "cue40", "signed_cue", "cue_soft", "cue_span", "keep10", "keep20", "keep35", "keep50", "importance", "importance_soft", "entity_cue") else None
    if mask_type in ("swap_first", "swap_cap") and not dynamic_masking:
        raise ValueError(f"mask_type {mask_type} is implemented on the dynamic-masking path only (set dynamic_masking: true)")
    if bert_mlm and model_type != "mlm":
        raise ValueError("bert_mlm_masking=True is only valid with model_type: mlm")

    print("Pre-tokenizing validation dataset for SSL...")
    split['test'] = preprocess_dataset_ssl(
        split['test'],
        tokenizer,
        max_length=cfg["max_length"],
        text_keys=text_keys,
        mask_probability=mask_prob,
        span_mean=span_len,
        mask_type=("random_token" if mask_type in ("swap_first", "swap_cap") else mask_type),   # swap_first/swap_cap: static path has no pair flags -> val batch gets random_token
        bert_mlm_masking=bert_mlm,
        ent_spans_file=ent_spans,
    )

    if dynamic_masking:
        print("Pre-tokenizing training dataset for SSL (dynamic masking — will remask each epoch)...")
        # remask_fresh (opt-in; absent = unchanged): a genuinely new mask every epoch, in parallel,
        # with only one epoch's mask cache on disk (see remask_ssl_dataset). Needs remask_cache_dir.
        remask_fresh = bool(cfg.get("remask_fresh", False))
        _nproc = int(os.environ.get("SSL_TOKENIZE_PROC", "1")) if remask_fresh else None
        train_orig = tokenize_only_ssl(split['train'], tokenizer, max_length=cfg["max_length"], text_keys=text_keys,
                                       num_proc=_nproc, ent_spans_file=ent_spans,
                                       swap_flags=(mask_type in ("swap_first", "swap_cap")))   # swap_first / swap_cap (PAWS, opt-in)
        if remask_fresh:
            split['train'] = train_orig   # sizes the loader only; epoch 1 masks it inside the loop
        else:
            split['train'] = remask_ssl_dataset(train_orig, tokenizer, mask_prob, span_len, mask_type,
                                              bert_mlm_masking=cfg.get('bert_mlm_masking', False))
    else:
        print("Pre-tokenizing training dataset for SSL...")
        split['train'] = preprocess_dataset_ssl(
            split['train'],
            tokenizer,
            max_length=cfg["max_length"],
            text_keys=text_keys,
            mask_probability=mask_prob,
            span_mean=span_len,
            mask_type=mask_type,
            bert_mlm_masking=bert_mlm,
            ent_spans_file=ent_spans,
        )

    #split = ds['train'].train_test_split(test_size=cfg["val_split"], seed=cfg["seed"])
    train_loader, val_loader = make_dataloaders(split["train"], split["test"], cfg["batch_size"], cfg["num_workers"], pre_tokenized=True, ssl_mode=True)
    print(f"Train dataset size: {len(split['train'])}")
    print(f"Test dataset size: {len(split['test'])}")
    # grad_accum: micro-batches per optimizer step. Default 1 reproduces the original loop
    # exactly (zero_grad/step/scheduler/EMA every micro-batch), so no existing task changes.
    # Only the DAPT biomed configs set it, to reach Gururangan's effective batch of 2048 on
    # a 40GB card where the physical batch cannot exceed 128 at seq 512.
    accum = max(1, int(cfg.get("grad_accum", 1)))
    # total_steps counts OPTIMIZER steps, not micro-batches: it drives the 6% warmup, the
    # linear decay and auto_lr_pred below, all of which are per-optimizer-step quantities.
    total_steps = cfg["epochs"] * len(train_loader) // accum
    # schedule_total_steps: size warmup/decay (and auto_lr_pred) to an explicit optimizer-step
    # count instead of epochs*len. For runs capped by max_steps mid-epoch -- e.g. DAPT at the
    # paper's 12.5K steps, ~1.6 passes of the corpus -- so the linear decay ends exactly at the
    # cap. Absent = unchanged for every existing task.
    if cfg.get("schedule_total_steps"):
        total_steps = int(cfg["schedule_total_steps"])
    # Auto-scale predictor LR by sqrt of total steps; reference: 4000 steps → 5e-4
    # Prevents instability on large datasets, allows faster learning on small ones.
    auto_lr_pred = 5e-4 * math.sqrt(4000 / max(1, total_steps))
    lr_pred = cfg.get("lr_pred", float(max(2e-4, min(5e-4, auto_lr_pred))))
    print(f"Predictor LR: {lr_pred:.2e} (total_steps={total_steps})")
    pred_warmup_steps = int(cfg.get("pred_warmup_steps", 0))   # gated: 0 = unchanged
    opt, scheduler = build_optimizer_and_scheduler(jepa, lr_pred=lr_pred, lr_enc=cfg["lr"], model_type=model_type, total_steps=total_steps,
                                                   enc_delay_steps=pred_warmup_steps)
    if pred_warmup_steps > 0:
        print(f"Predictor warmup: encoder frozen for the first {pred_warmup_steps} optimizer steps")
    # amp_dtype: "float16" (default, unchanged for every existing task) or "bfloat16".
    # bf16 carries fp32's exponent range, which removes the overflow the kNN term is
    # exposed to: jepa.py inverts pinv(cov + 1e-4*I), and with use_cls the covariance is
    # rank-deficient by construction (rank <= batch-1 in 768 dims), so the ridge sets the
    # smallest eigenvalue and |C_inv| lands around 1e4. The forward stays in fp16 range
    # (measured peak |sim| ~5e2), but the backward scales with C_inv and is then multiplied
    # by the GradScaler factor. bert-base on BERT's corpus at batch 256 went nan at step
    # 36992/54774 that way. bf16 needs no loss scaling, so the scaler is disabled with it.
    amp_dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16}[cfg.get("amp_dtype", "float16")]
    use_amp = cfg.get("mixed_precision", True)
    scaler = GradScaler("cuda", enabled=use_amp and amp_dtype is torch.float16)
    if use_amp:
        print(f"AMP dtype: {amp_dtype} (GradScaler {'on' if scaler.is_enabled() else 'off'})")

    best = float("inf")
    patience = 0
    cfg["ckpt_dir"] = os.path.abspath(cfg["ckpt_dir"])
    os.makedirs(cfg["ckpt_dir"], exist_ok=True)
    loss_history = []
    max_steps = cfg.get("max_steps")   # optional global-step cap; None = full run
    global_step = 0
    micro_step = 0                     # micro-batches seen; global_step advances every `accum`

    for epoch in range(1, cfg["epochs"] + 1):
        if dynamic_masking:
            if remask_fresh:
                import glob
                _rdir = cfg["remask_cache_dir"]; os.makedirs(_rdir, exist_ok=True)
                split['train'] = None; train_loader = None   # drop last epoch's mask before deleting it
                for _f in glob.glob(os.path.join(_rdir, "remask_ep*.arrow")):
                    os.remove(_f)
                split['train'] = remask_ssl_dataset(train_orig, tokenizer, mask_prob, span_len, mask_type,
                                                  bert_mlm_masking=cfg.get('bert_mlm_masking', False),
                                                  fresh_seed=cfg["seed"] * 1000 + epoch, num_proc=_nproc,
                                                  cache_file_name=os.path.join(_rdir, f"remask_ep{epoch}.arrow"))
            else:
                split['train'] = remask_ssl_dataset(train_orig, tokenizer, mask_prob, span_len, mask_type,
                                              bert_mlm_masking=cfg.get('bert_mlm_masking', False))
            train_loader, _ = make_dataloaders(split["train"], split["test"], cfg["batch_size"], cfg["num_workers"], pre_tokenized=True, ssl_mode=True)
        running, steps = 0.0, 0
        pbar = tqdm(train_loader, desc=f"Pretrain {epoch}/{cfg['epochs']}", leave=False)
        if model_type != "mlm":
            # `epochs` currently drives three unrelated things: total_steps (line 308), the
            # 6% warmup derived from it, and this EMA ramp. So run length silently changes
            # the target-encoder regime: a 1-epoch run never leaves 0.9961, while a 5-epoch
            # run reaches 0.9993 by epoch 3 and 0.9999 by epoch 4 -- a nearly frozen target.
            # That is the leading explanation for the bert-corpus arm collapsing (BEIR 2.06)
            # while its own epoch-2 checkpoint was fine (28.85). These two keys decouple the
            # ramp from the run; both default to the existing behaviour exactly.
            #   ema_schedule: "cosine" (default) | "constant" -> pin at cfg["ema_decay"]
            #   ema_ramp_epochs: denominator for the cosine ramp (default cfg["epochs"])
            if cfg.get("ema_schedule", "cosine") == "constant":
                jepa.ema_decay = cfg["ema_decay"]
            else:
                jepa.ema_decay = jepa.get_ema_decay(epoch, cfg.get("ema_ramp_epochs", cfg["epochs"]))
            # Log the value only when a run opted out of the default schedule, so the logs of
            # every pre-existing task stay byte-identical. Without this there is no way to
            # confirm from the outside that the override actually took.
            if cfg.get("ema_schedule", "cosine") != "cosine" or "ema_ramp_epochs" in cfg:
                print(f"[Pretrain] epoch {epoch}: ema_decay={jepa.ema_decay:.6f} "
                      f"(schedule={cfg.get('ema_schedule','cosine')})", flush=True)
        for batch in pbar:
            # Data is already tokenized
            keep_tok = {
                "input_ids": batch["keep_input_ids"].to(dev, non_blocking=True),
                "attention_mask": batch["keep_attention_mask"].to(dev, non_blocking=True),
            }
            target_tok = {
                "input_ids": batch["target_input_ids"].to(dev, non_blocking=True),
                "attention_mask": batch["target_attention_mask"].to(dev, non_blocking=True),
            }
            if "keep_token_type_ids" in batch:
                keep_tok["token_type_ids"] = batch["keep_token_type_ids"].to(dev, non_blocking=True)
                target_tok["token_type_ids"] = batch["target_token_type_ids"].to(dev, non_blocking=True)

            if model_type == "mlm" and bert_mlm:
                # Original-BERT MLM: the ENCODER SEES THE CORRUPTED sequence (target_tok) and
                # predicts the original ids at the selected positions, which mlm_labels marks
                # (-100 elsewhere). Needed because the legacy branch below feeds the *unmasked*
                # text and scores tokens already visible in the input -- a copy task, not MLM.
                mlm_labels = batch["mlm_labels"].to(dev, non_blocking=True)
                mask = (mlm_labels != -100).float()
                mlm_input = target_tok
                target_for_mlm = {
                    "input_ids": mlm_labels,
                    "attention_mask": target_tok["attention_mask"],
                }
            elif model_type == "mlm":
                # LEGACY path (bert_mlm_masking=False), kept byte-compatible for reproducibility:
                # - Context is keep_tok (original unmasked text)
                # - Target is keep_input_ids (original tokens to predict)
                # - Mask indicates positions where target_input_ids has MASK tokens
                mask_token_id = tokenizer.mask_token_id
                mask = (target_tok["input_ids"] == mask_token_id).float()
                # Also respect attention_mask (exclude padding)
                mask = mask * target_tok["attention_mask"].float()

                # Create target dict with original tokens
                mlm_input = keep_tok
                target_for_mlm = {
                    "input_ids": keep_tok["input_ids"],  # Original tokens to predict
                    "attention_mask": keep_tok["attention_mask"]
                }
            else:
                # Standard JEPA: use attention_mask
                mask = target_tok["attention_mask"].float().to(dev, non_blocking=True)
                target_for_mlm = None  # Not used for standard JEPA

            if micro_step % accum == 0:
                opt.zero_grad(set_to_none=True)
            with amp.autocast("cuda", enabled=use_amp, dtype=amp_dtype):
                if model_type == "mlm":
                    # MLMJEPA: predict original token IDs at masked positions
                    if bert_mlm:
                        loss, pred, target = jepa(mlm_input, target_for_mlm, mask,
                                                  ignore_index=-100, sparse_loss=True)
                    else:
                        pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else -100
                        loss, pred, target = jepa(mlm_input, target_for_mlm, mask, ignore_index=pad_token_id)
                else:
                    # Standard JEPA: reconstruct embeddings
                    loss, pred, target = jepa(keep_tok, target_tok, mask)

            # Divide by accum so the accumulated gradient is the MEAN over the effective
            # batch -- what one large batch would have produced. Without this the effective
            # learning rate would silently scale with accum.
            scaler.scale(loss / accum).backward()
            micro_step += 1

            running += float(loss.item()); steps += 1
            pbar.set_postfix(loss=f"{running/steps:.4f}")

            if micro_step % accum:
                continue   # mid-accumulation: no optimizer/scheduler/EMA tick, no checkpoint

            if global_step < pred_warmup_steps:
                for p in opt.param_groups[1]["params"]:
                    p.grad = None   # encoder untouched (no AdamW state/decay) during predictor warmup
            scaler.step(opt)
            scaler.update()
            scheduler.step()
            # Only update target encoder for standard JEPA (not for BERT-style MLM)
            if model_type != "mlm":
                jepa.update_target()

            global_step += 1
            # optional step-interval checkpointing (config: save_every_steps). Saves the
            # encoder at each multiple so downstream eval can use any milestone.
            _save_every = cfg.get("save_every_steps")
            if _save_every and global_step % _save_every == 0 and model_type != "mlm":
                _exp = cfg.get("experiment_name", "")
                _slug = cfg.get("model_name", "").replace("-", "_").replace("/", "_")
                _pfx = f"{_exp}_{_slug}_" if _exp else (f"{_slug}_" if _slug else "")
                _sp = os.path.join(cfg["ckpt_dir"], f"step{global_step}_{_pfx}.pt")
                save_ckpt({"step": global_step,
                           "encoder_state_dict": jepa.encoder.state_dict(),
                           "jepa_state_dict": jepa.state_dict()}, _sp)
                print(f"[Pretrain] saved step checkpoint: {_sp}")
            if max_steps is not None and global_step >= max_steps:
                break

        epoch_loss = running / max(1, steps)
        loss_history.append({"epoch": epoch, "loss": epoch_loss})
        print(f"[Pretrain] epoch {epoch}: loss={epoch_loss:.4f}")

        # checkpoint & early stopping on train loss (SSL)
        exp_name = cfg.get("experiment_name", "")
        model_slug = cfg.get("model_name", "").replace("-", "_").replace("/", "_")
        prefix = f"{exp_name}_{model_slug}_" if exp_name else f"{model_slug}_" if model_slug else ""

        path = os.path.join(cfg["ckpt_dir"], f"{prefix}epoch{epoch}_loss{epoch_loss:.4f}.pt")
        # save_every_epoch: keep one checkpoint per epoch (the block below was commented out, so
        # only the overwritten best_*.pt survived). Gated; absent = unchanged for every other task.
        # Works for model_type mlm too, unlike the save_every_steps hook.
        # save_epoch_interval: N -> only every N-th epoch (TAPT ladder, 2026-09-28). Gated too.
        _ep_iv = cfg.get("save_epoch_interval")
        if cfg.get("save_every_epoch") or (_ep_iv and epoch % int(_ep_iv) == 0):
            save_ckpt({
                "epoch": epoch,
                "loss": epoch_loss,
                "encoder_state_dict": jepa.encoder.state_dict(),
                "jepa_state_dict": jepa.state_dict()
            }, path)
            print(f"[Pretrain] saved epoch checkpoint: {path}")
        # save_ckpt({
        #     "epoch": epoch,
        #     "loss": epoch_loss,
        #     "encoder_state_dict": jepa.encoder.state_dict(),
        #     "jepa_state_dict": jepa.state_dict()
        # }, path)

        if epoch_loss + 1e-6 < best:
            best = epoch_loss
            patience = 0
            if use_lora:
                # LoRA run: save only the small adapter (base is frozen). Part 2
                # (finetune_xlm.py) reloads it with PeftModel + merge_and_unload.
                best_adapter_dir = os.path.join(cfg["ckpt_dir"], f"best_{prefix}adapter")
                jepa.encoder.save_pretrained(best_adapter_dir)
                print(f"[Pretrain] saved LoRA adapter: {best_adapter_dir}")
            else:
                best_ckpt_path = os.path.join(cfg["ckpt_dir"], f"best_{prefix}.pt")
                save_ckpt({
                    "epoch": epoch,
                    "loss": epoch_loss,
                    "encoder_state_dict": jepa.encoder.state_dict(),
                    "jepa_state_dict": jepa.state_dict()
                }, best_ckpt_path)
        else:
            patience += 1
            if patience >= cfg["early_stop_patience"]:
                print(f"Early stopping at epoch {epoch} (best={best:.4f}).")
                break

        if max_steps is not None and global_step >= max_steps:
            print(f"Reached max_steps={max_steps} (global_step={global_step}); stopping pretrain.")
            break

    # save pretrain loss history to results
    import json, datetime
    exp_name = cfg.get("experiment_name", "")
    model_slug = cfg.get("model_name", "").replace("-", "_").replace("/", "_")
    prefix = f"{exp_name}_{model_slug}_" if exp_name else f"{model_slug}_" if model_slug else ""
    results_dir = cfg.get("results_dir", "results")
    os.makedirs(results_dir, exist_ok=True)
    task = cfg.get("dataset_config", cfg.get("task", "pretrain"))
    fname = f"{exp_name}_{model_slug}_{task}.json" if exp_name else f"pretrain_{model_slug}_{task}.json"
    result = {
        "model": cfg.get("model_name", ""),
        "task": task,
        "experiment": exp_name,
        "epochs": loss_history,
        "timestamp": datetime.datetime.now().isoformat(),
    }
    with open(os.path.join(results_dir, fname), "w") as f:
        json.dump(result, f, indent=2)
    print(f"Saved pretrain results to {os.path.join(results_dir, fname)}")

    if use_lora:
        return os.path.join(cfg["ckpt_dir"], f"best_{prefix}adapter")
    return os.path.join(cfg["ckpt_dir"], f"best_{prefix}.pt")

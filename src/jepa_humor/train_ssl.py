import os
from tqdm import tqdm
import torch
from torch.cuda.amp import autocast, GradScaler
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
    sample_dataset,
    load_anli_splits,
    load_snli_mnli_for_anli,
    load_composite_nli_dataset,
    load_swag,
    expand_swag_for_pretraining,
)

def build_optimizer_and_scheduler(jepa, lr_pred=1e-4, lr_enc=5e-5, weight_decay=0.01, total_steps=1000, warmup_ratio=0.06, model_type="jepa"):
    # param groups: predictor/vocab_projection (higher LR), encoder (lower LR), target is frozen
    if model_type == "mlm":
        # MLMJEPA: has vocab_projection instead of predictor
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

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
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

    # Choose model type: "jepa" (embedding reconstruction) or "mlm" (token ID prediction)
    model_type = cfg.get("model_type", "jepa").lower()
    
    if model_type == "mlm":
        # BERT-style MLM: single encoder + vocab projection (no EMA target encoder)
        jepa = MLMJEPA(
            base, 
            vocab_size=tokenizer.vocab_size,
            hidden_size=base.config.hidden_size,
            use_cls=True
        ).to(dev)
        print("Using BERT-style MLM (single encoder, no EMA target encoder)")
    else:
        # Original JEPA that reconstructs embeddings
        embedding_matrix = base.get_input_embeddings().weight.detach().cpu().numpy()
        jepa = JEPA(
            base, 
            hidden_size=base.config.hidden_size,
            pred_hid=512, 
            ema_decay=cfg["ema_decay"], 
            use_cls=True
        ).to(dev)
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
    else:
        ds = load_hf_splits(cfg["dataset"], config=dataset_config)
    
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
    else:
        split['test'] = ds['validation']
    
    # Get text keys from config (defaults to ["question", "sentence"])
    if is_anli or dataset_config == "snli_mnli_anli":
        text_keys = cfg.get("text_keys", ["premise", "hypothesis"])  # ANLI uses premise/hypothesis like MNLI
    elif dataset_config == 'mnli':
        text_keys = cfg.get("text_keys", ["premise", "hypothesis"])
    elif is_swag or dataset_config == "swag":
        text_keys = cfg.get("text_keys", ["question", "sentence"])
    else:
        text_keys = cfg.get("text_keys", ["question", "sentence"])
    print(f"Using text keys: {text_keys}")
    
    # Pre-tokenize datasets once before training
    mask_prob = cfg.get("mask_ratio", 0.15)
    span_len = cfg.get("mean_span_len", 3.0)
    mask_type = cfg.get("mask_type", "random_span")

    print("Pre-tokenizing training dataset for SSL...")
    split['train'] = preprocess_dataset_ssl(
        split['train'],
        tokenizer,
        max_length=cfg["max_length"],
        text_keys=text_keys,
        mask_probability=mask_prob,
        span_mean=span_len,
        mask_type=mask_type,
    )
    print("Pre-tokenizing validation dataset for SSL...")
    split['test'] = preprocess_dataset_ssl(
        split['test'],
        tokenizer,
        max_length=cfg["max_length"],
        text_keys=text_keys,
        mask_probability=mask_prob,
        span_mean=span_len,
        mask_type=mask_type,
        
    )
    
    #split = ds['train'].train_test_split(test_size=cfg["val_split"], seed=cfg["seed"])
    train_loader, val_loader = make_dataloaders(split["train"], split["test"], cfg["batch_size"], cfg["num_workers"], pre_tokenized=True, ssl_mode=True)
    print(f"Train dataset size: {len(split['train'])}")
    print(f"Test dataset size: {len(split['test'])}")
    opt, scheduler = build_optimizer_and_scheduler(jepa, lr_pred=1e-3, lr_enc=cfg["lr"], model_type=model_type)
    scaler = GradScaler(enabled=cfg.get("mixed_precision", True))

    best = float("inf")
    patience = 0

    for epoch in range(1, cfg["epochs"] + 1):
        running, steps = 0.0, 0
        pbar = tqdm(train_loader, desc=f"Pretrain {epoch}/{cfg['epochs']}", leave=False)
        jepa.ema_decay = jepa.get_ema_decay(epoch, cfg["epochs"])
        for batch in pbar:
            # Data is already tokenized
            keep_tok = {
                "input_ids": batch["keep_input_ids"].to(dev, non_blocking=True),
                "attention_mask": batch["keep_attention_mask"].to(dev, non_blocking=True)
            }
            target_tok = {
                "input_ids": batch["target_input_ids"].to(dev, non_blocking=True),
                "attention_mask": batch["target_attention_mask"].to(dev, non_blocking=True)
            }
            
            if model_type == "mlm":
                # For MLM: 
                # - Context is keep_tok (original unmasked text)
                # - Target is keep_input_ids (original tokens to predict)
                # - Mask indicates positions where target_input_ids has MASK tokens
                mask_token_id = tokenizer.mask_token_id
                mask = (target_tok["input_ids"] == mask_token_id).float()
                # Also respect attention_mask (exclude padding)
                mask = mask * target_tok["attention_mask"].float()
                
                # Create target dict with original tokens
                target_for_mlm = {
                    "input_ids": keep_tok["input_ids"],  # Original tokens to predict
                    "attention_mask": keep_tok["attention_mask"]
                }
            else:
                # Standard JEPA: use attention_mask
                mask = target_tok["attention_mask"].float().to(dev, non_blocking=True)
                target_for_mlm = None  # Not used for standard JEPA

            opt.zero_grad(set_to_none=True)
            with amp.autocast("cuda", enabled=cfg.get("mixed_precision", True)):
                if model_type == "mlm":
                    # MLMJEPA: predict original token IDs at masked positions
                    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else -100
                    loss, pred, target = jepa(keep_tok, target_for_mlm, mask, ignore_index=pad_token_id)
                else:
                    # Standard JEPA: reconstruct embeddings
                    loss, pred, target = jepa(keep_tok, target_tok, mask)
            
          

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            scheduler.step()
            # Only update target encoder for standard JEPA (not for BERT-style MLM)
            if model_type != "mlm":
                jepa.update_target()

            running += float(loss.item()); steps += 1
            pbar.set_postfix(loss=f"{running/steps:.4f}")

        epoch_loss = running / max(1, steps)
        print(f"[Pretrain] epoch {epoch}: loss={epoch_loss:.4f}")

        # checkpoint & early stopping on train loss (SSL)
        os.makedirs(cfg["ckpt_dir"], exist_ok=True)
        exp_name = cfg.get("experiment_name", "")
        prefix = f"{exp_name}_" if exp_name else ""
        
        path = os.path.join(cfg["ckpt_dir"], f"{prefix}jepa_epoch{epoch}_loss{epoch_loss:.4f}.pt")
        # save_ckpt({
        #     "epoch": epoch,
        #     "loss": epoch_loss,
        #     "encoder_state_dict": jepa.encoder.state_dict(),
        #     "jepa_state_dict": jepa.state_dict()
        # }, path)

        if epoch_loss + 1e-6 < best:
            best = epoch_loss
            patience = 0
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

    # return path to best encoder
    exp_name = cfg.get("experiment_name", "")
    prefix = f"{exp_name}_" if exp_name else ""
    return os.path.join(cfg["ckpt_dir"], f"best_{prefix}.pt")

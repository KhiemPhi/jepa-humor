import os
from tqdm import tqdm
import torch
from torch.cuda.amp import autocast, GradScaler
from transformers import AutoTokenizer, AutoModel
from .utils import set_seed, device, save_ckpt
from .jepa import JEPA
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import os
from torch import amp
import torch.nn.functional as F
import math

def build_optimizer_and_scheduler(jepa, lr_pred=1e-3, lr_enc=5e-5, weight_decay=0.01, total_steps=1000, warmup_ratio=0.06):
    # param groups: predictor (higher LR), encoder (lower LR), target is frozen
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

    
    embedding_matrix = base.get_input_embeddings().weight.detach().cpu().numpy()

    jepa = JEPA(base, hidden_size=base.config.hidden_size,
                pred_hid=512, ema_decay=cfg["ema_decay"], use_cls=True).to(dev)
    jepa.train()

    from .data import load_hf_splits, make_dataloaders, preprocess_dataset_ssl, sample_dataset, load_anli_splits
    dataset_config = cfg.get("dataset_config", None)
    is_anli = dataset_config == "anli" or (dataset_config is None and "anli" in cfg["dataset"].lower())
    
    if is_anli:
        # ANLI: aggregate training sets from specified rounds
        train_rounds = cfg.get("train_rounds", [1, 2, 3])  # Default: use all rounds
        if isinstance(train_rounds, int):
            train_rounds = [train_rounds]  # Convert single int to list
        ds = load_anli_splits(cfg["dataset"], train_rounds=train_rounds, seed=cfg["seed"])
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
    else:
        split['test'] = ds['validation']
    
    # Get text keys from config (defaults to ["question", "sentence"])
    if is_anli:
        text_keys = cfg.get("text_keys", ["premise", "hypothesis"])  # ANLI uses premise/hypothesis like MNLI
    elif dataset_config == 'mnli':
        text_keys = cfg.get("text_keys", ["premise", "hypothesis"])
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
    opt, scheduler = build_optimizer_and_scheduler(jepa, lr_pred=1e-3, lr_enc=cfg["lr"])
    scaler = GradScaler(enabled=cfg.get("mixed_precision", True))

    best = float("inf")
    patience = 0

    for epoch in range(1, cfg["epochs"] + 1):
        running, steps = 0.0, 0
        pbar = tqdm(train_loader, desc=f"Pretrain {epoch}/{cfg['epochs']}", leave=False)
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
            # Create mask from target attention_mask (same as make_second_sent_mask_views does)
            mask = target_tok["attention_mask"].float().to(dev, non_blocking=True)
            # # Decode a single tensor (e.g. from keep_tok)
            # decoded_keep = tokenizer.batch_decode(
            #     keep_tok["input_ids"], 
            #     skip_special_tokens=True
            # )

            # # Decode from target_tok as well
            # decoded_target = tokenizer.batch_decode(
            #     target_tok["input_ids"], 
            #     skip_special_tokens=True
            # )
         

            opt.zero_grad(set_to_none=True)
            with amp.autocast("cuda", enabled=cfg.get("mixed_precision", True)):
                loss, pred, target = jepa(keep_tok, target_tok, mask)
            
          

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            scheduler.step()
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
        save_ckpt({
            "epoch": epoch,
            "loss": epoch_loss,
            "encoder_state_dict": jepa.encoder.state_dict(),
            "jepa_state_dict": jepa.state_dict()
        }, path)

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

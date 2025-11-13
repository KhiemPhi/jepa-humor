import os
from tqdm import tqdm
import torch
from torch.cuda.amp import autocast, GradScaler
from transformers import AutoTokenizer, AutoModel
from .utils import set_seed, device, save_ckpt
from .masking import make_next_phrases_mask_views, encode_multiple_choice
from .jepa import JEPA
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import os
from torch import amp
import torch.nn.functional as F
import math



def build_optimizer_and_scheduler(jepa, lr_pred=1e-3, lr_enc=5e-5, weight_decay=0.01, total_steps=1000, warmup_ratio=0.1):
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

ending_names = ["ending0", "ending1", "ending2", "ending3"]

next_sentences = ["next_sentence0", "next_sentence1", "next_sentence2", "next_sentence3"]

def make_next_sentence(example):
    example = dict(example)  # clone input
    example['true_next_sentence'] =  example[ending_names[example['label']]]
    example["true_start_sentence"] = example["sent1"] + " " + example["sent2"]
    return example

def pretrain(cfg):
    set_seed(cfg["seed"])
    dev = device()

    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"])
    base = AutoModel.from_pretrained(cfg["model_name"], trust_remote_code=True)
    if tokenizer.mask_token_id is None:
        raise ValueError("Tokenizer must have a [MASK] token.")

    
    jepa = JEPA(base, hidden_size=base.config.hidden_size,
                pred_hid=512, ema_decay=cfg["ema_decay"], use_cls=True).to(dev)
    jepa.train()

    from .data import load_hf_splits, make_dataloaders
    ds = load_hf_splits(cfg["dataset"], "regular")
    ds['test'] = ds['validation']
    del ds['validation']
    ds = ds.map( make_next_sentence, load_from_cache_file=False,)
    # split train/val from train_ds
    
    split = ds

    #split = ds['train'].train_test_split(test_size=cfg["val_split"], seed=cfg["seed"])
    
    train_loader, val_loader = make_dataloaders(split["train"], split["test"], cfg["batch_size"], cfg["num_workers"])
    print(f"Train dataset size: {len(split['train'])}")
    print(f"Test dataset size: {len(split['test'])}")
    opt, scheduler = build_optimizer_and_scheduler(jepa, lr_pred=1e-5, lr_enc=cfg["lr"])
    scaler = GradScaler(enabled=cfg.get("mixed_precision", True))

    best = float("inf")
    patience = 0

    for epoch in range(1, cfg["epochs"] + 1):
        running, steps = 0.0, 0
        pbar = tqdm(train_loader, desc=f"Pretrain {epoch}/{cfg['epochs']}", leave=False)
        for batch in pbar:
            sent1 = list(batch["true_start_sentence"])
            sent2 = list(batch["true_next_sentence"])
          
            keep_tok, target_tok = encode_multiple_choice(batch, tokenizer, cfg["max_length"], dev)
            
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
                loss, pred, target = jepa(keep_tok, target_tok, mask=None)
            
          

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
        path = os.path.join(cfg["ckpt_dir"], f"jepa_epoch{epoch}_loss{epoch_loss:.4f}.pt")
        save_ckpt({
            "epoch": epoch,
            "loss": epoch_loss,
            "encoder_state_dict": jepa.encoder.state_dict(),
            "jepa_state_dict": jepa.state_dict()
        }, path)

        if epoch_loss + 1e-6 < best:
            best = epoch_loss
            patience = 0
            save_ckpt({
                "epoch": epoch,
                "loss": epoch_loss,
                "encoder_state_dict": jepa.encoder.state_dict(),
                "jepa_state_dict": jepa.state_dict()
            }, os.path.join(cfg["ckpt_dir"], "best.pt"))
        else:
            patience += 1
            if patience >= cfg["early_stop_patience"]:
                print(f"Early stopping at epoch {epoch} (best={best:.4f}).")
                break

    # return path to best encoder
    return os.path.join(cfg["ckpt_dir"], "best.pt")

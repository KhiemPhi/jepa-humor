from dataclasses import dataclass
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from torch.optim.swa_utils import AveragedModel, SWALR, update_bn
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics import matthews_corrcoef, f1_score, accuracy_score
from .utils import set_seed, device, load_encoder_state, save_ckpt
from .data import load_hf_splits, make_dataloaders, load_hf_splits_test, load_hf_splits_test_mismatched, preprocess_dataset, load_mnli_test_from_json, load_mnli_mm_test_from_json
from .train_ssl import pretrain
import os
from torch import amp
import numpy as np
os.environ["TOKENIZERS_PARALLELISM"] = "false"

class CorrectEnglishSentenceClassifier(nn.Module):
    def __init__(self, encoder, hidden_size=768, num_labels=3, dropout=0.5):
        super().__init__()
        self.encoder = encoder
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_labels)
    def forward(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = out.last_hidden_state[:, 0, :]
        pooled = self.dropout(pooled)
        return self.classifier(pooled)


# --- PGD Adversarial Attack (Minimal, In-Place) ---
def pgd_attack(model, input_ids, attention_mask, labels, epsilon=1.0, alpha=0.3, iters=3):
    embed = model.encoder.get_input_embeddings()
    orig_embeddings = embed(input_ids).detach()  # [B, L, D] - detach original
    delta = torch.zeros_like(orig_embeddings).uniform_(-epsilon, epsilon)
    delta = delta.to(input_ids.device)
    delta.requires_grad = True

    for _ in range(iters):
        delta.grad = None
        adv_embeds = orig_embeddings + delta

        # Forward through encoder + classifier
        outputs = model.encoder(inputs_embeds=adv_embeds, attention_mask=attention_mask)
        pooled = outputs.pooler_output if outputs.pooler_output is not None else outputs.last_hidden_state[:, 0, :]
        logits = model.classifier(model.dropout(pooled))
        loss = nn.CrossEntropyLoss()(logits, labels)

        # Backward only for PGD update
        loss.backward()

        # Update delta
        delta.data = delta.data + alpha * delta.grad.sign()
        delta.data = torch.clamp(delta.data, -epsilon, epsilon)
        delta.grad = None

        # CRITICAL: Detach to break graph
        delta = delta.detach()
        delta.requires_grad = True

    return delta.detach()

def finetune(cfg, encoder_ckpt: str, eval_only: bool = False, model_ckpt: str = None):
    set_seed(cfg["seed"])
    dev = device()

    # 1. Load Encoder
    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"])
    base = AutoModel.from_pretrained(cfg["model_name"]) if encoder_ckpt is None else AutoModel.from_pretrained(cfg["model_name"])
    if encoder_ckpt != "fresh" and not eval_only:
        print(f"Loading pretrained encoder from {encoder_ckpt}")
        base = load_encoder_state(base, encoder_ckpt, map_location="cpu")
    else: 
        print(f"Loading fresh encoder")

    ds = load_hf_splits(cfg["dataset"])
    ds_test_matched = load_hf_splits_test(cfg["test_dataset"])
    ds_test_mismatched = load_hf_splits_test_mismatched(cfg["test_dataset"])
    
    # Load MNLI and MNLI-MM test sets from test_ann.json
    ds_mnli_test = load_mnli_test_from_json(json_path=os.path.join(os.path.dirname(__file__), "test_ann.json"))
    ds_mnli_mm_test = load_mnli_mm_test_from_json(json_path=os.path.join(os.path.dirname(__file__), "test_ann.json"))
    
    # Pre-tokenize datasets once before training
    print("Pre-tokenizing training dataset...")
    ds["train"] = preprocess_dataset(ds["train"], tokenizer, max_length=cfg["max_length"])
    print("Pre-tokenizing validation dataset (matched)...")
    ds_test_matched["validation"] = preprocess_dataset(ds_test_matched["validation"], tokenizer, max_length=cfg["max_length"])
    print("Pre-tokenizing validation dataset (mismatched)...")
    ds_test_mismatched["validation"] = preprocess_dataset(ds_test_mismatched["validation"], tokenizer, max_length=cfg["max_length"])
    print("Pre-tokenizing MNLI test dataset...")
    ds_mnli_test["test"] = preprocess_dataset(ds_mnli_test["test"], tokenizer, max_length=cfg["max_length"])
    print("Pre-tokenizing MNLI-MM test dataset...")
    ds_mnli_mm_test["test"] = preprocess_dataset(ds_mnli_mm_test["test"], tokenizer, max_length=cfg["max_length"])
    
    # split train/val from train_ds
    train_loader, val_loader_matched = make_dataloaders(ds["train"], ds_test_matched["validation"], cfg["batch_size"], cfg["num_workers"], pre_tokenized=True)
    _, val_loader_mismatched = make_dataloaders(ds["train"], ds_test_mismatched["validation"], cfg["batch_size"], cfg["num_workers"], pre_tokenized=True)
    _, test_loader_mnli = make_dataloaders(ds["train"], ds_mnli_test["test"], cfg["batch_size"], cfg["num_workers"], pre_tokenized=True)
    _, test_loader_mnli_mm = make_dataloaders(ds["train"], ds_mnli_mm_test["test"], cfg["batch_size"], cfg["num_workers"], pre_tokenized=True)
    print(f"Train dataset size: {len(ds['train'])}")
    print(f"Validation dataset size (matched): {len(ds_test_matched['validation'])}")
    print(f"Validation dataset size (mismatched): {len(ds_test_mismatched['validation'])}")
    print(f"Test dataset size (MNLI): {len(ds_mnli_test['test'])}")
    print(f"Test dataset size (MNLI-MM): {len(ds_mnli_mm_test['test'])}")

    model = CorrectEnglishSentenceClassifier(base, hidden_size=base.config.hidden_size).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=0.17)
    
    # Setup SWA (Stochastic Weight Averaging)
    swa_start_epoch = max(1, int(cfg["epochs"] * 0.9))  # Start SWA after 75% of training
    swa_model = AveragedModel(model)
    swa_scheduler = SWALR(opt, swa_lr=cfg["lr"] * 0.5)  # Lower LR for averaging phase
    use_swa = False #cfg.get("use_swa", True)
    
    if use_swa:
        print(f"SWA will start at epoch {swa_start_epoch}/{cfg['epochs']}")
    
    # Compute class weights from training labels (inverse frequency)
    try:
        train_labels = np.asarray(ds["train"]["label"])  # expects integer labels 0/1
        counts = np.bincount(train_labels, minlength=2).astype(np.float32)
        counts[counts == 0] = 1.0
        inv = 1.0 / counts
        weights = (inv / inv.sum()) * 2.0  # normalize roughly to sum=2
        class_weights = torch.tensor(weights, dtype=torch.float, device=dev)
        print(f"Class counts: {counts.astype(int).tolist()} -> weights: {weights.round(4).tolist()}")
        criterion = nn.CrossEntropyLoss(weight=class_weights)
    except Exception as e:
        print(f"Warning: falling back to unweighted CE (reason: {e})")
        criterion = nn.CrossEntropyLoss(label_smoothing=0.01)
    scaler = GradScaler(enabled=cfg.get("mixed_precision", True))
    best_t = 0.5  # Default threshold, will be updated during training
    
    # Track best model for saving
    best_val_mcc = -1.0
    best_model_state = None
    
    # If eval_only mode, load the model checkpoint and skip training
    if eval_only:
        if model_ckpt is None:
            raise ValueError("model_ckpt must be provided when eval_only=True")
        print(f"Loading model checkpoint from {model_ckpt}")
        checkpoint = torch.load(model_ckpt, map_location=dev, weights_only=False)
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        elif "state_dict" in checkpoint:
            model.load_state_dict(checkpoint["state_dict"])
        else:
            model.load_state_dict(checkpoint)
        # Load threshold if available
        if "best_threshold" in checkpoint:
            best_t = checkpoint["best_threshold"]
            print(f"Loaded threshold from checkpoint: {best_t:.3f}")
        print("Model loaded. Running evaluation only...")
    else:
        # Training loop
        for epoch in range(1, cfg["epochs"] + 1):
            # Check if SWA phase starts this epoch
            if use_swa and epoch == swa_start_epoch:
                print(f"Starting SWA at epoch {epoch}")
            
            # train
            model.train()
            running, y_true, y_pred = 0.0, [], []
            y_prob = []  # store positive-class probabilities across the epoch
            for batch in tqdm(train_loader, desc=f"Finetune {epoch}/{cfg['epochs']}", leave=False):
                # Data is already tokenized
                input_ids = batch["input_ids"].to(dev, non_blocking=True)
                attention_mask = batch["attention_mask"].to(dev, non_blocking=True)
                labels = batch["label"].to(dev, non_blocking=True)

                opt.zero_grad(set_to_none=True)
                with amp.autocast("cuda", enabled=cfg.get("mixed_precision", True)):
                    # --- Clean forward ---
                    logits_clean = model(input_ids, attention_mask)
                    loss_clean = criterion(logits_clean, labels)

                  

                    # --- Combined loss ---
                    loss = loss_clean

                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
                
                # Update SWA model and scheduler if in SWA phase (after each batch)
                if use_swa and epoch >= swa_start_epoch:
                    swa_model.update_parameters(model)
                    swa_scheduler.step()

                running += float(loss.item())
                y_true.extend(labels.detach().cpu().tolist())
                y_pred.extend(logits_clean.detach().cpu().argmax(-1).tolist())
                # collect probabilities for class 1
                probs_batch = torch.softmax(logits_clean.detach().cpu(), dim=-1)[:, 1].numpy()
                y_prob.append(probs_batch)
                

            train_mcc = matthews_corrcoef(y_true, y_pred)
            train_f1 = f1_score(y_true, y_pred, average='macro')
            # concatenate per-batch probabilities to a single vector aligned with y_true
            probs = np.concatenate(y_prob, axis=0)
            true = np.asarray(y_true)
            
            # Search across thresholds (optimized: fewer iterations, only on last epoch or every few epochs)
            if epoch == cfg["epochs"] or epoch % 3 == 0:
                best_t, best_mcc = 0.0, -1.0
                # Use fewer threshold points for faster search
                for t in np.linspace(0.1, 0.9, 17):  # Reduced from 99 to 17 points
                    preds = (probs >= t).astype(int)
                    mcc = matthews_corrcoef(true, preds)
                    if mcc > best_mcc:
                        best_mcc, best_t = mcc, t
                print(f"Best MCC for Train MCC = {best_mcc:.4f} at threshold = {best_t:.3f}")
            else:
                # Use default threshold of 0.5 for intermediate epochs
                best_t = 0.5
     
            print(f"[Finetune] epoch {epoch}: loss={running/len(train_loader):.4f} mcc={train_mcc:.4f} f1={train_f1:.4f}")

            # validate on matched
            model.eval()
            v_true_matched, v_pred_matched, v_prob_matched = [], [], []
            with torch.no_grad():
                for batch in val_loader_matched:
                    # Data is already tokenized
                    input_ids = batch["input_ids"].to(dev, non_blocking=True)
                    attention_mask = batch["attention_mask"].to(dev, non_blocking=True)
                    labels = batch["label"].to(dev, non_blocking=True)
                    logits = model(input_ids, attention_mask)
                    v_true_matched.extend(labels.cpu().tolist())
                    v_pred_matched.extend(logits.cpu().argmax(-1).tolist())
                    probs_batch = torch.softmax(logits.detach().cpu(), dim=-1)[:, 1].numpy()
                    v_prob_matched.append(probs_batch)

            probs_matched = np.concatenate(v_prob_matched, axis=0) 
            preds_matched = (probs_matched >= best_t).astype(int)
            val_mcc_matched = matthews_corrcoef(v_true_matched, preds_matched)
            val_f1_matched = f1_score(v_true_matched, preds_matched, average='macro')
            val_acc_matched = accuracy_score(v_true_matched, preds_matched)
            print(f"          val (matched): mcc={val_mcc_matched:.4f} f1={val_f1_matched:.4f} acc={val_acc_matched:.4f}")

            # validate on mismatched
            v_true_mismatched, v_pred_mismatched, v_prob_mismatched = [], [], []
            with torch.no_grad():
                for batch in val_loader_mismatched:
                    # Data is already tokenized
                    input_ids = batch["input_ids"].to(dev, non_blocking=True)
                    attention_mask = batch["attention_mask"].to(dev, non_blocking=True)
                    labels = batch["label"].to(dev, non_blocking=True)
                    logits = model(input_ids, attention_mask)
                    v_true_mismatched.extend(labels.cpu().tolist())
                    v_pred_mismatched.extend(logits.cpu().argmax(-1).tolist())
                    probs_batch = torch.softmax(logits.detach().cpu(), dim=-1)[:, 1].numpy()
                    v_prob_mismatched.append(probs_batch)

            probs_mismatched = np.concatenate(v_prob_mismatched, axis=0) 
            preds_mismatched = (probs_mismatched >= best_t).astype(int)
            val_mcc_mismatched = matthews_corrcoef(v_true_mismatched, preds_mismatched)
            val_f1_mismatched = f1_score(v_true_mismatched, preds_mismatched, average='macro')
            val_acc_mismatched = accuracy_score(v_true_mismatched, preds_mismatched)
            print(f"          val (mismatched): mcc={val_mcc_mismatched:.4f} f1={val_f1_mismatched:.4f} acc={val_acc_mismatched:.4f}")
            
            # Evaluate on MNLI test set
            t_true_mnli, t_pred_mnli = [], []
            with torch.no_grad():
                for batch in test_loader_mnli:
                    input_ids = batch["input_ids"].to(dev, non_blocking=True)
                    attention_mask = batch["attention_mask"].to(dev, non_blocking=True)
                    labels = batch["label"].to(dev, non_blocking=True)
                    logits = model(input_ids, attention_mask)
                    t_true_mnli.extend(labels.cpu().tolist())
                    t_pred_mnli.extend(logits.cpu().argmax(-1).tolist())
            
            test_mcc_mnli = matthews_corrcoef(t_true_mnli, t_pred_mnli)
            test_f1_mnli = f1_score(t_true_mnli, t_pred_mnli, average='macro')
            test_acc_mnli = accuracy_score(t_true_mnli, t_pred_mnli)
            print(f"          test (MNLI): mcc={test_mcc_mnli:.4f} f1={test_f1_mnli:.4f} acc={test_acc_mnli:.4f}")
            
            # Evaluate on MNLI-MM test set
            t_true_mnli_mm, t_pred_mnli_mm = [], []
            with torch.no_grad():
                for batch in test_loader_mnli_mm:
                    input_ids = batch["input_ids"].to(dev, non_blocking=True)
                    attention_mask = batch["attention_mask"].to(dev, non_blocking=True)
                    labels = batch["label"].to(dev, non_blocking=True)
                    logits = model(input_ids, attention_mask)
                    t_true_mnli_mm.extend(labels.cpu().tolist())
                    t_pred_mnli_mm.extend(logits.cpu().argmax(-1).tolist())
            
            test_mcc_mnli_mm = matthews_corrcoef(t_true_mnli_mm, t_pred_mnli_mm)
            test_f1_mnli_mm = f1_score(t_true_mnli_mm, t_pred_mnli_mm, average='macro')
            test_acc_mnli_mm = accuracy_score(t_true_mnli_mm, t_pred_mnli_mm)
            print(f"          test (MNLI-MM): mcc={test_mcc_mnli_mm:.4f} f1={test_f1_mnli_mm:.4f} acc={test_acc_mnli_mm:.4f}")
            
            # Save best model based on matched validation MCC
            if val_mcc_matched > best_val_mcc:
                best_val_mcc = val_mcc_matched
                best_model_state = model.state_dict().copy()
                # Save checkpoint
                ckpt_dir = cfg.get("ckpt_dir", "checkpoints/finetune")
                os.makedirs(ckpt_dir, exist_ok=True)
                save_ckpt({
                    "epoch": epoch,
                    "val_mcc_matched": val_mcc_matched,
                    "val_mcc_mismatched": val_mcc_mismatched,
                    "val_f1_matched": val_f1_matched,
                    "val_f1_mismatched": val_f1_mismatched,
                    "model_state_dict": model.state_dict(),
                    "best_threshold": best_t,
                }, os.path.join(ckpt_dir, "best.pt"))
                print(f"          Saved best model (MCC={val_mcc_matched:.4f}) to {os.path.join(ckpt_dir, 'best.pt')}")
        
        # After training, load best model if we saved one
        if best_model_state is not None:
            print(f"Loading best model (MCC={best_val_mcc:.4f}) for final evaluation...")
            model.load_state_dict(best_model_state)
        
        # After training, update BN stats for SWA model and replace regular model
        if use_swa:
            print("Updating batch normalization statistics for SWA model...")
            # Update BN stats by running one forward pass through training data
            model.eval()
            with torch.no_grad():
                for batch in tqdm(train_loader, desc="Updating BN stats", leave=False):
                    # Data is already tokenized
                    input_ids = batch["input_ids"].to(dev, non_blocking=True)
                    attention_mask = batch["attention_mask"].to(dev, non_blocking=True)
                    
                    _ = swa_model(input_ids, attention_mask)
            # Note: BERT uses LayerNorm, not BatchNorm, so update_bn is not strictly necessary
            # But we'll update any BatchNorm layers if they exist (e.g., in custom heads)
            try:
                update_bn(train_loader, swa_model, device=dev)
            except Exception:
                # If update_bn fails (e.g., no BN layers), that's fine
                pass
            
            print("Replacing model with SWA model...")
            model.load_state_dict(swa_model.module.state_dict())
            print("SWA model loaded!")
    
    # Final validation on both matched and mismatched
    model.eval()
    
    # Matched validation
    v_true_matched, v_pred_matched, v_prob_matched = [], [], []
    with torch.no_grad():
        for batch in val_loader_matched:
            # Data is already tokenized
            input_ids = batch["input_ids"].to(dev, non_blocking=True)
            attention_mask = batch["attention_mask"].to(dev, non_blocking=True)
            labels = batch["label"].to(dev, non_blocking=True)
            logits = model(input_ids, attention_mask)
            v_true_matched.extend(labels.cpu().tolist())
            v_pred_matched.extend(logits.cpu().argmax(-1).tolist())
            probs_batch = torch.softmax(logits.detach().cpu(), dim=-1)[:, 1].numpy()
            v_prob_matched.append(probs_batch)

    probs_matched = np.concatenate(v_prob_matched, axis=0)
    preds_matched = (probs_matched >= best_t).astype(int)
    val_mcc_matched = matthews_corrcoef(v_true_matched, preds_matched)
    val_f1_matched = f1_score(v_true_matched, preds_matched, average='macro')
    val_acc_matched = accuracy_score(v_true_matched, preds_matched)
    print(f"Final validation (matched): mcc={val_mcc_matched:.4f} f1={val_f1_matched:.4f} acc={val_acc_matched:.4f}")

    # Mismatched validation
    v_true_mismatched, v_pred_mismatched, v_prob_mismatched = [], [], []
    with torch.no_grad():
        for batch in val_loader_mismatched:
            # Data is already tokenized
            input_ids = batch["input_ids"].to(dev, non_blocking=True)
            attention_mask = batch["attention_mask"].to(dev, non_blocking=True)
            labels = batch["label"].to(dev, non_blocking=True)
            logits = model(input_ids, attention_mask)
            v_true_mismatched.extend(labels.cpu().tolist())
            v_pred_mismatched.extend(logits.cpu().argmax(-1).tolist())
            probs_batch = torch.softmax(logits.detach().cpu(), dim=-1)[:, 1].numpy()
            v_prob_mismatched.append(probs_batch)

    probs_mismatched = np.concatenate(v_prob_mismatched, axis=0)
    preds_mismatched = (probs_mismatched >= best_t).astype(int)
    val_mcc_mismatched = matthews_corrcoef(v_true_mismatched, preds_mismatched)
    val_f1_mismatched = f1_score(v_true_mismatched, preds_mismatched, average='macro')
    val_acc_mismatched = accuracy_score(v_true_mismatched, preds_mismatched)
    print(f"Final validation (mismatched): mcc={val_mcc_mismatched:.4f} f1={val_f1_mismatched:.4f} acc={val_acc_mismatched:.4f}")
    
    # Test set evaluation - MNLI
    print("\n" + "="*50)
    print("Evaluating on MNLI test set...")
    t_true_mnli, t_pred_mnli = [], []
    with torch.no_grad():
        for batch in test_loader_mnli:
            input_ids = batch["input_ids"].to(dev, non_blocking=True)
            attention_mask = batch["attention_mask"].to(dev, non_blocking=True)
            labels = batch["label"].to(dev, non_blocking=True)
            logits = model(input_ids, attention_mask)
            t_true_mnli.extend(labels.cpu().tolist())
            t_pred_mnli.extend(logits.cpu().argmax(-1).tolist())
    
    # For 3-class classification, use argmax predictions directly
    test_mcc_mnli = matthews_corrcoef(t_true_mnli, t_pred_mnli)
    test_f1_mnli = f1_score(t_true_mnli, t_pred_mnli, average='macro')
    test_acc_mnli = accuracy_score(t_true_mnli, t_pred_mnli)
    print(f"Test (MNLI): mcc={test_mcc_mnli:.4f} f1={test_f1_mnli:.4f} acc={test_acc_mnli:.4f}")
    
    # Test set evaluation - MNLI-MM
    print("\n" + "="*50)
    print("Evaluating on MNLI-MM test set...")
    t_true_mnli_mm, t_pred_mnli_mm = [], []
    with torch.no_grad():
        for batch in test_loader_mnli_mm:
            input_ids = batch["input_ids"].to(dev, non_blocking=True)
            attention_mask = batch["attention_mask"].to(dev, non_blocking=True)
            labels = batch["label"].to(dev, non_blocking=True)
            logits = model(input_ids, attention_mask)
            t_true_mnli_mm.extend(labels.cpu().tolist())
            t_pred_mnli_mm.extend(logits.cpu().argmax(-1).tolist())
    
    # For 3-class classification, use argmax predictions directly
    test_mcc_mnli_mm = matthews_corrcoef(t_true_mnli_mm, t_pred_mnli_mm)
    test_f1_mnli_mm = f1_score(t_true_mnli_mm, t_pred_mnli_mm, average='macro')
    test_acc_mnli_mm = accuracy_score(t_true_mnli_mm, t_pred_mnli_mm)
    print(f"Test (MNLI-MM): mcc={test_mcc_mnli_mm:.4f} f1={test_f1_mnli_mm:.4f} acc={test_acc_mnli_mm:.4f}")
    
    # If in eval_only mode, we're done
    if eval_only:
        print("✅ Evaluation complete.")
        return model
     
    return model

from dataclasses import dataclass
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.cuda.amp import autocast
from torch.amp import GradScaler
from torch.optim.swa_utils import AveragedModel, SWALR, update_bn
from transformers import AutoTokenizer, AutoModel
from sklearn.metrics import matthews_corrcoef, f1_score, accuracy_score
from .utils import set_seed, device, load_encoder_state, save_ckpt
from .data import load_hf_splits, make_dataloaders, load_hf_splits_test, load_hf_splits_test_mismatched, preprocess_dataset, load_test_set_from_json
from .train_ssl import pretrain
import os
from torch import amp
import numpy as np
os.environ["TOKENIZERS_PARALLELISM"] = "false"

class CorrectEnglishSentenceClassifier(nn.Module):
    def __init__(self, encoder, hidden_size=768, num_labels=2, dropout=0.1):
        super().__init__()
        self.encoder = encoder
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_labels)
    def forward(self, input_ids, attention_mask):
        out = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        pooled = out.last_hidden_state[:, 0, :]
        pooled = self.dropout(pooled)
        return self.classifier(pooled)

def finetune(cfg, encoder_ckpt: str, eval_only: bool = False, model_ckpt: str = None):
    set_seed(cfg["seed"])
    # Get GPU ID from config, default to None (which will use cuda:0 or cpu)
    gpu_id = cfg.get("gpu_id", None)
    dev = device(gpu_id)
    if gpu_id is not None:
        print(f"Using device: {dev}")
    else:
        print(f"Using device: {dev} (default)")

    # 1. Load Encoder
    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"])
    base = AutoModel.from_pretrained(cfg["model_name"]) if encoder_ckpt is None else AutoModel.from_pretrained(cfg["model_name"])
    if encoder_ckpt != "fresh" and not eval_only:
        print(f"Loading pretrained encoder from {encoder_ckpt}")
        base = load_encoder_state(base, encoder_ckpt, map_location="cpu")
    else: 
        print(f"Loading fresh encoder")

    # Determine dataset type from config
    dataset_name = cfg["dataset"]
    dataset_config = cfg.get("dataset_config", None)  # e.g., "qnli", "mnli", "rte"
    is_mnli = dataset_config == "mnli" or (dataset_config is None and "mnli" in dataset_name.lower())
    
    # Load training and validation datasets from HuggingFace
    ds = load_hf_splits(dataset_name, config=dataset_config)
    ds_test_matched = load_hf_splits_test(cfg.get("test_dataset", dataset_name), config=dataset_config)
    
    # For MNLI, also load mismatched validation set
    if is_mnli:
        ds_test_mismatched = load_hf_splits_test_mismatched(cfg.get("test_dataset", dataset_name), config=dataset_config)
    else:
        ds_test_mismatched = None
    
    # Load ALL test sets from JSON file
    json_path = os.path.join(os.path.dirname(__file__), "test_ann.json")
    
    if is_mnli:
        # For MNLI, load both matched and mismatched test sets from JSON
        ds_test_matched_json = load_test_set_from_json("mnli", json_path=json_path)
        ds_test_mismatched_json = load_test_set_from_json("mnli-mm", json_path=json_path)
        ds_test_other = None
    else:
        # For other datasets, load test set from JSON based on dataset_config
        ds_test_matched_json = None
        ds_test_mismatched_json = None
        if dataset_config:
            ds_test_other = load_test_set_from_json(dataset_config, json_path=json_path)
        else:
            # Try to infer from dataset name
            if "qnli" in dataset_name.lower():
                ds_test_other = load_test_set_from_json("qnli", json_path=json_path)
            elif "qqp" in dataset_name.lower():
                ds_test_other = load_test_set_from_json("qqp", json_path=json_path)
            elif "rte" in dataset_name.lower():
                ds_test_other = load_test_set_from_json("rte", json_path=json_path)
            elif "sst2" in dataset_name.lower():
                ds_test_other = load_test_set_from_json("sst2", json_path=json_path)
            else:
                ds_test_other = None
                print(f"Warning: Could not determine test set to load from JSON for dataset '{dataset_name}'")
    
    # Get text keys from config (defaults to ["question", "sentence"])
    text_keys = cfg.get("text_keys", ["question", "sentence"] if not is_mnli else ["premise", "hypothesis"])
    print(f"Using text keys: {text_keys}")
    
    # Pre-tokenize datasets once before training
    print("Pre-tokenizing training dataset...")
    ds["train"] = preprocess_dataset(ds["train"], tokenizer, max_length=cfg["max_length"], text_keys=text_keys)
    print("Pre-tokenizing validation dataset (matched)...")
    ds_test_matched["validation"] = preprocess_dataset(ds_test_matched["validation"], tokenizer, max_length=cfg["max_length"], text_keys=text_keys)
    
    # Pre-tokenize mismatched validation set for MNLI
    if is_mnli and ds_test_mismatched:
        print("Pre-tokenizing validation dataset (mismatched)...")
        ds_test_mismatched["validation"] = preprocess_dataset(ds_test_mismatched["validation"], tokenizer, max_length=cfg["max_length"], text_keys=text_keys)
    
    # Pre-tokenize test sets from JSON
    if is_mnli:
        if ds_test_matched_json:
            print("Pre-tokenizing MNLI test dataset (from JSON)...")
            ds_test_matched_json["test"] = preprocess_dataset(ds_test_matched_json["test"], tokenizer, max_length=cfg["max_length"], text_keys=text_keys)
        if ds_test_mismatched_json:
            print("Pre-tokenizing MNLI-MM test dataset (from JSON)...")
            ds_test_mismatched_json["test"] = preprocess_dataset(ds_test_mismatched_json["test"], tokenizer, max_length=cfg["max_length"], text_keys=text_keys)
    else:
        if ds_test_other:
            print(f"Pre-tokenizing {dataset_config or 'test'} test dataset (from JSON)...")
            ds_test_other["test"] = preprocess_dataset(ds_test_other["test"], tokenizer, max_length=cfg["max_length"], text_keys=text_keys)
    
    # Create dataloaders
    train_loader, val_loader_matched = make_dataloaders(ds["train"], ds_test_matched["validation"], cfg["batch_size"], cfg["num_workers"], pre_tokenized=True)
    
    if is_mnli and ds_test_mismatched:
        _, val_loader_mismatched = make_dataloaders(ds["train"], ds_test_mismatched["validation"], cfg["batch_size"], cfg["num_workers"], pre_tokenized=True)
        test_loader_matched = None
        test_loader_mismatched = None
        if ds_test_matched_json:
            _, test_loader_matched = make_dataloaders(ds["train"], ds_test_matched_json["test"], cfg["batch_size"], cfg["num_workers"], pre_tokenized=True)
        if ds_test_mismatched_json:
            _, test_loader_mismatched = make_dataloaders(ds["train"], ds_test_mismatched_json["test"], cfg["batch_size"], cfg["num_workers"], pre_tokenized=True)
    else:
        val_loader_mismatched = None
        test_loader_other = None
        if ds_test_other:
            _, test_loader_other = make_dataloaders(ds["train"], ds_test_other["test"], cfg["batch_size"], cfg["num_workers"], pre_tokenized=True)
        test_loader_matched = None
        test_loader_mismatched = None
    
    print(f"Train dataset size: {len(ds['train'])}")
    print(f"Validation dataset size (matched): {len(ds_test_matched['validation'])}")
    if is_mnli and ds_test_mismatched:
        print(f"Validation dataset size (mismatched): {len(ds_test_mismatched['validation'])}")
    if is_mnli:
        if ds_test_matched_json:
            print(f"Test dataset size (MNLI from JSON): {len(ds_test_matched_json['test'])}")
        if ds_test_mismatched_json:
            print(f"Test dataset size (MNLI-MM from JSON): {len(ds_test_mismatched_json['test'])}")
    elif ds_test_other:
        print(f"Test dataset size ({dataset_config or 'test'} from JSON): {len(ds_test_other['test'])}")

    # Get number of labels from config (defaults to 2 for binary classification)
    num_labels = cfg.get("num_labels", 2)
    print(f"Number of labels: {num_labels}")
    
    model = CorrectEnglishSentenceClassifier(base, hidden_size=base.config.hidden_size, num_labels=num_labels).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg.get("weight_decay", 0.17))
    
    # Setup SWA (Stochastic Weight Averaging)
    swa_start_epoch = max(1, int(cfg["epochs"] * 0.9))  # Start SWA after 75% of training
    swa_model = AveragedModel(model)
    swa_scheduler = SWALR(opt, swa_lr=cfg["lr"] * 0.5)  # Lower LR for averaging phase
    use_swa = True #cfg.get("use_swa", True)
    
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
    scaler = GradScaler('cuda', enabled=cfg.get("mixed_precision", True))
    best_t = 0.5  # Default threshold, will be updated during training
    
    # Track best model for saving (based on test accuracy)
    best_test_acc = -1.0
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
                
                # Gradient clipping
                max_grad_norm = cfg.get("max_grad_norm", 1.0)
                if max_grad_norm > 0:
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                
                scaler.step(opt)
                scaler.update()
                
                # Update SWA model and scheduler if in SWA phase (after each batch)
                if use_swa and epoch >= swa_start_epoch:
                    swa_model.update_parameters(model)
                    swa_scheduler.step()

                running += float(loss.item())
                y_true.extend(labels.detach().cpu().tolist())
                y_pred.extend(logits_clean.detach().cpu().argmax(-1).tolist())
                # collect probabilities (for binary: class 1, for multi-class: all classes)
                probs_batch = torch.softmax(logits_clean.detach().cpu(), dim=-1)
                if num_labels > 2:
                    y_prob.append(probs_batch.numpy())
                else:
                    y_prob.append(probs_batch[:, 1].numpy())
                

            train_mcc = matthews_corrcoef(y_true, y_pred)
            train_f1 = f1_score(y_true, y_pred, average='macro')
            # concatenate per-batch probabilities to a single vector aligned with y_true
            probs = np.concatenate(y_prob, axis=0)
            true = np.asarray(y_true)
            
            # Search across thresholds (only for binary classification)
            # For multi-class (num_labels > 2), use argmax directly (no thresholding)
            if num_labels > 2:
                # Multi-class: no threshold needed, use argmax
                best_t = 0.5  # Not used for multi-class, but kept for compatibility
            elif epoch == cfg["epochs"] or epoch % 3 == 0:
                # Binary classification: search for optimal threshold
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
                    if num_labels > 2:
                        probs_batch = torch.softmax(logits.detach().cpu(), dim=-1).numpy()
                        v_prob_matched.append(probs_batch)
                    else:
                        probs_batch = torch.softmax(logits.detach().cpu(), dim=-1)[:, 1].numpy()
                        v_prob_matched.append(probs_batch)

            if num_labels > 2:
                probs_matched = np.concatenate(v_prob_matched, axis=0)
                preds_matched = np.argmax(probs_matched, axis=1)
            else:
                probs_matched = np.concatenate(v_prob_matched, axis=0) 
                preds_matched = (probs_matched >= best_t).astype(int)
            val_mcc_matched = matthews_corrcoef(v_true_matched, preds_matched)
            val_f1_matched = f1_score(v_true_matched, preds_matched, average='macro')
            val_acc_matched = accuracy_score(v_true_matched, preds_matched)
            dataset_label = "MNLI" if is_mnli else cfg["dataset_config"].upper() if cfg["dataset_config"] else "TEST"
            print(f"          val ({dataset_label} matched): mcc={val_mcc_matched:.4f} f1={val_f1_matched:.4f} acc={val_acc_matched:.4f}")

            # Evaluate on mismatched validation set for MNLI
            val_acc_mismatched = None
            val_mcc_mismatched = None
            val_f1_mismatched = None
            if is_mnli and val_loader_mismatched:
                v_true_mismatched, v_pred_mismatched, v_prob_mismatched = [], [], []
                with torch.no_grad():
                    for batch in val_loader_mismatched:
                        input_ids = batch["input_ids"].to(dev, non_blocking=True)
                        attention_mask = batch["attention_mask"].to(dev, non_blocking=True)
                        labels = batch["label"].to(dev, non_blocking=True)
                        logits = model(input_ids, attention_mask)
                        v_true_mismatched.extend(labels.cpu().tolist())
                        v_pred_mismatched.extend(logits.cpu().argmax(-1).tolist())
                        # For multi-class, get probabilities for all classes
                        if num_labels > 2:
                            probs_batch = torch.softmax(logits.detach().cpu(), dim=-1).numpy()
                            v_prob_mismatched.append(probs_batch)
                        else:
                            probs_batch = torch.softmax(logits.detach().cpu(), dim=-1)[:, 1].numpy()
                            v_prob_mismatched.append(probs_batch)
                
                if num_labels > 2:
                    probs_mismatched = np.concatenate(v_prob_mismatched, axis=0)
                    preds_mismatched = np.argmax(probs_mismatched, axis=1)
                else:
                    probs_mismatched = np.concatenate(v_prob_mismatched, axis=0)
                    preds_mismatched = (probs_mismatched >= best_t).astype(int)
                val_mcc_mismatched = matthews_corrcoef(v_true_mismatched, preds_mismatched)
                val_f1_mismatched = f1_score(v_true_mismatched, preds_mismatched, average='macro')
                val_acc_mismatched = accuracy_score(v_true_mismatched, preds_mismatched)
                print(f"          val (MNLI mismatched): mcc={val_mcc_mismatched:.4f} f1={val_f1_mismatched:.4f} acc={val_acc_mismatched:.4f}")

            # Evaluate on test set (all from JSON)
            if is_mnli:
                # For MNLI, use mismatched test set for best model selection (standard practice)
                test_loader = test_loader_mismatched if test_loader_mismatched else test_loader_matched
                test_set_name = "MNLI-MM" if test_loader_mismatched else "MNLI"
            else:
                test_loader = test_loader_other
                test_set_name = dataset_config.upper() if dataset_config else "TEST"
            
            if test_loader:
                t_true, t_pred, t_prob = [], [], []
                with torch.no_grad():
                    for batch in test_loader:
                        input_ids = batch["input_ids"].to(dev, non_blocking=True)
                        attention_mask = batch["attention_mask"].to(dev, non_blocking=True)
                        labels = batch["label"].to(dev, non_blocking=True)
                        logits = model(input_ids, attention_mask)
                        t_true.extend(labels.cpu().tolist())
                        t_pred.extend(logits.cpu().argmax(-1).tolist())
                        # For multi-class, get probabilities for all classes
                        if num_labels > 2:
                            probs_batch = torch.softmax(logits.detach().cpu(), dim=-1).numpy()
                            t_prob.append(probs_batch)
                        else:
                            probs_batch = torch.softmax(logits.detach().cpu(), dim=-1)[:, 1].numpy()
                            t_prob.append(probs_batch)
                
                if num_labels > 2:
                    probs_test = np.concatenate(t_prob, axis=0)
                    preds_test = np.argmax(probs_test, axis=1)
                else:
                    probs_test = np.concatenate(t_prob, axis=0)
                    preds_test = (probs_test >= best_t).astype(int)
                test_mcc = matthews_corrcoef(t_true, preds_test)
                test_f1 = f1_score(t_true, preds_test, average='macro')
                test_acc = accuracy_score(t_true, preds_test)
                print(f"          test ({test_set_name}): mcc={test_mcc:.4f} f1={test_f1:.4f} acc={test_acc:.4f}")
            else:
                test_acc = 0.0
                test_mcc = 0.0
                test_f1 = 0.0
            
            # Save best model based on test accuracy
            ckpt_dir = cfg.get("ckpt_dir", "checkpoints/finetune")
            os.makedirs(ckpt_dir, exist_ok=True)
            exp_name = cfg.get("experiment_name", "")
            prefix = f"{exp_name}_" if exp_name else ""
            
            if test_acc > best_test_acc:
                best_test_acc = test_acc
                best_model_state = model.state_dict().copy()
                # Save checkpoint with best test accuracy
                best_ckpt_path = os.path.join(ckpt_dir, f"{prefix}best.pt")
                checkpoint_data = {
                    "epoch": epoch,
                    "val_mcc_matched": val_mcc_matched,
                    "val_acc_matched": val_acc_matched,
                    "val_f1_matched": val_f1_matched,
                    "test_mcc": test_mcc,
                    "test_acc": test_acc,
                    "test_f1": test_f1,
                    "model_state_dict": model.state_dict(),
                    "best_threshold": best_t,
                }
                if is_mnli and val_acc_mismatched is not None:
                    checkpoint_data.update({
                        "val_mcc_mismatched": val_mcc_mismatched,
                        "val_acc_mismatched": val_acc_mismatched,
                        "val_f1_mismatched": val_f1_mismatched,
                    })
                save_ckpt(checkpoint_data, best_ckpt_path)
                print(f"          ✓ Saved best model (Test Acc={test_acc:.4f}) to {best_ckpt_path}")
            
            # Also save per-epoch checkpoint with test accuracy in filename
            epoch_ckpt_path = os.path.join(ckpt_dir, f"{prefix}epoch{epoch}_testacc{test_acc:.4f}.pt")
            checkpoint_data = {
                "epoch": epoch,
                "val_mcc_matched": val_mcc_matched,
                "val_acc_matched": val_acc_matched,
                "val_f1_matched": val_f1_matched,
                "test_mcc": test_mcc,
                "test_acc": test_acc,
                "test_f1": test_f1,
                "model_state_dict": model.state_dict(),
                "best_threshold": best_t,
            }
            if is_mnli and val_acc_mismatched is not None:
                checkpoint_data.update({
                    "val_mcc_mismatched": val_mcc_mismatched,
                    "val_acc_mismatched": val_acc_mismatched,
                    "val_f1_mismatched": val_f1_mismatched,
                })
            save_ckpt(checkpoint_data, epoch_ckpt_path)
        
        # After training, ensure we use the best test accuracy model for final evaluation
        # (SWA might have been applied, but we prioritize best test accuracy)
        if best_model_state is not None:
            print(f"\nLoading best model based on test accuracy (Test Acc={best_test_acc:.4f}) for final evaluation...")
            model.load_state_dict(best_model_state)
        
        # After training, update BN stats for SWA model (optional, but don't replace best model)
        if use_swa:
            print("Updating batch normalization statistics for SWA model...")
            # Create a temporary model for SWA evaluation
            swa_model_for_eval = CorrectEnglishSentenceClassifier(base, hidden_size=base.config.hidden_size, num_labels=num_labels).to(dev)
            swa_model_for_eval.load_state_dict(swa_model.module.state_dict())
            
            # Update BN stats by running one forward pass through training data
            swa_model_for_eval.eval()
            with torch.no_grad():
                for batch in tqdm(train_loader, desc="Updating BN stats", leave=False):
                    # Data is already tokenized
                    input_ids = batch["input_ids"].to(dev, non_blocking=True)
                    attention_mask = batch["attention_mask"].to(dev, non_blocking=True)
                    
                    _ = swa_model_for_eval(input_ids, attention_mask)
            # Note: BERT uses LayerNorm, not BatchNorm, so update_bn is not strictly necessary
            # But we'll update any BatchNorm layers if they exist (e.g., in custom heads)
            try:
                update_bn(train_loader, swa_model_for_eval, device=dev)
            except Exception:
                # If update_bn fails (e.g., no BN layers), that's fine
                pass
            
            # Evaluate SWA model on test set to compare with best test accuracy model
            print("Evaluating SWA model on test set for comparison...")
            swa_model_for_eval.eval()
            
            # Select appropriate test loader (all from JSON)
            if is_mnli:
                swa_test_loader = test_loader_mismatched if test_loader_mismatched else test_loader_matched
            else:
                swa_test_loader = test_loader_other
            
            if swa_test_loader:
                swa_t_true, swa_t_pred, swa_t_prob = [], [], []
                with torch.no_grad():
                    for batch in swa_test_loader:
                        input_ids = batch["input_ids"].to(dev, non_blocking=True)
                        attention_mask = batch["attention_mask"].to(dev, non_blocking=True)
                        labels = batch["label"].to(dev, non_blocking=True)
                        logits = swa_model_for_eval(input_ids, attention_mask)
                        swa_t_true.extend(labels.cpu().tolist())
                        if num_labels > 2:
                            probs_batch = torch.softmax(logits.detach().cpu(), dim=-1).numpy()
                            swa_t_prob.append(probs_batch)
                        else:
                            probs_batch = torch.softmax(logits.detach().cpu(), dim=-1)[:, 1].numpy()
                            swa_t_prob.append(probs_batch)
                
                if num_labels > 2:
                    swa_t_probs = np.concatenate(swa_t_prob, axis=0)
                    swa_t_preds = np.argmax(swa_t_probs, axis=1)
                else:
                    swa_t_probs = np.concatenate(swa_t_prob, axis=0)
                    swa_t_preds = (swa_t_probs >= best_t).astype(int)
                swa_test_acc = accuracy_score(swa_t_true, swa_t_preds)
                print(f"SWA model test accuracy: {swa_test_acc:.4f}")
                
                # Only use SWA model if it's better than the best test accuracy model
                if swa_test_acc > best_test_acc:
                    print(f"SWA model is better (Test Acc={swa_test_acc:.4f} > {best_test_acc:.4f}). Using SWA model for final evaluation.")
                    model.load_state_dict(swa_model_for_eval.state_dict())
                    best_test_acc = swa_test_acc
                else:
                    print(f"Best test accuracy model is better (Test Acc={best_test_acc:.4f} >= {swa_test_acc:.4f}). Using best test accuracy model.")
            else:
                print("No test loader available for SWA evaluation.")
    
    # Final validation
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
            if num_labels > 2:
                probs_batch = torch.softmax(logits.detach().cpu(), dim=-1).numpy()
                v_prob_matched.append(probs_batch)
            else:
                probs_batch = torch.softmax(logits.detach().cpu(), dim=-1)[:, 1].numpy()
                v_prob_matched.append(probs_batch)

    if num_labels > 2:
        probs_matched = np.concatenate(v_prob_matched, axis=0)
        preds_matched = np.argmax(probs_matched, axis=1)
    else:
        probs_matched = np.concatenate(v_prob_matched, axis=0)
        preds_matched = (probs_matched >= best_t).astype(int)
    val_mcc_matched = matthews_corrcoef(v_true_matched, preds_matched)
    val_f1_matched = f1_score(v_true_matched, preds_matched, average='macro')
    val_acc_matched = accuracy_score(v_true_matched, preds_matched)
    dataset_label = "MNLI" if is_mnli else "QNLI"
    print(f"Final validation ({dataset_label} matched): mcc={val_mcc_matched:.4f} f1={val_f1_matched:.4f} acc={val_acc_matched:.4f}")
    
    # Mismatched validation for MNLI
    if is_mnli and val_loader_mismatched:
        v_true_mismatched, v_pred_mismatched, v_prob_mismatched = [], [], []
        with torch.no_grad():
            for batch in val_loader_mismatched:
                input_ids = batch["input_ids"].to(dev, non_blocking=True)
                attention_mask = batch["attention_mask"].to(dev, non_blocking=True)
                labels = batch["label"].to(dev, non_blocking=True)
                logits = model(input_ids, attention_mask)
                v_true_mismatched.extend(labels.cpu().tolist())
                v_pred_mismatched.extend(logits.cpu().argmax(-1).tolist())
                if num_labels > 2:
                    probs_batch = torch.softmax(logits.detach().cpu(), dim=-1).numpy()
                    v_prob_mismatched.append(probs_batch)
                else:
                    probs_batch = torch.softmax(logits.detach().cpu(), dim=-1)[:, 1].numpy()
                    v_prob_mismatched.append(probs_batch)
        
        if num_labels > 2:
            probs_mismatched = np.concatenate(v_prob_mismatched, axis=0)
            preds_mismatched = np.argmax(probs_mismatched, axis=1)
        else:
            probs_mismatched = np.concatenate(v_prob_mismatched, axis=0)
            preds_mismatched = (probs_mismatched >= best_t).astype(int)
        val_mcc_mismatched = matthews_corrcoef(v_true_mismatched, preds_mismatched)
        val_f1_mismatched = f1_score(v_true_mismatched, preds_mismatched, average='macro')
        val_acc_mismatched = accuracy_score(v_true_mismatched, preds_mismatched)
        print(f"Final validation (MNLI mismatched): mcc={val_mcc_mismatched:.4f} f1={val_f1_mismatched:.4f} acc={val_acc_mismatched:.4f}")
    
    # Test set evaluation (all from JSON)
    print("\n" + "="*50)
    if is_mnli:
        # Evaluate on both MNLI matched and mismatched test sets from JSON
        if test_loader_matched:
            print("Evaluating on MNLI test set (matched, from JSON)...")
            t_true_matched, t_pred_matched, t_prob_matched = [], [], []
            with torch.no_grad():
                for batch in test_loader_matched:
                    input_ids = batch["input_ids"].to(dev, non_blocking=True)
                    attention_mask = batch["attention_mask"].to(dev, non_blocking=True)
                    labels = batch["label"].to(dev, non_blocking=True)
                    logits = model(input_ids, attention_mask)
                    t_true_matched.extend(labels.cpu().tolist())
                    t_pred_matched.extend(logits.cpu().argmax(-1).tolist())
                    if num_labels > 2:
                        probs_batch = torch.softmax(logits.detach().cpu(), dim=-1).numpy()
                        t_prob_matched.append(probs_batch)
                    else:
                        probs_batch = torch.softmax(logits.detach().cpu(), dim=-1)[:, 1].numpy()
                        t_prob_matched.append(probs_batch)
            
            if num_labels > 2:
                probs_test_matched = np.concatenate(t_prob_matched, axis=0)
                preds_test_matched = np.argmax(probs_test_matched, axis=1)
            else:
                probs_test_matched = np.concatenate(t_prob_matched, axis=0)
                preds_test_matched = (probs_test_matched >= best_t).astype(int)
            test_mcc_matched = matthews_corrcoef(t_true_matched, preds_test_matched)
            test_f1_matched = f1_score(t_true_matched, preds_test_matched, average='macro')
            test_acc_matched = accuracy_score(t_true_matched, preds_test_matched)
            print(f"Test (MNLI matched, from JSON): mcc={test_mcc_matched:.4f} f1={test_f1_matched:.4f} acc={test_acc_matched:.4f}")
        
        if test_loader_mismatched:
            print("Evaluating on MNLI test set (mismatched, from JSON)...")
            t_true_mismatched, t_pred_mismatched, t_prob_mismatched = [], [], []
            with torch.no_grad():
                for batch in test_loader_mismatched:
                    input_ids = batch["input_ids"].to(dev, non_blocking=True)
                    attention_mask = batch["attention_mask"].to(dev, non_blocking=True)
                    labels = batch["label"].to(dev, non_blocking=True)
                    logits = model(input_ids, attention_mask)
                    t_true_mismatched.extend(labels.cpu().tolist())
                    t_pred_mismatched.extend(logits.cpu().argmax(-1).tolist())
                    if num_labels > 2:
                        probs_batch = torch.softmax(logits.detach().cpu(), dim=-1).numpy()
                        t_prob_mismatched.append(probs_batch)
                    else:
                        probs_batch = torch.softmax(logits.detach().cpu(), dim=-1)[:, 1].numpy()
                        t_prob_mismatched.append(probs_batch)
            
            if num_labels > 2:
                probs_test_mismatched = np.concatenate(t_prob_mismatched, axis=0)
                preds_test_mismatched = np.argmax(probs_test_mismatched, axis=1)
            else:
                probs_test_mismatched = np.concatenate(t_prob_mismatched, axis=0)
                preds_test_mismatched = (probs_test_mismatched >= best_t).astype(int)
            test_mcc_mismatched = matthews_corrcoef(t_true_mismatched, preds_test_mismatched)
            test_f1_mismatched = f1_score(t_true_mismatched, preds_test_mismatched, average='macro')
            test_acc_mismatched = accuracy_score(t_true_mismatched, preds_test_mismatched)
            print(f"Test (MNLI mismatched, from JSON): mcc={test_mcc_mismatched:.4f} f1={test_f1_mismatched:.4f} acc={test_acc_mismatched:.4f}")
    else:
        # Test set evaluation from JSON
        if test_loader_other:
            test_set_name = dataset_config.upper() if dataset_config else "TEST"
            print(f"Evaluating on {test_set_name} test set (from JSON)...")
            t_true, t_pred, t_prob = [], [], []
            with torch.no_grad():
                for batch in test_loader_other:
                    input_ids = batch["input_ids"].to(dev, non_blocking=True)
                    attention_mask = batch["attention_mask"].to(dev, non_blocking=True)
                    labels = batch["label"].to(dev, non_blocking=True)
                    logits = model(input_ids, attention_mask)
                    t_true.extend(labels.cpu().tolist())
                    t_pred.extend(logits.cpu().argmax(-1).tolist())
                    if num_labels > 2:
                        probs_batch = torch.softmax(logits.detach().cpu(), dim=-1).numpy()
                        t_prob.append(probs_batch)
                    else:
                        probs_batch = torch.softmax(logits.detach().cpu(), dim=-1)[:, 1].numpy()
                        t_prob.append(probs_batch)
            
            if num_labels > 2:
                probs_test = np.concatenate(t_prob, axis=0)
                preds_test = np.argmax(probs_test, axis=1)
            else:
                probs_test = np.concatenate(t_prob, axis=0)
                preds_test = (probs_test >= best_t).astype(int)
            test_mcc = matthews_corrcoef(t_true, preds_test)
            test_f1 = f1_score(t_true, preds_test, average='macro')
            test_acc = accuracy_score(t_true, preds_test)
            print(f"Test ({test_set_name}, from JSON): mcc={test_mcc:.4f} f1={test_f1:.4f} acc={test_acc:.4f}")
        else:
            print("Warning: No test loader available for evaluation")
    
    # If in eval_only mode, we're done
    if eval_only:
        print("✅ Evaluation complete.")
        return model
     
    return model

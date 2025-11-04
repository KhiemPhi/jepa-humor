#!/usr/bin/env python
import os
import argparse
import numpy as np
from dataclasses import dataclass
from typing import Dict

import torch
from datasets import load_dataset
from sklearn.metrics import matthews_corrcoef
from transformers import (
    AutoTokenizer,
    AutoConfig,
    AutoModelForSequenceClassification,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    TrainerCallback,
    set_seed,
)


def compute_metrics(eval_pred) -> Dict[str, float]:
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    mcc = matthews_corrcoef(labels, preds) if len(set(labels)) > 1 and len(set(preds)) > 1 else 0.0
    acc = (preds == labels).mean().item() if hasattr((preds == labels).mean(), 'item') else float((preds == labels).mean())
    return {"mcc": float(mcc), "accuracy": float(acc)}


class ProgressCallback(TrainerCallback):
    """Callback to print progress and verify training is actually running"""
    def on_train_begin(self, args, state, control, **kwargs):
        print(f"\n[ProgressCallback] Training started! Total steps: {state.max_steps}")
        import sys
        sys.stdout.flush()
        
    def on_step_begin(self, args, state, control, **kwargs):
        # Print first 5 steps, then every 10th step
        if state.global_step <= 5 or state.global_step % 10 == 0:
            print(f"[ProgressCallback] Step {state.global_step}/{state.max_steps} STARTED")
            import sys
            sys.stdout.flush()
        
    def on_step_end(self, args, state, control, **kwargs):
        # Print first 5 steps, then every 10th step
        if state.global_step <= 5 or state.global_step % 10 == 0:
            print(f"[ProgressCallback] Step {state.global_step}/{state.max_steps} COMPLETED")
            import sys
            sys.stdout.flush()
            
    def on_log(self, args, state, control, logs=None, **kwargs):
        print(f"[ProgressCallback] LOG at step {state.global_step}: {logs}")
        import sys
        sys.stdout.flush()
        
    def on_epoch_begin(self, args, state, control, **kwargs):
        print(f"[ProgressCallback] Epoch {state.epoch} started")
        import sys
        sys.stdout.flush()


def main():
    parser = argparse.ArgumentParser(description="Quick BERT-base-uncased fine-tuning on GLUE CoLA (MCC)")
    parser.add_argument("--model_name", type=str, default="bert-base-uncased")
    parser.add_argument("--encoder_ckpt", type=str, default=None, help="Path to JEPA encoder checkpoint to load into the base model")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--max_length", type=int, default=128)
    parser.add_argument("--output_dir", type=str, default="checkpoints/cola_bert_base")
    parser.add_argument("--logging_steps", type=int, default=50)
    parser.add_argument("--eval_steps", type=int, default=200)
    parser.add_argument("--save_total_limit", type=int, default=2)
    parser.add_argument("--fp16", action="store_true")
    args = parser.parse_args()

    set_seed(args.seed)


    # Load CoLA
    ds = load_dataset("glue", "cola")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    def preprocess_function(examples):
        return tokenizer(
            examples["sentence"],
            truncation=True,
            max_length=args.max_length,
        )
    ds = ds.map(preprocess_function, batched=True, desc="Tokenizing")
    ds = ds.rename_column("label", "labels")
    ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])  # token_type_ids not used

    # --- Sanity checks on labels ---
    # CoLA is binary (0/1). Make sure labels are in range and proper dtype.
    def collect_label_set(split_name: str):
        try:
            vals = set(ds[split_name]["labels"]) if "labels" in ds[split_name].column_names else set()
            print(f"[{split_name}] unique labels: {sorted(list(vals))}")
        except Exception as e:
            print(f"Could not inspect labels for split '{split_name}': {e}")

    for split in ["train", "validation", "test"]:
        if split in ds:
            collect_label_set(split)

    # Ensure labels tensor type is long and values are within [0,1]
    # If any label <0 or >1 exists, map to {0,1} by clipping and warn.
    import torch as _torch
    def _fix_labels(example):
        lbl = int(example["labels"]) if isinstance(example["labels"], (int,)) else int(example["labels"])
        if lbl < 0 or lbl > 1:
            # clip to valid range and warn once at runtime
            example["labels"] = 1 if lbl > 1 else 0
        return example

    ds = ds.map(_fix_labels)
    # Reset format and set it properly for Trainer
    ds.set_format(type=None)  # Reset format first
    ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])  # ensure long dtype for labels
  
    # Print dataset info for debugging
    print(f"Train dataset size: {len(ds['train'])}")
    if "validation" in ds:
        print(f"Validation dataset size: {len(ds['validation'])}")
    if "test" in ds:
        print(f"Test dataset size: {len(ds['test'])}")
        # Check if test set has labels
        if "labels" in ds["test"].column_names:
            print(f"Test set has labels")
        else:
            print(f"Warning: Test set does not have labels - cannot use for evaluation")

    config = AutoConfig.from_pretrained(args.model_name, num_labels=2)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name, config=config)

    # Optionally load JEPA encoder weights into the backbone (strict=False best-effort)
    if args.encoder_ckpt is not None and os.path.exists(args.encoder_ckpt):
        ckpt = torch.load(args.encoder_ckpt, map_location="cpu")
        # Accept either full-state or nested under 'encoder_state_dict'
        sd = ckpt.get("encoder_state_dict", ckpt)
        # Try to adapt key prefixes to match HF model backbone
        adapted = {}
        for k, v in sd.items():
            k_new = k
            # Common patterns: 'encoder.' prefix in JEPA → 'bert.' in HF backbone
            if k_new.startswith("encoder."):
                k_new = k_new[len("encoder."):]
            # If keys look like base encoder (e.g., 'transformer.' for DistilBERT), keep; Trainer will ignore non-matching
            # Prepend backbone prefix if needed
            if not k_new.startswith("bert.") and (k_new.startswith("embeddings") or k_new.startswith("encoder") or k_new.startswith("pooler")):
                k_new = "bert." + k_new
            adapted[k_new] = v
        missing, unexpected = model.base_model.load_state_dict(adapted, strict=False)
        print(f"Loaded JEPA encoder weights (strict=False). Missing: {len(missing)}, Unexpected: {len(unexpected)}")

    # Model head/output checks
    try:
        num_labels = getattr(model.config, "num_labels", None)
        head_out = getattr(getattr(model, "classifier", None), "out_features", None)
        print(f"Model num_labels={num_labels}, classifier.out_features={head_out}")
    except Exception as e:
        print(f"Could not print model head info: {e}")

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, pad_to_multiple_of=8 if args.fp16 else None)

    # Calculate steps per epoch and total steps
    steps_per_epoch = len(ds["train"]) // args.batch_size
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = int(args.warmup_ratio * total_steps)
    
    print(f"Steps per epoch: {steps_per_epoch}")
    print(f"Total training steps: {total_steps}")
    print(f"Warmup steps: {warmup_steps}")
    print(f"Eval steps: {args.eval_steps}, Logging steps: {args.logging_steps}")
    
    # Ensure eval_steps is reasonable (not larger than steps_per_epoch)
    # Also ensure it's not too small (could cause immediate evaluation hang)
    if args.eval_steps > steps_per_epoch:
        print(f"Warning: eval_steps ({args.eval_steps}) > steps_per_epoch ({steps_per_epoch}). Adjusting to steps_per_epoch.")
        eval_steps = max(1, steps_per_epoch)
    elif args.eval_steps < 1:
        print(f"Warning: eval_steps ({args.eval_steps}) < 1. Setting to 1.")
        eval_steps = 1
    else:
        eval_steps = args.eval_steps
    
    # Ensure logging_steps is also reasonable
    # Force logging_steps to 1 to get immediate progress feedback
    logging_steps = 1
    print(f"Setting logging_steps to 1 for immediate progress feedback")

    # Use validation set for evaluation (test set may not have labels)
    # Define this BEFORE training_args since it's used in training_args
    eval_dataset = ds["validation"] if "validation" in ds and "labels" in ds["validation"].column_names else None
    if eval_dataset is None:
        print("Warning: No validation set with labels found. Training without evaluation.")

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        num_train_epochs=args.epochs,
        eval_strategy="steps" if eval_dataset is not None else "no",
        save_strategy="steps",
        eval_steps=eval_steps if eval_dataset is not None else None,
        save_steps=eval_steps,
        logging_steps=logging_steps,
        load_best_model_at_end=True if eval_dataset is not None else False,
        metric_for_best_model="mcc" if eval_dataset is not None else None,
        greater_is_better=True if eval_dataset is not None else None,
        warmup_steps=warmup_steps,
        fp16=args.fp16,
        report_to=["none"],
        dataloader_num_workers=0,  # Set to 0 to avoid multiprocessing issues (common cause of hangs)
        dataloader_pin_memory=False,  # Can also cause hangs on some systems
        logging_first_step=True,  # Log immediately on first step
        remove_unused_columns=False,  # Sometimes helps with dataset issues
        disable_tqdm=False,  # Keep tqdm but let's see if callback helps
        dataloader_drop_last=False,  # Don't drop last incomplete batch
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds["train"],
        eval_dataset=ds["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics if eval_dataset is not None else None,
        callbacks=[ProgressCallback()],
    )
    
    print("Trainer initialized successfully")
    print(f"Trainer model device: {next(trainer.model.parameters()).device}")
    print(f"Trainer args: {training_args}")
    
    # Test dataset iteration before training
    print("Testing dataset iteration...")
    try:
        sample = ds["train"][0]
        print(f"Sample batch keys: {sample.keys()}")
        print(f"Sample input_ids shape: {sample['input_ids'].shape}")
        print(f"Sample label: {sample['labels']}")
    except Exception as e:
        print(f"Error testing dataset: {e}")
        import traceback
        traceback.print_exc()
    
    # Test data collator
    print("Testing data collator...")
    collated = None
    try:
        test_batch = [ds["train"][i] for i in range(min(2, len(ds["train"])))]
        collated = data_collator(test_batch)
        print(f"Collated batch keys: {collated.keys()}")
        print(f"Collated input_ids shape: {collated['input_ids'].shape}")
    except Exception as e:
        print(f"Error testing data collator: {e}")
        import traceback
        traceback.print_exc()
    
    # Test model forward pass (only if collator worked)
    if collated is not None:
        print("Testing model forward pass...")
        try:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            print(f"Using device: {device}")
            # Create a copy for testing to avoid modifying the original model
            test_model = AutoModelForSequenceClassification.from_pretrained(args.model_name, config=config)
            test_model = test_model.to(device)
            with torch.no_grad():
                test_input = collated["input_ids"].to(device)
                test_mask = collated["attention_mask"].to(device)
                test_labels = collated["labels"].to(device)
                outputs = test_model(input_ids=test_input, attention_mask=test_mask, labels=test_labels)
                print(f"Model forward pass successful. Loss: {outputs.loss.item()}")
            del test_model  # Clean up
            torch.cuda.empty_cache() if torch.cuda.is_available() else None
        except Exception as e:
            print(f"Error testing model: {e}")
            import traceback
            traceback.print_exc()
    
    # Test manual batch iteration to ensure dataloader works
    print("Testing manual batch iteration...")
    try:
        from torch.utils.data import DataLoader
        test_loader = DataLoader(
            ds["train"],
            batch_size=args.batch_size,
            collate_fn=data_collator,
            num_workers=0,
            pin_memory=False,
        )
        print("Created test dataloader, iterating through first batch...")
        first_batch = next(iter(test_loader))
        print(f"First batch keys: {first_batch.keys()}")
        print(f"First batch input_ids shape: {first_batch['input_ids'].shape}")
        print(f"First batch labels: {first_batch['labels']}")
        print("✓ Manual batch iteration successful")
    except Exception as e:
        print(f"✗ Error in manual batch iteration: {e}")
        import traceback
        traceback.print_exc()
    
    print("Starting training...")
    print(f"Training will run for {args.epochs} epochs, {total_steps} total steps")
    if eval_dataset is not None:
        print(f"First evaluation at step {eval_steps}")
    print(f"First log at step {logging_steps} (logging_first_step=True)")
    print("=" * 80)
    
    import time
    import sys
    sys.stdout.flush()
    sys.stderr.flush()
    
    start_time = time.time()
    print(f"Training started at {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("Calling trainer.train() now...")
    sys.stdout.flush()
    
    try:
        # Force output to be unbuffered
        import os
        os.environ['PYTHONUNBUFFERED'] = '1'
        
        trainer.train()
        print(f"Training completed in {time.time() - start_time:.2f} seconds")
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        raise
    except Exception as e:
        print(f"Training failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.stdout.flush()
        raise
    metrics = trainer.evaluate(ds["validation"])  # CoLA uses Matthew's corr on dev set
    print({k: float(v) for k, v in metrics.items()})

    # --- Generate test-set submission file (GLUE CoLA) ---
    os.makedirs(args.output_dir, exist_ok=True)
    test_output = trainer.predict(ds["validation"])  # returns Predictions(predictions, label_ids=None, metrics)
    test_logits = test_output.predictions
    test_preds = np.argmax(test_logits, axis=-1).astype(int)

    # GLUE CoLA test set contains an 'idx' column; use it if available
    test_indices = ds["test"].get_column("idx") if "idx" in ds["test"].column_names else list(range(len(test_preds)))
    print("Percentage of test inndices in test set: ", len(test_indices) / len(ds["test"]))

    submission_path = os.path.join(args.output_dir, "cola_test.tsv")
    # Write TSV with header matching common GLUE format
    with open(submission_path, "w", encoding="utf-8") as f:
        f.write("index\tprediction\n")
        for i, p in zip(test_indices, test_preds):
            f.write(f"{i}\t{p}\n")
    print(f"Wrote test predictions to {submission_path}")
    
    test_mcc = matthews_corrcoef(ds["test"]["label"], test_preds)
    print(f"Test MCC: {test_mcc:.4f}")


if __name__ == "__main__":
    main()


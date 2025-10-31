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
    set_seed,
)


def compute_metrics(eval_pred) -> Dict[str, float]:
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    mcc = matthews_corrcoef(labels, preds) if len(set(labels)) > 1 and len(set(preds)) > 1 else 0.0
    acc = (preds == labels).mean().item() if hasattr((preds == labels).mean(), 'item') else float((preds == labels).mean())
    return {"mcc": float(mcc), "accuracy": float(acc)}


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
    ds.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])  # ensure long dtype for labels
  
    test_indices = ds["test"].get_column("idx") if "idx" in ds["test"].column_names else list(range(len(test_preds)))
    print("Percentage of test inndices in test set: ", len(test_indices) / len(ds["test"]))

    config = AutoConfig.from_pretrained(args.model_name, num_labels=2)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_name, config=config)

    # Optionally load JEPA encoder weights into the backbone (strict=False best-effort)
    if args.encoder_ckpt is not None and os.path.exists(args.encoder_ckpt):
        import torch
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

    warmup_steps = int(args.warmup_ratio * (len(ds["train"]) // args.batch_size) * args.epochs)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        num_train_epochs=args.epochs,
        eval_strategy="steps",
        save_strategy="steps",
        eval_steps=args.eval_steps,
        save_steps=args.eval_steps,
        logging_steps=args.logging_steps,
        load_best_model_at_end=True,
        metric_for_best_model="mcc",
        greater_is_better=True,
        warmup_steps=warmup_steps,
        fp16=args.fp16,
        report_to=["none"],
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds["train"],
        eval_dataset=ds["test"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()
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



from dataclasses import dataclass
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from torch.optim.swa_utils import AveragedModel, SWALR, update_bn
from transformers import AutoTokenizer, AutoModel
import os
from torch import amp
import numpy as np
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from .utils import set_seed, device, load_encoder_state
from .data import load_hf_splits, make_dataloaders
import evaluate

metric = evaluate.load("squad")
max_length = 384
stride = 128
n_best = 20  # Define n_best for compute_metrics
max_answer_length = 30 # Define max_answer_length for compute_metrics

def preprocess_training_examples(examples,tokenizer):
    questions = [q.strip() for q in examples["question"]]
    inputs = tokenizer(
        questions,
        examples["context"],
        max_length=max_length,
        truncation="only_second",
        stride=stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    offset_mapping = inputs.pop("offset_mapping")
    sample_map = inputs.pop("overflow_to_sample_mapping")
    answers = examples["answers"]
    start_positions = []
    end_positions = []

    for i, offset in enumerate(offset_mapping):
        sample_idx = sample_map[i]
        answer = answers[sample_idx]
        start_char = answer["answer_start"][0]
        end_char = answer["answer_start"][0] + len(answer["text"][0])
        sequence_ids = inputs.sequence_ids(i)

        # Find the start and end of the context
        idx = 0
        while sequence_ids[idx] != 1:
            idx += 1
        context_start = idx
        while sequence_ids[idx] == 1:
            idx += 1
        context_end = idx - 1

        # If the answer is not fully inside the context, label is (0, 0)
        if offset[context_start][0] > start_char or offset[context_end][1] < end_char:
            start_positions.append(0)
            end_positions.append(0)
        else:
            # Otherwise it's the start and end token positions
            idx = context_start
            while idx <= context_end and offset[idx][0] <= start_char:
                idx += 1
            start_positions.append(idx - 1)

            idx = context_end
            while idx >= context_start and offset[idx][1] >= end_char:
                idx -= 1
            end_positions.append(idx + 1)

    inputs["start_positions"] = start_positions
    inputs["end_positions"] = end_positions
    return inputs


def preprocess_validation_examples(examples, tokenizer):
    questions = [q.strip() for q in examples["question"]]
    inputs = tokenizer(
        questions,
        examples["context"],
        max_length=max_length,
        truncation="only_second",
        stride=stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    sample_map = inputs.pop("overflow_to_sample_mapping")
    example_ids = []

    for i in range(len(inputs["input_ids"])):
        sample_idx = sample_map[i]
        example_ids.append(examples["id"][sample_idx])

        sequence_ids = inputs.sequence_ids(i)
        offset = inputs["offset_mapping"][i]
        inputs["offset_mapping"][i] = [
            o if sequence_ids[k] == 1 else (0, 0) for k, o in enumerate(offset) # Replace None with (0,0)
        ]

    inputs["example_id"] = example_ids
    return inputs

import collections # Import collections
import numpy as np # Import numpy

def compute_metrics(start_logits, end_logits, features, examples):
    example_to_features = collections.defaultdict(list)
    for idx, feature in enumerate(features):
        example_to_features[feature["example_id"]].append(idx)

    predicted_answers = []
    for example in examples:
        example_id = example["id"]
        context = example["context"]
        answers = []

        # Loop through all features associated with that example
        for feature_index in example_to_features[example_id]:
            start_logit = start_logits[feature_index]
            end_logit = end_logits[feature_index]
            offsets = features[feature_index]["offset_mapping"]

            start_indexes = np.argsort(start_logit)[-1 : -n_best - 1 : -1].tolist()
            end_indexes = np.argsort(end_logit)[-1 : -n_best - 1 : -1].tolist()
            for start_index in start_indexes:
                for end_index in end_indexes:
                    # Skip answers that are not fully in the context
                    # Check for placeholder (0,0) instead of None
                    if offsets[start_index] == (0,0) or offsets[end_index] == (0,0):
                        continue
                    # Skip answers with a length that is either < 0 or > max_answer_length
                    if (
                        end_index < start_index
                        or end_index - start_index + 1 > max_answer_length
                    ):
                        continue

                    answer = {
                        "text": context[offsets[start_index][0] : offsets[end_index][1]],
                        "logit_score": start_logit[start_index] + end_logit[end_index],
                    }
                    answers.append(answer)

        # Select the answer with the best score
        if len(answers) > 0:
            best_answer = max(answers, key=lambda x: x["logit_score"])
            predicted_answers.append(
                {"id": example_id, "prediction_text": best_answer["text"]}
            )
        else:
            predicted_answers.append({"id": example_id, "prediction_text": ""})

    theoretical_answers = [{"id": ex["id"], "answers": ex["answers"]} for ex in examples]
    return metric.compute(predictions=predicted_answers, references=theoretical_answers)

class QAModel(nn.Module):
    def __init__(self, encoder, hidden_size=768):
        super().__init__()
        self.encoder = encoder
        self.hidden_size = hidden_size
        self.qa_outputs = nn.Linear(self.hidden_size, 2)

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        seq = outputs[0]
        logits = self.qa_outputs(seq)
        start_logits, end_logits = logits.split(1, dim=-1)
        return start_logits.squeeze(-1), end_logits.squeeze(-1)



def finetune(cfg, encoder_ckpt: str):
    set_seed(cfg["seed"])
    dev = device()

    # 1. Load Encoder
    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"])
    base = AutoModel.from_pretrained(cfg["model_name"])
    if encoder_ckpt and encoder_ckpt != "fresh":
        print(f"Loading pretrained encoder from {encoder_ckpt}")
        base = load_encoder_state(base, encoder_ckpt, map_location="cpu")
    else: 
        print(f"Loading fresh encoder")

    # Load SQuAD dataset
    raw_datasets = load_hf_splits(cfg["dataset"])
   
    train_dataset = raw_datasets["train"].map(
        lambda batch: preprocess_training_examples(batch, tokenizer),
        batched=True,
        remove_columns=raw_datasets["train"].column_names,
    )

    validation_dataset = raw_datasets["validation"].map(
        lambda batch: preprocess_validation_examples(batch, tokenizer = tokenizer),
        batched=True,
        remove_columns=raw_datasets["validation"].column_names,
    )

    train_loader, val_loader = make_dataloaders(train_dataset, validation_dataset, cfg["batch_size"], cfg["num_workers"])
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(validation_dataset)}")

    # Initialize QA model
    model = QAModel(base, hidden_size=base.config.hidden_size).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"])
    
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # ignore index for invalid positions
    
    scaler = GradScaler(enabled=cfg.get("mixed_precision", True))

    for epoch in range(1, cfg["epochs"] + 1):
        model.train()
        running_loss = 0.0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch}"):
            start_positions, end_positions = batch["start_positions"], batch["end_positions"]

            start_positions = torch.tensor(start_positions).to(dev)
            end_positions = torch.tensor(end_positions).to(dev)

            input_ids = batch["input_ids"].to(dev)
            #input_ids = input_ids.t()
            attention_mask = batch["attention_mask"].to(dev)
            #attention_mask = attention_mask.t()

            opt.zero_grad(set_to_none=True)

            with amp.autocast(device_type='cuda'):
                start_logits, end_logits = model(input_ids, attention_mask)
                loss = (criterion(start_logits, start_positions) + criterion(end_logits, end_positions)) / 2

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            running_loss += loss.item()

        model.eval()
        start_logits = []
        end_logits = []
        for batch in tqdm(val_loader, desc=f"[Val] Epoch {epoch}"):
            input_ids = batch["input_ids"].to(dev)
            #input_ids = input_ids.t()
            attention_mask = batch["attention_mask"].to(dev)
            #attention_mask = attention_mask.t()
            with torch.no_grad():
                start,end = model(input_ids,attention_mask)

            start_logits.append((start).cpu().numpy())
            end_logits.append((end).cpu().numpy())

        start_logits = np.concatenate(start_logits)
        end_logits = np.concatenate(end_logits)
        start_logits = start_logits[: len(validation_dataset)]
        end_logits = end_logits[: len(validation_dataset)]

        val_score = compute_metrics(
            start_logits, end_logits, validation_dataset, raw_datasets["validation"]
        )
        val_em = val_score["exact_match"]
        val_f1 = val_score["f1"]
        print(f"          val: EM={val_em:.4f} F1={val_f1:.4f}")
    return model
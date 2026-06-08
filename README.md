# JEPA-RTE Agent

An autonomous experimentation agent that iteratively improves RoBERTa embeddings via a Joint Embedding Predictive Architecture (JEPA) training regime. The agent modifies `scripts/jepa.py` for architectural and loss changes, and tunes `configs/pretrain_rte.yml` / `configs/finetune_rte.yml` for training hyperparameters — then evaluates results end-to-end.

---

## Overview

The agent applies JEPA-style self-supervised pretraining to RoBERTa, followed by supervised fine-tuning on the RTE (Recognizing Textual Entailment) task. Each experiment cycle involves:

1. Modifying the model architecture and/or loss function in `scripts/jepa.py`
2. Adjusting pretraining and fine-tuning configurations in the corresponding YAML files
3. Running the full pretraining → fine-tuning pipeline
4. Evaluating results and deciding whether to keep or discard the change

---

## Repository Structure

```
.
├── scripts/
│   └── jepa.py                  # Model architecture and loss functions (agent modifies this)
├── configs/
│   ├── pretrain_rte.yml         # Pretraining hyperparameters (agent modifies this)
│   └── finetune_rte.yml         # Fine-tuning hyperparameters (agent modifies this)
├── checkpoints/
│   └── pretrain/                # Saved pretraining checkpoints
├── results.tsv                  # Experiment log (tab-separated, untracked by git)
├── README.md
└── program.md                   # Agent operational instructions
```

---

## Architecture

The agent operates over two model classes defined in `scripts/jepa.py`:

### `JEPA`
A dual-encoder architecture inspired by the JEPA paper:
- **Online encoder**: Encodes masked/context inputs and passes hidden states through a lightweight predictor MLP.
- **Target encoder**: An EMA (Exponential Moving Average) copy of the online encoder. Parameters are not updated via backprop — only via EMA.
- **Predictor**: A network that maps context embeddings into the target embedding space. By default a two-layer MLP, but can be replaced with a more expressive architecture such as a Transformer (e.g. a small stack of self-attention layers), enabling the predictor to model token-level interactions when reconstructing target embeddings.
- **Loss**: Symmetric contrastive loss (`jepa_loss`) over normalized embeddings, optionally restricted to masked token positions.

### `MLMJEPA`
A BERT-style masked language model variant:
- Single encoder with a linear vocabulary projection head.
- No EMA target encoder.
- Loss: Cross-entropy over masked token positions only.

---

## Usage

### Running a Full Experiment

```bash
python -m scripts.pretrain --config configs/pretrain_rte.yml && \
python -m scripts.finetune \
  --config configs/finetune_rte.yml \
  --encoder_ckpt checkpoints/pretrain/best_pretrain_rte_exp1_.pt
```

### Agent Execution

Launch the agent to run experiments autonomously:

```bash
python agent.py
```

The agent will loop indefinitely, proposing changes, running experiments, and logging results to `results.tsv` until manually interrupted.

---

## Experiment Logging

Results are recorded in `results.tsv` (tab-separated). Do **not** commit this file.

| Column | Description |
|---|---|
| `commit` | Short git commit hash (7 chars) |
| `val_acc` | Validation accuracy on RTE after fine-tuning |
| `memory_gb` | Peak VRAM usage in GB |
| `status` | `keep`, `discard`, or `crash` |
| `description` | Brief description of the experiment |

Example:

```
commit	val_acc	memory_gb	status	description
a1b2c3d	0.6425	6.2	keep	baseline
b2c3d4e	0.6571	6.4	keep	increase EMA decay to 0.999
c3d4e5f	0.6310	6.2	discard	switch predictor to 3-layer MLP
```

---

## Constraints

- **Do not modify** `scripts/pretrain.py`, `scripts/finetune.py`, or the evaluation harness.
- **Do not add new dependencies.** Only packages already available in the environment may be used.
- The only files the agent is permitted to edit are `scripts/jepa.py`, `configs/pretrain_rte.yml`, and `configs/finetune_rte.yml`.

---

## References

- Assran et al., [*Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture*](https://arxiv.org/abs/2301.08243) (I-JEPA, 2023)
- Liu et al., [*RoBERTa: A Robustly Optimized BERT Pretraining Approach*](https://arxiv.org/abs/1907.11692) (2019)
- Wang et al., [*GLUE: A Multi-Task Benchmark and Analysis Platform for Natural Language Understanding*](https://arxiv.org/abs/1804.07461) — RTE task (2018)




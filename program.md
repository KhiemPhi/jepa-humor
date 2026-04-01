# JEPA-RTE Agent — Operational Program

This document defines the operational protocol for the JEPA-RTE experimentation agent. The agent autonomously modifies model architecture and training configuration to improve RoBERTa embeddings on the RTE task using a JEPA-style pretraining regime.

---

## Setup

Before beginning experimentation, complete the following steps:

1. **Agree on a run tag**: Propose a tag based on today's date (e.g. `mar31`). The branch `jepa-rte/<tag>` must not already exist.
2. **Create the branch**: `git checkout -b jepa-rte/<tag>` from current master.
3. **Read the in-scope files** for full context:
   - `README.md` — repository overview and constraints.
   - `scripts/jepa.py` — model architecture (`JEPA`, `MLMJEPA`) and loss functions. **This file may be modified.**
   - `configs/pretrain_rte.yml` — pretraining configuration. **This file may be modified.**
   - `configs/finetune_rte.yml` — fine-tuning configuration. **This file may be modified.**
4. **Initialize `results.tsv`**: Create the file with the header row only. The baseline will be recorded after the first run.
5. **Confirm setup** before proceeding to experimentation.

---

## In-Scope Files

The agent is permitted to modify **only** the following files:

| File | Purpose |
|---|---|
| `scripts/jepa.py` | Model architecture, loss functions, EMA logic |
| `configs/pretrain_rte.yml` | Pretraining hyperparameters (LR, batch size, masking ratio, EMA decay, epochs, etc.) |
| `configs/finetune_rte.yml` | Fine-tuning hyperparameters (LR, scheduler, epochs, frozen layers, etc.) |

**Do not modify** any other file, including training scripts, evaluation code, or data preparation utilities.

---

## Running an Experiment

Each experiment executes the full pretraining → fine-tuning pipeline:

```bash
python -m scripts.pretrain --config configs/pretrain_rte.yml && \
python -m scripts.finetune \
  --config configs/finetune_rte.yml \
  --encoder_ckpt checkpoints/pretrain/best_pretrain_rte_exp1_.pt
```

Redirect all output to a log file:

```bash
python -m scripts.pretrain --config configs/pretrain_rte.yml > run.log 2>&1 && \
python -m scripts.finetune \
  --config configs/finetune_rte.yml \
  --encoder_ckpt checkpoints/pretrain/best_pretrain_rte_exp1_.pt >> run.log 2>&1
```

Extract the key metric after completion:

```bash
grep "^val_acc:" run.log
```

---

## Experimentation Loop

**The goal is to maximize `val_acc` on RTE.** Once the experiment loop begins, the agent runs autonomously and does not pause to ask the user for input.

```
LOOP FOREVER:

1. Review the current git state (branch, last commit, results.tsv).
2. Propose an experimental change to one or more of the in-scope files.
3. Apply the change directly to the file(s).
4. git commit the change with a descriptive message.
5. Run the experiment and redirect output to run.log.
6. Extract val_acc: grep "^val_acc:" run.log
7. If grep returns nothing, the run crashed:
     - Run: tail -n 50 run.log
     - If the crash is trivial (typo, missing import), fix and re-run.
     - If the idea is fundamentally broken, log as "crash" and move on.
8. Record the result in results.tsv.
9. Decision:
     - val_acc improved → keep the commit, advance the branch.
     - val_acc equal or worse → git reset --hard HEAD~1 (revert to previous commit).
```

---

## What to Experiment With

The agent should explore changes across two dimensions:

### Architecture (`scripts/jepa.py`)
- Switching between `JEPA` (EMA dual-encoder + contrastive loss) and `MLMJEPA` (single encoder + MLM head)
- Predictor architecture — ranging from shallow MLPs to deep MLPs to full Transformer predictors (e.g. a small stack of `nn.TransformerEncoderLayer` blocks); a Transformer predictor can model token-level interactions across the sequence when reconstructing target embeddings, more faithfully reflecting the spirit of the original JEPA paper
- Predictor MLP depth, width, and activation functions (when using an MLP predictor)
- Predictor Transformer configuration: number of layers, number of attention heads, feedforward dimension, dropout
- EMA decay rate (`ema_decay`) — higher values slow target encoder updates
- Masking strategies passed into `forward()` (e.g. span masking vs. random token masking)
- Temperature scaling in `jepa_loss`
- Encoder layer freezing strategies (freeze all but last N transformer layers)
- Adding a projection head between encoder output and predictor input
- Combining JEPA contrastive loss with auxiliary MLM loss

### Training Configuration (`pretrain_rte.yml` / `finetune_rte.yml`)
- Learning rate and scheduler (warmup steps, cosine decay)
- Batch size
- Masking ratio (fraction of tokens masked for context vs. target)
- Number of pretraining epochs
- Number of fine-tuning epochs
- Weight decay and gradient clipping
- Frozen vs. unfrozen encoder layers during fine-tuning

---

## Simplicity Criterion

All else being equal, prefer simpler solutions. Specifically:

- A marginal improvement that adds significant complexity is generally not worth keeping.
- Removing code that yields equivalent or better performance is always a win.
- When in doubt, favor the approach that is easier to reason about.

---

## Logging Results

Log every completed experiment to `results.tsv` (tab-separated). Do **not** commit this file to git.

**Header:**
```
commit	val_acc	memory_gb	status	description
```

**Columns:**

| Column | Format | Notes |
|---|---|---|
| `commit` | 7-char hex | Short git hash |
| `val_acc` | `0.000000` | RTE validation accuracy; use `0.000000` for crashes |
| `memory_gb` | `0.0` | Peak VRAM in GB (divide MB by 1024); use `0.0` for crashes |
| `status` | string | One of: `keep`, `discard`, `crash` |
| `description` | string | Brief description — no commas, no tabs |

**Example:**
```
commit	val_acc	memory_gb	status	description
a1b2c3d	0.642500	6.2	keep	baseline
b2c3d4e	0.657100	6.4	keep	increase EMA decay to 0.999
c3d4e5f	0.631000	6.2	discard	switch predictor to 3-layer MLP
d4e5f6g	0.000000	0.0	crash	double predictor width (OOM)
```

---

## Failure Handling

| Situation | Action |
|---|---|
| Crash (OOM, bug) — trivially fixable | Fix and re-run |
| Crash — fundamentally broken idea | Log as `crash`, revert, move on |
| Run exceeds 30 minutes | Kill process, treat as crash |
| No improvement after many iterations | Try more radical architectural changes; re-read reference papers |

---

## Autonomy Policy

Once experimentation begins, the agent **does not stop** to ask the user for guidance. The agent continues running experiments indefinitely until manually interrupted. If ideas are exhausted, the agent should:

- Re-read `scripts/jepa.py` for overlooked opportunities
- Revisit near-miss experiments from `results.tsv` and explore adjacent variations
- Consider combining previously successful individual changes
- Consult the I-JEPA and RoBERTa papers for architectural inspiration

---

## References

- Assran et al., *Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture* (I-JEPA, 2023) — https://arxiv.org/abs/2301.08243
- Liu et al., *RoBERTa: A Robustly Optimized BERT Pretraining Approach* (2019) — https://arxiv.org/abs/1907.11692
- Wang et al., *GLUE Benchmark — RTE Task* (2018) — https://arxiv.org/abs/1804.07461

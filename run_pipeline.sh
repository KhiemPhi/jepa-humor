#!/bin/bash
# Pipeline:
#   Phase 1 — pretrain+finetune MNLI/QQP/QNLI/SST2 in parallel (GPUs 0-3).
#             MNLI signals a flag when its finetune completes.
#   Phase 2 — finetune RTE/MRPC/STS-B from the MNLI checkpoint (GPUs 4-6),
#             starts as soon as MNLI is ready.
#   Phase 3 — eval all 7 tasks and write GLUE submission TSVs.
set -euo pipefail

mkdir -p logs checkpoints/pretrain checkpoints/finetune glue_submission

MNLI_FT_CKPT=checkpoints/finetune/finetune_mnli_exp1_FacebookAI_roberta_large_best_test.pt
MNLI_DONE=/tmp/mnli_ft_done_$$
rm -f "$MNLI_DONE"

# ── Phase 1: pretrain → finetune (parallel) ───────────────────────────────────

# MNLI — GPU 0; touches flag when finetune finishes so phase 2 can start early
(
  python -m scripts.pretrain --config configs/pretrain_mnli.yml --gpu_id 0 && \
  python -m scripts.finetune --config configs/finetune_mnli.yml \
    --encoder_ckpt checkpoints/pretrain/best_pretrain_mnli_exp1_FacebookAI_roberta_large_.pt \
    --gpu_id 0 && \
  touch "$MNLI_DONE"
) 2>&1 | tee logs/mnli.log &

# SST2 — GPU 1
(
  python -m scripts.pretrain --config configs/pretrain_sst2.yml --gpu_id 1 && \
  python -m scripts.finetune --config configs/finetune_sst2.yml \
    --encoder_ckpt checkpoints/pretrain/best_pretrain_sst2_exp1_FacebookAI_roberta_large_.pt \
    --gpu_id 1
) 2>&1 | tee logs/sst2.log &

# QNLI — GPU 2
(
  python -m scripts.pretrain --config configs/pretrain_qnli.yml --gpu_id 2 && \
  python -m scripts.finetune --config configs/finetune_qnli.yml \
    --encoder_ckpt checkpoints/pretrain/best_pretrain_qnli_exp1_FacebookAI_roberta_large_.pt \
    --gpu_id 2
) 2>&1 | tee logs/qnli.log &

# QQP — GPU 3
(
  python -m scripts.pretrain --config configs/pretrain_qqp.yml --gpu_id 3 && \
  python -m scripts.finetune --config configs/finetune_qqp.yml \
    --encoder_ckpt checkpoints/pretrain/best_pretrain_qqp_exp1_FacebookAI_roberta_large_.pt \
    --gpu_id 3
) 2>&1 | tee logs/qqp.log &

# ── Phase 2: finetune RTE/MRPC/STS-B from MNLI checkpoint ────────────────────
# Each waits for MNLI finetune to finish, then starts immediately.

(
  until [ -f "$MNLI_DONE" ]; do sleep 30; done
  python -m scripts.finetune --config configs/finetune_rte.yml \
    --encoder_ckpt "$MNLI_FT_CKPT" --gpu_id 4
) 2>&1 | tee logs/rte.log &

(
  until [ -f "$MNLI_DONE" ]; do sleep 30; done
  python -m scripts.finetune --config configs/finetune_mrpc.yml \
    --encoder_ckpt "$MNLI_FT_CKPT" --gpu_id 5
) 2>&1 | tee logs/mrpc.log &

(
  until [ -f "$MNLI_DONE" ]; do sleep 30; done
  python -m scripts.finetune_stsb --config configs/finetune_stsb.yml \
    --encoder_ckpt "$MNLI_FT_CKPT" --gpu_id 6
) 2>&1 | tee logs/stsb.log &

wait
rm -f "$MNLI_DONE"

# ── Phase 3: eval all 7 tasks → GLUE submission TSVs ─────────────────────────

python -m scripts.eval --config configs/finetune_mnli.yml \
  --ckpt checkpoints/finetune/finetune_mnli_exp1_FacebookAI_roberta_large_best_test.pt \
  --gpu_id 0 --output_dir glue_submission 2>&1 | tee logs/eval_mnli.log &

python -m scripts.eval --config configs/finetune_sst2.yml \
  --ckpt checkpoints/finetune/finetune_sst2_exp1_FacebookAI_roberta_large_best_test.pt \
  --gpu_id 1 --output_dir glue_submission 2>&1 | tee logs/eval_sst2.log &

python -m scripts.eval --config configs/finetune_qnli.yml \
  --ckpt checkpoints/finetune/finetune_qnli_exp1_FacebookAI_roberta_large_best_test.pt \
  --gpu_id 2 --output_dir glue_submission 2>&1 | tee logs/eval_qnli.log &

python -m scripts.eval --config configs/finetune_qqp.yml \
  --ckpt checkpoints/finetune/finetune_qqp_exp1_FacebookAI_roberta_large_best_test.pt \
  --gpu_id 3 --output_dir glue_submission 2>&1 | tee logs/eval_qqp.log &

python -m scripts.eval --config configs/finetune_rte.yml \
  --ckpt checkpoints/finetune/finetune_rte_exp1_roberta_large_best_test.pt \
  --gpu_id 4 --output_dir glue_submission 2>&1 | tee logs/eval_rte.log &

python -m scripts.eval --config configs/finetune_mrpc.yml \
  --ckpt checkpoints/finetune/finetune_mrpc_exp1_roberta_large_best_val.pt \
  --gpu_id 5 --output_dir glue_submission 2>&1 | tee logs/eval_mrpc.log &

python -m scripts.eval --config configs/finetune_stsb.yml \
  --ckpt checkpoints/finetune/finetune_stsb_exp1_roberta_large_best_val.pt \
  --gpu_id 6 --output_dir glue_submission 2>&1 | tee logs/eval_stsb.log &

wait
echo "Done. GLUE submission TSVs in glue_submission/"

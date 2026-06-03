#!/bin/bash
# Fine-tune RTE / STS-B / MRPC from the MNLI best-val checkpoint,
# then run eval.py for all 7 GLUE tasks to produce submission TSVs.
#
# Run from the project root:  bash run_mnli_transfer_and_submit.sh
set -euo pipefail

GPU=0          # GPU to use for all steps
OUT="glue_submission"

MNLI_CKPT="checkpoints/finetune/finetune_mnli_exp1_FacebookAI_roberta_large_best_val.pt"
RTE_CKPT="checkpoints/finetune/finetune_rte_exp1_roberta_large_best_val.pt"
STSB_CKPT="checkpoints/finetune/finetune_stsb_exp1_roberta_large_best.pt"
MRPC_CKPT="checkpoints/finetune/finetune_mrpc_exp1_roberta_large_best_val.pt"

mkdir -p logs "$OUT"

# ── 1. Fine-tune RTE from MNLI ───────────────────────────────────────────────
echo "=== [1/3] Fine-tuning RTE from MNLI best-val ==="
python -m scripts.finetune \
    --config configs/finetune_rte.yml \
    --encoder_ckpt "$MNLI_CKPT" \
    --gpu_id $GPU \
    2>&1 | tee logs/rte_mnli_transfer.log

# ── 2. Fine-tune STS-B from MNLI ─────────────────────────────────────────────
echo ""
echo "=== [2/3] Fine-tuning STS-B from MNLI best-val ==="
python -m scripts.finetune_stsb \
    --config configs/finetune_stsb.yml \
    --encoder_ckpt "$MNLI_CKPT" \
    --gpu_id $GPU \
    2>&1 | tee logs/stsb_mnli_transfer.log

# ── 3. Fine-tune MRPC from MNLI ──────────────────────────────────────────────
echo ""
echo "=== [3/3] Fine-tuning MRPC from MNLI best-val ==="
python -m scripts.finetune \
    --config configs/finetune_mrpc.yml \
    --encoder_ckpt "$MNLI_CKPT" \
    --gpu_id $GPU \
    2>&1 | tee logs/mrpc_mnli_transfer.log

# ── 4. GLUE submission TSVs ───────────────────────────────────────────────────
echo ""
echo "=== Generating GLUE submission TSVs → $OUT ==="

python -m scripts.eval \
    --config configs/finetune_sst2.yml \
    --ckpt checkpoints/finetune/finetune_sst2_exp1_FacebookAI_roberta_large_best_val.pt \
    --gpu_id $GPU --output_dir "$OUT" \
    2>&1 | tee logs/eval_sst2.log

python -m scripts.eval \
    --config configs/finetune_mnli.yml \
    --ckpt "$MNLI_CKPT" \
    --gpu_id $GPU --output_dir "$OUT" \
    2>&1 | tee logs/eval_mnli.log

python -m scripts.eval \
    --config configs/finetune_qnli.yml \
    --ckpt checkpoints/finetune/finetune_qnli_exp1_FacebookAI_roberta_large_best_val.pt \
    --gpu_id $GPU --output_dir "$OUT" \
    2>&1 | tee logs/eval_qnli.log

python -m scripts.eval \
    --config configs/finetune_qqp.yml \
    --ckpt checkpoints/finetune/finetune_qqp_exp1_FacebookAI_roberta_large_best_val.pt \
    --gpu_id $GPU --output_dir "$OUT" \
    2>&1 | tee logs/eval_qqp.log

python -m scripts.eval \
    --config configs/finetune_rte.yml \
    --ckpt "$RTE_CKPT" \
    --gpu_id $GPU --output_dir "$OUT" \
    2>&1 | tee logs/eval_rte.log

python -m scripts.eval \
    --config configs/finetune_stsb.yml \
    --ckpt "$STSB_CKPT" \
    --gpu_id $GPU --output_dir "$OUT" \
    2>&1 | tee logs/eval_stsb.log

python -m scripts.eval \
    --config configs/finetune_mrpc.yml \
    --ckpt "$MRPC_CKPT" \
    --gpu_id $GPU --output_dir "$OUT" \
    2>&1 | tee logs/eval_mrpc.log

echo ""
echo "✅ Done. GLUE submission TSVs are in $OUT/"
ls "$OUT"/*.tsv

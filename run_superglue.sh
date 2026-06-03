#!/bin/bash
# run_superglue.sh
# Runs all SuperGLUE tasks except boolq, cb, copa, multirc, wic.
#   - RTE, WSC, ReCoRD : pretrain then finetune
#   - AX-b, AX-g       : eval-only using the MNLI checkpoint
set -e

GPU=${GPU:-7}
MNLI_CKPT="billionare_file/checkpoints_roberta_large/finetune/finetune_mnli_exp1_FacebookAI_roberta_large_best.pt"

mkdir -p logs

# ── 1. SuperGLUE RTE ──────────────────────────────────────────────────────────
echo "===== [1/5] SuperGLUE RTE: pretrain ====="
python -m scripts.pretrain \
    --config configs/pretrain_superglue_rte.yml \
    --gpu_id $GPU \
    2>&1 | tee logs/pretrain_superglue_rte.log

RTE_ENCODER="checkpoints/pretrain/best_pretrain_superglue_rte_exp1_FacebookAI_roberta_large_.pt"
echo "===== [1/5] SuperGLUE RTE: finetune ====="
python -m scripts.finetune_superglue \
    --config configs/finetune_superglue_rte.yml \
    --encoder_ckpt $RTE_ENCODER \
    --gpu_id $GPU \
    2>&1 | tee logs/finetune_superglue_rte.log

# ── 2. WSC ────────────────────────────────────────────────────────────────────
echo "===== [2/5] WSC: pretrain ====="
python -m scripts.pretrain \
    --config configs/pretrain_wsc.yml \
    --gpu_id $GPU \
    2>&1 | tee logs/pretrain_wsc.log

WSC_ENCODER="checkpoints/pretrain/best_pretrain_wsc_exp1_FacebookAI_roberta_large_.pt"
echo "===== [2/5] WSC: finetune ====="
python -m scripts.finetune_superglue \
    --config configs/finetune_wsc.yml \
    --encoder_ckpt $WSC_ENCODER \
    --gpu_id $GPU \
    2>&1 | tee logs/finetune_wsc.log

# ── 3. ReCoRD ─────────────────────────────────────────────────────────────────
echo "===== [3/5] ReCoRD: pretrain ====="
python -m scripts.pretrain \
    --config configs/pretrain_rccr.yml \
    --gpu_id $GPU \
    2>&1 | tee logs/pretrain_rccr.log

RCCR_ENCODER="checkpoints/pretrain/best_pretrain_rccr_exp1_FacebookAI_roberta_large_.pt"
echo "===== [3/5] ReCoRD: finetune ====="
python -m scripts.finetune_superglue \
    --config configs/finetune_rccr.yml \
    --encoder_ckpt $RCCR_ENCODER \
    --gpu_id $GPU \
    2>&1 | tee logs/finetune_rccr.log

# ── 4. AX-b (eval only) ───────────────────────────────────────────────────────
echo "===== [4/5] AX-b: eval only ====="
python -m scripts.eval_superglue \
    --config configs/eval_axb.yml \
    --ckpt $MNLI_CKPT \
    --gpu_id $GPU \
    --output_dir superglue_results \
    2>&1 | tee logs/eval_axb.log

# ── 5. AX-g (eval only) ───────────────────────────────────────────────────────
echo "===== [5/5] AX-g: eval only ====="
python -m scripts.eval_superglue \
    --config configs/eval_axg.yml \
    --ckpt $MNLI_CKPT \
    --gpu_id $GPU \
    --output_dir superglue_results \
    2>&1 | tee logs/eval_axg.log

echo "===== All SuperGLUE tasks complete ====="
echo "Results:"
echo "  Finetune logs  : logs/"
echo "  AX-b / AX-g    : superglue_results/"
echo "  Checkpoints    : checkpoints/finetune/"
echo "  Training plots : results/"

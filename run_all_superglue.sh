#!/bin/bash
# =============================================================================
# run_all_superglue.sh
#
# SuperGLUE pipeline: finetune → eval for all tasks (pretrain step removed).
# Each task finetunes from its EXISTING pretrain checkpoint in
# checkpoints/pretrain/ (the pretrain step is assumed already done).
# After completion, roberta_large_superglue/ contains all TSV files ready
# for SuperGLUE leaderboard submission.
#
# Usage:
#   bash run_all_superglue.sh          # uses GPU=7
#   GPU=0 bash run_all_superglue.sh    # override GPU
# =============================================================================
set -euo pipefail

GPU=${GPU:-7}
CKPT_P="checkpoints/pretrain"
OUT="roberta_large_superglue"

mkdir -p logs "$OUT"

step() { echo; echo "════════════════════════════════════════════════"; echo "  $*"; echo "════════════════════════════════════════════════"; }

# ── Helper: finetune a task from its existing pretrain checkpoint ─────────────
run_task() {
    local TASK=$1
    local FINETUNE_CFG=$2
    local ENCODER_CKPT=$3   # path to existing pretrain output checkpoint

    step "[$TASK] Finetune"
    python -m scripts.finetune_superglue \
        --config "$FINETUNE_CFG" \
        --encoder_ckpt "$ENCODER_CKPT" \
        --gpu_id $GPU \
        2>&1 | tee "logs/finetune_${TASK}.log"
}

# =============================================================================
# 1. BoolQ
# =============================================================================
run_task boolq   configs/finetune_boolq.yml \
    "${CKPT_P}/best_pretrain_boolq_exp1_FacebookAI_roberta_large_.pt"

# =============================================================================
# 2. CB
# =============================================================================
run_task cb      configs/finetune_cb.yml \
    "${CKPT_P}/best_pretrain_cb_exp1_FacebookAI_roberta_large_.pt"

# =============================================================================
# 3. COPA
# =============================================================================
run_task copa    configs/finetune_copa.yml \
    "${CKPT_P}/best_pretrain_copa_exp1_FacebookAI_roberta_large_.pt"

# =============================================================================
# 4. MultiRC
# =============================================================================
run_task multirc configs/finetune_multirc.yml \
    "${CKPT_P}/best_pretrain_multirc_exp1_FacebookAI_roberta_large_.pt"

# =============================================================================
# 5. WiC  (uses finetune_superglue_wic — word-in-context tokenisation)
# =============================================================================
run_task wic     configs/finetune_wic.yml \
    "${CKPT_P}/best_pretrain_wic_exp1_FacebookAI_roberta_large_.pt"

# =============================================================================
# 6. WSC  (MLM scoring — uses finetune_wnli internally)
# =============================================================================
run_task wsc     configs/finetune_wsc.yml \
    "${CKPT_P}/best_pretrain_wsc_exp1_FacebookAI_roberta_large_.pt"

# =============================================================================
# 7. RTE  (binary NLI on super_glue/rte)
# =============================================================================
run_task rte     configs/finetune_superglue_rte.yml \
    "${CKPT_P}/best_pretrain_superglue_rte_exp1_FacebookAI_roberta_large_.pt"

# =============================================================================
# 8. ReCoRD  (entity-scoring reading comprehension)
# =============================================================================
run_task record  configs/finetune_rccr.yml \
    "${CKPT_P}/best_pretrain_rccr_exp1_FacebookAI_roberta_large_.pt"

# =============================================================================
# 9+10. AX-b and AX-g  (eval-only — zero-shot from MNLI checkpoint)
# =============================================================================
# No pretrain / finetune needed; the MNLI model is already available.

# =============================================================================
# EVAL: all 10 tasks → roberta_large_superglue/
# =============================================================================
step "Evaluating all tasks → $OUT/"
python -m scripts.eval_superglue_tasks \
    --gpu_id $GPU \
    --output_dir "$OUT" \
    2>&1 | tee "logs/eval_all_superglue.log"

# =============================================================================
# Summary
# =============================================================================
step "Done"
echo "Leaderboard TSVs:"
ls -1 "${OUT}/"*.tsv 2>/dev/null | sed 's/^/  /'
echo
echo "Validation metrics:"
for f in "${OUT}"/*_eval.json; do
    task=$(basename "$f" _eval.json)
    python3 -c "
import json, sys
d = json.load(open('$f'))
val = d.get('validation') or d.get('test_metrics') or {}
metrics = '  '.join(f\"{k}={v:.4f}\" for k,v in val.items() if isinstance(v,float))
print(f'  {\"$task\":<10} {metrics}')
" 2>/dev/null
done

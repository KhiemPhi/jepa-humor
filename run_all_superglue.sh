#!/bin/bash
# =============================================================================
# run_all_superglue.sh
#
# Full SuperGLUE pipeline: pretrain → finetune → eval for all 10 tasks.
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
MNLI_CKPT="checkpoints/finetune/finetune_mnli_exp1_FacebookAI_roberta_large_best_test.pt"
OUT="roberta_large_superglue"

mkdir -p logs "$OUT"

step() { echo; echo "════════════════════════════════════════════════"; echo "  $*"; echo "════════════════════════════════════════════════"; }

# ── Helper: pretrain then finetune a task ─────────────────────────────────────
run_task() {
    local TASK=$1
    local PRETRAIN_CFG=$2
    local FINETUNE_CFG=$3
    local ENCODER_CKPT=$4   # path to pretrain output checkpoint

    step "[$TASK] Pretrain"
    python -m scripts.pretrain \
        --config "$PRETRAIN_CFG" \
        --gpu_id $GPU \
        2>&1 | tee "logs/pretrain_${TASK}.log"

    step "[$TASK] Finetune"
    python -m scripts.finetune_superglue \
        --config "$FINETUNE_CFG" \
        --encoder_ckpt "$ENCODER_CKPT" \
        --gpu_id $GPU \
        2>&1 | tee "logs/finetune_${TASK}.log"
}


# =============================================================================
# 8. ReCoRD  (entity-scoring reading comprehension) — pretrain already done
# =============================================================================
step "[record] Finetune"
python -m scripts.finetune_superglue \
    --config configs/finetune_rccr.yml \
    --encoder_ckpt "${CKPT_P}/best_pretrain_rccr_exp1_FacebookAI_roberta_large_.pt" \
    --gpu_id $GPU \
    2>&1 | tee "logs/finetune_record.log"

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

#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# run_superglue_sweep.sh
#
# SuperGLUE finetune hyperparameter sweep over (batch_size × learning_rate).
#
#     batch_size ∈ {8, 16, 32, 64}   ×   lr ∈ {1e-5, 2e-5, 3e-5}   → 12 combos.
#
# Two-phase execution:
#   PHASE 1 — all non-ReCoRD tasks (light→heavy by update budget) across all
#             12 combos.  Ordering: WSC, CB, RTE, COPA, WiC, BoolQ, MultiRC.
#   PHASE 2 — ReCoRD across all 12 combos, then eval_superglue_tasks per combo.
#
# "Hold epochs constant" across batch sizes — update budget rescaled so every
# task trains the same number of epochs at each batch size:
#   * Update-budget tasks (boolq, cb, copa, multirc, wic, rte; base bs=16):
#       total_num_update(bs) = round(base × 16 / bs);  warmup scaled the same.
#   * WSC (finetune_wnli; fixed 2000-update budget at base bs=8):
#       total_num_updates(bs) = round(2000 × 8 / bs);  warmup is ratio-based.
#   * ReCoRD (epochs-based, base bs=32): budget auto-scales with batch size
#       (epochs=10 held); only its fixed warmup is rescaled (× 32/bs).
#
# Per-combination outputs namespaced by bs<bs>_lr<lr>:
#   checkpoints/finetune_superglue_bs8_lr1e-5/
#   results_superglue_bs8_lr1e-5/
#   superglue_submission_bs8_lr1e-5/
# Per-task logs: logs/sg_<combo>_<task>.log  and  logs/sg_<combo>_eval.log
#
# Resumable: a task whose checkpoint already exists is skipped; a combo whose
# 8 submission TSVs already exist is skipped in Phase 2.
#
# Usage:
#   bash run_superglue_sweep.sh
#   GPU=0 bash run_superglue_sweep.sh
# ─────────────────────────────────────────────────────────────────────────────
set -uo pipefail
cd "$(dirname "$0")"

GPU=${GPU:-7}
PY=python
export PYTHONUNBUFFERED=1

mkdir -p logs

# ── Pretrain encoders (combo-independent; assumed already trained) ────────────
CKPT_P=checkpoints/pretrain
PRE_BOOLQ=$CKPT_P/best_pretrain_boolq_exp1_FacebookAI_roberta_large_.pt
PRE_CB=$CKPT_P/best_pretrain_cb_exp1_FacebookAI_roberta_large_.pt
PRE_COPA=$CKPT_P/best_pretrain_copa_exp1_FacebookAI_roberta_large_.pt
PRE_MULTIRC=$CKPT_P/best_pretrain_multirc_exp1_FacebookAI_roberta_large_.pt
PRE_WIC=$CKPT_P/best_pretrain_wic_exp1_FacebookAI_roberta_large_.pt
PRE_WSC=$CKPT_P/best_pretrain_wsc_exp1_FacebookAI_roberta_large_.pt
PRE_RTE=$CKPT_P/best_pretrain_superglue_rte_exp1_FacebookAI_roberta_large_.pt
PRE_RECORD=$CKPT_P/best_pretrain_rccr_exp1_FacebookAI_roberta_large_.pt

for ck in "$PRE_BOOLQ" "$PRE_CB" "$PRE_COPA" "$PRE_MULTIRC" \
          "$PRE_WIC" "$PRE_WSC" "$PRE_RTE" "$PRE_RECORD"; do
    [ -f "$ck" ] || { echo "FATAL: missing pretrain checkpoint: $ck"; exit 1; }
done

scale() {  # base base_bs BS  → round(base × base_bs / BS)
    echo $(( ($1 * $2 + $3 / 2) / $3 ))
}

# ft <task> <config> <encoder_ckpt> <ckpt_basename> [key=value overrides...]
# Uses globals: TAG CKPT_DIR RES_DIR LR BS GPU
ft() {
    local task=$1 cfg=$2 enc=$3 ckname=$4; shift 4
    if [ -f "$CKPT_DIR/$ckname" ]; then
        echo "[skip ft $task]  checkpoint exists → $CKPT_DIR/$ckname"
        return 0
    fi
    echo "──────── [$TAG] finetune $task ────────"
    $PY -m scripts.finetune_superglue \
        --config "$cfg" --encoder_ckpt "$enc" --gpu_id "$GPU" \
        ckpt_dir="$CKPT_DIR" results_dir="$RES_DIR" lr="$LR" batch_size="$BS" "$@" \
        2>&1 | tee "logs/sg_${TAG}_${task}.log"
}

BATCH_SIZES=(8 16 32 64)
LR_NUMS=(1 2 3)        # ×e-5
SUB_TSVS=(BoolQ.tsv CB.tsv COPA.tsv MultiRC.tsv WiC.tsv WSC.tsv RTE.tsv ReCoRD.tsv)

# ── PHASE 1: all non-ReCoRD tasks, all combos, light → heavy ─────────────────
echo "==================== PHASE 1: non-ReCoRD tasks, all 12 combos ===================="
for BS in "${BATCH_SIZES[@]}"; do for LRN in "${LR_NUMS[@]}"; do
    LR="${LRN}.0e-5"
    TAG="bs${BS}_lr${LRN}e-5"
    CKPT_DIR="checkpoints/finetune_superglue_${TAG}"
    RES_DIR="results_superglue_${TAG}"
    SUB_DIR="superglue_submission_${TAG}"
    mkdir -p "$CKPT_DIR" "$RES_DIR" "$SUB_DIR"

    echo "── [PHASE 1] $TAG : non-ReCoRD tasks ──"

    # Ordered light → heavy by update budget at base bs
    ft wsc     configs/finetune_wsc.yml           "$PRE_WSC"     finetune_wsc_exp1_FacebookAI_roberta_large_best.pt \
        total_num_updates=$(scale 2000 8 "$BS")
    ft cb      configs/finetune_cb.yml            "$PRE_CB"      finetune_cb_exp1_FacebookAI_roberta_large_best.pt \
        total_num_update=$(scale 1875 16 "$BS")   warmup_updates=$(scale 113 16 "$BS")
    ft rte     configs/finetune_superglue_rte.yml "$PRE_RTE"     finetune_superglue_rte_exp1_FacebookAI_roberta_large_rte_best.pt \
        total_num_update=$(scale 2036 16 "$BS")   warmup_updates=$(scale 122 16 "$BS")
    ft copa    configs/finetune_copa.yml          "$PRE_COPA"    finetune_copa_exp1_FacebookAI_roberta_large_copa_best.pt \
        total_num_update=$(scale 3000 16 "$BS")   warmup_updates=$(scale 150 16 "$BS")
    ft wic     configs/finetune_wic.yml           "$PRE_WIC"     finetune_wic_exp1_FacebookAI_roberta_large_best.pt \
        total_num_update=$(scale 3400 16 "$BS")   warmup_updates=$(scale 204 16 "$BS")
    ft boolq   configs/finetune_boolq.yml         "$PRE_BOOLQ"   finetune_boolq_exp1_FacebookAI_roberta_large_best.pt \
        total_num_update=$(scale 5900 16 "$BS")   warmup_updates=$(scale 354 16 "$BS")
    ft multirc configs/finetune_multirc.yml       "$PRE_MULTIRC" finetune_multirc_exp1_FacebookAI_roberta_large_multirc_best.pt \
        total_num_update=$(scale 17000 16 "$BS")  warmup_updates=$(scale 1020 16 "$BS")

    echo "── [PHASE 1] $TAG done ──"
done; done

# ── PHASE 2: ReCoRD for all combos, then eval per combo ──────────────────────
echo "==================== PHASE 2: ReCoRD + eval, all 12 combos ===================="
for BS in "${BATCH_SIZES[@]}"; do for LRN in "${LR_NUMS[@]}"; do
    LR="${LRN}.0e-5"
    TAG="bs${BS}_lr${LRN}e-5"
    CKPT_DIR="checkpoints/finetune_superglue_${TAG}"
    RES_DIR="results_superglue_${TAG}"
    SUB_DIR="superglue_submission_${TAG}"
    mkdir -p "$CKPT_DIR" "$RES_DIR" "$SUB_DIR"

    # Skip if all 8 submission TSVs already exist
    done_combo=1
    for t in "${SUB_TSVS[@]}"; do [ -f "$SUB_DIR/$t" ] || done_combo=0; done
    if [ "$done_combo" = 1 ]; then
        echo "==================== COMBO $TAG already complete → $SUB_DIR (skip) ===================="
        continue
    fi

    echo "==================== COMBO $TAG : ReCoRD + eval on GPU $GPU ===================="

    ft record  configs/finetune_rccr.yml  "$PRE_RECORD"  finetune_rccr_exp1_FacebookAI_roberta_large_record_best.pt \
        warmup_updates=$(scale 1800 32 "$BS")

    echo "==================== COMBO $TAG : SuperGLUE submission (eval, ax excluded) ===================="
    $PY -m scripts.eval_superglue_tasks \
        --gpu_id "$GPU" --ckpt_dir "$CKPT_DIR" --output_dir "$SUB_DIR" \
        --tasks boolq cb copa multirc wic wsc rte record \
        2>&1 | tee "logs/sg_${TAG}_eval.log"

    echo "==================== COMBO $TAG COMPLETE → $SUB_DIR ===================="
    ls "$SUB_DIR"/*.tsv 2>/dev/null | sed 's/^/  /' || true
done; done

echo
echo "ALL 12 COMBINATIONS COMPLETE."
echo "Submissions: superglue_submission_bs{8,16,32,64}_lr{1,2,3}e-5/"

#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# GLUE finetune hyperparameter sweep over (learning_rate × batch_size).
#
# Grid: batch_size ∈ {16, 32} × lr ∈ {1e-5, 2e-5, 3e-5}  → 6 combinations.
# Combinations run SEQUENTIALLY. Each combination uses exactly 4 GPUs (ids 0-3),
# modelled as 4 task QUEUES — one per GPU. The fast tasks (CoLA, WNLI) finish
# first and are replaced on their GPU by SST2 / QNLI:
#
#   GPU 0:  MNLI ───────────────▶ RTE        (RTE after MNLI's best_val)
#   GPU 1:  QQP  ───────────────▶ MRPC       (MRPC waits for MNLI's best_val)
#   GPU 2:  CoLA ▶ SST2 ────────▶ STSB       (STSB waits for MNLI's best_val)
#   GPU 3:  WNLI ▶ QNLI
#
# RTE/MRPC/STSB are fine-tuned FROM this combo's MNLI best_val checkpoint
# (gated on a flag MNLI raises the moment that checkpoint is written).
# Phase 3 then runs eval.py → GLUE submission TSVs for every task, incl. AX.
#
# Pretrain checkpoints for MNLI/QQP/QNLI/SST2 already exist and are reused.
# CoLA and WNLI have no pretrain checkpoint, so they are pretrained ONCE up
# front (pretraining is independent of the finetune lr/batch grid) and reused.
#
# Outputs are namespaced per combination by  bs<batch>_lr<lr>:
#   checkpoints/finetune_bs16_lr1e-5/ , results_bs16_lr1e-5/ , glue_submission_bs16_lr1e-5/
# Per-task logs: logs/<combo>_<task>.log  and  logs/<combo>_eval_<task>.log
#
# Usage (one line, from project root):
#   bash run_glue_sweep.sh
# ─────────────────────────────────────────────────────────────────────────────
set -uo pipefail
cd "$(dirname "$0")"
mkdir -p logs checkpoints/pretrain

PY=python
export PYTHONUNBUFFERED=1   # stream prints/tqdm live (Python block-buffers stdout when piped)

# ── Smoke-test toggle ────────────────────────────────────────────────────────
# SMOKE=1 runs the WHOLE pipeline on a tiny sample so it finishes in ~minutes,
# to validate wiring before the real multi-day run. It is fully namespaced
# (smoke_* dirs, *_smoke pretrain ckpts) so it never collides with real outputs.
#   SMOKE=1 bash run_glue_sweep.sh
# Tunables (defaults shown):  SMOKE_PCT=2 (finetune %)   SMOKE_LIMIT=64 (eval rows)
SMOKE="${SMOKE:-0}"
SMOKE_PCT="${SMOKE_PCT:-2}"
SMOKE_LIMIT="${SMOKE_LIMIT:-64}"

EXTRA_FT=()        # extra finetune overrides
EVAL_LIMIT=()      # extra eval args
PRE_EXTRA=()       # extra pretrain overrides (cola/wnli)
PRE_SFX=""         # pretrain experiment_name suffix (cola/wnli) — isolates smoke ckpts
TAGPRE=""          # output-dir tag prefix
if [ "$SMOKE" = "1" ]; then
    echo "### SMOKE TEST: finetune sample_percent=${SMOKE_PCT}, epochs=1; eval --limit ${SMOKE_LIMIT} ###"
    EXTRA_FT=(sample_percent="$SMOKE_PCT" epochs=1 total_num_updates=20)
    EVAL_LIMIT=(--limit "$SMOKE_LIMIT")
    PRE_EXTRA=(epochs=1)
    PRE_SFX="_smoke"
    TAGPRE="smoke_"
fi

# ── Pretrain encoders (combo-independent) ────────────────────────────────────
PRE_MNLI=checkpoints/pretrain/best_pretrain_mnli_exp1_FacebookAI_roberta_large_.pt
PRE_QQP=checkpoints/pretrain/best_pretrain_qqp_exp1_FacebookAI_roberta_large_.pt
PRE_QNLI=checkpoints/pretrain/best_pretrain_qnli_exp1_FacebookAI_roberta_large_.pt
PRE_SST2=checkpoints/pretrain/best_pretrain_sst2_exp1_FacebookAI_roberta_large_.pt
# CoLA/WNLI are pretrained here; smoke runs use *_smoke ckpts so they never clobber real ones.
COLA_EXP="pretrain_cola_exp1${PRE_SFX}"
WNLI_EXP="pretrain_wnli_exp1${PRE_SFX}"
PRE_COLA="checkpoints/pretrain/best_${COLA_EXP}_roberta_large_.pt"
PRE_WNLI="checkpoints/pretrain/best_${WNLI_EXP}_roberta_large_.pt"

# Pretrain CoLA & WNLI once if their checkpoints are missing (GPUs 0,1).
if [ ! -f "$PRE_COLA" ] || [ ! -f "$PRE_WNLI" ]; then
    echo "==================== Pretraining CoLA & WNLI (one-time) ===================="
    [ -f "$PRE_COLA" ] || $PY -m scripts.pretrain --config configs/pretrain_cola.yml --gpu_id 0 experiment_name="$COLA_EXP" "${PRE_EXTRA[@]}" 2>&1 | tee "logs/pretrain_cola${PRE_SFX}.log" &
    [ -f "$PRE_WNLI" ] || $PY -m scripts.pretrain --config configs/pretrain_wnli.yml --gpu_id 1 experiment_name="$WNLI_EXP" "${PRE_EXTRA[@]}" 2>&1 | tee "logs/pretrain_wnli${PRE_SFX}.log" &
    wait
fi

# Preflight: every encoder we depend on must exist before the sweep starts.
for ck in "$PRE_MNLI" "$PRE_QQP" "$PRE_QNLI" "$PRE_SST2" "$PRE_COLA" "$PRE_WNLI"; do
    [ -f "$ck" ] || { echo "FATAL: missing pretrain checkpoint: $ck"; exit 1; }
done

# ── Helpers (use loop globals OVR, TAG, SUB_DIR; inherited by () subshells) ───
ft  () { $PY -m scripts.finetune      --config "$1" --encoder_ckpt "$2" --gpu_id "$3" "${OVR[@]}" 2>&1 | tee "logs/${TAG}_$4.log"; }
ftw () { $PY -m scripts.finetune_wnli --config "$1" --encoder_ckpt "$2" --gpu_id "$3" "${OVR[@]}" 2>&1 | tee "logs/${TAG}_$4.log"; }
run_eval () {   # <config> <ckpt> <gpu> <name> [extra key=value ...]
    local cfg=$1 ckpt=$2 gpu=$3 name=$4; shift 4
    if [ ! -f "$ckpt" ]; then echo "[skip eval $name] missing checkpoint: $ckpt"; return; fi
    $PY -m scripts.eval --config "$cfg" --ckpt "$ckpt" --gpu_id "$gpu" \
        --output_dir "$SUB_DIR" "${EVAL_LIMIT[@]}" "$@" 2>&1 | tee "logs/${TAG}_eval_${name}.log"
}

# ── Sweep ────────────────────────────────────────────────────────────────────
BATCH_SIZES=(16 32)
LR_NUMS=(1 2 3)        # ×e-5

for BS in "${BATCH_SIZES[@]}"; do
for LRN in "${LR_NUMS[@]}"; do
    LR="${LRN}.0e-5"                 # YAML-float form; "1e-5" would parse as a string
    TAG="${TAGPRE}bs${BS}_lr${LRN}e-5"
    CKPT_DIR="checkpoints/finetune_${TAG}"
    RES_DIR="results_${TAG}"
    SUB_DIR="glue_submission_${TAG}"
    FLAG="/tmp/mnli_done_${TAG}_$$"
    mkdir -p "$CKPT_DIR" "$RES_DIR" "$SUB_DIR"
    rm -f "$FLAG"

    OVR=(ckpt_dir="$CKPT_DIR" results_dir="$RES_DIR" lr="$LR" batch_size="$BS" "${EXTRA_FT[@]}")
    MNLI_FT="$CKPT_DIR/finetune_mnli_exp1_FacebookAI_roberta_large_best_val.pt"

    echo "==================== COMBO $TAG : finetuning (4 GPUs) ===================="

    # GPU 0:  MNLI ─▶ RTE
    ( ft configs/finetune_mnli.yml "$PRE_MNLI" 0 mnli
      touch "$FLAG"
      if [ -f "$MNLI_FT" ]; then ft configs/finetune_rte.yml "$MNLI_FT" 0 rte
      else echo "[skip rte] MNLI checkpoint missing: $MNLI_FT" >> "logs/${TAG}_rte.log"; fi
    ) &

    # GPU 1:  QQP ─▶ MRPC  (MRPC waits for MNLI best_val)
    ( ft configs/finetune_qqp.yml "$PRE_QQP" 1 qqp
      until [ -f "$FLAG" ]; do sleep 30; done
      if [ -f "$MNLI_FT" ]; then ft configs/finetune_mrpc.yml "$MNLI_FT" 1 mrpc
      else echo "[skip mrpc] MNLI checkpoint missing: $MNLI_FT" >> "logs/${TAG}_mrpc.log"; fi
    ) &

    # GPU 2:  CoLA ─▶ SST2 ─▶ STSB  (STSB waits for MNLI best_val)
    ( ft configs/finetune_cola.yml "$PRE_COLA" 2 cola
      ft configs/finetune_sst2.yml "$PRE_SST2" 2 sst2
      until [ -f "$FLAG" ]; do sleep 30; done
      if [ -f "$MNLI_FT" ]; then ft configs/finetune_stsb.yml "$MNLI_FT" 2 stsb
      else echo "[skip stsb] MNLI checkpoint missing: $MNLI_FT" >> "logs/${TAG}_stsb.log"; fi
    ) &

    # GPU 3:  WNLI ─▶ QNLI
    ( ftw configs/finetune_wnli.yml "$PRE_WNLI" 3 wnli
      ft  configs/finetune_qnli.yml "$PRE_QNLI" 3 qnli
    ) &

    wait
    rm -f "$FLAG"

    echo "==================== COMBO $TAG : GLUE submission (eval) ===================="

    QQP_FT="$CKPT_DIR/finetune_qqp_exp1_FacebookAI_roberta_large_best_val.pt"
    QNLI_FT="$CKPT_DIR/finetune_qnli_exp1_FacebookAI_roberta_large_best_val.pt"
    SST2_FT="$CKPT_DIR/finetune_sst2_exp1_FacebookAI_roberta_large_best_val.pt"
    COLA_FT="$CKPT_DIR/finetune_cola_exp1_roberta_large_best_val.pt"
    RTE_FT="$CKPT_DIR/finetune_rte_exp1_roberta_large_best_val.pt"
    MRPC_FT="$CKPT_DIR/finetune_mrpc_exp1_roberta_large_best_val.pt"
    STSB_FT="$CKPT_DIR/finetune_stsb_exp1_roberta_large_best_val.pt"
    WNLI_FT="$CKPT_DIR/finetune_wnli_exp1_roberta_large_best.pt"

    # Eval in waves of 4 GPUs.
    run_eval configs/finetune_mnli.yml "$MNLI_FT" 0 mnli                   &
    run_eval configs/finetune_mnli.yml "$MNLI_FT" 1 ax   dataset_config=ax &
    run_eval configs/finetune_qqp.yml  "$QQP_FT"  2 qqp                    &
    run_eval configs/finetune_qnli.yml "$QNLI_FT" 3 qnli                   &
    wait
    run_eval configs/finetune_sst2.yml "$SST2_FT" 0 sst2 &
    run_eval configs/finetune_cola.yml "$COLA_FT" 1 cola &
    run_eval configs/finetune_rte.yml  "$RTE_FT"  2 rte  &
    run_eval configs/finetune_mrpc.yml "$MRPC_FT" 3 mrpc &
    wait
    run_eval configs/finetune_stsb.yml "$STSB_FT" 0 stsb &
    run_eval configs/finetune_wnli.yml "$WNLI_FT" 1 wnli &
    wait

    echo "==================== COMBO $TAG COMPLETE → $SUB_DIR ===================="
    ls "$SUB_DIR"/*.tsv 2>/dev/null || true
done
done

echo "ALL 6 COMBINATIONS COMPLETE."

#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# run_glue_mnli_transfer_sweep.sh
#
# Nested GLUE sweep (2 levels), MNLI-transfer for RTE/STS-B/MRPC.
#
#   OUTER grid  bs{16,32,64} × lr{1,2,3}e-5  (9 combos)  → MNLI + QQP/QNLI/SST2/CoLA/WNLI
#   INNER grid  bs{16,32,64} × lr{1,2,3}e-5  (9 combos)  nested in EACH outer combo
#               → RTE/STS-B/MRPC finetuned FROM that outer combo's MNLI best_val,
#                 using the INNER (bs,lr). Output dirs:
#                   checkpoints/finetune_<outer>/<inner>/
#                   results_<outer>/<inner>/
#                   glue_submission_<outer>/<inner>/
#
# OUTER main tasks (MNLI/QQP/QNLI/SST2/CoLA/WNLI) are finetuned for ALL 9 combos
# (Phase A). Phase B then runs the 9×9×3 = 243 inner RTE/STSB/MRPC.
# Every finetune is followed by eval → GLUE submission TSV(s).
#
# GLUE finetune is epoch-based (epochs=10, ratio warmup) for all tasks except WNLI
# (fixed total_num_updates=2000, left as-is, matching run_glue_sweep.sh) — so larger
# batch sizes need NO budget rescaling; only batch_size + lr are overridden.
#
# Resumable: a finetune whose best ckpt exists is skipped; an eval whose TSV exists
# is skipped. Runs on 4 GPUs (0-3) as 4 round-robin queues.
#
# Usage:
#   bash run_glue_mnli_transfer_sweep.sh           # real run
#   DRYRUN=1 bash run_glue_mnli_transfer_sweep.sh  # print the job list, run nothing
#   GPUS="0 1" bash run_glue_mnli_transfer_sweep.sh
# ─────────────────────────────────────────────────────────────────────────────
set -uo pipefail
cd "$(dirname "$0")"
mkdir -p logs
PY=python
export PYTHONUNBUFFERED=1
DRYRUN="${DRYRUN:-0}"
read -r -a GPUS <<<"${GPUS:-0 1 2 3}"

# ── Pretrain encoders for the OUTER main tasks (combo-independent) ─────────────
PRE_MNLI=checkpoints/pretrain/best_pretrain_mnli_exp1_FacebookAI_roberta_large_.pt
PRE_QQP=checkpoints/pretrain/best_pretrain_qqp_exp1_FacebookAI_roberta_large_.pt
PRE_QNLI=checkpoints/pretrain/best_pretrain_qnli_exp1_FacebookAI_roberta_large_.pt
PRE_SST2=checkpoints/pretrain/best_pretrain_sst2_exp1_FacebookAI_roberta_large_.pt
PRE_COLA=checkpoints/pretrain/best_pretrain_cola_exp1_roberta_large_.pt
PRE_WNLI=checkpoints/pretrain/best_pretrain_wnli_exp1_roberta_large_.pt
for ck in "$PRE_MNLI" "$PRE_QQP" "$PRE_QNLI" "$PRE_SST2" "$PRE_COLA" "$PRE_WNLI"; do
    [ -f "$ck" ] || { echo "FATAL: missing pretrain checkpoint: $ck"; exit 1; }
done

# ── Per-task naming (matches existing checkpoints/ + submission TSVs) ──────────
ckname() { case $1 in
    mnli) echo finetune_mnli_exp1_FacebookAI_roberta_large_best_val.pt;;
    qqp)  echo finetune_qqp_exp1_FacebookAI_roberta_large_best_val.pt;;
    qnli) echo finetune_qnli_exp1_FacebookAI_roberta_large_best_val.pt;;
    sst2) echo finetune_sst2_exp1_FacebookAI_roberta_large_best_val.pt;;
    cola) echo finetune_cola_exp1_roberta_large_best_val.pt;;
    wnli) echo finetune_wnli_exp1_roberta_large_best.pt;;
    rte)  echo finetune_rte_exp1_roberta_large_best_val.pt;;
    mrpc) echo finetune_mrpc_exp1_roberta_large_best_val.pt;;
    stsb) echo finetune_stsb_exp1_roberta_large_best_val.pt;;
esac; }
cfgof() { echo "configs/finetune_$1.yml"; }
tsvof() { case $1 in     # sentinel TSV used for eval-skip
    rte) echo RTE.tsv;; mrpc) echo MRPC.tsv;; stsb) echo STS-B.tsv;;
    mnli) echo MNLI-m.tsv;; qqp) echo QQP.tsv;; qnli) echo QNLI.tsv;;
    sst2) echo SST-2.tsv;; cola) echo CoLA.tsv;; wnli) echo WNLI.tsv;;
esac; }

# ── Single job: finetune (if needed) then eval → TSV. Tab-separated spec. ──────
#   spec = type \t task \t enc \t lr \t bs \t ckdir \t rsdir \t subdir \t tag
#   type ∈ ft | ftwnli | ax
run_spec() {  # <spec> <gpu>
    local spec=$1 gpu=$2 typ task enc lr bs ckdir rsdir subdir tag
    IFS=$'\t' read -r typ task enc lr bs ckdir rsdir subdir tag <<<"$spec"
    mkdir -p "$ckdir" "$rsdir" "$subdir"
    if [ "$typ" = ax ]; then
        local mnli_ck="$ckdir/$(ckname mnli)"
        [ -f "$subdir/AX.tsv" ] && { echo "[skip ax $tag]"; return; }
        [ -f "$mnli_ck" ]      || { echo "[skip ax $tag] no MNLI ckpt"; return; }
        $PY -m scripts.eval --config configs/finetune_mnli.yml --ckpt "$mnli_ck" \
            --gpu_id "$gpu" --output_dir "$subdir" dataset_config=ax \
            2>&1 | tee "logs/glue_mt_${tag}_eval_ax.log"
        return
    fi
    local cfg ckf tsv prog=scripts.finetune
    [ "$typ" = ftwnli ] && prog=scripts.finetune_wnli
    cfg=$(cfgof "$task"); ckf="$ckdir/$(ckname "$task")"; tsv="$subdir/$(tsvof "$task")"
    # finetune
    if [ -f "$ckf" ]; then
        echo "[skip ft $tag/$task] ckpt exists"
    else
        $PY -m "$prog" --config "$cfg" --encoder_ckpt "$enc" --gpu_id "$gpu" \
            ckpt_dir="$ckdir" results_dir="$rsdir" lr="$lr" batch_size="$bs" \
            2>&1 | tee "logs/glue_mt_${tag}_${task}.log"
    fi
    # eval → submission TSV
    if [ -f "$tsv" ]; then
        echo "[skip eval $tag/$task] TSV exists"
    elif [ -f "$ckf" ]; then
        $PY -m scripts.eval --config "$cfg" --ckpt "$ckf" --gpu_id "$gpu" \
            --output_dir "$subdir" 2>&1 | tee "logs/glue_mt_${tag}_eval_${task}.log"
    else
        echo "[skip eval $tag/$task] finetune produced no ckpt ($ckf)"
    fi
}

# ── Run the global JOBS array across GPUS (round-robin, 1 queue per GPU) ───────
JOBS=()
run_jobs() {  # <phase-label>
    echo "==================== $1 : ${#JOBS[@]} jobs on GPUs [${GPUS[*]}] ===================="
    if [ "$DRYRUN" = 1 ]; then
        local n=0 spec
        for spec in "${JOBS[@]}"; do
            printf '  [%3d] %s\n' "$n" "$(tr '\t' ' ' <<<"$spec")"; n=$((n+1))
        done
        return
    fi
    local i g
    for i in "${!GPUS[@]}"; do
        g="${GPUS[$i]}"
        (
            local j=$i
            while [ "$j" -lt "${#JOBS[@]}" ]; do
                run_spec "${JOBS[$j]}" "$g"
                j=$(( j + ${#GPUS[@]} ))
            done
        ) &
    done
    wait
}
add() { JOBS+=("$(printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s' "$@")"); }

# ── PHASE A: OUTER main tasks (MNLI/QQP/QNLI/SST2/CoLA/WNLI) for ALL bs ────────
JOBS=()
for BS in 32 64; do for LRN in 1 2 3; do
    LR="${LRN}.0e-5"; TAG="bs${BS}_lr${LRN}e-5"
    CK="checkpoints/finetune_${TAG}"; RS="results_${TAG}"; SUB="glue_submission_${TAG}"
    add ft     mnli "$PRE_MNLI" "$LR" "$BS" "$CK" "$RS" "$SUB" "$TAG"
    add ft     qqp  "$PRE_QQP"  "$LR" "$BS" "$CK" "$RS" "$SUB" "$TAG"
    add ft     qnli "$PRE_QNLI" "$LR" "$BS" "$CK" "$RS" "$SUB" "$TAG"
    add ft     sst2 "$PRE_SST2" "$LR" "$BS" "$CK" "$RS" "$SUB" "$TAG"
    add ft     cola "$PRE_COLA" "$LR" "$BS" "$CK" "$RS" "$SUB" "$TAG"
    add ftwnli wnli "$PRE_WNLI" "$LR" "$BS" "$CK" "$RS" "$SUB" "$TAG"
done; done
run_jobs "PHASE A (outer main tasks, all bs)"

# AX diagnostic (needs each combo's MNLI ckpt; run after Phase A)
JOBS=()
for BS in 32 64; do for LRN in 1 2 3; do
    TAG="bs${BS}_lr${LRN}e-5"
    add ax mnli - - - "checkpoints/finetune_${TAG}" "results_${TAG}" "glue_submission_${TAG}" "$TAG"
done; done
run_jobs "PHASE A.ax (AX diagnostic, all bs)"

# ── PHASE B: INNER RTE/STS-B/MRPC, from each outer combo's MNLI best_val ───────
JOBS=()
for OB in 32 64; do for OL in 1 2 3; do
    OUTER="bs${OB}_lr${OL}e-5"
    MNLI_CK="checkpoints/finetune_${OUTER}/$(ckname mnli)"
    if [ ! -f "$MNLI_CK" ]; then
        echo "[skip outer $OUTER] missing MNLI best_val: $MNLI_CK"; continue
    fi
    for IB in 16 32 64; do for IL in 1 2 3; do
        INNER="bs${IB}_lr${IL}e-5"; ILR="${IL}.0e-5"
        CK="checkpoints/finetune_${OUTER}/${INNER}"
        RS="results_${OUTER}/${INNER}"
        SUB="glue_submission_${OUTER}/${INNER}"
        TAG="${OUTER}__${INNER}"
        add ft rte  "$MNLI_CK" "$ILR" "$IB" "$CK" "$RS" "$SUB" "$TAG"
        add ft mrpc "$MNLI_CK" "$ILR" "$IB" "$CK" "$RS" "$SUB" "$TAG"
        add ft stsb "$MNLI_CK" "$ILR" "$IB" "$CK" "$RS" "$SUB" "$TAG"
    done; done
done; done
run_jobs "PHASE B (inner RTE/STSB/MRPC, MNLI-transfer)"

echo "NESTED GLUE MNLI-TRANSFER SWEEP COMPLETE."

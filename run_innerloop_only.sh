#!/bin/bash
# Phase-B-only runner: inner RTE/STSB/MRPC MNLI-transfer for the outer MNLI
# checkpoints that still exist. Reuses functions from run_glue_mnli_transfer_sweep.sh.
set -uo pipefail
cd "$(dirname "$0")"
mkdir -p logs
PY=python
export PYTHONUNBUFFERED=1
DRYRUN="${DRYRUN:-0}"
read -r -a GPUS <<<"${GPUS:-0 1 2 3}"

ckname() { case $1 in
    mnli) echo finetune_mnli_exp1_FacebookAI_roberta_large_best_val.pt;;
    rte)  echo finetune_rte_exp1_roberta_large_best_val.pt;;
    mrpc) echo finetune_mrpc_exp1_roberta_large_best_val.pt;;
    stsb) echo finetune_stsb_exp1_roberta_large_best_val.pt;;
esac; }
cfgof() { echo "configs/finetune_$1.yml"; }
tsvof() { case $1 in
    rte) echo RTE.tsv;; mrpc) echo MRPC.tsv;; stsb) echo STS-B.tsv;;
esac; }

run_spec() {  # <spec> <gpu>
    local spec=$1 gpu=$2 typ task enc lr bs ckdir rsdir subdir tag
    IFS=$'\t' read -r typ task enc lr bs ckdir rsdir subdir tag <<<"$spec"
    mkdir -p "$ckdir" "$rsdir" "$subdir"
    local cfg ckf tsv prog=scripts.finetune
    cfg=$(cfgof "$task"); ckf="$ckdir/$(ckname "$task")"; tsv="$subdir/$(tsvof "$task")"
    if [ -f "$ckf" ]; then
        echo "[skip ft $tag/$task] ckpt exists"
    else
        $PY -m "$prog" --config "$cfg" --encoder_ckpt "$enc" --gpu_id "$gpu" \
            ckpt_dir="$ckdir" results_dir="$rsdir" lr="$lr" batch_size="$bs" \
            2>&1 | tee "logs/glue_mt_${tag}_${task}.log"
    fi
    if [ -f "$tsv" ]; then
        echo "[skip eval $tag/$task] TSV exists"
    elif [ -f "$ckf" ]; then
        $PY -m scripts.eval --config "$cfg" --ckpt "$ckf" --gpu_id "$gpu" \
            --output_dir "$subdir" 2>&1 | tee "logs/glue_mt_${tag}_eval_${task}.log"
    else
        echo "[skip eval $tag/$task] finetune produced no ckpt ($ckf)"
    fi
}

JOBS=()
run_jobs() {
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

JOBS=()
for OB in 16 32 64; do for OL in 1 2 3; do
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
run_jobs "PHASE B only (inner RTE/STSB/MRPC, MNLI-transfer)"
echo "INNER-LOOP-ONLY SWEEP COMPLETE."

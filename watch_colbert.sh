#!/bin/bash
# Reports new ColBERT events (finetune-done + per-dataset BEIR results) since the
# last snapshot, then exits so it can be re-armed. Exits ALL_DONE when both jobs
# have exited and nothing new remains. Manages its own snapshot file.
cd "$(dirname "$0")"
SNAP=.colbert_watch_snapshot
FT_PRE=experiments/colbert/colbert_msmarco_exp1/none/run1/checkpoints/colbert-300000
FT_FRESH=experiments/colbert/colbert_msmarco_exp1_fresh/none/run1/checkpoints/colbert-300000

state() {
    [ -e "$FT_PRE" ]   && echo "FT:pretrained (slurm_colbert.sh) finetuning COMPLETE"
    [ -e "$FT_FRESH" ] && echo "FT:fresh (slurm_colbert_fresh.sh) finetuning COMPLETE"
    for f in results/beir_colbert/*.json; do
        [ -e "$f" ] && echo "RES:$(basename "$f")"
    done
}

while true; do
    cur=$(state | sort)
    prev=$(cat "$SNAP" 2>/dev/null)
    if [ "$cur" != "$prev" ]; then
        comm -13 <(printf '%s\n' "$prev" | grep .) <(printf '%s\n' "$cur" | grep .)
        printf '%s\n' "$cur" > "$SNAP"
        exit 0
    fi
    if ! pgrep -f 'slurm_colbert.*\.sh' >/dev/null; then
        echo "ALL_DONE both_jobs_exited"; exit 0
    fi
    sleep 60
done

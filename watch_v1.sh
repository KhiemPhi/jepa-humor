#!/bin/bash
# Unified ColBERT-v1 watcher: pings on (1) errors, (2) finetune completion,
# (3) each per-dataset eval result. Snapshot-based: prints NEW events then exits
# so it can be re-armed. Manages its own snapshot file.
cd /home/tudao/jepa-humor
SNAP=.v1_watch_snapshot
CKF=experiments_v1/colbert_fresh/train.py/msmarco/checkpoints/colbert-300000.dnn
CKJ=experiments_v1/colbert_jepa/train.py/msmarco/checkpoints/colbert-300000.dnn

state() {
  # (2) finetune done
  [ -f "$CKF" ] && echo "FINETUNE-DONE: fresh"
  [ -f "$CKJ" ] && echo "FINETUNE-DONE: jepa"
  # (3) per-dataset eval results
  for f in results_beir_v1/*/*.txt; do
    [ -e "$f" ] && echo "RESULT: ${f#results_beir_v1/}"
  done
  # (1a) errors surfaced in any monitored log (reported once per log)
  for lg in logs/v1train_fresh.log logs/v1train_jepa.log logs/v1eval_fresh.log logs/v1eval_jepa.log; do
    [ -f "$lg" ] || continue
    grep -qE 'Traceback \(most recent call last\)|CUDA out of memory|RuntimeError|AssertionError|\[FAIL ' "$lg" \
      && echo "ERROR-IN-LOG: $lg"
  done
  # (1b) a finetune proc died WITHOUT producing its checkpoint (e.g. OOM SIGKILL, no traceback)
  for tag in fresh jepa; do
    ck=$CKF; [ "$tag" = jepa ] && ck=$CKJ
    [ -f "$ck" ] && continue
    pgrep -f "experiment colbert_${tag}" >/dev/null || echo "TRAIN-DIED-NO-CKPT: $tag"
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
  sleep 60
done

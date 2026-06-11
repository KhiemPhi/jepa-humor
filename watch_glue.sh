#!/bin/bash
# GLUE headdrop=0.0 sweep watcher: pings on ERRORS (and on completion).
# Snapshot-based: prints new events then exits so it can be re-armed.
cd /home/tudao/jepa-humor
SNAP=.glue_watch_snapshot
SWEEPLOG=logs/glue_headdrop0_sweep.log

state() {
  grep -q "NESTED GLUE MNLI-TRANSFER SWEEP COMPLETE" "$SWEEPLOG" 2>/dev/null && echo "SWEEP-COMPLETE"
  for lg in "$SWEEPLOG"; do
    [ -f "$lg" ] || continue
    grep -qE 'Traceback \(most recent call last\)|CUDA out of memory|OutOfMemoryError|FATAL:|finetune produced no ckpt' "$lg" \
      && echo "ERROR-IN: $lg"
  done
  if ! pgrep -f 'run_glue_mnli_transfer_sweep' >/dev/null; then
    grep -q "NESTED GLUE MNLI-TRANSFER SWEEP COMPLETE" "$SWEEPLOG" 2>/dev/null || echo "SWEEP-DIED-INCOMPLETE"
  fi
}

while true; do
  cur=$(state | sort); prev=$(cat "$SNAP" 2>/dev/null)
  if [ "$cur" != "$prev" ]; then
    comm -13 <(printf '%s\n' "$prev" | grep .) <(printf '%s\n' "$cur" | grep .)
    printf '%s\n' "$cur" > "$SNAP"; exit 0
  fi
  sleep 60
done

#!/usr/bin/env bash
# Emit one event per new/updated BEIR result JSON; flag when colbert jobs leave the
# queue; and emit COLBERT-ERROR when a slurm_colbert.sh / slurm_colbert_fresh.sh run
# (or its BEIR eval) writes a crash/failure signature to logs/colbert_*.{out,err}.
cd /home/tudao/jepa-humor || exit 1
DIR=results/beir_colbert
shopt -s nullglob
declare -A seen
# Seed with existing files so only NEW results from now on are reported.
for f in "$DIR"/*.json; do
  seen["$f"]=$(stat -c %Y "$f" 2>/dev/null || echo 0)
done

# ── Colbert run-failure watch ────────────────────────────────────────────────
# slurm_colbert.sh        -> logs/colbert_<jobid>.{out,err}
# slurm_colbert_fresh.sh  -> logs/colbert_fresh_<jobid>.{out,err}   (+ *_eval_* logs)
# (colbert runs on another server but writes here; squeue isn't available, so we
#  watch the logs.) Seed each existing log's byte size: only content appended from
# now on is scanned, so we report failures going forward (no replay of old logs).
declare -A logsize
for lf in logs/colbert_*.out logs/colbert_*.err; do
  logsize["$lf"]=$(stat -c %s "$lf" 2>/dev/null || echo 0)
done
ERR_RE='Traceback \(most recent call last\)|[A-Za-z.]*Error\b|CUDA out of memory|OutOfMemory|Exception:|Segmentation fault|core dumped|Aborted|[Kk]illed|oom-kill|[Oo]ut [Oo]f [Mm]emory|slurmstepd: error|srun: error|CANCELLED|DUE TO TIME LIMIT|No space left on device|FAILED'

jobs_were_present=0
while true; do
  for f in "$DIR"/*.json; do
    m=$(stat -c %Y "$f" 2>/dev/null || echo 0)
    if [ "${seen[$f]:-}" != "$m" ]; then
      seen["$f"]="$m"
      python3 - "$f" <<'PY' 2>/dev/null || echo "RESULT (parse-pending) $f"
import json,sys
f=sys.argv[1]
d=json.load(open(f))
m=d.get("metrics",{})
g=lambda k: m.get(k,0)
print(f"RESULT {d.get('dataset',f)} | ndcg@10={g('ndcg@10'):.4f} ndcg@100={g('ndcg@100'):.4f} "
      f"recall@10={g('recall@10'):.4f} recall@100={g('recall@100'):.4f} map@10={g('map@10'):.4f} | {f}")
PY
    fi
  done
  # Emit COLBERT-ERROR for any failure signature newly appended to a colbert run log.
  for lf in logs/colbert_*.out logs/colbert_*.err; do
    cur=$(stat -c %s "$lf" 2>/dev/null || echo 0)
    prev=${logsize["$lf"]:-0}
    if [ "$cur" -gt "$prev" ]; then
      hit=$(tail -c "+$((prev+1))" "$lf" 2>/dev/null | tr '\r' '\n' | grep -aE "$ERR_RE" | head -3)
      [ -n "$hit" ] && { echo "COLBERT-ERROR in $lf:"; printf '%s\n' "$hit" | sed 's/^/    /'; }
      logsize["$lf"]="$cur"
    elif [ "$cur" -lt "$prev" ]; then
      logsize["$lf"]="$cur"   # log rotated/truncated; reset offset
    fi
  done
  n=$(squeue -u "$USER" -h -o '%j' 2>/dev/null | grep -ic colbert || true)
  [ "$n" -gt 0 ] && jobs_were_present=1
  if [ "$jobs_were_present" -eq 1 ] && [ "$n" -eq 0 ]; then
    echo "JOBS-DONE: no colbert jobs left in squeue (all finished or stopped)"
    jobs_were_present=0
  fi
  sleep 30
done

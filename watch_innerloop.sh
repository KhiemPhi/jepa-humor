#!/bin/bash
# Blocks until an outer combo (not in $1) reaches 27/27 inner TSVs, prints its
# name, and exits. Exits with DONE_ALL if the runner died with nothing new.
cd "$(dirname "$0")"
DONE="${1:-}"   # comma-separated combos already reported
COMBOS="bs32_lr1e-5 bs32_lr2e-5 bs64_lr1e-5 bs64_lr2e-5"
count() { ls glue_submission_$1/*/{RTE.tsv,MRPC.tsv,STS-B.tsv} 2>/dev/null | wc -l; }
while true; do
    for O in $COMBOS; do
        case ",$DONE," in *",$O,"*) continue;; esac
        [ "$(count "$O")" -ge 27 ] && { echo "COMPLETE $O"; exit 0; }
    done
    if ! pgrep -f run_innerloop_only.sh >/dev/null; then
        echo "DONE_ALL runner_exited"; exit 0
    fi
    sleep 60
done

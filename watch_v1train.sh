#!/bin/bash
# Blocks until a not-yet-reported colbert-300000.dnn appears, prints which, exits.
# Exits DONE_ALL if both training procs are gone with nothing new.
cd /home/tudao/jepa-humor
DONE="${1:-}"   # comma-sep already-reported tags
declare -A CK=(
  [fresh]=experiments_v1/colbert_fresh/train.py/msmarco/checkpoints/colbert-300000.dnn
  [jepa]=experiments_v1/colbert_jepa/train.py/msmarco/checkpoints/colbert-300000.dnn
)
while true; do
  for tag in fresh jepa; do
    case ",$DONE," in *",$tag,"*) continue;; esac
    [ -f "${CK[$tag]}" ] && { echo "DONE $tag"; exit 0; }
  done
  if ! pgrep -f 'colbert.train' >/dev/null; then echo "DONE_ALL train_exited"; exit 0; fi
  sleep 60
done

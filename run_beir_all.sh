#!/bin/bash
# run_beir_all.sh <tag> <checkpoint.dnn> <gpu>
# Thin wrapper around beir-ColBERT's EXACT 5-step pipeline (data_prep, index,
# index_faiss, retrieve, beir_eval), looping all public BEIR datasets fast->slow.
# Repo code is untouched; this only invokes its modules. Resumable per dataset.
set -uo pipefail
TAG=$1; CKPT=$2; GPU=$3
REPO=/home/tudao/jepa-humor/beir-ColBERT
PY=/home/tudao/miniconda3/envs/beir/bin/python
WORK=/home/tudao/jepa-humor/beir_work/$TAG
OUT=$WORK/output; IDX=$WORK/indices; RANKD=$WORK/rankings
RES=/home/tudao/jepa-humor/results_beir_v1/$TAG
mkdir -p "$OUT" "$IDX" "$RANKD" "$RES"
cd "$REPO"
export OMP_NUM_THREADS=6

# fast -> slow (public BEIR only; BioASQ/Signal-1M/TREC-NEWS/Robust04 are not auto-downloadable)
DATASETS=(nfcorpus scifact arguana scidocs fiqa trec-covid webis-touche2020 quora \
          nq dbpedia-entity hotpotqa fever climate-fever msmarco)

splitof(){ case $1 in msmarco) echo dev;; *) echo test;; esac; }

# one full pipeline for <dataset> with optional <data_dir> (used by cqadupstack subsets)
eval_one(){
  local ds=$1 name=$2 data_dir=${3:-}
  local split; split=$(splitof "$ds")
  local col=$OUT/$name/collection.tsv que=$OUT/$name/queries.tsv
  local rank=$RANKD/$name res=$RES/$name.txt
  [ -f "$res" ] && { echo "[skip $name] result exists"; return; }
  echo "==================== BEIR: $name (split=$split) ===================="
  local ddopt=""; [ -n "$data_dir" ] && ddopt="--data_dir $data_dir"
  # 0. data prep
  $PY -m colbert.data_prep --dataset "$ds" --split "$split" $ddopt \
        --collection "$col" --queries "$que" || { echo "[FAIL prep $name]"; return; }
  # 1. index (single-proc; upstream uses torch.distributed.launch)
  CUDA_VISIBLE_DEVICES=$GPU $PY -m colbert.index --root "$OUT" \
        --doc_maxlen 300 --mask-punctuation --bsize 128 --amp \
        --checkpoint "$CKPT" --index_root "$IDX" --index_name "$name" \
        --collection "$col" --experiment "$name" || { echo "[FAIL index $name]"; return; }
  # 2. faiss index (auto --partitions per corpus size)
  CUDA_VISIBLE_DEVICES=$GPU $PY -m colbert.index_faiss --index_root "$IDX" \
        --index_name "$name" --sample 0.3 --root "$OUT" --experiment "$name" \
        || { echo "[FAIL faiss $name]"; return; }
  # capture the auto-chosen partition count so retrieve finds the same index file
  local P; P=$(ls "$IDX/$name"/ivfpq.*.faiss 2>/dev/null | head -1 | grep -oP 'ivfpq\.\K[0-9]+')
  [ -z "$P" ] && { echo "[FAIL faiss-name $name]"; return; }
  # 3. retrieve (faiss_depth=100, depth=100, nprobe=32 -- the paper's end-to-end retrieval)
  CUDA_VISIBLE_DEVICES=$GPU $PY -m colbert.retrieve --amp --doc_maxlen 300 \
        --mask-punctuation --bsize 256 --queries "$que" --nprobe 32 --partitions "$P" \
        --faiss_depth 100 --depth 100 --index_root "$IDX" --index_name "$name" \
        --checkpoint "$CKPT" --root "$OUT" --experiment "$name" --ranking_dir "$rank" \
        || { echo "[FAIL retrieve $name]"; return; }
  # 4. BEIR eval
  $PY -m colbert.beir_eval --dataset "$ds" --split "$split" ${data_dir:+--data_dir $data_dir} \
        --collection "$col" --rankings "$rank/ranking.tsv" 2>&1 | tee "$res"
}

for ds in "${DATASETS[@]}"; do eval_one "$ds" "$ds"; done

# CQADupStack: 12 subsets (download once, then per-subset)
CQA=$WORK/cqadupstack_dl
if [ ! -d "$CQA/cqadupstack" ]; then
  $PY - <<PYEOF
from beir import util
util.download_and_unzip("https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/cqadupstack.zip", "$CQA")
PYEOF
fi
for sub in android english gaming gis mathematica physics programmers stats tex unix webmasters wordpress; do
  eval_one cqadupstack "cqadupstack_${sub}" "$CQA/cqadupstack/$sub"
done

echo "ALL BEIR DATASETS DONE for $TAG. Results in $RES/"

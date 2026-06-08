#!/bin/bash
#SBATCH --job-name=colbert-msmarco-fresh
#SBATCH --output=logs/colbert_fresh_%j.out
#SBATCH --error=logs/colbert_fresh_%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=80G
#SBATCH --time=5-00:00
#SBATCH --nodelist=h100

set -e   # abort on any non-zero exit so training failure stops eval

mkdir -p logs results/beir_colbert

# CUDA toolkit for ColBERT's JIT-compiled indexing extensions (needs CUDA_HOME)
export CUDA_HOME=/shared/software/cuda/12.6
export PATH="$CUDA_HOME/bin:$PATH"
export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$LD_LIBRARY_PATH"

# ColBERT path_ = {root}/{experiment}/{script_name_}/{name}/checkpoints/
# script_name_ is "none" because checkpoints are saved from spawned worker
# processes where __main__ has no __file__ (see colbert/infra/config/settings.py)
COLBERT_CKPT=experiments/colbert/colbert_msmarco_exp1_fresh/none/run1/checkpoints/colbert-300000
EVAL="python colbert/eval_beir.py --checkpoint ${COLBERT_CKPT} --results_dir results/beir_colbert"

# ── 1. Fine-tune ColBERT on MS MARCO (fresh bert-base-uncased, no pretrain) ──
python colbert/train.py \
    --config colbert/config.yml \
    --checkpoint bert-base-uncased \
    --experiment colbert_msmarco_exp1_fresh

# ── 2. Benchmark trained ColBERT on all public BEIR datasets ─────────────────

# Helper: run <short_name> <hf_dataset> <hf_qrels> [extra args]
run() {
    local name=$1 ds=$2 qrels=$3
    shift 3
    echo "=== BEIR: ${name} ==="
    ${EVAL} \
        --dataset        "${ds}" \
        --qrels_dataset  "${qrels}" \
        --experiment_name "colbert_fresh_${name}" \
        "$@"
}

run trec-covid    BeIR/trec-covid        BeIR/trec-covid-qrels
run nfcorpus      BeIR/nfcorpus          BeIR/nfcorpus-qrels
run nq            BeIR/nq                BeIR/nq-qrels
run hotpotqa      BeIR/hotpotqa          BeIR/hotpotqa-qrels
run fiqa          BeIR/fiqa              BeIR/fiqa-qrels
run arguana       BeIR/arguana           BeIR/arguana-qrels
run touche2020    BeIR/webis-touche2020  BeIR/webis-touche2020-qrels
run quora         BeIR/quora             BeIR/quora-qrels
run dbpedia       BeIR/dbpedia-entity    BeIR/dbpedia-entity-qrels
run scidocs       BeIR/scidocs           BeIR/scidocs-qrels
run fever         BeIR/fever             BeIR/fever-qrels
run climate-fever BeIR/climate-fever     BeIR/climate-fever-qrels
run scifact       BeIR/scifact           BeIR/scifact-qrels

run msmarco BeIR/msmarco BeIR/msmarco-qrels --qrels_split validation

for subset in android english gaming gis mathematica physics programmers stats tex unix webmasters wordpress; do
    echo "=== BEIR: cqadupstack/${subset} ==="
    ${EVAL} \
        --dataset        BeIR/cqadupstack \
        --qrels_dataset  BeIR/cqadupstack-qrels \
        --dataset_config "${subset}" \
        --experiment_name "colbert_fresh_cqadupstack_${subset}"
done

echo "All BEIR evaluations complete. Results in results/beir_colbert/."
echo "CQADupstack nDCG@10: average the 12 colbert_fresh_cqadupstack_* result files."

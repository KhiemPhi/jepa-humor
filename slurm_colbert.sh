#!/bin/bash
#SBATCH --job-name=colbert-msmarco
#SBATCH --output=logs/colbert_%j.out
#SBATCH --error=logs/colbert_%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=80G
#SBATCH --time=5-00:00
#SBATCH --nodelist=h100

set -e   # abort on any non-zero exit so training failure stops eval

mkdir -p logs results/beir_colbert

PRETRAIN_CKPT=checkpoints/pretrain/best_pretrain_msmarco_exp1_bert_base_uncased_.pt
HF_ENCODER=checkpoints/pretrain/hf_encoder
# ColBERT path_ = {root}/{experiment}/{script_name_}/{name}/checkpoints/
# script_name_ for `python colbert/train.py` run from project root = "colbert.train"
COLBERT_CKPT=experiments/colbert/colbert_msmarco_exp1/colbert.train/run1/checkpoints/colbert-300000
EVAL="python colbert/eval_beir.py --checkpoint ${COLBERT_CKPT} --results_dir results/beir_colbert"

# ── 1. Pretrain encoder on MS MARCO ─────────────────────────────────────────
python scripts/pretrain.py --config configs/pretrain_ms_marco.yml

# ── 2. Convert pretrained checkpoint to HuggingFace format for ColBERT ──────
python -c "
import torch
from transformers import AutoModel, AutoTokenizer
from jepa_humor.utils import load_encoder_state

model = AutoModel.from_pretrained('bert-base-uncased')
load_encoder_state(model, '${PRETRAIN_CKPT}')
model.save_pretrained('${HF_ENCODER}')
AutoTokenizer.from_pretrained('bert-base-uncased').save_pretrained('${HF_ENCODER}')
print('Saved HF encoder to ${HF_ENCODER}')
"

# ── 3. Fine-tune ColBERT on MS MARCO (starting from our pretrained encoder) ──
python colbert/train.py --config colbert/config.yml

# ── 4. Benchmark trained ColBERT on all public BEIR datasets ────────────────

# Helper: run <short_name> <hf_dataset> <hf_qrels> [extra args]
run() {
    local name=$1 ds=$2 qrels=$3
    shift 3
    echo "=== BEIR: ${name} ==="
    ${EVAL} \
        --dataset        "${ds}" \
        --qrels_dataset  "${qrels}" \
        --experiment_name "colbert_${name}" \
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
        --experiment_name "colbert_cqadupstack_${subset}"
done

echo "All BEIR evaluations complete. Results in results/beir_colbert/."
echo "CQADupstack nDCG@10: average the 12 colbert_cqadupstack_* result files."

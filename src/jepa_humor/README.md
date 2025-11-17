# JEPA-Humor

Self-supervised JEPA-style pretraining on CreativeLang/ColBERT_Humor_Detection, then supervised finetuning for humor detection.

## Setup
```bash
pip install -e .
# or: pip install -r requirements.txt
python -m scripts.pretrain --config configs/pretrain.yml && python -m scripts.finetune --config configs/finetune.yml --encoder_ckpt checkpoints/pretrain/best.pt
python -m scripts.finetune --config configs/finetune_fresh.yml --encoder_ckpt fresh

python -m scripts.pretrain --config configs/pretrain_mnli.yml && python -m scripts.finetune --config configs/finetune_mnli.yml --encoder_ckpt checkpoints/pretrain/best_pretrain_mnli_exp1_.pt
python -m scripts.pretrain --config configs/pretrain_qnli.yml && python -m scripts.finetune --config configs/finetune_mnli.yml --encoder_ckpt checkpoints/pretrain/best_pretrain_qnli_exp1_.pt
python -m scripts.pretrain --config configs/pretrain_rte.yml && python -m scripts.finetune --config configs/finetune_rte.yml --encoder_ckpt checkpoints/pretrain/best_pretrain_rte_exp1_.pt
python -m scripts.pretrain --config configs/pretrain_sst2.yml && python -m scripts.finetune --config configs/finetune_sst2.yml --encoder_ckpt checkpoints/pretrain/best_pretrain_sst2_exp1_.pt
python -m scripts.pretrain --config configs/pretrain_qqp.yml && python -m scripts.finetune --config configs/finetune_qqp.yml --encoder_ckpt checkpoints/pretrain/best_pretrain_qqp_exp1_.pt

BERT-Base Pretrained results
---- MRPC with Finetune on Cola Pretrained ---- 
Replacing model with SWA model...                                                                                                                 
SWA model loaded!
Final validation: mcc=0.6969 f1=0.9110, val_acc=0.8725

---- MRPC with Finetune on MRPC Pretrained---- 
Replacing model with SWA model...                                                                                                                 
SWA model loaded!
Final validation: mcc=0.6849 f1=0.9072, val_acc=0.8676

Advesarial-Glue when compared to BERT-Base

----------------------------------------------------
Test-Set MNLI (+0)
==================================================
Evaluating on MNLI test set (matched, from JSON)...
Test (MNLI matched, from JSON): mcc=-0.0679 f1=0.2778 acc=0.2872
Evaluating on MNLI test set (mismatched, from JSON)...
Test (MNLI mismatched, from JSON): mcc=-0.0942 f1=0.2668 acc=0.2678
----------------------------------------------------

----------------------------------------------------
Test-Set QNLI(+8)
==================================================
Evaluating on QNLI test set...
Test (QNLI): mcc=-0.0471 f1=0.4697 acc=0.4721 # Random Span Masking 
----------------------------------------------------

----------------------------------------------------
Test-Set RTE(+10)
==================================================
Evaluating on RTE test set (from JSON)...
Test (RTE, from JSON): mcc=-0.1755 f1=0.4253 acc=0.4474  # Random Span Masking
Evaluating on RTE test set (from JSON)...
Test (RTE, from JSON): mcc=-0.0580 f1=0.4679 acc=0.5066  # Advesarial Masking (task-aware masking)
----------------------------------------------------


----------------------------------------------------
Test-Set SST-2 (Sentiment Aware Masking)
==================================================



----------------------------------------------------


----------------------------------------------------
Test-Set QQP (+15)
==================================================
Evaluating on QQP test set (from JSON) ...
Test (QQP, from JSON): mcc=-0.1606 f1=4191, acc==53.32 # Random Span Masking
----------------------------------------------------




System MNLI-(m/mm) QQP QNLI SST-2 CoLA STS-B MRPC RTE Average
392k 363k 108k 67k 8.5k 5.7k 3.5k 2.5k -
Pre-OpenAI SOTA 80.6/80.1 66.1 82.3 93.2 35.0 81.0 86.0 61.7 74.0
BiLSTM+ELMo+Attn 76.4/76.1 64.8 79.8 90.4 36.0 73.3 84.9 56.8 71.0
OpenAI GPT 82.1/81.4 70.3 87.4 91.3 45.4 80.0 82.3 56.0 75.1
BERTBASE 84.6/83.4 71.2 90.5 93.5 52.1 85.8 88.9 66.4 79.6
BERTLARGE 86.7/85.9 72.1 92.7 94.9 60.5 86.5 89.3 70.1 82.1

Table 4: Model performance on AdvGLUE test set. BERT (Large) and RoBERTa (Large) are fine-tuned
using different random seeds and thus different from the surrogate models used for adversarial text generation.
For MNLI, we report the test accuracy on the matched and mismatched test sets; for QQP, we report accuracy
and F1; and for other tasks, we report the accuracy. All values are reported by percentage (%). We also report
the macro-average (Avg) of per-task scores for different models. (Complete results are listed in our leaderboard.)
Model SST-2 MNLI RTE QNLI QQP Avg Avg Avg
AdvGLUE AdvGLUE AdvGLUE AdvGLUE AdvGLUE AdvGLUE GLUE ∆ ↓
State-of-the-art Pre-trained Language Models
BERT (Large) 33.03 28.72/27.05 40.46 39.77 37.91/16.56 33.68 85.76 52.08
ELECTRA (Large) 58.59 14.62/20.22 23.03 57.54 61.37/42.40 41.69 93.16 51.47
RoBERTa (Large) 58.52 50.78/39.62 45.39 52.48 57.11/41.80 50.21 91.44 41.23
T5 (Large) 60.56 48.43/38.98 62.83 57.64 63.03/55.68 56.82 90.39 33.57
ALBERT (XXLarge) 66.83 51.83/44.17 73.03 63.84 56.40/32.35 59.22 91.87 32.65
DeBERTa (Large) 57.89 58.36/52.46 78.95 57.85 60.43/47.98 60.86 92.67 31.81
Robust Training Methods for Pre-trained Language Models
SMART (BERT) 25.21 26.89/23.32 38.16 34.61 36.49/20.24 30.29 85.70 55.41
SMART (RoBERTa) 50.92 45.56/36.07 70.39 52.17 64.22/44.28 53.71 92.62 38.91
FreeLB (RoBERTa) 61.69 31.59/27.60 62.17 62.29 42.18/31.07 50.47 92.28 41.81
InfoBERT (RoBERTa) 47.61 50.39/41.26 39.47 54.86 49.29/35.54 46.04 89.06 43.0
# JEPA-Humor

Self-supervised JEPA-style pretraining on CreativeLang/ColBERT_Humor_Detection, then supervised finetuning for humor detection.

## Setup
```bash
pip install -e .
# or: pip install -r requirements.txt
python -m scripts.pretrain --config configs/pretrain.yml && python -m scripts.finetune --config configs/finetune.yml --encoder_ckpt checkpoints/pretrain/best.pt
python -m scripts.finetune --config configs/finetune_fresh.yml --encoder_ckpt fresh

python -m scripts.pretrain --config configs/pretrain_mnli.yml && python -m scripts.finetune --config configs/finetune_mnli.yml --encoder_ckpt checkpoints/pretrain/best_pretrain_mnli_exp1_.pt
python -m scripts.pretrain --config configs/pretrain_qnli.yml && python -m scripts.finetune --config configs/finetune_qnli.yml --encoder_ckpt checkpoints/pretrain/best_pretrain_qnli_exp1_.pt
python -m scripts.pretrain --config configs/pretrain_rte.yml && python -m scripts.finetune --config configs/finetune_rte.yml --encoder_ckpt checkpoints/pretrain/best_pretrain_rte_exp1_.pt
python -m scripts.pretrain --config configs/pretrain_sst2.yml && python -m scripts.finetune --config configs/finetune_sst2.yml --encoder_ckpt checkpoints/pretrain/best_pretrain_sst2_exp1_.pt
python -m scripts.pretrain --config configs/pretrain_qqp.yml && python -m scripts.finetune --config configs/finetune_qqp.yml --encoder_ckpt checkpoints/pretrain/best_pretrain_qqp_exp1_.pt

--Zero Shot--
python -m scripts.finetune --config configs/finetune_qqp.yml --encoder_ckpt checkpoints/pretrain/best_pretrain_qqp_exp1_.pt --eval_only
python -m scripts.finetune --config configs/finetune_sst2.yml --encoder_ckpt checkpoints/pretrain/best_pretrain_sst2_exp1_.pt --eval_only
python -m scripts.finetune --config configs/finetune_rte.yml --encoder_ckpt checkpoints/pretrain/best_pretrain_rte_exp1_.pt --eval_only
python -m scripts.finetune --config configs/finetune_qnli.yml --encoder_ckpt /home/khiem/nlp/checkpoints/pretrain/pretrain_adv_mask_qnli_exp1_jepa_epoch5_loss4.7826.pt--eval_only
python -m scripts.finetune --config configs/finetune_mnli.yml --encoder_ckpt /home/khiem/nlp/checkpoints/pretrain/pretrain_mnli_exp1_jepa_epoch5_loss3.9961.pt --eval_only


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
Test (QNLI): mcc=-0.0471 f1=0.4697 acc=0.4721 # Zero-shot w/Adversarial Masking
----------------------------------------------------

----------------------------------------------------
Test-Set RTE(+10) (Pretrain+Finetune) (+10)
==================================================
Evaluating on RTE test set (from JSON)...
Test (RTE, from JSON): mcc=-0.1755 f1=0.4253 acc=0.4474  # Random Span Masking
Evaluating on RTE test set (from JSON)...
Test (RTE, from JSON): mcc=-0.1755 f1=0.4253 acc=0.4474  # Zero-shot w/ Advesarial Masking
Evaluating on RTE test set (from JSON)...
Test (RTE, from JSON): mcc=-0.0580 f1=0.4679 acc=0.5066  # Advesarial Masking (task-aware masking)
----------------------------------------------------


----------------------------------------------------
Test-Set SST-2 (Sentiment Aware Masking) (+24)
==================================================
Evaluating on SST2 test set (from JSON)...
Test (SST2, from JSON): mcc=-0.0017 f1=0.4046 acc=0.5761
----------------------------------------------------


----------------------------------------------------
Test-Set QQP (+22)
==================================================
Evaluating on QQP test set (from JSON) ...
Test (QQP, from JSON): mcc=-0.1606 f1=4191, acc==53.32 # Advesarial Masking QQP 
Evaluating on QQP test set (from JSON) ...
Test (QQP, from JSON): mcc=-0.1606 f1=55.34, acc==62.09 # Advesarial Masking QQP
----------------------------------------------------

# JEPA-Humor

Self-supervised JEPA-style pretraining on CreativeLang/ColBERT_Humor_Detection, then supervised finetuning for humor detection.

## Setup
```bash
pip install -e .
# or: pip install -r requirements.txt
python -m scripts.pretrain --config configs/pretrain.yml && python -m scripts.finetune --config configs/finetune.yml --encoder_ckpt checkpoints/pretrain/best.pt
python -m scripts.finetune --config configs/finetune_fresh.yml --encoder_ckpt fresh

python -m scripts.pretrain --config configs/pretrain_mnli.yml && python -m scripts.finetune --config configs/finetune_mnli.yml --encoder_ckpt checkpoints/pretrain/best_pretrain_mnli_exp1_.pt
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

Advesarial-Glue

----------------------------------------------------
Test-Set MNLI
==================================================

----------------------------------------------------

----------------------------------------------------
Test-Set QNLI
==================================================
Evaluating on QNLI test set...
Test (QNLI): mcc=-0.0471 f1=0.4697 acc=0.4721
----------------------------------------------------

----------------------------------------------------
Test-Set RTE
==================================================
Evaluating on RTE test set (from JSON)...
Test (RTE, from JSON): mcc=-0.1755 f1=0.4253 acc=0.4474  
----------------------------------------------------


----------------------------------------------------
Test-Set SST-2
==================================================

----------------------------------------------------


----------------------------------------------------
Test-Set QQP
==================================================

----------------------------------------------------
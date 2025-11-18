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
python -m scripts.finetune --config configs/finetune_qnli.yml --model_ckpt /home/khiem/nlp/checkpoints/pretrain/pretrain_adv_mask_qnli_exp1_jepa_epoch5_loss4.7826.pt--eval_only
python -m scripts.finetune --config configs/finetune_mnli.yml --encoder_ckpt /home/khiem/nlp/checkpoints/pretrain/pretrain_mnli_exp1_jepa_epoch5_loss3.9961.pt --eval_only

# MNLI
python -m scripts.finetune --config configs/finetune_qqp.yml --model_ckpt /home/khiem/nlp/checkpoints/pretrain/pretrain_mnli_exp1_jepa_epoch4_loss0.8451.pt --eval_only
python -m scripts.finetune --config configs/finetune_qnli.yml --model_ckpt /home/khiem/nlp/checkpoints/pretrain/pretrain_mnli_exp1_jepa_epoch4_loss0.8451.pt --eval_only && python -m scripts.finetune --config configs/finetune_rte.yml --model_ckpt /home/khiem/nlp/checkpoints/pretrain/pretrain_mnli_exp1_jepa_epoch4_loss0.8451.pt --eval_only && python -m scripts.finetune --config configs/finetune_sst2.yml --model_ckpt /home/khiem/nlp/checkpoints/pretrain/pretrain_mnli_exp1_jepa_epoch4_loss0.8451.pt --eval_only

#QNLI
python -m scripts.finetune --config configs/finetune_qqp.yml --model_ckpt checkpoints/pretrain/best_pretrain_adv_mask_qnli_exp1_.pt --eval_only && python -m scripts.finetune --config configs/finetune_rte.yml --model_ckpt checkpoints/pretrain/best_pretrain_adv_mask_qnli_exp1_.pt --eval_only && python -m scripts.finetune --config configs/finetune_sst2.yml --model_ckpt checkpoints/pretrain/best_pretrain_adv_mask_qnli_exp1_.pt --eval_only && python -m scripts.finetune --config configs/finetune_mnli.yml --model_ckpt checkpoints/pretrain/best_pretrain_adv_mask_qnli_exp1_.pt --eval_only
checkpoints/pretrain/best_pretrain_qqp_exp1_.pt

# QQP
python -m scripts.finetune --config configs/finetune_qnli.yml --model_ckpt checkpoints/pretrain/best_pretrain_qqp_exp1_.pt  --eval_only && python -m scripts.finetune --config configs/finetune_sst2.yml --model_ckpt checkpoints/pretrain/best_pretrain_qqp_exp1_.pt --eval_only && python -m scripts.finetune --config configs/finetune_rte.yml --model_ckpt checkpoints/pretrain/best_pretrain_qqp_exp1_.pt--eval_only && python -m scripts.finetune --config configs/finetune_mnli.yml --model_ckptcheckpoints/pretrain/best_pretrain_qqp_exp1_.pt --eval_only

#RTE
python -m scripts.finetune --config configs/finetune_qnli.yml --model_ckpt checkpoints/pretrain/best_pretrain_rte_exp1_.pt --eval_only && python -m scripts.finetune --config configs/finetune_mnli.yml --model_ckpt /home/khiem/nlp/checkpoints/pretrain/best_pretrain_rte_exp1_.pt --eval_only && python -m scripts.finetune --config configs/finetune_sst2.yml --model_ckpt /home/khiem/nlp/checkpoints/pretrain/best_pretrain_rte_exp1_.pt --eval_only && python -m scripts.finetune --config configs/finetune_qqp.yml --model_ckpt /home/khiem/nlp/checkpoints/pretrain/best_pretrain_rte_exp1_.pt --eval_only

#SST2
python -m scripts.finetune --config configs/finetune_qnli.yml --model_ckpt /home/khiem/nlp/checkpoints/pretrain/best_pretrain_sst2_exp1_.pt --eval_only && python -m scripts.finetune --config configs/finetune_rte.yml --model_ckpt /home/khiem/nlp/checkpoints/pretrain/best_pretrain_sst2_exp1_.pt --eval_only && python -m scripts.finetune --config configs/finetune_qqp.yml --model_ckpt /home/khiem/nlp/checkpoints/pretrain/pretrain_sst2_exp1_.pt --eval_only

----------------------------------------------------
Test-Set MNLI (SOA: Matched 0.289, Unmached: 0.270) +4, +2
==================================================
Evaluating on MNLI test set (matched, from JSON)...
test (MNLI): mcc=0.0096 f1=0.3272 acc=0.3394
test (MNLI-MM): mcc=-0.0559 f1=0.2901 acc=0.2905
----------------------------------------------------

----------------------------------------------------
Test-Set QNLI(SOA: acc= 39.77) + 8
==================================================
Evaluating on QNLI test set...
Test (QNLI): mcc=-0.0471 f1=0.4697 acc=0.4721 # QNLI
Evaluating on QNLI test set (from JSON)...
Test (QNLI, from JSON): mcc=0.0358 f1=0.3050 acc=0.4132 # QQP
Evaluating on QNLI test set (from JSON)...
Test (QNLI, from JSON): mcc=-0.0507 f1=0.4318 acc=0.4390 #SST2
Evaluating on QNLI test set (from JSON)...
Test (QNLI, from JSON): mcc=-0.0860 f1=0.4463 acc=0.5072 #RTE

----------------------------------------------------
Test-Set RTE(+10) (Pretrain+Finetune) (SOA: 0.4046) + 10
==================================================
Evaluating on RTE test set (from JSON)...
Test (RTE, from JSON): mcc=-0.1755 f1=0.4253 acc=0.4474   (RTE-Random Span Masking)
Test (RTE, from JSON): mcc=-0.0580 f1=0.4679 acc=0.5066   (RTE-Task Aware)
Evaluating on RTE test set (from JSON)...
Test (RTE, from JSON): mcc=-0.0662 f1=0.4648 acc=0.5000  QNLI 
Evaluating on RTE test set (from JSON)...
Test (RTE, from JSON): mcc=0.0584 f1=0.3656 acc=0.4342  QQP 
Evaluating on RTE test set (from JSON)...
Test (RTE, from JSON): mcc=0.0304 f1=0.5075 acc=0.5099 SST2 
----------------------------------------------------


----------------------------------------------------
Test-Set SST-2 (Sentiment Aware Masking) (SOA: acc=0.3303) (-10)
==================================================
Evaluating on SST2 test set (from JSON)...
Test (SST2, from JSON): mcc=-0.0017 f1=0.4046 acc=0.5761 (RTE)
Evaluating on SST2 test set (from JSON)...
Test (SST2, from JSON): mcc=-0.0407 f1=0.4522 acc=0.5359 (QNLI)
Evaluating on SST2 test set (from JSON)...
Test (SST2, from JSON): mcc=-0.0445 f1=0.2928 acc=0.4141 (QQP)
Evaluating on SST2 test set (from JSON)...
Test (SST2, from JSON): mcc=-0.5156 f1=0.2385 acc=0.2401 (SST-2)
----------------------------------------------------


----------------------------------------------------
Test-Set QQP (SOA: acc=0.3791, f1=0.1656 ) + 15
==================================================
Evaluating on QQP test set (from JSON) ...
Test (QQP, from JSON): mcc=-0.1606 f1=0.4191, acc==0.5332 (QQP)
Evaluating on QQP test set (from JSON)...
Test (QQP, from JSON): mcc=0.0440 f1=0.5190 acc=0.5735 (RTE)
Evaluating on QQP test set (from JSON)...
Test (QQP, from JSON): mcc=0.0871 f1=0.5435 acc=0.6185 (QNLI)
Evaluating on QQP test set (from JSON)...
Test (QQP, from JSON): mcc=0.0973 f1=0.5294 acc=0.6754 (SST2)
----------------------------------------------------

Zero-shot transfer tests:
--------------------------------------------------------
MNLI -> QQP
Evaluating on QQP test set (from JSON)...
Test (QQP, from JSON): mcc=0.0635 f1=0.2435 acc=0.3057

MNLI -> QNLI
Evaluating on QNLI test set (from JSON)...
Test (QNLI, from JSON): mcc=0.0295 f1=0.3774 acc=0.5940

MNLI -> RTE
Evaluating on RTE test set (from JSON)...
Test (RTE, from JSON): mcc=-0.0254 f1=0.3904 acc=0.5855

MNLI -> SST2
Evaluating on SST2 test set (from JSON)...
Test (SST2, from JSON): mcc=0.0068 f1=0.3886 acc=0.5817
-------------------------------------------------------


Zero-shot transfer tests:
--------------------------------------------------------
QNLI -> QQP
Evaluating on QQP test set (from JSON)...
Test (QQP, from JSON): mcc=0.0635 f1=0.2572 acc=0.3104

QNLI -> MNLI matched
Evaluating on MNLI test set (matched, from JSON)...
Test (MNLI matched, from JSON): mcc=-0.0019 f1=0.1576 acc=0.3042
Evaluating on MNLI test set (mismatched, from JSON)...
Test (MNLI mismatched, from JSON): mcc=0.0299 f1=0.1803 acc=0.3333

QNLI -> RTE
Evaluating on RTE test set (from JSON)...
Test (RTE, from JSON): mcc=-0.0254 f1=0.3732 acc=0.5954

QNLI -> SST2
Evaluating on SST2 test set (from JSON)...
Test (SST2, from JSON): mcc=0.0068 f1=0.3631 acc=0.5831
-------------------------------------------------------


Zero-shot transfer tests:
--------------------------------------------------------
RTE -> QQP
Evaluating on QQP test set (from JSON)...
Test (QQP, from JSON): mcc=0.0000 f1=0.2285 acc=0.2962

RTE -> MNLI matched
aluating on MNLI test set (matched, from JSON)...
Test (MNLI matched, from JSON): mcc=0.0299 f1=0.1581 acc=0.3055
Evaluating on MNLI test set (mismatched, from JSON)...
Test (MNLI mismatched, from JSON): mcc=0.0098 f1=0.1739 acc=0.3297

RTE -> QNLI
Evaluating on RTE test set (from JSON)...
Test (QNLI, from JSON): mcc=0.0572 f1=0.3987 acc=0.5971

RTE -> SST2
Evaluating on SST2 test set (from JSON)...
Test (SST2, from JSON): mcc=0.0000 f1=0.3689 acc=0.5845
-------------------------------------------------------


Zero-shot transfer tests:
--------------------------------------------------------
SST2 -> MNLI
Evaluating on MNLI test set (matched, from JSON)...
Test (MNLI matched, from JSON): mcc=-0.0028 f1=0.1599 acc=0.3042
Evaluating on MNLI test set (mismatched, from JSON)...
Test (MNLI mismatched, from JSON): mcc=-0.0155 f1=0.1799 acc=0.3224

SST2 -> QNLI
Evaluating on QNLI test set (from JSON)...
Test (QNLI, from JSON): mcc=0.0000 f1=0.3722 acc=0.5930

SST2 -> QQP
Evaluating on QQP test set (from JSON)...
Test (QQP, from JSON): mcc=0.0000 f1=0.2285 acc=0.2962


SST2 -> RTE
Evaluating on RTE test set (from JSON)...
Test (RTE, from JSON): mcc=0.0000 f1=0.3732 acc=0.5954
-------------------------------------------------------

Zero-shot transfer tests:
------------------------------------------------------
QQP -> SST2
Test (SST2, from JSON): mcc=-0.0042 f1=0.4298 acc=0.5676

QQP -> QNLI
Evaluating on QNLI test set (from JSON)...
Test (QNLI, from JSON): mcc=0.0122 f1=0.3769 acc=0.5930

QQP -> MNLI 
Test (MNLI matched, from JSON): mcc=0.0495 f1=0.3426 acc=0.3564
Evaluating on MNLI test set (mismatched, from JSON)...
Test (MNLI mismatched, from JSON): mcc=0.0245 f1=0.3245 acc=0.3388

QQP -> RTE
Evaluating on RTE test set (from JSON)...
Test (RTE, from JSON): mcc=0.0000 f1=0.3732 acc=0.5954


ANLI Dataset
----------------------------

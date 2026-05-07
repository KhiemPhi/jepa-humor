# JEPA-Humor

Self-supervised JEPA-style pretraining on CreativeLang/ColBERT_Humor_Detection, then supervised finetuning for humor detection.

   
python -m scripts.pretrain --config configs/pretrain_sst2.yml && python 
-m scripts.finetune --config configs/finetune_sst2.yml --encoder_ckpt checkpoints/pretrain/best_pretrain_sst2_
exp1_.pt

for task in rte sst2 qnli mnli qqp; do python -m scripts.pretrain --config  configs/pretrain_${task}.yml && python -m scripts.finetune --config configs/finetune_${task}.yml  --encoder_ckpt checkpoints/pretrain/best_pretrain_${task}_exp1_.pt; done 

for task in qnli mnli qqp; do python -m scripts.pretrain --config configs/pretrain_${task}.yml --gpu_id 2 dynamic_masking=true && python -m scripts.finetune --config configs/finetune_${task}.yml --gpu_id 2 dynamic_masking=true --encoder_ckpt checkpoints/pretrain/best_pretrain_${task}_exp1_dynamic_masking_.pt; done

for task in mnli; do python -m scripts.finetune --config configs/finetune_${task}.yml --gpu_id 4 --encoder_ckpt checkpoints/pretrain/best_pretrain_${task}_exp1_.pt; done 

tmux new -s jepa_goat
tmux attach -t jepa_goat
tmux attach -t rte, mnli
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
python -m scripts.pretrain --config configs/pretrain_swag.yml && python -m scripts.finetune --config configs/finetune_swag.yml --encoder_ckpt checkpoints/pretrain/best_pretrain_swag_exp1_.pt

#ANLI Zero Shot
python -m scripts.finetune --config configs/finetune_anli.yml --model_ckpt checkpoints/pretrain/best_pretrain_anli_exp1_.pt --eval_only
python -m scripts.finetune --config configs/finetune_qqp.yml --model_ckpt checkpoints/pretrain/best_pretrain_anli_exp1_.pt --eval_only
python -m scripts.finetune --config configs/finetune_qnli.yml --model_ckpt checkpoints/pretrain/best_pretrain_anli_exp1_.pt --eval_only
python -m scripts.finetune --config configs/finetune_mnli.yml --model_ckpt checkpoints/pretrain/best_pretrain_anli_exp1_.pt --eval_only
python -m scripts.finetune --config configs/finetune_sst2.yml --model_ckpt checkpoints/pretrain/best_pretrain_anli_exp1_.pt --eval_only
python -m scripts.finetune --config configs/finetune_rte.yml --model_ckpt checkpoints/pretrain/best_pretrain_anli_exp1_.pt --eval_only
python -m scripts.finetune --confing configs/finetune_anli.yml --model_ckpt  checkpoints/pretrain/best_pretrain_snli_mnli_anli_exp1_.pt --eval_only

python -m scripts.finetune --config configs/finetune_rte.yml --model_ckpt /home/khiem/nlp/checkpoints/pretrain/pretrain_qqp_exp1_jepa_epoch3_loss3.6761.pt
python -m scripts.finetune --config configs/finetune_sst2.yml --model_ckpt checkpoints/pretrain/best_pretrain_composite_nli_exp1_.pt --eval_only

python -m scripts.finetune --config configs/finetune_sst2.yml --model_ckpt checkpoints/pretrain/best_pretrain_rte_mlm_exp1_.pt --eval_only
python -m scripts.finetune --config configs/finetune_sst2.yml --model_ckpt checkpoints/pretrain/best_pretrain_rte_exp1_.pt --eval_only

python -m scripts.finetune --config configs/finetune_rte.yml --model_ckpt checkpoints/pretrain/best_pretrain_rte_mlm_exp1_.pt --eval_only &&
python -m scripts.finetune --config configs/finetune_rte.yml --model_ckpt checkpoints/pretrain/best_pretrain_rte_exp1_.pt --eval_only


python -m scripts.finetune --config configs/finetune_qqp.yml --model_ckpt checkpoints/pretrain/best_pretrain_rte_mlm_exp1_.pt --eval_only &&
python -m scripts.finetune --config configs/finetune_qqp.yml --model_ckpt checkpoints/pretrain/best_pretrain_rte_exp1_.pt --eval_only

python -m scripts.finetune --config configs/finetune_anli.yml --model_ckpt checkpoints/pretrain/best_pretrain_rte_mlm_exp1_.pt --eval_only &&
python -m scripts.finetune --config configs/finetune_anli.yml --model_ckpt checkpoints/pretrain/best_pretrain_rte_exp1_.pt --eval_only

python -m scripts.finetune --config configs/finetune_rte --model_ckpt /home/khiem/nlp/checkpoints/pretrain/pretrain_sst2_exp1_jepa_epoch5_loss1.3255.pt --eval_only
python -m scripts.finetune --config configs/finetune_rte.yml --model_ckpt checkpoints/best_pretrain_sst2_mlm_exp1_.pt --eval_only



python -m scripts.finetune --config configs/finetune_rte --model_ckpt  /home/khiem/nlp/checkpoints/finetune/finetune_qnli_exp1_best.pt --eval_only










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
Test (QQP, from JSON): mcc=-0.1606 f1=0.4191, acc==0.5332 (QQP) -> random
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
==================================================
Evaluating on QQP test set (from JSON)...
Test (QQP, from JSON): mcc=-0.0751 f1=0.2271 acc=0.2938

MNLI -> QNLI
Evaluating on QNLI test set (from JSON)...
Test (QNLI, from JSON): mcc=0.0295 f1=0.3774 acc=0.5940
==================================================
Evaluating on QNLI test set (from JSON)...
Test (QNLI, from JSON): mcc=0.1772 f1=0.5885 acc=0.6012




MNLI -> SST2
Evaluating on SST2 test set (from JSON)...
Test (SST2, from JSON): mcc=0.0068 f1=0.3886 acc=0.5817
Test (SST2, from JSON): mcc=0.0468 f1=0.5164 acc=0.5542 -3



Zero-Shot MLM QQP -> ANLI
==================================================
Evaluating on ANLI test_r1...
Test (test_r1): f1=0.1665 acc=0.3330
Evaluating on ANLI test_r2...
Test (test_r2): f1=0.1665 acc=0.3330
Evaluating on ANLI test_r3...
Test (test_r3): f1=0.1654 acc=0.3300

Zero-Shot My Method QQP -> ANLI
==================================================
Evaluating on ANLI test_r1...
Test (test_r1): f1=0.2360 acc=0.3200
Evaluating on ANLI test_r2...
Test (test_r2): f1=0.2361 acc=0.3190
Evaluating on ANLI test_r3...
Test (test_r3): f1=0.2463 acc=0.3342

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

==================================================
Evaluating on ANLI test_r1...
Test (test_r1): mcc=-0.0198 f1=0.2548 acc=0.3230
Evaluating on ANLI test_r2...
Test (test_r2): mcc=-0.0282 f1=0.2434 acc=0.3190
Evaluating on ANLI test_r3...
Test (test_r3): mcc=-0.0090 f1=0.2588 acc=0.3267
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


RTE -> ANLI
Evaluating on ANLI test_r1...
Test (test_r1): mcc=0.0047 f1=0.1851 acc=0.3340
Evaluating on ANLI test_r2...
Test (test_r2): mcc=-0.0381 f1=0.1762 acc=0.3260
Evaluating on ANLI test_r3...
Test (test_r3): mcc=0.0113 f1=0.1777 acc=0.3317
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

SST2 -> ANLI 
==================================================
Evaluating on ANLI test_r1...
Test (test_r1): mcc=-0.0338 f1=0.2181 acc=0.3190
Evaluating on ANLI test_r2...
Test (test_r2): mcc=0.0055 f1=0.2172 acc=0.3350
Evaluating on ANLI test_r3...
Test (test_r3): mcc=-0.0321 f1=0.1960 acc=0.3200

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

QQP -> ANLI
Evaluating on ANLI test_r1...
Test (test_r1): mcc=0.0342 f1=0.2921 acc=0.3510
Evaluating on ANLI test_r2...
Test (test_r2): mcc=-0.0022 f1=0.2834 acc=0.3320
Evaluating on ANLI test_r3...
Test (test_r3): mcc=0.0187 f1=0.2785 acc=0.3408


ANLI Dataset
----------------------------

 
A1+A2+A3 (120k data points)
test (test_r1): mcc=0.2308 f1=0.4772 acc=0.4840
test (test_r2): mcc=0.1633 f1=0.4269 acc=0.4390
test (test_r3): mcc=0.1247 f1=0.4098 acc=0.4158


A1+A2

A1

A1+A2+A3+SNLI+M
Evaluating on ANLI test_r1...
Test (test_r1): mcc=0.0000 f1=0.1665 acc=0.3330
Evaluating on ANLI test_r2...
Test (test_r2): mcc=0.0000 f1=0.1665 acc=0.3330
Evaluating on ANLI test_r3...
Test (test_r3): mcc=-0.0083 f1=0.1672 acc=0.3292
✅ Evaluation complete.


SNLI + MNLI Dev


-----------------------------------------------







Zero-shot transfer tests

ANLI -> ANLI
==================================================
Evaluating on ANLI test_r1...
Test (test_r1): mcc=-0.0083 f1=0.2594 acc=0.3290
Evaluating on ANLI test_r2...
Test (test_r2): mcc=0.0056 f1=0.2566 acc=0.3360
Evaluating on ANLI test_r3...
Test (test_r3): mcc=0.0116 f1=0.2875 acc=0.3383

ANLI -> RTE
==================================================
Evaluating on RTE test set (from JSON)...
Test (RTE, from JSON): mcc=-0.1012 f1=0.4289 acc=0.5164

ANLI -> QQP
==================================================
Evaluating on QQP test set (from JSON)...
Test (QQP, from JSON): mcc=-0.0167 f1=0.4051 acc=0.4052

ANLI -> QNLI
==================================================
Evaluating on QNLI test set (from JSON)...
Test (QNLI, from JSON): mcc=0.0274 f1=0.4845 acc=0.5682

ANLI -> MNLI
Evaluating on MNLI test set (matched, from JSON)...
Test (MNLI matched, from JSON): mcc=0.0088 f1=0.2985 acc=0.3251
Evaluating on MNLI test set (mismatched, from JSON)...
Test (MNLI mismatched, from JSON): mcc=0.0128 f1=0.2954 acc=0.3452

ANLI -> SST2
==================================================
Evaluating on SST2 test set (from JSON)...
Test (SST2, from JSON): mcc=-0.0292 f1=0.4308 acc=0.5556
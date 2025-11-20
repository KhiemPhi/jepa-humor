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


Jepa: [Finetune] epoch 1: loss=1.9561 EM=0.3437 F1=0.5020                                                           
          val: loss=1.4250 EM=0.4246 F1=0.5900
[Finetune] epoch 2: loss=1.3205 EM=0.4342 F1=0.6079                                                           
          val: loss=1.2857 EM=0.4388 F1=0.6088
Starting SWA at epoch 3
[Finetune] epoch 3: loss=1.1180 EM=0.4622 F1=0.6337                                                           
          val: loss=1.2593 EM=0.4473 F1=0.6188


Base:     val: EM=79.4986 F1=87.2234


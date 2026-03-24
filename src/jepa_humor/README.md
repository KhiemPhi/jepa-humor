# JEPA-Humor

Self-supervised JEPA-style pretraining on CreativeLang/ColBERT_Humor_Detection, then supervised finetuning for humor detection.

## Setup
```bash
pip install -e .
# or: pip install -r requirements.txt
python -m scripts.pretrain --config configs/pretrain.yml && python -m scripts.finetune --config configs/finetune.yml --encoder_ckpt checkpoints/pretrain/best.pt
python -m scripts.finetune --config configs/finetune_fresh.yml --encoder_ckpt fresh


No Jepa:  Epoch 15: Train Loss=0.1653, Pearson=0.9608, Spearman=0.9556, Final=0.9582                                    
          val: Pearson=0.9008, Spearman=0.8982, Final=0.8995
          best STS-B score so far: 0.9027
✅ Finetune done.



Jepa: Epoch 15: Train Loss=0.1445, Pearson=0.9659, Spearman=0.9617, Final=0.9638                                    
          val: Pearson=0.9041, Spearman=0.9026, Final=0.9034
          best STS-B score so far: 0.9059
✅ Finetune done.
# JEPA-Humor

Self-supervised JEPA-style pretraining on CreativeLang/ColBERT_Humor_Detection, then supervised finetuning for humor detection.

## Setup
```bash
pip install -e .
# or: pip install -r requirements.txt
python -m scripts.pretrain --config configs/pretrain.yml && python -m scripts.finetune --config configs/finetune.yml --encoder_ckpt checkpoints/pretrain/best.pt
python -m scripts.finetune --config configs/finetune_fresh.yml --encoder_ckpt fresh


Jepa, pretrain 2 epoch, constrastive loss:
Evaluating SWA model vs best single epoch model...
Best single epoch: 7 with MCC: 0.6719
SWA model MCC: 0.6157 (threshold=0.250)
Final validation: mcc=0.6157 f1=0.8934
✅ Finetune done.

No Jepa:
Evaluating SWA model vs best single epoch model...
Best single epoch: 7 with MCC: 0.6648
SWA model MCC: 0.6157 (threshold=0.310)
Final validation: mcc=0.6157 f1=0.8930
✅ Finetune done.
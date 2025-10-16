# JEPA-Humor

Self-supervised JEPA-style pretraining on CreativeLang/ColBERT_Humor_Detection, then supervised finetuning for humor detection.

## Setup
```bash
pip install -e .
# or: pip install -r requirements.txt
python -m scripts.pretrain --config configs/pretrain.yaml
python -m scripts.finetune --config configs/finetune.yaml --encoder_ckpt checkpoints/pretrain/best.pt

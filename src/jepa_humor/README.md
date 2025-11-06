# JEPA-Humor

Self-supervised JEPA-style pretraining on CreativeLang/ColBERT_Humor_Detection, then supervised finetuning for humor detection.

## Setup
```bash
pip install -e .
# or: pip install -r requirements.txt
python -m scripts.pretrain --config configs/pretrain.yml && python -m scripts.finetune --config configs/finetune.yml --encoder_ckpt checkpoints/pretrain/best.pt
python -m scripts.finetune --config configs/finetune_fresh.yml --encoder_ckpt fresh



Comparision Bert_base_score = [Finetune] epoch 15: loss=0.0205 acc=1.0000 f1=1.0000                                                                                             
          val: acc=0.8393 f1=0.8284
          Best val: acc=0.8393 f1=0.8284

Bert_large_score: [Finetune] epoch 15: loss=0.0051 acc=1.0000 f1=1.0000                                                                                             
          val: acc=0.8571 f1=0.8502
          Best val: acc=0.8750 f1=0.8821

Bert_base + JEPA: pretrain 10: [Finetune] epoch 15: loss=0.0163 acc=1.0000 f1=1.0000                                                                                                  
          val: acc=0.85393 f1=0.8135
          Best val: acc=0.8571 f1=0.8650
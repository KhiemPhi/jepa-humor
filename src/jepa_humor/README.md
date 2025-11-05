# JEPA-Humor

Self-supervised JEPA-style pretraining on CreativeLang/ColBERT_Humor_Detection, then supervised finetuning for humor detection.

## Setup
```bash
pip install -e .
# or: pip install -r requirements.txt
python -m scripts.pretrain --config configs/pretrain.yml && python -m scripts.finetune --config configs/finetune.yml --encoder_ckpt checkpoints/pretrain/best.pt
python -m scripts.finetune --config configs/finetune_fresh.yml --encoder_ckpt fresh



Comparision score = [Finetune] epoch 15: loss=0.0075 acc=1.0000 f1=1.0000                                                                                                  
          val: acc=0.8214 f1=0.8397
          Best val: acc=0.8214 f1=0.8397



          epoch6 [Finetune] epoch 15: loss=0.0196 acc=1.0000 f1=1.0000                                                                                                  
          val: acc=0.8571 f1=0.8272
          Best val: acc=0.8571 f1=0.8272
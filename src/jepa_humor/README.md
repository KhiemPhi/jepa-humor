# JEPA-Humor

Self-supervised JEPA-style pretraining on CreativeLang/ColBERT_Humor_Detection, then supervised finetuning for humor detection.

## Setup
```bash
pip install -e .
# or: pip install -r requirements.txt
python -m scripts.pretrain --config configs/pretrain.yml && python -m scripts.finetune --config configs/finetune.yml --encoder_ckpt checkpoints/pretrain/best.pt
python -m scripts.finetune --config configs/finetune_fresh.yml --encoder_ckpt fresh


Jepa: [Finetune] epoch 4: loss=0.0962 | acc=0.9779                    
          val: acc=0.7112 (best=0.7329)
[Finetune] epoch 5: loss=0.0534 | acc=0.9876                    
          val: acc=0.7004 (best=0.7329)
  ✓ New best run! Validation acc: 0.7329



  No Jepa:
  [Finetune] epoch 4: loss=0.0815 | acc=0.9795                                          
          val: acc=0.7798 (best=0.7798)
[Finetune] epoch 5: loss=0.0375 | acc=0.9924                                          
          val: acc=0.7762 (best=0.7798)

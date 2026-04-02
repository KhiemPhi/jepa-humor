# JEPA-Humor

Self-supervised JEPA-style pretraining on CreativeLang/ColBERT_Humor_Detection, then supervised finetuning for humor detection.

## Setup
```bash
pip install -e .
# or: pip install -r requirements.txt
python -m scripts.pretrain --config configs/pretrain.yml && python -m scripts.finetune --config configs/finetune.yml --encoder_ckpt checkpoints/pretrain/best.pt
python -m scripts.finetune --config configs/finetune_fresh.yml --encoder_ckpt fresh


          val: acc=0.6318 (best=0.6318)
[Finetune] epoch 2: loss=0.0340 | acc=0.9932                                                          
          val: acc=0.5921 (best=0.6318)
[Finetune] epoch 3: loss=0.0167 | acc=0.9968                                                          
          val: acc=0.6137 (best=0.6318)`````
[Finetune] epoch 4: loss=0.0036 | acc=0.9992                                                          
          val: acc=0.6173 (best=0.6318)
[Finetune] epoch 5: loss=0.0055 | acc=0.9988                                                          
          val: acc=0.6065 (best=0.6318)
  Run 9 best val acc: 0.6318 (overall best: 0.6679)

============================================================
Best run validation accuracy: 0.6679
============================================================

✅ Finetune done.



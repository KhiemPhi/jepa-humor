# JEPA-Humor

Self-supervised JEPA-style pretraining on CreativeLang/ColBERT_Humor_Detection, then supervised finetuning for humor detection.

## Setup
```bash
pip install -e .
# or: pip install -r requirements.txt
python -m scripts.pretrain --config configs/pretrain.yml && python -m scripts.finetune --config configs/finetune.yml --encoder_ckpt checkpoints/pretrain/best.pt
python -m scripts.finetune --config configs/finetune_fresh.yml --encoder_ckpt fresh



Comparision  Run 4 best val acc: 0.5185 (overall best: 0.5556)

============================================================
Run 5/10
============================================================
/workspace/jepa-humor/src/jepa_humor/finetune.py:112: FutureWarning: `torch.cuda.amp.GradScaler(args...)` is deprecated. Please use `torch.amp.GradScaler('cuda', args...)` instead.
  scaler = GradScaler(enabled=cfg.get("mixed_precision", True))
[Finetune] epoch 1: loss=0.4395 | acc=0.9426                                                                   
 val: acc=0.4198 (best=0.4198)
[Finetune] epoch 2: loss=0.3716 | acc=0.9880                                                                   
 val: acc=0.3951 (best=0.4198)
[Finetune] epoch 3: loss=0.3565 | acc=0.9936                                                                   
 val: acc=0.3951 (best=0.4198)
[Finetune] epoch 4: loss=0.3492 | acc=0.9964                                                                   
 val: acc=0.4074 (best=0.4198)


 No PDG attack:Run 3 best val acc: 0.4568 (overall best: 0.4938)



JEPA:
Run 4 best val acc: 0.4444 (overall best: 0.6667)

============================================================
Run 5/10
============================================================
/workspace/jepa-humor/src/jepa_humor/finetune.py:112: FutureWarning: `torch.cuda.amp.GradScaler(args...)` is deprecated. Please use `torch.amp.GradScaler('cuda', args...)` instead.
  scaler = GradScaler(enabled=cfg.get("mixed_precision", True))
[Finetune] epoch 1: loss=0.4832 | acc=0.8414                                                                   
 val: acc=0.4444 (best=0.4444)
[Finetune] epoch 2: loss=0.3665 | acc=0.9904                                                                   
 val: acc=0.4198 (best=0.4444)
[Finetune] epoch 3: loss=0.3597 | acc=0.9968                                                                   
 val: acc=0.3827 (best=0.4444)
[Finetune] epoch 4: loss=0.3477 | acc=0.9984                                                                   
 val: acc=0.3951 (best=0.4444)


No PGD attack: Run 3 best val acc: 0.3210 (overall best: 0.5802)
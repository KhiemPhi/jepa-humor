# JEPA-Humor

Self-supervised JEPA-style pretraining on CreativeLang/ColBERT_Humor_Detection, then supervised finetuning for humor detection.

## Setup
```bash
pip install -e .
# or: pip install -r requirements.txt
python -m scripts.pretrain --config configs/pretrain.yml && python -m scripts.finetune --config configs/finetune.yml --encoder_ckpt checkpoints/pretrain/best.pt
python -m scripts.finetune --config configs/finetune_fresh.yml --encoder_ckpt fresh



Comparision F1 = 0.982

50/50
Finetune] epoch 1: loss=0.0720 acc=0.9736 f1=0.9736                                                                                              
          val: acc=0.9847 f1=0.9847
[Finetune] epoch 2: loss=0.0226 acc=0.9925 f1=0.9925                                                                                              
          val: acc=0.9835 f1=0.9833
[Finetune] epoch 3: loss=0.0109 acc=0.9962 f1=0.9962                                                                                              
          val: acc=0.9850 f1=0.9850
[Finetune] epoch 4: loss=0.0057 acc=0.9980 f1=0.9980   

80/20 
[Finetune] epoch 1: loss=0.0632 acc=0.9766 f1=0.9766                                                                                              
          val: acc=0.9850 f1=0.9849
[Finetune] epoch 2: loss=0.0221 acc=0.9925 f1=0.9925                                                                                              
          val: acc=0.9852 f1=0.9851
[Finetune] epoch 3: loss=0.0106 acc=0.9963 f1=0.9963                                                                                              
          val: acc=0.9859 f1=0.9858
[Finetune] epoch 4: loss=0.0075 acc=0.9974 f1=0.9974                                                                                              
          val: acc=0.9852 f1=0.9852
[Finetune] epoch 5: loss=0.0055 acc=0.9980 f1=0.9981                                                                                              
          val: acc=0.9852 f1=0.9851

20/80
[Finetune] epoch 1: loss=0.0934 acc=0.9666 f1=0.9666                                                                                              
          val: acc=0.9774 f1=0.9770
[Finetune] epoch 2: loss=0.0265 acc=0.9908 f1=0.9908                                                                                              
          val: acc=0.9812 f1=0.9810
[Finetune] epoch 3: loss=0.0108 acc=0.9965 f1=0.9965                                                                                              
          val: acc=0.9805 f1=0.9806

BERT-Base Pretrained results
---- MRPC with Finetune on Cola Pretrained ---- 
Replacing model with SWA model...                                                                                                                 
SWA model loaded!
Final validation: mcc=0.6969 f1=0.9110, val_acc=0.8725

---- MRPC with Finetune on MRPC Pretrained---- 
Replacing model with SWA model...                                                                                                                 
SWA model loaded!
Final validation: mcc=0.6849 f1=0.9072, val_acc=0.8676

QQP-Adv-Glue: Acc=51.28 F1=36.67
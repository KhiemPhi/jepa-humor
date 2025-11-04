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


Best MCC for Train MCC = 0.3688 at threshold = 0.400                                                                                                                                                                                                                              
[Finetune] epoch 1: loss=0.5734 mcc=0.3572 f1=0.7663
          val: mcc=0.5191 f1=0.8693
Best MCC for Train MCC = 0.6576 at threshold = 0.440                                                                                                                                                                                                                              
[Finetune] epoch 2: loss=0.3810 mcc=0.6494 f1=0.8840
          val: mcc=0.5157 f1=0.8716
Best MCC for Train MCC = 0.7762 at threshold = 0.360                                                                                                                                                                                                                              
[Finetune] epoch 3: loss=0.2616 mcc=0.7580 f1=0.9212
          val: mcc=0.5495 f1=0.8777
Best MCC for Train MCC = 0.8602 at threshold = 0.330                                                                                                                                                                                                                              
[Finetune] epoch 4: loss=0.1730 mcc=0.8507 f1=0.9528
          val: mcc=0.5754 f1=0.8831
Best MCC for Train MCC = 0.8938 at threshold = 0.330                                                                                                                                                                                                                              
[Finetune] epoch 5: loss=0.1365 mcc=0.8890 f1=0.9652
          val: mcc=0.5825 f1=0.8824
Best MCC for Train MCC = 0.9227 at threshold = 0.360                                                                                                                                                                                                                              
[Finetune] epoch 6: loss=0.1009 mcc=0.9180 f1=0.9748
          val: mcc=0.5685 f1=0.8803
Best MCC for Train MCC = 0.9395 at threshold = 0.250                                                                                                                                                                                                                              
[Finetune] epoch 7: loss=0.0808 mcc=0.9317 f1=0.9792
          val: mcc=0.6014 f1=0.8882
Best MCC for Train MCC = 0.9511 at threshold = 0.460                                                                                                                                                                                                                              
[Finetune] epoch 8: loss=0.0674 mcc=0.9498 f1=0.9848
          val: mcc=0.5851 f1=0.8830
Best MCC for Train MCC = 0.9567 at threshold = 0.290                                                                                                                                                                                                                              
[Finetune] epoch 9: loss=0.0583 mcc=0.9542 f1=0.9861
          val: mcc=0.5521 f1=0.8783
Best MCC for Train MCC = 0.9609 at threshold = 0.300                                                                                                                                                                                                                              
[Finetune] epoch 10: loss=0.0521 mcc=0.9570 f1=0.9869
          val: mcc=0.5855 f1=0.8857
Best MCC for Train MCC = 0.9675 at threshold = 0.230                                                                                                                                                                                                                              
[Finetune] epoch 11: loss=0.0426 mcc=0.9632 f1=0.9889
          val: mcc=0.5752 f1=0.8836
Best MCC for Train MCC = 0.9693 at threshold = 0.320                                                                                                                                                                                                                              
[Finetune] epoch 12: loss=0.0436 mcc=0.9666 f1=0.9899
          val: mcc=0.5931 f1=0.8880
Best MCC for Train MCC = 0.9776 at threshold = 0.330                                                                                                                                                                                                                              
[Finetune] epoch 13: loss=0.0319 mcc=0.9742 f1=0.9922
          val: mcc=0.5755 f1=0.8845
Best MCC for Train MCC = 0.9777 at threshold = 0.460                                                                                                                                                                                                                              
[Finetune] epoch 14: loss=0.0292 mcc=0.9772 f1=0.9932
          val: mcc=0.5702 f1=0.8821
Best MCC for Train MCC = 0.9796 at threshold = 0.330                                                                                                                                                                                                                              
[Finetune] epoch 15: loss=0.0329 mcc=0.9775 f1=0.9932
          val: mcc=0.5677 f1=0.8812
✅ Finetune done.


[Finetune] epoch 1: loss=0.7210 mcc=0.0016 f1=0.5919
          val: mcc=0.0588 f1=0.0219
Best MCC for Train MCC = 0.1112 at threshold = 0.510                                                                                                                                                                                                                              
[Finetune] epoch 2: loss=0.6873 mcc=0.1063 f1=0.6354
          val: mcc=0.3261 f1=0.8032
Best MCC for Train MCC = 0.2889 at threshold = 0.450                                                                                                                                                                                                                              
[Finetune] epoch 3: loss=0.6276 mcc=0.2769 f1=0.7252
          val: mcc=0.4102 f1=0.8489
Best MCC for Train MCC = 0.4280 at threshold = 0.420                                                                                                                                                                                                                              
[Finetune] epoch 4: loss=0.5547 mcc=0.4149 f1=0.7979
          val: mcc=0.4815 f1=0.8615
Best MCC for Train MCC = 0.4938 at threshold = 0.460                                                                                                                                                                                                                              
[Finetune] epoch 5: loss=0.5077 mcc=0.4901 f1=0.8245
          val: mcc=0.4992 f1=0.8638
Best MCC for Train MCC = 0.5364 at threshold = 0.450                                                                                                                                                                                                                              
[Finetune] epoch 6: loss=0.4769 mcc=0.5251 f1=0.8354
          val: mcc=0.5169 f1=0.8684
Best MCC for Train MCC = 0.5841 at threshold = 0.430                                                                                                                                                                                                                              
[Finetune] epoch 7: loss=0.4421 mcc=0.5720 f1=0.8529
          val: mcc=0.4997 f1=0.8661
Best MCC for Train MCC = 0.6142 at threshold = 0.400                                                                                                                                                                                                                              
[Finetune] epoch 8: loss=0.4144 mcc=0.6122 f1=0.8671
          val: mcc=0.5183 f1=0.8702
Best MCC for Train MCC = 0.6363 at threshold = 0.450                                                                                                                                                                                                                              
[Finetune] epoch 9: loss=0.3957 mcc=0.6231 f1=0.8740
          val: mcc=0.5351 f1=0.8726
Best MCC for Train MCC = 0.6593 at threshold = 0.380                                                                                                                                                                                                                              
[Finetune] epoch 10: loss=0.3754 mcc=0.6512 f1=0.8824
          val: mcc=0.5347 f1=0.8729
✅ Finetune done.

/home/khiem/nlp/src/jepa_humor/finetune.py:87: FutureWarning: `torch.cuda.amp.GradScaler(args...)` is deprecated. Please use `torch.amp.GradScaler('cuda', args...)` instead.
  scaler = GradScaler(enabled=cfg.get("mixed_precision", True))
[Finetune] epoch 1: loss=0.1067 | acc=0.9928                                                                                                                                                                                    
          val: acc=0.6137 (best=0.6137)
[Finetune] epoch 2: loss=0.0293 | acc=0.9952                                                                                                                                                                                    
          val: acc=0.6498 (best=0.6498)
[Finetune] epoch 3: loss=0.0095 | acc=0.9984                                                                                                                                                                                    
          val: acc=0.6534 (best=0.6534)
[Finetune] epoch 4: loss=0.0033 | acc=0.9996                                                                                                                                                                                    
          val: acc=0.6498 (best=0.6534)
  Run 9 best val acc: 0.6534 (overall best: 0.6679)

============================================================
Best run validation accuracy: 0.6679
============================================================

>>> np.mean( [0.6715, 0.6137, 0.6245, 0.6534, 0.6282, 0.6173, 0.6173, 0.6606, 0.6426] ) 
0.6365666666666666
>>> np.mean( [0.5957, 0.6137, 0.5993, 0.6318, 0.6426, 0.6679, 0.6462, 0.6426, 0.6534] )
0.6325777777777777
>>> np.mean( [0.6390, 0.6498, 0.6498, 0.6426, 0.6462, 0.6643, 0.6390, 0.6606, 0.6318] )
0.6470111111111111

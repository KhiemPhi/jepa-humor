python -m scripts.finetune --config configs/finetune_anli.yml --model_ckpt checkpoints/pretrain/best_pretrain_mnli_exp1_.pt --eval_only
python -m scripts.finetune --config configs/finetune_anli.yml --model_ckpt checkpoints/pretrain/best_pretrain_qnli.pt --eval_only
python -m scripts.finetune --config configs/finetune_anli.yml --model_ckpt checkpoints/pretrain/best_pretrain_rte_exp1_.pt --eval_only
python -m scripts.finetune --config configs/finetune_anli.yml --model_ckpt checkpoints/pretrain/best_pretrain_qqp_exp1_.pt --eval_only
python -m scripts.finetune --config configs/finetune_anli.yml --model_ckpt checkpoints/pretrain/best_pretrain_sst2_exp1_.pt --eval_only
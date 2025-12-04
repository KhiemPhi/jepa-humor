python -m scripts.finetune --config configs/finetune_anli.yml --model_ckpt checkpoints/pretrain/best_pretrain_anli_exp1_.pt --eval_only #bug
python -m scripts.finetune --config configs/finetune_qqp.yml --model_ckpt checkpoints/pretrain/best_pretrain_anli_exp1_.pt --eval_only
python -m scripts.finetune --config configs/finetune_qnli.yml --model_ckpt checkpoints/pretrain/best_pretrain_anli_exp1_.pt --eval_only
python -m scripts.finetune --config configs/finetune_mnli.yml --model_ckpt checkpoints/pretrain/best_pretrain_anli_exp1_.pt --eval_only
python -m scripts.finetune --config configs/finetune_sst2.yml --model_ckpt checkpoints/pretrain/best_pretrain_anli_exp1_.pt --eval_only



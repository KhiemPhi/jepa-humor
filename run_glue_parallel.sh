#!/bin/bash
# Run pretrain → finetune for 5 GLUE tasks in parallel, one task per GPU.
set -euo pipefail

mkdir -p logs

TASKS=(rte sst2 qnli mnli qqp)
GPUS=(0    1    2    3    4  )

for i in "${!TASKS[@]}"; do
    TASK=${TASKS[$i]}
    GPU=${GPUS[$i]}
    ENCODER="checkpoints/pretrain/best_pretrain_${TASK}_exp1_FacebookAI_roberta_large_.pt"

    (
        echo "[GPU $GPU | $TASK] pretrain started"
        python -m scripts.pretrain \
            --config "configs/pretrain_${TASK}.yml" \
            --gpu_id $GPU

        echo "[GPU $GPU | $TASK] finetune started"
        python -m scripts.finetune \
            --config "configs/finetune_${TASK}.yml" \
            --encoder_ckpt "$ENCODER" \
            --gpu_id $GPU

        echo "[GPU $GPU | $TASK] done"
    ) 2>&1 | tee "logs/${TASK}_gpu${GPU}.log" &
done

wait
echo "All 5 tasks complete."

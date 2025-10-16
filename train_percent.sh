#!/bin/bash

# Base config
CONFIG="config.yml"

# Test size variations
TEST_SIZES=(0.2 0.5 0.7)

for TEST in "${TEST_SIZES[@]}"; do
    
    # Unique run name
    RUN_NAME="test${TEST}"
    
    # Output dirs
    PRETRAIN_CKPT_DIR="checkpoints/pretrain_${RUN_NAME}"
    PRETRAIN_LOG_DIR="runs/pretrain_${RUN_NAME}"
    FINETUNE_CKPT_DIR="checkpoints/finetune_${RUN_NAME}"
    FINETUNE_LOG_DIR="runs/finetune_${RUN_NAME}"
    
    mkdir -p "$PRETRAIN_CKPT_DIR" "$PRETRAIN_LOG_DIR" "$FINETUNE_CKPT_DIR" "$FINETUNE_LOG_DIR"
    
    # -------------------
    # Step 1: Pretraining
    # -------------------
    TMP_PRETRAIN="tmp_pretrain_${RUN_NAME}.yml"
    cp "$CONFIG" "$TMP_PRETRAIN"
    
    sed -i "s/^test_size:.*/test_size: ${TEST}/" "$TMP_PRETRAIN"
    sed -i "s|^log_dir:.*|log_dir: ${PRETRAIN_LOG_DIR}|" "$TMP_PRETRAIN"
    sed -i "s|^ckpt_dir:.*|ckpt_dir: ${PRETRAIN_CKPT_DIR}|" "$TMP_PRETRAIN"
    
    echo ">>> Pretraining with test_size=${TEST}"
    python pretrain.py --config "$TMP_PRETRAIN"
    
    # Assuming pretrain.py saves as "best.ckpt"
    PRETRAIN_CKPT="${PRETRAIN_CKPT_DIR}/best.ckpt"
    
    # -------------------
    # Step 2: Finetuning
    # -------------------
    TMP_FINETUNE="tmp_finetune_${RUN_NAME}.yml"
    cp "$CONFIG" "$TMP_FINETUNE"
    
    sed -i "s/^test_size:.*/test_size: ${TEST}/" "$TMP_FINETUNE"
    sed -i "s|^log_dir:.*|log_dir: ${FINETUNE_LOG_DIR}|" "$TMP_FINETUNE"
    sed -i "s|^ckpt_dir:.*|ckpt_dir: ${FINETUNE_CKPT_DIR}|" "$TMP_FINETUNE"
    
    echo ">>> Finetuning with test_size=${TEST}"
    python finetune.py --config "$TMP_FINETUNE" --ckpt "$PRETRAIN_CKPT"
    
    # Log completion
    echo "Finished pretrain+finetune with test_size=${TEST}" >> experiment_log.txt
done

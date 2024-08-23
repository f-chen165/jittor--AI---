#!/bin/bash
export HF_ENDPOINT="https://hf-mirror.com"

MODEL_NAME="stabilityai/stable-diffusion-2-1"
BASE_INSTANCE_DIR="./B"
OUTPUT_DIR_PREFIX="style_8_19_text_14_45_700_4/style_"        # 最后的两个数字表示unet_rl = 2e-4, text_rl = 6e-5
SEED_DIR_PREFIX="style_6_18_text_14_65/style_"
RESOLUTION=512
TRAIN_BATCH_SIZE=1
GRADIENT_ACCUMULATION_STEPS=1
CHECKPOINTING_STEPS=500
LEARNING_RATE=1e-4
TEXT_LEARNING_RATE=4e-5
LR_SCHEDULER="constant"
LR_WARMUP_STEPS=0
MAX_TRAIN_STEPS=700
# SEED=1234
GPU_COUNT=8
MAX_NUM=28


for ((folder_number = 0; folder_number < $MAX_NUM; folder_number += $GPU_COUNT - 4)); do
    for ((gpu_id = 4; gpu_id < GPU_COUNT; gpu_id++)); do
        current_folder_number=$((folder_number + gpu_id - 4))
        if [ $current_folder_number -gt $MAX_NUM ]; then
            break
        fi
        INSTANCE_DIR="${BASE_INSTANCE_DIR}/$(printf "%02d" $current_folder_number)/images"
        OUTPUT_DIR="${OUTPUT_DIR_PREFIX}$(printf "%02d" $current_folder_number)"
        SEED_DIR="${SEED_DIR_PREFIX}$(printf "%02d" $current_folder_number)"
        CUDA_VISIBLE_DEVICES=$gpu_id
        PROMPT="$(printf "style%02d" $current_folder_number)"   # 反斜杠转义空格
        # PROMPT="\"Vincent van Gogh Work Style\""   
        echo $PROMPT     

        COMMAND="CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python train_text_unet.py \
            --pretrained_model_name_or_path=$MODEL_NAME \
            --instance_data_dir=$INSTANCE_DIR \
            --output_dir=$OUTPUT_DIR \
            --instance_prompt=$PROMPT \
            --resolution=$RESOLUTION \
            --train_batch_size=$TRAIN_BATCH_SIZE \
            --gradient_accumulation_steps=$GRADIENT_ACCUMULATION_STEPS \
            --learning_rate=$LEARNING_RATE \
            --lr_scheduler=$LR_SCHEDULER \
            --lr_warmup_steps=$LR_WARMUP_STEPS \
            --max_train_steps=$MAX_TRAIN_STEPS \
            --learning_rate_text=$TEXT_LEARNING_RATE\
            --train_text_encoder=False"
            # --seed=$SEED"
            # --seed_dir=$SEED_DIR  \
            # --pretrained_model_name_or_path=$MODEL_NAME \

        eval $COMMAND &
        sleep 3
    done
    wait
done
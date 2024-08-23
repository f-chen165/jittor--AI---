#!/bin/bash
export HF_ENDPOINT="https://hf-mirror.com"

current_timestamp=$(date +%s)
MODEL_NAME="stabilityai/stable-diffusion-2-1"
BASE_INSTANCE_DIR="./B"                        # 风格图像的训练数据集路径
OUTPUT_DIR_PREFIX="style_weight/style_"        # 权重文件保存路径
RESOLUTION=512
TRAIN_BATCH_SIZE=1
GRADIENT_ACCUMULATION_STEPS=1
CHECKPOINTING_STEPS=600
LEARNING_RATE=9e-5          # unet学习率
TEXT_LEARNING_RATE=4e-5     # clip text encoder学习率
ADDCE_LEARNING_RATE=1e-3    # prompt特征增强模型学习率
LR_SCHEDULER="constant"
LR_WARMUP_STEPS=0
MAX_TRAIN_STEPS=700         # 训练的总批次数
# SEED=1234
GPU_COUNT=1     # GPU数目
MAX_NUM=28      # 风格总数


for ((folder_number = 0; folder_number < $MAX_NUM; folder_number += $GPU_COUNT - 0)); do
    for ((gpu_id = 0; gpu_id < GPU_COUNT; gpu_id++)); do
        current_folder_number=$((folder_number + gpu_id - 0))
        if [ $current_folder_number -gt $MAX_NUM ]; then
            break
        fi
        INSTANCE_DIR="${BASE_INSTANCE_DIR}/$(printf "%02d" $current_folder_number)/images"
        OUTPUT_DIR="${OUTPUT_DIR_PREFIX}$(printf "%02d" $current_folder_number)"    # _${current_timestamp}
        CUDA_VISIBLE_DEVICES=$gpu_id
        PROMPT="$(printf "style%02d" $current_folder_number)"   # 反斜杠转义空格
        # PROMPT="\"Vincent van Gogh Work Style\""   
        echo $PROMPT     

        COMMAND="CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python train_text_unet_ip_vae_encode.py \
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
            --learning_rate_text=$TEXT_LEARNING_RATE \
            --train_text_encoder=False \
            --learning_rate_addce=$ADDCE_LEARNING_RATE \
            --train_addce=False"
            # --seed=$SEED"
                        # --pretrained_model_name_or_path=$MODEL_NAME \

        eval $COMMAND &
        sleep 10
    done
    wait
done
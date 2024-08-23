#!/bin/bash
export HF_ENDPOINT="https://hf-mirror.com"
CUDA_VISIBLE_DEVICES=1
# COMMAND="CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python run_all_text.py"
# COMMAND="CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python run_all_text_ip_addpter.py"
COMMAND="CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python run_all_text_ip_addpter_add_text.py"
# COMMAND="CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES python run_all_text_ip_one_style.py"


eval $COMMAND 

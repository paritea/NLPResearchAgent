#!/bin/bash

# export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export CUDA_VISIBLE_DEVICES=0
# export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,max_split_size_mb:512
export TRITON_CACHE_DIR=/scratch1/asm6590/triton_cache
mkdir -p $TRITON_CACHE_DIR
export NCCL_DEBUG=INFO


# accelerate launch --config_file "configs/fsdp_config.yaml"  train_old.py
# accelerate launch --config_file "configs/deepspeed_config.yaml"  train_old.py
accelerate launch --config_file configs/fsdp_config.yaml train_2.py
# accelerate launch --config_file configs/fsdp_config.yaml gpu_check.py


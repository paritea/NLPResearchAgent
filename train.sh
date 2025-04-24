#!/bin/bash

export HF_HOME="/scratch1/asm6590/.cache/huggingface"
export HF_DATASETS_CACHE="/scratch1/asm6590/.cache/huggingface"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0
export TRITON_CACHE_DIR=/scratch1/asm6590/triton_cache
export NCCL_DEBUG=INFO

# python train_4.py --finetune_method=qlora
python train_4.py --finetune_method=sft
#!/bin/bash

export HF_HOME="/scratch1/asm6590/.cache/huggingface"
export HF_DATASETS_CACHE="/scratch1/asm6590/.cache/huggingface"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export CUDA_VISIBLE_DEVICES=0,1,2
export TRITON_CACHE_DIR=/scratch1/asm6590/triton_cache
export NCCL_DEBUG=INFO

python eval.py \
 --model_name=meta-llama/Llama-3.2-1B \
 --finetuned_ckpt_path=finetuned/Llama-3.2-1B_sft_epoch-2 \
 --num_beams=10 \
 --gpu_id=0
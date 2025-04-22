#!/bin/bash

export CUDA_VISIBLE_DEVICES=0
export TRITON_CACHE_DIR=/scratch1/asm6590/triton_cache
mkdir -p $TRITON_CACHE_DIR
export NCCL_DEBUG=INFO

python train_3.py


python -m transformers.convert_fsdp_model \
  --fsdp_checkpoint_path ./fsdp_finetuned_model_1 \
  --model_type gpt2 \
  --save_dir ./converted_model

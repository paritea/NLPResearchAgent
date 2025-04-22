import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,6,7"
# os.environ['TRANSFORMERS_CACHE'] = '/scratch1/asm6590/.cache/huggingface'
os.environ['HF_HOME'] = '/scratch1/asm6590/.cache/huggingface'
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    
import torch

from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from datasets import load_dataset
from peft import get_peft_model, LoraConfig, TaskType
import numpy as np

from dataset import create_datasets
from metrics import get_metrics_func

# Model checkpoint
model_name = "openai-community/gpt2"
model_cache = './hf_models'

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=model_cache)
tokenizer.pad_token = tokenizer.eos_token  # for GPT-like models

model = AutoModelForCausalLM.from_pretrained(model_name, cache_dir=model_cache)

peft_config = LoraConfig(
    task_type="CAUSAL_LM",
    inference_mode=False,
    r=8,
    lora_alpha=16,
    lora_dropout=0.1,
    bias="none",
    target_modules=["attn.c_attn", "attn.c_proj"]
    # target_modules=["all-linear"]
)

model = get_peft_model(model, peft_config)


training_args = TrainingArguments(
    output_dir="./results-lora",
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    # gradient_accumulation_steps=8,
    eval_strategy="epoch",
    logging_dir="./logs",
    save_total_limit=1,
    learning_rate=2e-4,
    num_train_epochs=1,
    report_to="none",
    
    bf16=True,
    bf16_full_eval=True,

    # # New additions:
    # fsdp="full_shard auto_wrap",
    # fsdp_config={
    #     "fsdp_auto_wrap_policy": "TRANSFORMER_BASED_WRAP",
    #     "fsdp_backward_prefetch": "BACKWARD_PRE",
    #     "fsdp_cpu_ram_efficient_loading": True,
    #     "fsdp_forward_prefetch": False,
    #     "fsdp_offload_params": False,
    #     "fsdp_sharding_strategy": "FULL_SHARD",
    #     "fsdp_state_dict_type": "SHARDED_STATE_DICT",
    #     "fsdp_sync_module_states": True,
    #     "fsdp_use_orig_params": False,
    #     "fsdp_limit_all_gathers": True,
    #     "fsdp_activation_checkpointing": True
    # }
)


train_dataset, eval_dataset = create_datasets(tokenizer)
metric_func = get_metrics_func(tokenizer)


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    tokenizer=tokenizer,
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    compute_metrics=metric_func,
)


if __name__ == '__main__':
    
    # Train
    trainer.train()

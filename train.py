import os
import torch
import argparse
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, get_scheduler, DataCollatorForLanguageModeling, 
    TrainingArguments, Trainer, BitsAndBytesConfig
)
from datasets import load_dataset
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm.auto import tqdm
from peft import get_peft_model, LoraConfig, TaskType

from dataset import create_dataset, data_collator
from metrics import get_metrics_func


device = torch.device('cuda:0')
# model_name = "openai-community/gpt2"
model_name = "google/gemma-3-1b-it"
model_cache = './hf_model_cache'
finetuned_ckpt_path = "finetuned/ckpt_{epoch}"


EPOCHS = 3
INITIAL_LR = 5e-5
BATCH_SIZE = 8

def main(args):
    # Single-GPU training
    
    # Load tokenizer, model, data
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=model_cache)
    tokenizer.pad_token = tokenizer.eos_token  # GPT2 has no pad token by default

    model_kwargs = {}
    if args.finetune_method == 'qlora':
        model_kwargs['quantization_config'] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",  # or "fp4"
            bnb_4bit_compute_dtype=torch.float16  # or torch.float16 if needed
        )
    if model_name == "google/gemma-3-1b-it":
        model_kwargs['attn_implementation'] = "eager"
        
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        cache_dir=model_cache,
        low_cpu_mem_usage=True,
        **model_kwargs
    )
    
    train_ds_tokenized = create_dataset(tokenizer, split='train')
    train_dataloader = DataLoader(
        train_ds_tokenized, 
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        collate_fn=data_collator
    )
    
    # Setup finetuning
    if args.finetune_method == 'sft':
        model.to(device)
    elif args.finetune_method in ['lora', 'qlora']:
        peft_config = LoraConfig(
            r=8,
            lora_alpha=32,
            target_modules=["c_attn", "q_proj", "v_proj"],  # update for your model
            lora_dropout=0.1,
            bias="none",
            task_type=TaskType.CAUSAL_LM
        )

        model = get_peft_model(model, peft_config)
        model.to(device)   
    
    
    # Optimizer and learning rate scheduler
    optimizer = AdamW(model.parameters(), lr=INITIAL_LR)
    num_training_steps = len(train_dataloader) * EPOCHS 

    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps
    )

    for epoch in range(EPOCHS):
        
        print(f"Epoch: {epoch}")
        progress_bar = tqdm(range(len(train_dataloader)))
        
        model.train()
        
        for batch in train_dataloader:
            
            batch = {k: v.to(device) for k, v in batch.items()}
            
            outputs = model(**batch)
            loss = outputs.loss
            
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            
            progress_bar.update(1)
            progress_bar.set_postfix(loss=loss.item())
            
        
        # Save the model 
        model.save_pretrained(
            finetuned_ckpt_path.format(epoch=epoch)
        )
            
    
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Fine-tuning script with method selection")
    parser.add_argument("--finetune_method", type=str, choices=["sft", "lora", "qlora"], default="sft", help="Fine-tuning method to use")
    parser.add_argument("--train_layers", type=int, default=0, help="Number of top layers to train (used only for SFT)")
    args = parser.parse_args()

    main(args)

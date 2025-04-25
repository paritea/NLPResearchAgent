import os
import torch
import argparse
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, get_scheduler, 
    BitsAndBytesConfig
)
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm
from peft import get_peft_model, LoraConfig, TaskType

from dataset import create_dataset, data_collator


model_cache = './hf_model_cache'
os.makedirs('finetuned', exist_ok=True)
finetuned_ckpt_path = "finetuned/{model}_{method}_epoch-{epoch}"

EPOCHS = 3
INITIAL_LR = 5e-5
BATCH_SIZE = 8

def main(args):
    # Single-GPU training
    
    model_name = args.model_name
    finetune_method = args.finetune_method
    device = torch.device(f"cuda:{args.gpu_id}")
    
    print("Training Info: ", model_name, finetune_method, device)
    
    # Load tokenizer, model, data
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=model_cache)
    tokenizer.pad_token = tokenizer.eos_token 

    model_kwargs = {}
    if finetune_method == 'qlora':
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
    if finetune_method == 'sft':
        model.to(device)
    elif finetune_method in ['lora', 'qlora']:
        # Peft model
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
    
    writer = SummaryWriter(log_dir=f"train_logs/{model_name.split('/')[1]}_{finetune_method}")

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
            
            writer.add_scalar("train/loss", loss.item(), epoch)
            
            progress_bar.update(1)
            progress_bar.set_postfix(loss=loss.item())
            
        
        # Save the model 
        model.save_pretrained(
            finetuned_ckpt_path.format(
                model=model_name.split('/')[1], method=finetune_method, epoch=epoch
            )
        )
    
    writer.close()
            
    
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, choices=["meta-llama/Llama-3.2-1B", "google/gemma-3-1b-it"])
    parser.add_argument("--finetune_method", type=str, choices=["sft", "lora", "qlora"])
    parser.add_argument("--gpu_id", type=int)
    args = parser.parse_args()

    main(args)

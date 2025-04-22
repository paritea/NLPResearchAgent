import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, get_scheduler, DataCollatorForLanguageModeling, TrainingArguments, Trainer
from datasets import load_dataset
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm.auto import tqdm

from dataset import create_datasets
from metrics import get_metrics_func

device = torch.device('cuda:0')


EPOCHS = 2

def main():
    # Initialize Accelerator with your FSDP config automatically loaded
    
    model_name = "openai-community/gpt2"
    model_cache = './hf_models'
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=model_cache)
    tokenizer.pad_token = tokenizer.eos_token  # GPT2 has no pad token by default

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        # torch_dtype=torch.float16,  # Use mixed precision if supported
        trust_remote_code=True,  # If using custom models
        cache_dir=model_cache,
        low_cpu_mem_usage=True
    )
    model.to(device)

    tokenized_ds = create_datasets(tokenizer)
        
    def data_collator(batch):
        
        input_ids = [each['input_ids'] for each in batch]
        attention_mask = [each['attention_mask'] for each in batch]
        labels = [each['labels'] for each in batch]
        
        input_ids = torch.tensor(input_ids) 
        attention_mask = torch.tensor(attention_mask) 
        labels = torch.tensor(labels) 
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels
        }
        
    train_dataloader = DataLoader(
        tokenized_ds['train'], 
        batch_size=8, 
        shuffle=True, 
        collate_fn=data_collator
    )
    eval_dataloader = DataLoader(
        tokenized_ds['validation'], 
        batch_size=8, 
        shuffle=False, 
        collate_fn=data_collator
    )

    # Optimizer and learning rate scheduler
    optimizer = AdamW(model.parameters(), lr=5e-5)
    num_training_steps = len(train_dataloader) * EPOCHS 

    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps
    )

    for epoch in range(EPOCHS):
        
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
        model.save_pretrained(f"std_finetuned_model_{epoch}")
            
    
if __name__ == "__main__":
    main()

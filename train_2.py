import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, get_scheduler, DataCollatorForLanguageModeling, TrainingArguments, Trainer
from datasets import load_dataset
from accelerate import Accelerator
from torch.utils.data import DataLoader
from torch.optim import AdamW
from tqdm.auto import tqdm

from dataset import create_datasets
from metrics import get_metrics_func


EPOCHS = 2

def main():
    # Initialize Accelerator with your FSDP config automatically loaded
    accelerator = Accelerator()
    
    accelerator.print(f"Hello from rank {accelerator.process_index} on device {accelerator.device}")


    # Load tokenizer and model with device_map="auto" for sharding
    # model_name = "gpt2"  # Replace with your 1B model checkpoint
    model_name = "openai-community/gpt2"
    model_cache = './hf_models'
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=model_cache)
    tokenizer.pad_token = tokenizer.eos_token  # GPT2 has no pad token by default

    # with accelerator.main_process_first():
    #     # model = AutoModelForCausalLM.from_pretrained(
    #     #     model_name,
    #     #     device_map="auto",  # or remove this if using FSDP
    #     #     torch_dtype=torch.float16,  # or bfloat16 depending on your config
    #     # )
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,  # Use mixed precision if supported
        trust_remote_code=True,  # If using custom models
        cache_dir=model_cache,
        low_cpu_mem_usage=True
    )

    # Load dataset and tokenize
    # dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    # tokenized_ds = create_datasets(tokenizer)
    
    with accelerator.main_process_first():
        tokenized_ds = create_datasets(tokenizer)
        
    # def tokenize_function(examples):
    #     return tokenizer(examples["text"], truncation=True, max_length=128, padding="max_length")

    # tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])

    # Prepare dataloader
    # train_dataloader = DataLoader(tokenized_dataset, batch_size=8, shuffle=True)
    # data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    
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
    num_training_steps = len(train_dataloader) * EPOCHS  # 3 epochs

    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps
    )

    # Prepare everything with accelerator
    model, optimizer, lr_scheduler, train_dataloader, eval_dataloader = accelerator.prepare(
        model, optimizer, lr_scheduler, train_dataloader, eval_dataloader
    )

    

    for epoch in range(EPOCHS):
        
        progress_bar = tqdm(range(len(train_dataloader)), disable=not accelerator.is_local_main_process)
        
        model.train()
        
        for batch in train_dataloader:
            # Move batch to device handled by accelerator
            # batch = {k: v.to(accelerator.device) for k, v in batch.items() if k in ["input_ids", "attention_mask"]}
            # outputs = model(**batch, labels=batch["input_ids"])
            outputs = model(**batch)
            loss = outputs.loss

            accelerator.backward(loss)
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            progress_bar.update(1)
            progress_bar.set_postfix(loss=loss.item())
            
        
        model.eval()
        
        for batch in eval_dataloader:
            
            outputs = model(**batch)
            
        # Save the model (handles FSDP sharded weights)
        if accelerator.is_main_process:
            accelerator.wait_for_everyone()
            unwrapped_model = accelerator.unwrap_model(model)
            # unwrapped_model.save_pretrained(f"fsdp_finetuned_model_{epoch}", save_function=accelerator.save)
            unwrapped_model.save_pretrained(f"finetuned_model_{epoch}")
            

# def main():
#     accelerator = Accelerator()
#     accelerator.print(f"Hello from rank {accelerator.process_index} on device {accelerator.device}")
    
if __name__ == "__main__":
    main()

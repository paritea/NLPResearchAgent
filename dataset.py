import torch
import datasets
from datasets import load_dataset, Split

hf_data_cache = 'hf_data_cache'

    
def create_dataset(tokenizer, split='train'):
    
    # Preprocessing functions
    def __preprocess_train(example):
        input_text = "Question: " + example["problem"] + "\nAnswer:"
        target_text = example["approach"]
        full_text = input_text + " " + target_text

        inputs = tokenizer(full_text, truncation=True, padding="max_length", max_length=512)
        inputs["labels"] = inputs["input_ids"].copy()
        return inputs

    def __preprocess_eval(example):
        input_text = "Question: " + example["problem"] + "\nAnswer:"
        target_text = example["approach"]
        inputs = tokenizer(input_text, truncation=True, padding="max_length", max_length=512)
        inputs["labels"] = tokenizer(target_text, truncation=True, padding="max_length", max_length=512)['input_ids']
        return inputs    

    dataset = None
    __preprocess_func = None
    
    if split == 'train':
        dataset = load_dataset(
            "csv", data_files={"train": "data/train.csv"}, cache_dir=hf_data_cache
        )[split]
        __preprocess_func = __preprocess_train
        
    elif split == 'validation':
        dataset = load_dataset(
            "csv", data_files={"validation": "data/validation.csv"},  cache_dir=hf_data_cache
        )[split]
        __preprocess_func = __preprocess_eval
    
    tokenized_ds = dataset.map(__preprocess_func, batched=False, remove_columns=['Unnamed: 0', 'problem', 'approach'])
    
    return tokenized_ds


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
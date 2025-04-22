from datasets import load_dataset


def create_datasets(tokenizer):

    # Example Dataset (replace with your own)
    # Dataset must have fields: "text1" (question) and "text2" (answer)
    root_dir = '/scratch1/asm6590/NLP_Research_Agent/'
    data_cache = root_dir + 'hf_data'
    
    dataset = load_dataset("csv", 
                        data_files={"train": root_dir + "data/train.csv", "validation": root_dir + "data/validation.csv"},
                        cache_dir=data_cache)

    # Preprocessing
    def __preprocess(example):
        input_text = "Question: " + example["problem"] + "\nAnswer:"
        target_text = example["approach"]
        full_text = input_text + " " + target_text

        inputs = tokenizer(full_text, truncation=True, padding="max_length", max_length=512)
        inputs["labels"] = inputs["input_ids"].copy()
        return inputs
    
    tokenized_ds = dataset.map(__preprocess, batched=False, remove_columns=['Unnamed: 0', 'problem', 'approach'])
    
    return tokenized_ds
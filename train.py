import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    
import torch
torch.cuda.set_device(0)

from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer, DataCollatorForLanguageModeling
from datasets import load_dataset
from peft import get_peft_model, LoraConfig, TaskType
import evaluate
import numpy as np

# Model checkpoint
model_name = "google/gemma-3-1b-it"

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token  # for GPT-like models



model = AutoModelForCausalLM.from_pretrained(model_name)

# Enable LoRA
peft_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=8,
    lora_alpha=16,
    lora_dropout=0.1,
    bias="none"
)
model = get_peft_model(model, peft_config)

# Example Dataset (replace with your own)
# Dataset must have fields: "text1" (question) and "text2" (answer)
dataset = load_dataset("json", data_files={"train": "train.json", "validation": "val.json"})

# Preprocessing
def preprocess(example):
    input_text = "Question: " + example["text1"] + "\nAnswer:"
    target_text = example["text2"]
    full_text = input_text + " " + target_text

    inputs = tokenizer(full_text, truncation=True, padding="max_length", max_length=512)
    inputs["labels"] = inputs["input_ids"].copy()
    return inputs

tokenized_ds = dataset.map(preprocess, batched=False)

# Metrics
bleu = evaluate.load("bleu")
rouge = evaluate.load("rouge")
meteor = evaluate.load("meteor")

def compute_metrics(eval_preds):
    preds, labels = eval_preds
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # Strip "Answer:" prefix if needed
    decoded_preds = [pred.split("Answer:")[-1].strip() for pred in decoded_preds]
    decoded_labels = [label.split("Answer:")[-1].strip() for label in decoded_labels]

    bleu_score = bleu.compute(predictions=decoded_preds, references=decoded_labels)["bleu"]
    rouge_score = rouge.compute(predictions=decoded_preds, references=decoded_labels)
    meteor_score = meteor.compute(predictions=decoded_preds, references=decoded_labels)["meteor"]

    return {
        "bleu": bleu_score,
        "rougeL": rouge_score["rougeL"],
        "meteor": meteor_score
    }

# Training setup
training_args = TrainingArguments(
    output_dir="./results-lora",
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=8,
    evaluation_strategy="epoch",
    logging_dir="./logs",
    save_total_limit=1,
    learning_rate=2e-4,
    num_train_epochs=3,
    fp16=torch.cuda.is_available(),
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_ds["train"],
    eval_dataset=tokenized_ds["validation"],
    tokenizer=tokenizer,
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    compute_metrics=compute_metrics,
)

# Train
trainer.train()

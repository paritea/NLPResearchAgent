import os
import torch
import argparse
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
)
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from peft import PeftModel
import evaluate
import json

from dataset import create_dataset, data_collator

# Metrics
bleu = evaluate.load("bleu")
rouge = evaluate.load("rouge")
meteor = evaluate.load("meteor")

os.makedirs('scores', exist_ok=True)
model_cache = './hf_model_cache'
BATCH_SIZE = 8


def main(args):
    
    model_name = args.model_name
    finetuned_ckpt_path = args.finetuned_ckpt_path
    finetune_method = finetuned_ckpt_path.split('/')[-1].split('_')[1]
    device = torch.device(f"cuda:{args.gpu_id}")

    # Load tokenizer, model, data
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=model_cache)
    tokenizer.pad_token = tokenizer.eos_token 
    tokenizer.padding_side = "left" # For Causal models

    model_kwargs = {
        'low_cpu_mem_usage': True
    }
    if finetune_method == 'qlora':
        model_kwargs['quantization_config'] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",  # or "fp4"
            bnb_4bit_compute_dtype=torch.float16  # or torch.float16 if needed
        )
    if model_name == "google/gemma-3-1b-it":
        model_kwargs['attn_implementation'] = "eager"

    model = None
    if finetune_method in ['lora', 'qlora']:
        # Peft model
        base_model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            cache_dir=model_cache,
            **model_kwargs
        )
        model = PeftModel.from_pretrained(base_model, finetuned_ckpt_path)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            finetuned_ckpt_path,
            **model_kwargs
        )
        
    eval_ds_tokenized = create_dataset(tokenizer, split='validation')
    eval_dataloader = DataLoader(
        eval_ds_tokenized, 
        batch_size=BATCH_SIZE, 
        shuffle=False, 
        collate_fn=data_collator
    )
    
    model.to(device)
    model.eval()

    progress_bar = tqdm(range(len(eval_dataloader)))

    all_preds = []
    all_labels = []
    
    gen_kwargs = {}
    
    if args.top_k != None or args.top_p != None:
        gen_kwargs['do_sample'] = True
        gen_kwargs['top_k'] = args.top_k
        gen_kwargs['top_p'] = args.top_p
        gen_kwargs['temperature'] = 0.8
    elif args.num_beams != None:
        gen_kwargs['num_beams'] = args.num_beams
        gen_kwargs['early_stopping'] = args.num_beams > 1
        gen_kwargs['no_repeat_ngram_size'] = 2
    
    for batch in eval_dataloader:
        
        model_inputs = {k: v.to(device) for k, v in batch.items() if k in ['input_ids', 'attention_mask']}
        
        preds = model.generate(
            **model_inputs,  max_new_tokens=30, pad_token_id=tokenizer.pad_token_id,
            **gen_kwargs
        )
        
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(batch['labels'], skip_special_tokens=True)
        
        all_preds += [pred.split("\nAnswer:", 1)[-1].strip() for pred in decoded_preds]
        all_labels += decoded_labels
        
        progress_bar.update(1)
        
    bleu_score = bleu.compute(predictions=all_preds, references=all_labels)["bleu"]
    rouge_score = rouge.compute(predictions=all_preds, references=all_labels)["rougeL"]
    meteor_score = meteor.compute(predictions=all_preds, references=all_labels)["meteor"]
    
    
    postfix = '_'.join(
        f"{k}-{gen_kwargs[k]}" for k in ['top_k', 'top_p', 'num_beams'] if gen_kwargs.get(k) != None
    )
    
    scores_save_path = f"scores/{finetuned_ckpt_path.split('/')[-1]}" + "_" + postfix + ".json"
            
    json.dump(
        {
            'bleu': bleu_score,
            'rouge-L': rouge_score,
            'meteor': meteor_score
        }, 
        open(scores_save_path, "w+")
    )
    
    
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, choices=["meta-llama/Llama-3.2-1B", "google/gemma-3-1b-it"])
    parser.add_argument("--finetuned_ckpt_path", type=str)
    parser.add_argument("--top_k", type=int, default=None)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=None)
    parser.add_argument("--gpu_id", type=int, default=0)
    args = parser.parse_args()
    
    main(args)
    
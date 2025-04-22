import evaluate

# Metrics
bleu = evaluate.load("bleu")
rouge = evaluate.load("rouge")
meteor = evaluate.load("meteor")

def get_metrics_func(tokenizer):
    
    def __compute_metrics(eval_preds):
        preds, labels = eval_preds
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

        # Strip "Answer:" prefix if needed
        decoded_preds = [pred.split("Answer:")[-1].strip() for pred in decoded_preds]
        decoded_labels = [label.split("Answer:")[-1].strip() for label in decoded_labels]

        bleu_score = bleu.compute(predictions=decoded_preds, references=decoded_labels)["bleu"]
        rouge_score = rouge.compute(predictions=decoded_preds, references=decoded_labels)["rougeL"]
        meteor_score = meteor.compute(predictions=decoded_preds, references=decoded_labels)["meteor"]

        return {
            "bleu": bleu_score,
            "rougeL": rouge_score,
            "meteor": meteor_score
        }
        
    return __compute_metrics
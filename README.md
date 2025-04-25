# NLPResearchAgent

scrape_dataset.py - This code calls the arxiv api to get summaries of different papers. 

create_dataset.py - This code takes the summaries obtained in scrape_dataset.py and extracts problem-approach pairs from it. 

train.py - This file contains the code for finetuning the models via LoRA, QLoRA, SFT.

eval.py - This file contains code for computing the metrics (bleu, rougel, meteor).

CSE-587_Final_project.pdf - Report for the final project.

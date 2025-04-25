import os, csv
# 0) Restrict to GPU 0
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# 1) (Optional) redirect HF cache if you hit quota
# os.environ["HF_HOME"] = "/scratch1/dsc5636/.cache/huggingface"

from huggingface_hub import login
from transformers import AutoTokenizer, AutoModelForCausalLM, TextGenerationPipeline
import pandas as pd, json

# 2) Authenticate
login(token="")

MODEL = "meta-llama/Llama-3.2-3B-Instruct"

# 3) Load tokenizer & model (using token= instead of use_auth_token=)
tokenizer = AutoTokenizer.from_pretrained(
    MODEL,
    use_fast=False,
    trust_remote_code=True,
    token=""
)
model = AutoModelForCausalLM.from_pretrained(
    MODEL,
    device_map="auto",     # sees only GPU 0 thanks to CUDA_VISIBLE_DEVICES
    torch_dtype="auto",
    trust_remote_code=True,
    token=""
)

# 4) Build the pipeline *without* the `device` arg
pipe = TextGenerationPipeline(
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=128,
    do_sample=False,
    top_p=1.0,         # disable nucleus sampling
    return_full_text=False,
    clean_up_tokenization_spaces=True
)

def extract_problem_and_approach(abstract: str) -> dict:
    prompt = (
        # --- FEW‑SHOT EXAMPLE ---
        "Example:\n"
        "Abstract:\n"
        "We explore how generating a chain of thought — a series of intermediate reasoning steps — "
        "significantly improves the ability of large language models to perform complex reasoning. "
        "In particular, we show how such reasoning abilities emerge naturally in sufficiently large "
        "language models via a simple method called chain of thought prompting, where a few chain of "
        "thought demonstrations are provided as exemplars in prompting. Experiments on three large "
        "language models show that chain of thought prompting improves performance on a range of "
        "arithmetic, commonsense, and symbolic reasoning tasks. The empirical gains can be striking. "
        "For instance, prompting a 540B‑parameter language model with just eight chain of thought exemplars "
        "achieves state‑of‑the‑art accuracy on the GSM8K benchmark of math word problems, surpassing even "
        "finetuned GPT‑3 with a verifier.\n\n"
        "JSON:\n"
        "{\n"
        '  "problem": "Large language models lack reliable complex‑reasoning abilities without intermediate reasoning steps, '
        'leading to poor performance on tasks like math word problems and commonsense reasoning.",\n'
        '  "approach": "Introduce chain‑of‑thought prompting—providing a few exemplar sequences of intermediate reasoning steps '
        'in the prompt—to elicit and improve the model’s reasoning performance."\n'
        "}\n\n"
        # --- END EXAMPLE, BEGIN NEW PROMPT ---
        "—\n\n"
        "Now you:\n"
        "Abstract:\n"
        f"{abstract}\n\n"
        "JSON:"
    )
    raw = pipe(prompt)[0]["generated_text"]
    print(raw)
    # strip off any echoed prompt
    return raw
    # json_str = raw[raw.find("{"):]
    # return json.loads(json_str)

if __name__ == "__main__":
    df = pd.read_csv("/scratch1/dsc5636/ProjectDLforNLP/dataset.csv")
    # example = extract_problem_and_approach(df["summary"].iloc[0])
    # Store results

    df["problem_solution"] = df["summary"].apply(lambda x: extract_problem_and_approach(x))
    
    # And save if you like
    df.to_csv("fin_dataset.csv", index=False)


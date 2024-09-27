"""
python3 src/dpo_llama_generation.py \
 --model_name=meta-llama/Llama-2-7b-hf \
 --adapter_dir=llama/llama-7b-etpc
 
 python3 src/dpo_llama_generation.py \
 --model_name=meta-llama/Llama-2-13b-hf \
 --adapter_dir=llama/llama-13b-etpc
"""

import argparse
import os
import json
from tqdm import tqdm

from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from rouge import Rouge

import torch
from datasets import load_dataset, DatasetDict
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel, PeftConfig, get_peft_model
from trl import DPOTrainer, DPOConfig

# Initialize Hugging Face Hub login (if needed)
from huggingface_hub import login
with open("token_file.txt", "r") as token_file:
    hf_token = token_file.read().strip()
login(token=hf_token)

def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Run DPO training")
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-2-7b-hf", help="Path to the model")
    parser.add_argument("--adapter_dir", type=str, default="llama/llama-7b-etpc", help="Directory of the PEFT adapter")
    return parser.parse_args()   

def load_data(filename):
    """Loads data from a file in JSON format.

    Args:
        filename (str): The path to the file to load.

    Returns:
        list: The loaded data as a list of dictionaries.

    Example:
        ```python
        filename = "data.json"
        data = load_data(filename)
        print(data)
        ```
    """
    with open(filename, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f.readlines()]
    return data

def generate_paraphrases(
    data, model, max_gen_len, temperature, top_p, max_batch_size, num_examples=3
):
    paraphrases = []

    for i in tqdm(range(0, num_examples, max_batch_size)):
        batch = data[i : i + max_batch_size]
        user_messages = [instance["messages"][0]["content"] for instance in batch]

        results = model.text_completion(
            user_messages,
            max_gen_len=max_gen_len,
            temperature=temperature,
            top_p=top_p,
        )

        paraphrases.extend([result["generation"] for result in results])

    return paraphrases


def evaluate(paraphrases, references):
    """
    Evaluates the quality of paraphrases compared to reference texts.

    Args:
        paraphrases (list): The generated paraphrases.
        references (list): The reference texts.

    Returns:
        dict: A dictionary containing the evaluation scores.

    Example:
        ```python
        paraphrases = ["Paraphrase 1", "Paraphrase 2"]
        references = ["Reference 1", "Reference 2"]

        scores = evaluate(paraphrases, references)
        print(scores)
        ```
    """

    rouge = Rouge()

    # ROUGE scores
    rouge_scores = rouge.get_scores(paraphrases, references, avg=True)

    # BLEU scores
    smoothie = SmoothingFunction().method4
    bleu_scores = [
        sentence_bleu([ref], paraphrase, smoothing_function=smoothie)
        for ref, paraphrase in zip(references, paraphrases)
    ]
    avg_bleu = sum(bleu_scores) / len(bleu_scores)

    return {
        "ROUGE-1": rouge_scores["rouge-1"]["f"],
        "ROUGE-2": rouge_scores["rouge-2"]["f"],
        "ROUGE-L": rouge_scores["rouge-l"]["f"],
        "BLEU": avg_bleu,
    }
def main(
    num_examples: int = 1000,
    temperature: float = 0.6,
    top_p: float = 0.9,
    max_seq_len: int = 2048,
    max_gen_len: int = 1024,
    max_batch_size: int = 4,
):    
    args = parse_arguments()
    
    # Check if CUDA is available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    quant_config = BitsAndBytesConfig(load_in_8bit=True)
     
    model = AutoModelForCausalLM.from_pretrained(args.model_name, quantization_config=quant_config)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    # Load PEFT adapters
    script_dir = os.path.dirname(os.path.abspath(__file__))
    adapter_dir = os.path.join(script_dir, args.adapter_dir)
    
    peft_config = PeftConfig.from_pretrained(adapter_dir, base_model_name_or_path=args.model_name)
    model = PeftModel.from_pretrained(model, adapter_dir)
    
    # Load reference adapter for DPO
    model.load_adapter(adapter_dir, adapter_name="reference")
    
    # Apply additional LoRA weights
    model = get_peft_model(model, peft_config=peft_config)
    
    # Move model to GPU after loading PEFT model
    model = model.to(device)
    
    torch.cuda.empty_cache()  # Clear GPU cache before starting
                       
    # Load dataset from JSONL files
    train_json_path = "out/generation_apty_ranked_train.jsonl"
    eval_json_path = "out/generation_apty_ranked_test.jsonl"
    
    train_dataset = load_dataset('json', data_files={'train': train_json_path})['train']
    validation_dataset = load_dataset('json', data_files={'validation': eval_json_path})['validation']
    ds=  DatasetDict({
        'train': train_dataset,
        'validation': validation_dataset,
    })
    
    # Prepare datasets for training
    train_dataset = ds['train']
    eval_dataset = ds['validation']

    # Setup trainer    
    training_args=DPOConfig(output_dir=f"./out/gen-models/dpo_{args.model_name}_{args.adapter_dir}",
                            per_device_train_batch_size=1,
                            gradient_accumulation_steps=1,
                            max_length=1024,
                            max_prompt_length=512,
                            fp16=True,
                            optim="adamw_8bit",        
                            )
    
    trainer = DPOTrainer(model,
                         args=training_args,
                         train_dataset=train_dataset,
                         eval_dataset=eval_dataset,
                         tokenizer=tokenizer,
                         peft_config=peft_config,
                         )

    # Start training
    trainer.train()

    # Save the trained model
    trainer.save_model("./out/gen-models/dpo_{args.model_name}_{args.adapter_dir}")
    
    # Load data and predict
    data_file= "out/generation_etpc_test.jsonl"
    test_data = load_data(data_file)
    generated_paraphrases = generate_paraphrases(
        test_data,
        model,
        max_gen_len=max_gen_len,
        temperature=temperature,
        top_p=top_p,
        max_batch_size=max_batch_size,
        num_examples=num_examples,
    )
    references = [item["messages"][1]["content"] for item in test_data[:num_examples]]
    scores = evaluate(generated_paraphrases, references)
    print(f"Model: {args.model_name}")
    print(f"Adapter: {args.adapter_dir}")
    print(scores)

if __name__ == "__main__":
    main()

import argparse
import os
import json
import logging
from tqdm import tqdm
import chardet
import csv
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from rouge import Rouge
from typing import List, Optional
import torch
from datasets import load_dataset, DatasetDict
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel, PeftConfig

# Initialize Hugging Face Hub login (if needed)
from huggingface_hub import login
with open("token_file.txt", "r") as token_file:
    hf_token = token_file.read().strip()
login(token=hf_token)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Run DPO training")
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-2-7b-hf", help="Path to the model")
    parser.add_argument("--etpc_dir", type=str, default="src/llama/llama-7b-etpc", help="Directory of the ETPC adapter")
    parser.add_argument("--dpo_dir", type=str, default="out/dpo_llama-7b_apty", help="Directory of the DPO adapter")
    return parser.parse_args()

def load_data(filename):
    """Loads data from a file in JSON format."""
    logging.info(f"Loading data from {filename}")
    with open(filename, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f.readlines()]
    logging.info(f"Loaded {len(data)} records from {filename}")
    return data

def read_sentences_from_files(data_dir):
    """
    Reads base sentences and their paraphrase types from text files in the specified directory.
    Only the first 10 sentences are returned for each file.

    Args:
        data_dir (str): The directory containing the text files to read from.

    Returns:
        dict[str, list[str]]: A dictionary where the keys are the paraphrase types and the values are lists of sentences.
    """
    sentences_by_type = {}
    for file_name in os.listdir(data_dir):
        if file_name.endswith(".txt"):
            file_path = os.path.join(data_dir, file_name)

            with open(file_path, 'rb') as file:
                raw_data = file.read()
                result = chardet.detect(raw_data)
                encoding = result['encoding']

            with open(file_path, 'r', encoding=encoding) as file:
                lines = file.readlines()
                paraphrase_type = lines[0].strip()
                sentences = [line.strip() for line in lines[1:11]]  # Only take the first 10 sentences
                sentences_by_type[paraphrase_type] = sentences
                

def process_model_generation(model_name_or_path, adapter_path, sentences_by_type, model_suffix, dataset_type):
    """
    Generates paraphrases using the specified model and returns them in a list.

    Args:
        model_name_or_path (str): The base model path or name.
        adapter_path (str): The adapter directory path.
        sentences_by_type (dict): Dictionary of sentences categorized by paraphrase type.
        model_suffix (str): Suffix to append to the model name in saved files.
        dataset_type (str): The type of dataset being used ("apty" or "etpc").

    Returns:
        list[dict]: A list of dictionaries containing generated paraphrases information.
    """
    model, tokenizer = load_model_and_tokenizer(model_name_or_path, adapter_path)
    all_paraphrases = []

    for paraphrase_type, sentences in sentences_by_type.items():
        print(f"Generating paraphrases for type: {paraphrase_type} using {model_suffix} Model")
        paraphrases = generate_paraphrases(model, tokenizer, sentences, dataset_type, paraphrase_type)
        
        for index, paraphrase in enumerate(paraphrases, start=1):
            entry = {
                "data": {
                    "Original": sentences[index - 1],
                    "APT": paraphrase_type,
                    "Paraphrase": paraphrase,
                    "Kind": model_suffix,
                    "Index": index
                }
            }
            all_paraphrases.append(entry)

    del model, tokenizer
    torch.cuda.empty_cache()

    return all_paraphrases


def load_model_and_tokenizer(model_name, adapter_dir=None):
    """
    Load a model and tokenizer, applying PEFT adapters if specified.

    Args:
        model_name (str): The name or path of the model to load.
        adapter_dir (str, optional): The path to the PEFT adapter to apply. Defaults to None.

    Returns:
        tuple: The loaded model and tokenizer.
    """    
    logging.info(f"Loading model and tokenizer: {model_name}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.float16,
        )    
   
    logging.info("Loading the model...")
    model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=bnb_config)
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
    tokenizer.pad_token = tokenizer.eos_token
    
    logging.info(f"Loading PEFT adapter from {adapter_dir}")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    adapter_dir = os.path.join(script_dir, adapter_dir)
    
    peft_config = PeftConfig.from_pretrained(adapter_dir)
    peft_config.base_model_name_or_path=model_name
    model = PeftModel.from_pretrained(model, adapter_dir, config=peft_config)
    
    logging.info("Loading reference adapter for DPO...")
    model.load_adapter(adapter_dir, adapter_name="reference")
    
    logging.info("Model moved to GPU")
    model = model.to(device)
    model.eval()    
    
    torch.cuda.empty_cache()  # Clear GPU cache
    
    return model, tokenizer

def generate_paraphrases(model, tokenizer, data, dataset_type="apty", paraphrase_type=None, max_gen_len=1024, temperature=0.6, top_p=0.9, batch_size=8):
    """
    Generate paraphrases for a list of sentences or prompts using the provided model and tokenizer.

    Args:
        model: The model to use for generating paraphrases.
        tokenizer: The tokenizer to use for encoding the input sentences.
        data: For ETPC, it's a list of prompts; for APTY, it's a list of sentences.
        dataset_type: Type of dataset ("apty" or "etpc") to distinguish prompt handling.
        paraphrase_type: For APTY dataset, the type of paraphrase to generate.
        max_gen_len: Maximum number of tokens to generate.
        temperature: Sampling temperature for generation.
        top_p: Top-p sampling probability.
        batch_size: Number of sentences/prompts to process in each batch.

    Returns:
        list: A list of generated paraphrases.
    """
    paraphrases = []
    device = next(model.parameters()).device

    for i in range(0, len(data), batch_size):
        batch = data[i:i + batch_size]
        
        # For APTY, create prompts using the paraphrase type
        if dataset_type == "apty":
            prompts = [
                f"Instruction: Given the following sentence, generate a paraphrase with the following type. "
                f"Sentence: {sentence} Paraphrase Type: {paraphrase_type}. Generated Paraphrase: "
                for sentence in batch
            ]
        # For ETPC, the prompts are already present
        elif dataset_type == "etpc":
            prompts = [instance["messages"][0]["content"] for instance in batch]

        inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_gen_len,
                top_p=top_p,
                temperature=temperature
            )

        paraphrases.extend([tokenizer.decode(output, skip_special_tokens=True).strip() for output in outputs])

    return paraphrases

def save_paraphrases_to_json(paraphrases, output_file):
    """
    Saves the generated paraphrases to a JSON file in the specified format.

    Args:
        paraphrases (list[dict]): A list of dictionaries containing paraphrase information.
        output_file (str): The file path to save the JSON file.
    """
    with open(output_file, 'w', encoding='utf-8') as file:
        json.dump(paraphrases, file, ensure_ascii=False, indent=4)

    print(f"Paraphrases saved to {output_file}.\n")
    

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

def save_metrics_to_csv(metrics, output_csv):
    """
    Save evaluation metrics to a CSV file.

    Args:
        metrics (list[dict]): A list of dictionaries with model and adapter information and evaluation metrics.
        output_csv (str): The file path for the output CSV.
    """
    fieldnames = ["Model", "Adapter", "ROUGE-1", "ROUGE-2", "ROUGE-L", "BLEU"]

    with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for metric in metrics:
            writer.writerow(metric)
    
    print(f"Evaluation metrics saved to {output_csv}.\n")

def main(
    num_examples: int = 1000,
):
    logging.basicConfig(
    filename='slurm_files/my_app.log',  # Specify the log file
    level=logging.INFO,  # Log level
    format='%(asctime)s - %(levelname)s - %(message)s'  # Log format with timestamp
    )
    
    logging.info("Parsing arguments...")
    args = parse_arguments()    
    
    # Load models
    data_dir = "src/basesentences"
    output_file = "out/generated_paraphrases-7b.json"
    output_csv = "out/evaluation_metrics-7b.csv"

    #####
    # APTY evaluation
    #####
    
    # Load sentences from text files
    logging.info("Reading APTY files...")
    sentences_by_type = read_sentences_from_files(data_dir)

    # Generate paraphrases for base model, ETPC adapter, and DPO adapter
    logging.info("Generating paraphrases for APTY")
    all_paraphrases = []
    all_paraphrases.extend(process_model_generation(args.model_name, None, sentences_by_type, "base_model", "apty"))
    all_paraphrases.extend(process_model_generation(args.model_name, args.etpc_dir, sentences_by_type, "etpc_model", "apty"))
    all_paraphrases.extend(process_model_generation(args.model_name, args.dpo_dir, sentences_by_type, "dpo_model", "apty"))

    # Save paraphrases to JSON
    logging.info("Saving APTY paraphrases...")
    save_paraphrases_to_json(all_paraphrases, output_file)
    
    #####
    # ETPC evaluation
    #####

    logging.info("Loading test data for paraphrase generation...")
    data_file = "out/generation_etpc_test.jsonl"
    test_data = load_data(data_file)
    
    generated_paraphrases = generate_paraphrases(
        test_data,
        model,
        tokenizer=tokenizer,
        max_gen_len=max_gen_len,
        temperature=temperature,
        top_p=top_p,
        max_batch_size=max_batch_size,
        num_examples=num_examples,
    )

    references = [item["messages"][1]["content"] for item in test_data[:num_examples]]
    logging.info(f"Generated {len(generated_paraphrases)} paraphrases, and found {len(references)} references.")

    logging.info("Evaluating paraphrases...")
    scores = evaluate(generated_paraphrases, references)

    # Save evaluation metrics to CSV
    save_metrics_to_csv(scores, output_csv)

if __name__ == "__main__":
    main()
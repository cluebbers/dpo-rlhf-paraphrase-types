import argparse
import os
import json
import logging
import torch
import chardet
import csv
from tqdm import tqdm
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from rouge import Rouge
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel, PeftConfig

# Initialize Hugging Face Hub login (if needed)
from huggingface_hub import login
with open("token_file.txt", "r") as token_file:
    hf_token = token_file.read().strip()
login(token=hf_token)

# Logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Generate paraphrases and evaluate models.")
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-2-7b-hf", help="Base model path.")
    parser.add_argument("--etpc_dir", type=str, default="src/llama/llama-7b-etpc", help="ETPC adapter directory.")
    parser.add_argument("--dpo_dir", type=str, default="out/dpo_llama-7b_apty", help="DPO adapter directory.")
    return parser.parse_args()

def load_data(filename, num_examples=None):
    """Loads data from a file in JSON format and limits the number of examples."""
    with open(filename, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f.readlines()]

    if num_examples:
        data = data[:num_examples]

    logging.info(f"Loaded {len(data)} examples from {filename}")
    return data


def read_sentences_from_files(data_dir):
    """Reads base sentences and paraphrase types from text files."""
    sentences_by_type = {}
    for file_name in os.listdir(data_dir):
        if file_name.endswith(".txt"):
            file_path = os.path.join(data_dir, file_name)

            with open(file_path, 'rb') as file:
                raw_data = file.read()
                encoding = chardet.detect(raw_data)['encoding']

            with open(file_path, 'r', encoding=encoding) as file:
                lines = file.readlines()
                paraphrase_type = lines[0].strip()
                sentences = [line.strip() for line in lines[1:11]]  # Limit to first 10 sentences
                sentences_by_type[paraphrase_type] = sentences
    return sentences_by_type


def load_model_and_tokenizer(model_name, adapter_dir=None):
    """Load a model and tokenizer, applying PEFT adapters if specified, and ensure padding configuration."""
    logging.info(f"Loading model and tokenizer: {model_name}")
    
    # Configure for bitsandbytes (4-bit quantization) and use bfloat16
    bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)

    # Load the base model with bfloat16 precision for better stability
    model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=bnb_config, torch_dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Ensure left-padding is used and add a padding token if missing
    tokenizer.padding_side = "left"

    # Add a padding token if it's missing and resize embeddings
    if tokenizer.pad_token is None:
        logging.info("Adding a padding token ('<pad>') to the tokenizer.")
        tokenizer.add_special_tokens({"pad_token": "<pad>"})
        model.resize_token_embeddings(len(tokenizer))

    # Set the model's padding token ID to the new padding token
    model.config.pad_token_id = tokenizer.pad_token_id

    # Load the adapter if specified
    if adapter_dir:
        logging.info(f"Loading PEFT adapter from {adapter_dir}")
        peft_config = PeftConfig.from_pretrained(adapter_dir)
        model = PeftModel.from_pretrained(model, adapter_dir, config=peft_config)

    model.eval()
    torch.cuda.empty_cache()
    
    return model, tokenizer

from tqdm import tqdm  # Import the tqdm library for the progress bar

def generate_paraphrases(
    model, 
    tokenizer, 
    data, 
    dataset_type="apty", 
    paraphrase_type=None, 
    temperature=0.6,  
    top_p=0.9,        
    batch_size=1,     
    max_length=512,   
    max_new_tokens=50 
):
    """
    Generate paraphrases for a list of sentences or prompts using the provided model and tokenizer.
    This version includes a progress bar and slices the output to remove the input prompt from the final paraphrase.
    """
    paraphrases = []  # List to store generated paraphrases
    device = next(model.parameters()).device  # Ensure model is on the correct device (e.g., CUDA)

    # Use tqdm to show progress during generation
    total_batches = (len(data) + batch_size - 1) // batch_size  # Calculate total number of batches
    progress_bar = tqdm(total=total_batches, desc="Generating Paraphrases", unit="batch")

    for i in range(0, len(data), batch_size):
        batch = data[i:i + batch_size]  # Slice the data into batches

        # Construct prompts based on dataset type
        if dataset_type == "apty":
            prompts = [
                f"Instruction: Given the following sentence, generate a paraphrase with the following type. "
                f"Sentence: {sentence} Paraphrase Type: {paraphrase_type}. Generated Paraphrase: "
                for sentence in batch
            ]
        elif dataset_type == "etpc":
            prompts = [instance["messages"][0]["content"] for instance in batch]

        # Tokenize inputs in batches with padding and truncation
        inputs = tokenizer(
            prompts, 
            return_tensors="pt",  # Return as PyTorch tensors
            padding=True,         # Pad shorter sequences
            truncation=True,      # Truncate longer sequences
            max_length=max_length # Maximum input length
        ).to(device)  # Move tokenized inputs to the correct device (GPU or CPU)

        # Store the length of the input tokens (prompt)
        input_token_len = inputs["input_ids"].shape[-1]

        with torch.no_grad():
            # Generate paraphrases in batches
            outputs = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],  # Pass the attention mask to handle padding
                max_new_tokens=max_new_tokens,  # Limit the number of new tokens generated
                top_p=top_p,                    # Top-p nucleus sampling
                temperature=temperature,        # Sampling temperature
                do_sample=True                  # Enable sampling (for stochastic generation)
            )

        # For each output, slice off the input prompt tokens and decode only the new generated tokens
        for output in outputs:
            generated_tokens = output[input_token_len:]  # Slice off the input prompt tokens
            paraphrase = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
            paraphrases.append(paraphrase)

        # Update progress bar after processing each batch
        progress_bar.update(1)

    progress_bar.close()  # Close the progress bar after the loop

    return paraphrases

def evaluate_paraphrases(paraphrases, references):
    """Evaluate paraphrases using ROUGE and BLEU scores."""
    rouge = Rouge()

    # ROUGE scores
    rouge_scores = rouge.get_scores(paraphrases, references, avg=True)

    # BLEU scores
    smoothie = SmoothingFunction().method4
    bleu_scores = [sentence_bleu([ref], paraphrase, smoothing_function=smoothie) for ref, paraphrase in zip(references, paraphrases)]
    avg_bleu = sum(bleu_scores) / len(bleu_scores)

    return {
        "ROUGE-1": rouge_scores["rouge-1"]["f"],
        "ROUGE-2": rouge_scores["rouge-2"]["f"],
        "ROUGE-L": rouge_scores["rouge-l"]["f"],
        "BLEU": avg_bleu,
    }

def save_metrics_to_csv(metrics, output_csv):
    """Save evaluation metrics to a CSV file."""
    fieldnames = ["Model", "Adapter", "ROUGE-1", "ROUGE-2", "ROUGE-L", "BLEU"]

    with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for metric in metrics:
            writer.writerow(metric)
    logging.info(f"Evaluation metrics saved to {output_csv}")


def save_paraphrases_to_json(paraphrases, output_file):
    """Saves the generated paraphrases to a JSON file."""
    with open(output_file, 'w', encoding='utf-8') as file:
        json.dump(paraphrases, file, ensure_ascii=False, indent=4)
    logging.info(f"Paraphrases saved to {output_file}")


def process_model_generation(model_name_or_path, adapter_path, apty_data, etpc_data, model_suffix, batch_size):
    """Generate paraphrases for both ETPC and APTY datasets using a specified model and adapter."""
    model, tokenizer = load_model_and_tokenizer(model_name_or_path, adapter_path)
    all_paraphrases = []

    # Generate paraphrases for APTY dataset
    for paraphrase_type, sentences in apty_data.items():
        paraphrases = generate_paraphrases(model, tokenizer, sentences, "apty", paraphrase_type, batch_size=batch_size)
        all_paraphrases.extend([{
            "Original": sentence,  # Only store the sentence
            "APT": paraphrase_type, 
            "Paraphrase": paraphrase, 
            "Kind": model_suffix,  # Indicate the model type (base, etpc, dpo)
            "Dataset": "APTY"      # Identify the dataset source
        } for sentence, paraphrase in zip(sentences, paraphrases)])

    # Generate paraphrases for ETPC dataset
    etpc_paraphrases = generate_paraphrases(model, tokenizer, etpc_data, "etpc", batch_size=batch_size)
    for index, paraphrase in enumerate(etpc_paraphrases):
        # Extract only the sentence part from the ETPC message content
        full_content = etpc_data[index]["messages"][0]["content"]

        # Safely extract the sentence and paraphrase types from the ETPC content
        if "Sentence:" in full_content and "Paraphrase Types:" in full_content:
            original_sentence = full_content.split("Sentence:")[1].split("Paraphrase Types:")[0].strip()
            paraphrase_types = full_content.split("Paraphrase Types:")[1].strip()  # Get everything after "Paraphrase Types:"
        else:
            # Fallback if format is unexpected
            original_sentence = full_content
            paraphrase_types = "Unknown"  # Use a default value if not found

        all_paraphrases.append({
            "Original": original_sentence,  # Store only the extracted sentence
            "APT": paraphrase_types,        # Store the extracted paraphrase types (APT)
            "Paraphrase": paraphrase,
            "Kind": model_suffix,           # Indicate the model type (base, etpc, dpo)
            "Dataset": "ETPC"               # Identify the dataset source (ETPC in this case)
        })

    del model, tokenizer
    torch.cuda.empty_cache()

    return all_paraphrases

def generate_and_evaluate(models, apty_data, etpc_data, output_csv, output_json, batch_size):
    """Main function to generate paraphrases for different models and evaluate them."""
    metrics = []
    all_paraphrases = []
    references = [item["messages"][0]["content"] for item in etpc_data]  # References for ETPC dataset

    for model_name, adapter, model_suffix in models:
        logging.info(f"Generating paraphrases for {model_suffix} model")
        paraphrases = process_model_generation(model_name, adapter, apty_data, etpc_data, model_suffix, batch_size)
        all_paraphrases.extend(paraphrases)

        # Filter ETPC-generated paraphrases using 'Dataset' field
        generated = [entry["Paraphrase"] for entry in paraphrases if entry["Dataset"] == "ETPC"]

        # Debug: Log lengths of generated paraphrases and references to check for mismatches
        logging.info(f"Generated ETPC paraphrases: {len(generated)}, References: {len(references)}")

        # Evaluate the paraphrases using ROUGE and BLEU
        scores = evaluate_paraphrases(generated, references)
        metrics.append({"Model": model_name, "Adapter": model_suffix, **scores})

    save_paraphrases_to_json(all_paraphrases, output_json)
    save_metrics_to_csv(metrics, output_csv)


def main():
    args = parse_arguments()

    # Set batch size and number of examples
    batch_size = 8  
    num_examples = 16  

    # Ensure that num_examples is divisible by batch_size
    if num_examples % batch_size != 0:
        raise ValueError(f"num_examples ({num_examples}) must be divisible by batch_size ({batch_size})")

    # Define output files
    output_csv = f"out/eval_{args.model_name.split('/')[-1]}.csv"
    output_json = f"out/generated_paraphrases_{args.model_name.split('/')[-1]}.json"
    
    # Load datasets, limiting ETPC data to num_examples
    apty_data = read_sentences_from_files("src/basesentences")
    etpc_data = load_data("out/generation_etpc_test.jsonl", num_examples=num_examples)

    # List of models to process: base, ETPC adapter, DPO adapter
    models = [
        (args.model_name, None, "base_model"),
        (args.model_name, args.etpc_dir, "etpc_model"),
        (args.model_name, args.dpo_dir, "dpo_model")
    ]

    # Generate paraphrases and evaluate models
    generate_and_evaluate(models, apty_data, etpc_data, output_csv, output_json, batch_size)


if __name__ == "__main__":
    main()

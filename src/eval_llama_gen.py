import argparse
import os
import json
import logging
from typing import Dict, List, Optional, Any
import torch
import chardet

from tqdm import tqdm
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from rouge import Rouge
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, PreTrainedModel, PreTrainedTokenizerBase
from peft import PeftModel, PeftConfig
import pandas as pd

# Initialize Hugging Face Hub login (if needed)
from huggingface_hub import login

with open("token_file.txt", "r") as token_file:
    hf_token = token_file.read().strip()
login(token=hf_token)

# Logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def parse_arguments():
    """Parse command-line arguments.

    This function defines and parses the command-line arguments for the script.
    It uses the argparse library to create an ArgumentParser object, which is
    then used to parse the command-line arguments.

    The arguments are as follows:

    --model_name: The path to the base model, which is used to generate the
        paraphrases. The default value is "meta-llama/Llama-2-7b-hf".
    --etpc_dir: The directory containing the ETPC adapter. The default value
        is "src/llama/llama-7b-etpc".
    --dpo_dir: The directory containing the DPO adapter. The default value is
        "out/dpo_llama-7b_apty".

    Returns:
        The parsed arguments as a Namespace object.
    """
    parser = argparse.ArgumentParser(description="Generate paraphrases and evaluate models.")
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-2-7b-hf", help="Base model path.")
    parser.add_argument("--etpc_dir", type=str, default="out/gen-models/llama-7b-etpc", help="ETPC adapter directory.")
    parser.add_argument("--dpo_dir", type=str, default="out/gen-models/dpo_meta-llama-Llama-7b-hf_sigmoid", help="DPO adapter directory.")
    parser.add_argument("--ipo_dir", type=str, default="out/gen-models/dpo_meta-llama-Llama-7b-hf_ipo", help="DPO adapter directory.")
    return parser.parse_args()

def load_data(filename, num_examples=None):
    """
    Loads data from a file in JSON format and limits the number of examples.

    Args:
        filename (str): The path to the file to load.
        num_examples (int, optional): The number of examples to load. Defaults to None.

    Returns:
        list: The loaded data as a list of dictionaries.
    """
    with open(filename, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f.readlines()]

    if num_examples:
        # Limit the number of examples to the specified number
        data = data[:num_examples]

    logging.info(f"Loaded {len(data)} examples from {filename}")
    return data

import chardet  # Make sure chardet is imported

def read_sentences_from_files(data_dir):
    """Reads base sentences and paraphrase types from text files.

    Args:
        data_dir (str): The path to the directory containing the text files.

    Returns:
        dict: A dictionary with the paraphrase type as key and a list of
            sentences as value.
    """
    sentences_by_type = {}
    for file_name in os.listdir(data_dir):
        if file_name.endswith(".txt"):
            file_path = os.path.join(data_dir, file_name)

            # Detect file encoding
            with open(file_path, 'rb') as file:
                raw_data = file.read()
                result = chardet.detect(raw_data)
                encoding = result['encoding']  # Use detected encoding

            # Read the file with the detected encoding
            with open(file_path, 'r', encoding=encoding) as file:
                # Read the first line as the paraphrase type and the following lines as sentences
                paraphrase_type = file.readline().strip()
                sentences = [line.strip() for line in file if line.strip()]  # Read remaining lines, removing empty ones
                sentences_by_type[paraphrase_type] = sentences[:10]  # Limit to the first 10 sentences

    return sentences_by_type

def load_model_and_tokenizer(model_name: str, adapter_dir: Optional[str] = None) -> tuple:
    """Load a model and tokenizer, applying PEFT adapters if specified, and ensure padding configuration.

    Args:
        model_name (str): The name of the model to load.
        adapter_dir (str, optional): The path to the PEFT adapter to apply. Defaults to None.

    Returns:
        tuple: The loaded model and tokenizer.
    """
    logging.info(f"Loading model and tokenizer: {model_name}")
    
    # Configure for bitsandbytes (4-bit quantization) and use bfloat16
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,  # Load in 4-bit precision
        bnb_4bit_compute_dtype=torch.bfloat16,  # Use bfloat16 for computations
    )

    # Load the base model with bfloat16 precision for better stability
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        torch_dtype=torch.bfloat16,  # Load in bfloat16 precision
        low_cpu_mem_usage=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    # Ensure left-padding is used and add a padding token if missing
    tokenizer.padding_side = "left"

    # Add a padding token if it's missing and resize embeddings
    if tokenizer.pad_token is None:
        logging.info("Adding a padding token ('<pad>') to the tokenizer.")
        tokenizer.add_special_tokens({"pad_token": "<pad>"})
        model.resize_token_embeddings(len(tokenizer))  # Resize embeddings to match the new vocabulary size

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

def generate_paraphrases(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    data: List[str],
    dataset_type: str = "apty",
    paraphrase_type: Optional[str] = None,
    temperature: float = 0.6,  
    top_p: float = 0.9,        
    batch_size: int = 1,     
    max_length: int = 512,   
    max_new_tokens: int = 50 
) -> List[str]:
    """
    Generate paraphrases for a list of sentences or prompts using the provided model and tokenizer.
    """
    paraphrases = []
    device = next(model.parameters()).device  

    total_batches = (len(data) + batch_size - 1) // batch_size
    progress_bar = tqdm(total=total_batches, desc="Generating Paraphrases", unit="batch")

    logging.info(f"Starting paraphrase generation: {len(data)} sentences, batch size = {batch_size}")

    for i in range(0, len(data), batch_size):
        batch = data[i:i + batch_size]  # Get batch
        logging.debug(f"Processing batch {i // batch_size + 1}/{total_batches}")

        if dataset_type == "apty":
            prompts = [
                f"Instruction: Given the following sentence, generate a paraphrase with the following type. "
                f"Sentence: {sentence} Paraphrase Type: {paraphrase_type}. Generated Paraphrase: "
                for sentence in batch
            ]
        elif dataset_type == "etpc":
            prompts = [instance["messages"][0]["content"] for instance in batch]

        # Tokenize and move to device
        inputs = tokenizer(
            prompts, 
            return_tensors="pt", 
            padding=True, 
            truncation=True, 
            max_length=max_length
        ).to(device)

        input_token_len = inputs["input_ids"].shape[-1]

        with torch.no_grad():
            outputs = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"], 
                max_new_tokens=max_new_tokens,  
                top_p=top_p,                    
                temperature=temperature,        
                do_sample=True                  
            )

        # Decode generated paraphrases
        for output in outputs:
            generated_tokens = output[input_token_len:]
            paraphrase = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
            # Log and track empty paraphrases
            if not paraphrase:
                logging.warning(f"Empty paraphrase generated for input: {prompts[outputs.index(output)]}")
            paraphrases.append(paraphrase)

        logging.debug(f"Generated {len(outputs)} paraphrases in batch {i // batch_size + 1}")
        progress_bar.update(1)

    progress_bar.close()
    logging.info(f"Finished paraphrase generation: {len(paraphrases)} paraphrases generated")

    return paraphrases

def evaluate_paraphrases(paraphrases: List[str], references: List[str]) -> Dict[str, float]:
    """
    Evaluate paraphrases using ROUGE and BLEU scores.

    Args:
        paraphrases (List[str]): The generated paraphrases.
        references (List[str]): The reference texts.

    Returns:
        Dict[str, float]: A dictionary containing the evaluation scores.
    """
    # ROUGE scores
    rouge = Rouge()
    valid_paraphrases = []
    valid_references = []

    # Collect only valid pairs for ROUGE evaluation
    for idx, (paraphrase, reference) in enumerate(zip(paraphrases, references)):
        try:
            # Ensure each pair is valid before adding to the evaluation
            rouge.get_scores(paraphrase, reference)
            valid_paraphrases.append(paraphrase)
            valid_references.append(reference)
        except ValueError as e:
            logging.warning(f"Skipping paraphrase at index {idx} due to ROUGE error: {e}")

    # Perform ROUGE evaluation on valid pairs
    if valid_paraphrases:
        rouge_scores = rouge.get_scores(valid_paraphrases, valid_references, avg=True)
    else:
        # Return zero scores if no valid paraphrases
        logging.warning("No valid paraphrases for ROUGE evaluation.")
        rouge_scores = {
            "rouge-1": {"f": 0},
            "rouge-2": {"f": 0},
            "rouge-l": {"f": 0},
        }

    # BLEU scores
    smoothie = SmoothingFunction().method4
    bleu_scores = [sentence_bleu([ref], paraphrase, smoothing_function=smoothie) for ref, paraphrase in zip(references, paraphrases)]
    avg_bleu = sum(bleu_scores) / len(bleu_scores)

    return {
        # ROUGE scores
        "ROUGE-1": rouge_scores["rouge-1"]["f"],
        "ROUGE-2": rouge_scores["rouge-2"]["f"],
        "ROUGE-L": rouge_scores["rouge-l"]["f"],

        # BLEU scores
        "BLEU": avg_bleu,
    }

def save_metrics_to_csv(metrics, output_csv):
    """
    Save evaluation metrics to a CSV file.

    Args:
        metrics (list): List of dictionaries containing evaluation metrics.
        output_csv (str): Path to the output CSV file.
    """
    fieldnames = ["Model", "Adapter", "ROUGE-1", "ROUGE-2", "ROUGE-L", "BLEU"]

    # Use pandas to write the CSV file
    pd.DataFrame(metrics).to_csv(output_csv, index=False, header=fieldnames)

    logging.info(f"Evaluation metrics saved to {output_csv}")
    
def save_paraphrases_to_json(paraphrases, output_file):
    """
    Saves the generated paraphrases to a JSON file in a human-readable format.

    Args:
        paraphrases (list): List of dictionaries containing the generated paraphrases.
        output_file (str): Path to the output JSON file.
    """
    with open(output_file, 'w', encoding='utf-8') as file:
        # Use json.dump with ensure_ascii=False to allow non-ASCII characters,
        # and indent=4 for pretty-printing
        json.dump(paraphrases, file, ensure_ascii=False, indent=4)
    logging.info(f"Paraphrases saved to {output_file}")
    
# Track loaded adapters globally
loaded_adapters = set()

def process_model_generation(
    model: PreTrainedModel, 
    tokenizer: PreTrainedTokenizerBase, 
    apty_data: Dict[str, List[str]], 
    etpc_data: List[Dict[str, Any]], 
    adapter_dir: Optional[str], 
    model_suffix: str, 
    batch_size: int
) -> List[Dict[str, str]]:
    """
    Generate paraphrases for both ETPC and APTY datasets using the provided model and adapter.

    Args:
        model (PreTrainedModel): Loaded base model.
        tokenizer (PreTrainedTokenizerBase): Loaded tokenizer.
        apty_data (Dict[str, List[str]]): Dictionary of paraphrase types and sentences for APTY dataset.
        etpc_data (List[Dict[str, Any]]): List of entries to generate paraphrases for ETPC dataset.
        adapter_dir (str): Path to the PEFT adapter to apply.
        model_suffix (str): Suffix to use for identifying the model type (base, etpc, dpo).
        batch_size (int): Batch size to use for generation.

    Returns:
        List[Dict[str, str]]: List of dictionaries containing generated paraphrases.
    """
    
    adapter_name = os.path.basename(adapter_dir) if adapter_dir else "base"

    # Check if the adapter is already loaded
    if adapter_name not in loaded_adapters:
        if adapter_dir:
            try:
                # Load and set the adapter
                logging.info(f"Loading PEFT adapter from {adapter_dir}")
                model.load_adapter(adapter_dir, adapter_name=adapter_name)
                model.set_adapter(adapter_name)
                loaded_adapters.add(adapter_name)
                logging.info(f"Adapter '{adapter_name}' activated successfully.")
            except Exception as e:
                logging.error(f"Failed to load or activate adapter: {e}")
                raise
        else:
            logging.info("No adapter specified. Using base model.")
    else:
        # Set the adapter if it's already loaded
        model.set_adapter(adapter_name)
        logging.info(f"Using already loaded adapter: {adapter_name}")

    all_paraphrases = []

    # Generate paraphrases for APTY dataset
    for paraphrase_type, sentences in apty_data.items():
        paraphrases = generate_paraphrases(model, tokenizer, sentences, "apty", paraphrase_type, batch_size=batch_size)
        
        # Log each paraphrase and its original sentence
        for idx, paraphrase in enumerate(paraphrases):
            if not paraphrase.strip():
                logging.warning(f"Empty paraphrase generated for APTY sentence at index {idx}: {sentences[idx]}")
            
            
        all_paraphrases.extend([{
            "Original": sentence,
            "APT": paraphrase_type,
            "Paraphrase": paraphrase,
            "Kind": model_suffix,
            "Dataset": "APTY"
        } for sentence, paraphrase in zip(sentences, paraphrases)])

    # Generate paraphrases for ETPC dataset
    etpc_paraphrases = generate_paraphrases(model, tokenizer, etpc_data, "etpc", batch_size=batch_size)
    for index, paraphrase in enumerate(etpc_paraphrases):
        full_content = etpc_data[index]["messages"][0]["content"]
        
        if not paraphrase.strip():
            logging.warning(f"Empty paraphrase for ETPC content at index {index}: {full_content}")


        # Safely extract the sentence and paraphrase types from the ETPC content
        if "Sentence:" in full_content and "Paraphrase Types:" in full_content:
            original_sentence = full_content.split("Sentence:")[1].split("Paraphrase Types:")[0].strip()
            paraphrase_types = full_content.split("Paraphrase Types:")[1].strip()
        else:
            original_sentence = full_content
            paraphrase_types = "Unknown"

        all_paraphrases.append({
            "Original": original_sentence,
            "APT": paraphrase_types,
            "Paraphrase": paraphrase,
            "Kind": model_suffix,
            "Dataset": "ETPC"
        })

    return all_paraphrases

def generate_and_evaluate(base_model, tokenizer, models, apty_data, etpc_data, output_csv, output_json, batch_size):
    """
    Main function to generate paraphrases for different models and evaluate them.
    """
    metrics = []
    all_paraphrases = []
    references = [item["messages"][0]["content"] for item in etpc_data]  # References for ETPC dataset

    for model_name, adapter_dir, model_suffix in models:
        logging.info(f"Starting generation for model: {model_suffix}")
        
        # Extract the adapter name from the adapter_dir (e.g., 'llama-7b-etpc')
        adapter_name = os.path.basename(adapter_dir) if adapter_dir else "base"

        paraphrases = process_model_generation(base_model, tokenizer, apty_data, etpc_data, adapter_dir, model_suffix, batch_size)
        all_paraphrases.extend(paraphrases)

        # Filter ETPC-generated paraphrases
        generated = [entry["Paraphrase"] for entry in paraphrases if entry["Dataset"] == "ETPC"]

         # Log and skip empty paraphrases
        for idx, (gen, ref) in enumerate(zip(generated, references)):
            if not gen.strip():
                logging.warning(f"Empty paraphrase for reference at index {idx}: {ref}")

        if len(generated) == 0:
            logging.warning(f"No paraphrases generated for model {model_suffix}")
            continue

        # Evaluate the paraphrases using ROUGE and BLEU
        scores = evaluate_paraphrases(generated, references)
        logging.info(f"Scores for {model_suffix}: {scores}")

        # Append the correct model and adapter names to the metrics
        metrics.append({
            "Model": model_name,  # Use the actual model name
            "Adapter": adapter_name,  # Use the extracted adapter name or 'base'
            **scores
        })

    # Save the results
    save_paraphrases_to_json(all_paraphrases, output_json)
    save_metrics_to_csv(metrics, output_csv)

def main():
    """
    Main function for evaluating the base model, ETPC adapter, and DPO adapter on the ETPC dataset.
    """
    args = parse_arguments()

    # Set batch size and number of examples
    batch_size = 10  
    num_examples = 1000 

    # Ensure that num_examples is divisible by batch_size
    if num_examples % batch_size != 0:
        raise ValueError(f"num_examples ({num_examples}) must be divisible by batch_size ({batch_size})")

    # Define output files
    output_csv = f"out/eval_{args.model_name.split('/')[-1]}.csv"
    output_json = f"out/generated_paraphrases_{args.model_name.split('/')[-1]}.json"
    
    # Load datasets, limiting ETPC data to num_examples
    apty_data = read_sentences_from_files("out/basesentences")
    etpc_data = load_data("out/generation_etpc_test.jsonl", num_examples=num_examples)

    # Load base model and tokenizer once
    base_model, tokenizer = load_model_and_tokenizer(args.model_name)

    # List of models to process: base, ETPC adapter, DPO adapter
    models = [
        (args.model_name, None, "base_model"),           # Base model (no adapter)
        (args.model_name, args.etpc_dir, "etpc_model"),  # ETPC adapter
        (args.model_name, args.dpo_dir, "dpo_model"),    # DPO adapter 
        (args.model_name, args.ipo_dir, "ipo_model"),    # IPO adapter       
    ]


    # Generate paraphrases and evaluate models
    generate_and_evaluate(base_model, tokenizer, models, apty_data, etpc_data, output_csv, output_json, batch_size)

if __name__ == "__main__":
    main()


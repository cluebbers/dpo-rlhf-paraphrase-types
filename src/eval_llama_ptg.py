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
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.1-8B", help="Base model path.")
    parser.add_argument("--etpc_dir", type=str, default="out/gen-models/llama-3.1-8b-etpc", help="ETPC adapter directory.")
    parser.add_argument("--dpo_dir", type=str, default="out/gen-models/dpo_out-gen-models-llama-3.1-8b-etpc_sigmoid", help="DPO adapter directory.")
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
    # After adding, ensure that pad_token_id is set
    assert tokenizer.pad_token_id is not None, "Padding token should be added and tokenizer's pad_token_id should not be None."
    
    # Set the model's padding token ID to the new padding token
    model.config.pad_token_id = tokenizer.pad_token_id or model.config.eos_token_id


    model.eval()
    torch.cuda.empty_cache()
    
    return model, tokenizer

def tokenize_data(tokenizer, prompts: List[str], max_length: int = 256) -> Dict[str, torch.Tensor]:
    """Tokenizes a list of prompts using the specified tokenizer."""
    return tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_length
    )
    
def generate_paraphrases(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    tokenized_inputs: Dict[str, torch.Tensor],
    input_token_len: int,
    temperature: float = 0.6,  
    top_p: float = 0.9,        
    batch_size: int = 10,     
    max_new_tokens: int = 50 
) -> List[str]:
    paraphrases = []
    device = next(model.parameters()).device
    input_ids = tokenized_inputs["input_ids"].to(device)
    attention_mask = tokenized_inputs["attention_mask"].to(device)
    total_batches = (input_ids.shape[0] + batch_size - 1) // batch_size
    progress_bar = tqdm(total=total_batches, desc="Generating Paraphrases", unit="batch")
    for i in range(0, input_ids.shape[0], batch_size):
        batch_input_ids = input_ids[i:i + batch_size]
        batch_attention_mask = attention_mask[i:i + batch_size]
        with torch.inference_mode():
            outputs = model.generate(
                input_ids=batch_input_ids,
                attention_mask=batch_attention_mask,
                max_new_tokens=max_new_tokens,
                top_p=top_p,
                temperature=temperature,
                do_sample=True
            )
        for output in outputs:
            generated_tokens = output[input_token_len:]
            paraphrase = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
            paraphrases.append(paraphrase)
        progress_bar.update(1)
    progress_bar.close()
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
   # fieldnames = ["Model", "Adapter", "ROUGE-1", "ROUGE-2", "ROUGE-L", "BLEU"]

    # Use pandas to write the CSV file
    pd.DataFrame(metrics).to_csv(output_csv, index=False)

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

def evaluate_paraphrases_individual(paraphrases: List[str], references: List[str]) -> List[Dict[str, Any]]:
    """
    Evaluate paraphrases using ROUGE and BLEU scores, returning individual scores per sentence pair.

    Args:
        paraphrases (List[str]): The generated paraphrases.
        references (List[str]): The reference texts.

    Returns:
        List[Dict[str, Any]]: A list containing the evaluation scores for each paraphrase-reference pair.
    """
    # ROUGE scores
    rouge = Rouge()
    valid_paraphrases = []
    valid_references = []
    individual_scores = []

    # Collect only valid pairs for ROUGE evaluation
    for idx, (paraphrase, reference) in enumerate(zip(paraphrases, references)):
        try:
            # Ensure each pair is valid before adding to the evaluation
            rouge_score = rouge.get_scores(paraphrase, reference)[0]  # Get individual scores for this pair
            smoothie = SmoothingFunction().method4
            bleu_score = sentence_bleu([reference.split()], paraphrase.split(), smoothing_function=smoothie)

            individual_scores.append({
                "rouge-1": rouge_score["rouge-1"]["f"],
                "rouge-2": rouge_score["rouge-2"]["f"],
                "rouge-l": rouge_score["rouge-l"]["f"],
                "bleu": bleu_score
            })

        except ValueError as e:
            logging.warning(f"Skipping paraphrase at index {idx} due to ROUGE error: {e}")
            individual_scores.append({
                "rouge-1": 0.0,
                "rouge-2": 0.0,
                "rouge-l": 0.0,
                "bleu": 0.0
            })

    return individual_scores
 
def process_model_generation(
    model: PreTrainedModel, 
    tokenizer: PreTrainedTokenizerBase, 
    tokenized_apty_data: Dict[str, Dict[str, torch.Tensor]], 
    tokenized_etpc_data: Dict[str, torch.Tensor], 
    etpc_references: List[str],
    model_suffix: str, 
    batch_size: int
) -> List[Dict[str, Any]]:
    grouped_paraphrases = {}

    # Generate paraphrases for APTY dataset
    for paraphrase_type, tokenized_inputs in tokenized_apty_data.items():
        input_token_len = tokenized_inputs["input_ids"].shape[-1]
        paraphrases = generate_paraphrases(
            model,
            tokenizer,
            tokenized_inputs,
            input_token_len,
            batch_size=batch_size
        )
        for original_sentence, paraphrase in zip(tokenized_inputs["original_sentences"], paraphrases):
            if original_sentence not in grouped_paraphrases:
                grouped_paraphrases[original_sentence] = {
                    "id": hash(original_sentence) % 1000,
                    "original": original_sentence,
                    "dataset": "APTY",
                    "List": []
                }
            grouped_paraphrases[original_sentence]["List"].append({
                "id": len(grouped_paraphrases[original_sentence]["List"]),
                "paraphrase": paraphrase,
                "model": model_suffix
            })

    # Generate paraphrases for ETPC dataset
    input_token_len = tokenized_etpc_data["input_ids"].shape[-1]
    etpc_paraphrases = generate_paraphrases(
        model,
        tokenizer,
        tokenized_etpc_data,
        input_token_len,
        batch_size=batch_size
    )
    for original_sentence, paraphrase, reference in zip(
        tokenized_etpc_data["original_sentences"], etpc_paraphrases, etpc_references
    ):
        evaluation = evaluate_paraphrases_individual([paraphrase], [reference])[0]
        if original_sentence not in grouped_paraphrases:
            grouped_paraphrases[original_sentence] = {
                "id": hash(original_sentence) % 1000,
                "original": original_sentence,
                "reference": reference,
                "dataset": "ETPC",
                "List": []
            }
        grouped_paraphrases[original_sentence]["List"].append({
            "id": len(grouped_paraphrases[original_sentence]["List"]),
            "paraphrase": paraphrase,
            "evaluation": evaluation,
            "model": model_suffix
        })
    return grouped_paraphrases

def generate_and_evaluate(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    models, 
    tokenized_apty_data, 
    tokenized_etpc_data, 
    etpc_references, 
    output_csv, 
    output_json, 
    batch_size
):
    metrics = []
    all_paraphrases = {}

    for model_name, adapter_dir, model_suffix in models:
        if adapter_dir:
            logging.info(f"Adding adapter {model_suffix} from {adapter_dir}")
            model.load_adapter(adapter_dir, adapter_name=model_suffix)
            model.set_adapter(model_suffix)

        # Generate paraphrases including evaluation metrics
        paraphrases = process_model_generation(
            model, tokenizer, tokenized_apty_data, tokenized_etpc_data, 
            etpc_references, model_suffix, batch_size
        )

        # Merge paraphrases into all_paraphrases dictionary
        for original_sentence, details in paraphrases.items():
            if original_sentence not in all_paraphrases:
                all_paraphrases[original_sentence] = details
            else:
                all_paraphrases[original_sentence]["List"].extend(details["List"])

        # Collect evaluation metrics for this model
        scores = [
            entry["evaluation"] for details in paraphrases.values()
            for entry in details["List"] if "evaluation" in entry
        ]
        if scores:
            avg_rouge_1 = sum(score["rouge-1"] for score in scores) / len(scores)
            avg_rouge_2 = sum(score["rouge-2"] for score in scores) / len(scores)
            avg_rouge_l = sum(score["rouge-l"] for score in scores) / len(scores)
            avg_bleu = sum(score["bleu"] for score in scores) / len(scores)
        else:
            avg_rouge_1 = avg_rouge_2 = avg_rouge_l = avg_bleu = 0

        metrics.append({
            "Model": model_name,
            "Adapter": model_suffix,
            "ROUGE-1": avg_rouge_1,
            "ROUGE-2": avg_rouge_2,
            "ROUGE-L": avg_rouge_l,
            "BLEU": avg_bleu,
        })
    
    json_paraphrases = [
        {
            "id": details["id"],
            "original": details["original"],
            "reference": details.get("reference"),
            "dataset": details["dataset"],
            "List": details["List"]
        }
        for details in all_paraphrases.values()
    ]

    save_paraphrases_to_json(json_paraphrases, output_json)
    save_metrics_to_csv(metrics, output_csv)

def main():
    args = parse_arguments()
    batch_size = 10  
    num_examples = 10
    if num_examples % batch_size != 0:
        raise ValueError(f"num_examples ({num_examples}) must be divisible by batch_size ({batch_size})")
    output_csv = f"out/gen-models/eval_{args.model_name.split('/')[-1]}.csv"
    output_json = f"out/gen-models/generated_paraphrases_{args.model_name.split('/')[-1]}.json"
    apty_data = read_sentences_from_files("out/basesentences")
    etpc_data = load_data("out/generation_etpc_test.jsonl", num_examples=num_examples)
    etpc_prompts = [instance["messages"][0]["content"] for instance in etpc_data]
    etpc_references = [instance["messages"][1]["content"] for instance in etpc_data]

    model, tokenizer = load_model_and_tokenizer(args.model_name)
    
    # Pre-tokenize APTY data
    tokenized_apty_data = {
        paraphrase_type: {
            **tokenize_data(tokenizer, [
                f"Instruction: Given the following sentence, generate a paraphrase with the following type. "
                f"Sentence: {sentence} Paraphrase Type: {paraphrase_type}. Generated Paraphrase: "
                for sentence in sentences
            ]),
            "original_sentences": sentences
        }
        for paraphrase_type, sentences in apty_data.items()
    }

    # Pre-tokenize ETPC data
    tokenized_etpc_data = tokenize_data(tokenizer, etpc_prompts)
    tokenized_etpc_data["original_sentences"] = etpc_prompts

    models = [
        (args.model_name, None, "base_model"),
        (args.model_name, args.etpc_dir, "etpc_model"),
        (args.model_name, args.dpo_dir, "dpo_model"),    # DPO adapter 
        #(args.model_name, args.ipo_dir, "ipo_model"),    # IPO adapter  
    ]

    generate_and_evaluate(
        model, tokenizer, models, tokenized_apty_data, tokenized_etpc_data, 
        etpc_references, output_csv, output_json, batch_size
    )
if __name__ == "__main__":
    main()


import argparse
import os
import json
import logging
import numpy as np
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

ALL_TYPES = {
    "Derivational Changes",
    "Inflectional Changes",
    "Modal Verb Changes",
    "Spelling changes",
    "Change of format",
    "Same Polarity Substitution (contextual)",
    "Same Polarity Substitution (habitual)",
    "Same Polarity Substitution (named ent.)",
    "Converse substitution",
    "Opposite polarity substitution (contextual)",
    "Opposite polarity substitution (habitual)",
    "Synthetic/analytic substitution",
    "Coordination changes",
    "Diathesis alternation",
    "Ellipsis",
    "Negation switching",
    "Subordination and nesting changes",
    "Direct/indirect style alternations",
    "Punctuation changes",
    "Syntax/discourse structure changes",
    "Entailment",
    "Identity",
    "Non-paraphrase",
    "Addition/Deletion",
    "Change of order",
    "Semantic-based",
}

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
    parser.add_argument("--etpc_dir", type=str, default="src/llama/llama-7b-etpc", help="ETPC adapter directory.")
    parser.add_argument("--dpo_dir", type=str, default="out/dpo_llama-7b_apty", help="DPO adapter directory.")
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

def classify(
    data, model, 
    max_gen_len, 
    temperature, 
    top_p, 
    max_batch_size, 
    num_examples=100
):
    """Classifies the data using the provided model and parameters. Now forwards max_batch_size prompts into the model instead of one.

    Args:
        data (list): The data to classify.
        model (object): The model to use for classification.
        max_gen_len (int): The maximum generation length.
        temperature (float): The temperature parameter for the model.
        top_p (float): The top_p parameter for the model.
        max_batch_size (int): The maximum batch size for the model.
        num_examples (int, optional): The number of examples to classify. Defaults to 100.

    Returns:
        tuple: The true and predicted labels.
    """
    y_true = []
    y_pred = []

    for i in tqdm(range(0, num_examples, max_batch_size)):
        batch = data[i : i + max_batch_size]
        user_messages = [instance["messages"][0]["content"] for instance in batch]
        true_response_labels = [
            set(instance["messages"][1]["content"].split(", ")) for instance in batch
        ]

        # Call the API and retry if it fails

        results = model.generate(
            user_messages,
            max_gen_len=max_gen_len,
            temperature=temperature,
            top_p=top_p,
        )

        predicted_response_labels = [
            set(result["generation"].split(", ")) for result in results
        ]

        if results:
            y_true.extend(true_response_labels)
            y_pred.extend(predicted_response_labels)

    return y_true, y_pred


def evaluate(y_true, y_pred):
    """Evaluates the performance of a classification model.

    Args:
        y_true (list): The true labels.
        y_pred (list): The predicted labels.

    Returns:
        tuple: A tuple containing the F1 score and accuracy.

    Example:
        ```python
        y_true = [[1, 0, 1], [0, 1, 0]]
        y_pred = [[1, 1, 0], [0, 1, 1]]

        acc = evaluate(y_true, y_pred)
        print(acc)
        ```
    """

    y_true_bin = [[1 if t in labels else 0 for t in ALL_TYPES] for labels in y_true]
    y_pred_bin = [[1 if t in labels else 0 for t in ALL_TYPES] for labels in y_pred]

    # Convert lists to numpy arrays for easier calculations
    y_true_np = np.array(y_true_bin)
    y_pred_np = np.array(y_pred_bin)

    # Calculate per-class accuracy
    acc = np.mean(np.equal(y_true_np, y_pred_np).astype(int))

    return acc
    
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

def classify_and_evaluate(test_data):
    y_true, y_pred = classify(
        test_data,
        model,
        max_gen_len=max_gen_len,
        temperature=temperature,
        top_p=top_p,
        max_batch_size=max_batch_size,
        num_examples=num_examples,
    )
    _, acc = evaluate(y_true, y_pred)
    
    
    return acc, y_pred

def main(ckpt_dir: str,
    tokenizer_path: str,
    data_file: str,
    num_examples: int = 1000,
    temperature: float = 0.6,
    top_p: float = 0.9,
    max_seq_len: int = 2048,
    max_gen_len: int = 1024,
    max_batch_size: int = 4,
    ):
    """
    Main function for evaluating the base model, ETPC adapter, and DPO adapter on the ETPC dataset.
    """
    args = parse_arguments()

    # Set batch size and number of examples
    batch_size = 10  
    num_examples = 10  

    # Ensure that num_examples is divisible by batch_size
    if num_examples % batch_size != 0:
        raise ValueError(f"num_examples ({num_examples}) must be divisible by batch_size ({batch_size})")

    # Define output files
    output_csv = f"out/eval_{args.model_name.split('/')[-1]}.csv"
    output_json = f"out/generated_paraphrases_{args.model_name.split('/')[-1]}.json"
    
    # Load datasets, limiting ETPC data to num_examples
    apty_data = load_data("out/detection_etpc_test.jsonl", num_examples=num_examples)
    etpc_data = load_data("out/detection_etpc_test.jsonl", num_examples=num_examples)

    # Load base model and tokenizer once
    base_model, tokenizer = load_model_and_tokenizer(args.model_name)

    # List of models to process: base, ETPC adapter, DPO adapter
    models = [
        (args.model_name, None, "base_model"),           # Base model (no adapter)
        (args.model_name, args.etpc_dir, "etpc_model"),  # ETPC adapter
        (args.model_name, args.dpo_dir, "dpo_model"),    # DPO adapter        
    ]
    
    # Generate paraphrases and evaluate models
    acc, y_pred = classify_and_evaluate(base_model, tokenizer, models, apty_data, etpc_data, output_csv, output_json, batch_size)

    print(f"Model: {ckpt_dir}")
    print(f"Eval set size: {len(y_pred)}")
    print(f"Accuracy: {acc:.2f}")
    
if __name__ == "__main__":
    main()


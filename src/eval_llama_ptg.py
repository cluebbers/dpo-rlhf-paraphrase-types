import argparse
import json
import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import torch
from datasets import load_dataset
# Initialize Hugging Face Hub login (if needed)
from huggingface_hub import login
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from rouge import Rouge
from tqdm import tqdm
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          BitsAndBytesConfig, PreTrainedModel,
                          PreTrainedTokenizerBase)

with open("token_file.txt", "r") as token_file:
    hf_token = token_file.read().strip()
login(token=hf_token)

# Logging configuration
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def parse_arguments() -> argparse.Namespace:
    """
    Parse command-line arguments for generating paraphrases and evaluating models.

    Args:
        --model_name (str): Path to the base model. Defaults to "meta-llama/Llama-3.1-8B".
        --etpc_dir (str): Directory containing the ETPC adapter. Defaults to "out/gen-models/llama-3.1-8b-etpc".
        --dpo_dir (str): Directory containing the DPO adapter. Defaults to "out/gen-models/dpo_out-gen-models-llama-3.1-8b-etpc_sigmoid".
        --ipo_dir (str): Directory containing the DPO adapter. Defaults to "out/gen-models/dpo_out-gen-models-llama-3.1-8b-etpc_ipo".

    Returns:
        argparse.Namespace: The parsed command line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Generate paraphrases and evaluate models."
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="meta-llama/Llama-3.1-8B",
        help="Base model path.",
    )
    parser.add_argument(
        "--etpc_dir",
        type=str,
        default="out/gen-models/llama-3.1-8b-etpc",
        help="ETPC adapter directory.",
    )
    parser.add_argument(
        "--dpo_dir",
        type=str,
        default="out/gen-models/dpo_out-gen-models-llama-3.1-8b-etpc_sigmoid",
        help="DPO adapter directory.",
    )
    parser.add_argument(
        "--ipo_dir",
        type=str,
        default="out/gen-models/dpo_out-gen-models-llama-3.1-8b-etpc_ipo",
        help="DPO adapter directory.",
    )

    args: argparse.Namespace = parser.parse_args()

    # Convert the string "None" to actual None
    if args.etpc_dir == "None":
        args.etpc_dir = None
    if args.dpo_dir == "None":
        args.dpo_dir = None
    if args.ipo_dir == "None":
        args.ipo_dir = None

    return args


def create_etpc_prompts(data: List[Dict[str, Any]]) -> Tuple[List[str], List[str]]:
    """
    Converts the ETPC data into prompts and references
    for paraphrase generation.

    Args:
        data (List[Dict[str, Any]]): The ETPC dataset.

    Returns:
        Tuple[List[str], List[str]]: A tuple containing the prompts and references.
    """
    prompts: List[str] = []
    references: List[str] = []
    for instance in data:
        # Skip instances without any paraphrase types
        if not instance["paraphrase_types"]:
            continue

        # Construct the prompt
        prompt = (
            "Given the following sentence, generate a paraphrase with the following types. "
            f"Sentence: {instance['sentence1']} "
            f"Paraphrase Types: {', '.join(instance['paraphrase_types'])}. "
            "Generated Paraphrase: "
        )
        prompts.append(prompt)
        # Add the reference paraphrase
        references.append(instance["sentence2"])

    return prompts, references


def read_sentences_by_type(
    data_dir: str, num_examples: int
) -> Dict[str, Dict[str, List[str]]]:
    """
    Reads all .txt files in the given directory and creates a dictionary where the keys are the paraphrase type names
    (from the first line of each file) and the values are dictionaries containing the paraphrase type name and a list of
    sentences (from the rest of the file). The list of sentences is truncated to num_examples.

    Args:
        data_dir (str): The directory path containing the .txt files.
        num_examples (int): The maximum number of sentences to read per paraphrase type.

    Returns:
        Dict[str, Dict[str, List[str]]]: A dictionary where the keys are the paraphrase type names
            and the values are dictionaries containing the paraphrase type name and a list of sentences.
    """
    sentences_by_type: Dict[str, Dict[str, List[str]]] = {}

    for file_name in os.listdir(data_dir):
        file_path = os.path.join(data_dir, file_name)
        if os.path.isfile(file_path) and file_name.endswith(".txt"):
            try:
                with open(file_path, "r", encoding="utf-8-sig") as file:
                    paraphrase_type = next(
                        file
                    ).strip()  # The first line is the paraphrase type name
                    sentences = [
                        line.strip() for line in file if line.strip()
                    ]  # The rest are the sentences
            except UnicodeDecodeError:
                logging.warning(
                    f"Failed to decode {file_name} with utf-8. Trying ISO-8859-1."
                )
                with open(file_path, "r", encoding="ISO-8859-1") as file:
                    paraphrase_type = next(
                        file
                    ).strip()  # The first line is the paraphrase type name
                    sentences = [
                        line.strip() for line in file if line.strip()
                    ]  # The rest are the sentences
            sentences = sentences[:num_examples]  # Truncate the list of sentences
            if paraphrase_type not in sentences_by_type:
                sentences_by_type[paraphrase_type] = {
                    "type": paraphrase_type,
                    "sentences": sentences[
                        :num_examples
                    ],  # Add the list of sentences to the dictionary
                }
            else:
                sentences_by_type[paraphrase_type]["sentences"].extend(
                    sentences[:num_examples]
                )
                # Ensure no duplicates if needed
                sentences_by_type[paraphrase_type]["sentences"] = list(
                    set(sentences_by_type[paraphrase_type]["sentences"])
                )

    return sentences_by_type


def load_tokenizer(model_name: str) -> PreTrainedTokenizerBase:
    """
    Load a tokenizer from the Hugging Face model hub.

    Args:
        model_name (str): The name of the model to load the tokenizer from.

    Returns:
        PreTrainedTokenizerBase: The loaded tokenizer, configured to pad on the
        left and use the fast tokenizer implementation. A pad token is added if
        it does not exist in the tokenizer's vocabulary.
    """

    # Load the tokenizer with specific settings
    tokenizer: PreTrainedTokenizerBase = AutoTokenizer.from_pretrained(
        model_name,
        padding_side="left",  
        use_fast=True,  
    )

    tokenizer.add_special_tokens({"pad_token": "<|finetune_right_pad_id|>"})
    tokenizer.pad_token = "<|finetune_right_pad_id|>"

    return tokenizer


def tokenize_data(
    tokenizer: PreTrainedTokenizerBase,
    prompts: List[str],
) -> Dict[str, torch.Tensor]:
    """
    Tokenize the prompts using the provided tokenizer.

    Args:
        tokenizer (PreTrainedTokenizerBase): The tokenizer to use.
        prompts (List[str]): The list of prompts to tokenize.

    Returns:
        Dict[str, torch.Tensor]: A dictionary containing the tokenized input IDs and attention mask.
    """
    # Tokenize the prompts
    tokenized_inputs: Dict[str, torch.Tensor] = tokenizer(
        prompts,
        return_tensors="pt",  # Return PyTorch tensors
        padding=True,  # Pad the inputs to the longest sequence
        truncation=True,  # Truncate long sequences
    )

    return tokenized_inputs


def generate_paraphrases(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    inputs: Dict[str, torch.Tensor],
    input_token_length: int,
    temperature: float = 0.6,
    top_p: float = 0.9,
    batch_size: int = 10,
    max_new_tokens: int = 50,
) -> List[str]:
    """
    Generate paraphrases using the provided model and tokenizer.

    Args:
        model (transformers.PreTrainedModel): The pre-trained language model for generation.
        tokenizer (transformers.PreTrainedTokenizerBase): Tokenizer used for encoding and decoding text.
        inputs (Dict[str, torch.Tensor]): Tokenized inputs containing input IDs and attention masks.
        input_token_length (int): Length of the input tokens to separate them from generated tokens.
        temperature (float, optional): Controls the randomness of predictions by scaling the logits.
        top_p (float, optional): Probability threshold for nucleus sampling.
        batch_size (int, optional): Number of examples per batch.
        max_new_tokens (int, optional): Maximum number of new tokens to generate.

    Returns:
        List[str]: List of generated paraphrases.
    """

    paraphrases = []
    device = model.device
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)
    total_batches = (input_ids.shape[0] + batch_size - 1) // batch_size

    # Create a progress bar to monitor the paraphrasing process
    progress_bar = tqdm(
        total=total_batches, desc="Generating Paraphrases", unit="batch"
    )

    for i in range(0, input_ids.shape[0], batch_size):
        batch_input_ids = input_ids[i : i + batch_size]
        batch_attention_mask = attention_mask[i : i + batch_size]

        # Generate paraphrases using the model in inference mode
        with torch.inference_mode():
            outputs = model.generate(
                input_ids=batch_input_ids,
                attention_mask=batch_attention_mask,
                max_new_tokens=max_new_tokens,
                top_p=top_p,
                temperature=temperature,
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

        # Decode the generated outputs into text and clean up special tokens
        for output in outputs:
            generated_tokens = output[input_token_length:]
            paraphrase = tokenizer.decode(
                generated_tokens, skip_special_tokens=True
            ).strip()
            paraphrases.append(paraphrase)

        # Update the progress bar for each batch processed
        progress_bar.update(1)

    progress_bar.close()
    return paraphrases


def save_metrics_to_csv(metrics: List[Dict[str, float]], output_csv: str) -> None:
    """
    Save evaluation metrics to a CSV file.

    Args:
        metrics (List[Dict[str, float]]): List of dictionaries containing evaluation metrics.
        output_csv (str): Path to the output CSV file.

    Returns:
        None
    """
    # Define the column names for the CSV file
    fieldnames = ["Model", "Adapter", "ROUGE-1", "ROUGE-2", "ROUGE-L", "BLEU"]

    # Convert the metrics list into a Pandas DataFrame with the specified columns
    pd.DataFrame(metrics, columns=fieldnames).to_csv(output_csv, index=False)

    # Log the completion of writing the metrics to the CSV file
    logging.info(f"Evaluation metrics saved to {output_csv}")


def save_paraphrases_to_json(
    paraphrases_list: List[Dict[str, str]], output_file_path: str
) -> None:
    """
    Save the generated paraphrases to a JSON file in a human-readable format.

    Args:
        paraphrases_list (List[Dict[str, str]]): List of dictionaries containing the generated paraphrases.
        output_file_path (str): Path to the output JSON file.

    Returns:
        None
    """
    # Open the output file in write mode, using UTF-8 encoding
    with open(output_file_path, "w", encoding="utf-8") as output_file:
        # Use the json.dump method to write the paraphrases_list to the output file
        # The ensure_ascii=False argument ensures that non-ASCII characters are preserved
        # The indent=4 argument formats the JSON output with indentation
        json.dump(paraphrases_list, output_file, ensure_ascii=False, indent=4)
    
    logging.info(f"Generated paraphrases saved to {output_file_path}")


def evaluate_individual_paraphrases(
    paraphrases: List[str], references: List[str]
) -> List[Dict[str, float]]:
    """
    Evaluate paraphrases using ROUGE and BLEU scores, returning individual scores per sentence pair.

    Args:
        paraphrases (List[str]): The generated paraphrases.
        references (List[str]): The reference texts.

    Returns:
        List[Dict[str, float]]: A list of dictionaries containing the evaluation scores for each paraphrase-reference pair.
            Each dictionary will contain the following keys:
                - "rouge-1": ROUGE-1 F-score
                - "rouge-2": ROUGE-2 F-score
                - "rouge-l": ROUGE-L F-score
                - "bleu": BLEU score
    """
    # Use the ROUGE library to calculate the ROUGE scores
    rouge = Rouge()
    scores = []

    # Iterate over the paraphrases and references, calculating the ROUGE and BLEU scores
    for paraphrase, reference in zip(paraphrases, references):
        try:
            # Calculate the ROUGE scores using the get_scores method
            rouge_score = rouge.get_scores(paraphrase, reference)[0]
            # Calculate the BLEU score using the sentence_bleu method
            # Use the smoothing function Smoothie to avoid division by zero
            smoothie = SmoothingFunction().method4
            bleu_score = sentence_bleu(
                [reference.split()], paraphrase.split(), smoothing_function=smoothie
            )

            # Append the scores to the scores list
            scores.append(
                {
                    "rouge-1": rouge_score["rouge-1"]["f"],
                    "rouge-2": rouge_score["rouge-2"]["f"],
                    "rouge-l": rouge_score["rouge-l"]["f"],
                    "bleu": bleu_score,
                }
            )
        except ValueError:
            # Append a dictionary with zeros if the ROUGE score calculation fails
            scores.append({"rouge-1": 0.0, "rouge-2": 0.0, "rouge-l": 0.0, "bleu": 0.0})

    return scores


def process_model_generation(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    tokenized_apty_data: Dict[str, Dict[str, torch.Tensor]],
    tokenized_etpc_data: Dict[str, torch.Tensor],
    etpc_references: List[str],
    model_suffix: str,
    batch_size: int,
) -> List[Dict[str, Any]]:
    """
    Process the model generation for APTY and ETPC datasets.

    Args:
        model (PreTrainedModel): The pre-trained language model for generation.
        tokenizer (PreTrainedTokenizerBase): Tokenizer used for encoding and decoding text.
        tokenized_apty_data (Dict[str, Dict[str, torch.Tensor]]): Tokenized inputs for APTY dataset.
        tokenized_etpc_data (Dict[str, torch.Tensor]): Tokenized inputs for ETPC dataset.
        etpc_references (List[str]): List of reference texts for ETPC dataset.
        model_suffix (str): Suffix to be added to the model name.
        batch_size (int): Number of examples per batch.

    Returns:
        List[Dict[str, Any]]: List of dictionaries containing the generated paraphrases,
            their corresponding original sentences, and evaluation metrics.
    """

    grouped_paraphrases: Dict[str, Dict[str, Any]] = {}

    # Generate paraphrases for APTY dataset
    for paraphrase_type, tokenized_inputs in tokenized_apty_data.items():
        input_token_len: int = tokenized_inputs["input_ids"].shape[-1]
        paraphrases: List[str] = generate_paraphrases(
            model, tokenizer, tokenized_inputs, input_token_len, batch_size=batch_size
        )
        # Use a combined key of original_sentence and paraphrase_type for APTY
        for original_sentence, paraphrase in zip(
            tokenized_inputs["original_sentences"], paraphrases
        ):
            key: str = (
                f"{original_sentence}_{paraphrase_type}"  # Combine original sentence and type
            )
            if key not in grouped_paraphrases:
                grouped_paraphrases[key] = {
                    "id": hash(key) % 1000,
                    "original": original_sentence,
                    "dataset": "APTY",
                    "APT": paraphrase_type,  # Include APT for APTY dataset.
                    "List": [],
                }
            grouped_paraphrases[key]["List"].append(
                {
                    "id": len(grouped_paraphrases[key]["List"]),
                    "paraphrase": paraphrase,
                    "model": model_suffix,
                }
            )

    # Generate paraphrases for ETPC dataset
    input_token_len: int = tokenized_etpc_data["input_ids"].shape[-1]
    etpc_paraphrases: List[str] = generate_paraphrases(
        model, tokenizer, tokenized_etpc_data, input_token_len, batch_size=batch_size
    )
    for original_sentence, paraphrase, reference in zip(
        tokenized_etpc_data["original_sentences"], etpc_paraphrases, etpc_references
    ):

        # Extract the part between "Sentence: " and " Paraphrase Types:"
        start: int = original_sentence.find("Sentence: ") + len("Sentence: ")
        end: int = original_sentence.find(" Paraphrase Types:")
        if start != -1 and end != -1:
            cleaned_sentence: str = original_sentence[start:end].strip()
        else:
            cleaned_sentence: str = original_sentence

        # Extract the part between "Paraphrase Types:" and " Generated Paraphrase:"
        apt_start: int = original_sentence.find("Paraphrase Types:") + len(
            "Paraphrase Types:"
        )
        apt_end: int = original_sentence.find(" Generated Paraphrase:")
        if apt_start != -1 and apt_end != -1:
            apt: str = original_sentence[apt_start:apt_end].strip()
        else:
            apt: str = ""

        evaluation: Dict[str, float] = evaluate_individual_paraphrases(
            [paraphrase], [reference]
        )[0]
        if cleaned_sentence not in grouped_paraphrases:
            grouped_paraphrases[cleaned_sentence] = {
                "id": hash(cleaned_sentence) % 1000,
                "original": cleaned_sentence,
                "APT": apt,
                "reference": reference,
                "dataset": "ETPC",
                "List": [],
            }
        grouped_paraphrases[cleaned_sentence]["List"].append(
            {
                "id": len(grouped_paraphrases[cleaned_sentence]["List"]),
                "paraphrase": paraphrase,
                "evaluation": evaluation,
                "model": model_suffix,
            }
        )
    return grouped_paraphrases


def generate_and_evaluate(
    tokenizer: PreTrainedTokenizerBase,
    models: List[Tuple[str, str, str]],
    tokenized_apty_data: Dict[str, List[Dict[str, Any]]],
    tokenized_etpc_data: Dict[str, List[Dict[str, Any]]],
    etpc_references: List[str],
    output_csv: str,
    output_json: str,
    batch_size: int,
) -> None:
    """
    Generate paraphrases for each model and adapter, evaluate them, and save the results to a CSV file.

    Args:
        tokenizer (PreTrainedTokenizerBase): The tokenizer to use.
        models (List[Tuple[str, str, str]]): List of tuples containing model name, adapter directory, and model type.
        tokenized_apty_data (Dict[str, List[Dict[str, Any]]]): Tokenized APTY data.
        tokenized_etpc_data (Dict[str, List[Dict[str, Any]]]): Tokenized ETPC data.
        etpc_references (List[str]): List of references for ETPC.
        output_csv (str): Output CSV file path.
        output_json (str): Output JSON file path.
        batch_size (int): Batch size to use for evaluation.

    Returns:
        None
    """
    metrics: List[Dict[str, float]] = []
    all_paraphrases: Dict[str, Dict[str, Any]] = {}

    # Configure the BitsAndBytes config for loading models in 4-bit precision
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,  # Load in 4-bit precision
        bnb_4bit_compute_dtype=torch.bfloat16,  # Use bfloat16 for computations
    )

    # Loop over each model and adapter
    for model_name, adapter_dir, model_suffix in models:
        logging.info(f"Loading model {model_suffix} from {model_name}")

        # Load the model and adapter
        if adapter_dir:
            if os.path.exists(
                os.path.join(adapter_dir, "model.safetensors.index.json")
            ):
                logging.info(f"Loading merged model {model_suffix} from {adapter_dir}")
                model = AutoModelForCausalLM.from_pretrained(
                    adapter_dir,
                    torch_dtype=torch.bfloat16,  # Load in bfloat16 precision
                    low_cpu_mem_usage=True,
                    attn_implementation="flash_attention_2",
                )
                model.to('cuda')
            else:
                logging.info(
                    f"Loading base model and adding adapter {model_suffix} from {adapter_dir}"
                )
                model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    quantization_config=bnb_config,
                    torch_dtype=torch.bfloat16,  # Load in bfloat16 precision
                    low_cpu_mem_usage=True,
                    attn_implementation="flash_attention_2",
                )
                model.load_adapter(adapter_dir, adapter_name=model_suffix)
                model.set_adapter(model_suffix)
        else:
            logging.info(f"Loading model {model_suffix} from {model_name}")
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=bnb_config,
                torch_dtype=torch.bfloat16,  # Load in bfloat16 precision
                low_cpu_mem_usage=True,
                attn_implementation="flash_attention_2",
            )
       
        # Resize the model's embeddings to handle new tokens
        model.resize_token_embeddings(len(tokenizer))
        model.config.pad_token_id = tokenizer.pad_token_id or model.config.eos_token_id

        # Put the model in evaluation mode
        model.eval()

        # Generate paraphrases including evaluation metrics
        paraphrases: Dict[str, Dict[str, Any]] = process_model_generation(
            model,
            tokenizer,
            tokenized_apty_data,
            tokenized_etpc_data,
            etpc_references,
            model_suffix,
            batch_size,
        )

        # Merge paraphrases into all_paraphrases dictionary
        for original_sentence, details in paraphrases.items():
            if original_sentence not in all_paraphrases:
                all_paraphrases[original_sentence] = details
            else:
                all_paraphrases[original_sentence]["List"].extend(details["List"])

        # Collect evaluation metrics for this model
        scores: List[Dict[str, float]] = [
            entry["evaluation"]
            for details in paraphrases.values()
            for entry in details["List"]
            if "evaluation" in entry
        ]
        if scores:
            avg_rouge_1 = sum(score["rouge-1"] for score in scores) / len(scores)
            avg_rouge_2 = sum(score["rouge-2"] for score in scores) / len(scores)
            avg_rouge_l = sum(score["rouge-l"] for score in scores) / len(scores)
            avg_bleu = sum(score["bleu"] for score in scores) / len(scores)
        else:
            avg_rouge_1 = avg_rouge_2 = avg_rouge_l = avg_bleu = 0

        metrics.append(
            {
                "Model": model_name,
                "Adapter": model_suffix,
                "ROUGE-1": round(avg_rouge_1, 4),
                "ROUGE-2": round(avg_rouge_2, 4),
                "ROUGE-L": round(avg_rouge_l, 4),
                "BLEU": round(avg_bleu, 4),
            }
        )

        # Clean up memory
        del model
        torch.cuda.empty_cache()

    # Save the paraphrases to a JSON file
    json_paraphrases = [
        {
            "id": details["id"],
            "original": details["original"],
            "APT": details["APT"],
            "reference": details.get("reference"),
            "dataset": details["dataset"],
            "List": details["List"],
        }
        for details in all_paraphrases.values()
    ]

    save_paraphrases_to_json(json_paraphrases, output_json)

    # Save the evaluation metrics to a CSV file
    save_metrics_to_csv(metrics, output_csv)


def main() -> None:
    """Main function to evaluate the generation capabilities of the models.

    Args:
        None

    Returns:
        None
    """
    args = parse_arguments()
    batch_size: int = 10
    num_examples: int = 10

    output_csv: str = f"out/gen-models/eval_{args.model_name.split('/')[-1]}.csv"
    output_json: str = (
        f"out/gen-models/generated_paraphrases_{args.model_name.split('/')[-1]}.json"
    )

    # Read sentences by type from the APTY dataset
    apty_data: Dict[str, Dict[str, Any]] = read_sentences_by_type(
        "out/basesentences", num_examples=num_examples
    )

    # Load the ETPC dataset and filter out sentences without paraphrases
    etpc_data = (
        load_dataset("jpwahle/etpc", split="train")
        .filter(lambda x: x["etpc_label"] == 1)
        .shuffle(seed=42)
        .select(range(num_examples))
    )

    # Create prompts and references from the test set
    etpc_prompts: List[str]
    etpc_references: List[str]
    etpc_prompts, etpc_references = create_etpc_prompts(etpc_data)

    tokenizer = load_tokenizer(args.model_name)

    # Pre-tokenize APTY data
    tokenized_apty_data: Dict[str, Dict[str, Any]] = {
        paraphrase_type: {
            **tokenize_data(
                tokenizer,
                [
                    "Instruction: Given the following sentence, generate a paraphrase with the following type. "
                    f"Sentence: {sentence} Paraphrase Type: {paraphrase_type}. "
                    "Generated Paraphrase: "
                    for sentence in sentences
                ],
            ),
            "original_sentences": sentences,  # Ensure this matches the number of input_ids
        }
        for paraphrase_type, details in apty_data.items()
        for sentences in [
            details["sentences"][:num_examples]
        ]  # Limit to `num_examples`
    }

    # Pre-tokenize ETPC data
    tokenized_etpc_data: Dict[str, Any] = tokenize_data(tokenizer, etpc_prompts)
    tokenized_etpc_data["original_sentences"] = etpc_prompts

    models: List[Tuple[str, Optional[str], str]] = [
        (args.model_name, None, "base_model"),
        (args.model_name, args.etpc_dir, "etpc_model"),
        (args.model_name, args.dpo_dir, "dpo_model"),  # DPO adapter
        (args.model_name, args.ipo_dir, "ipo_model"),  # IPO adapter
    ]

    # Evaluate the models on the APTY and ETPC datasets
    generate_and_evaluate(
        tokenizer,
        models,
        tokenized_apty_data,
        tokenized_etpc_data,
        etpc_references,
        output_csv,
        output_json,
        batch_size,
    )


if __name__ == "__main__":
    main()

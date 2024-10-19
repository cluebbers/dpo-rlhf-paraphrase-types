import argparse
import os
import json
import logging
from typing import Dict, List, Any
import torch

from tqdm import tqdm
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from rouge import Rouge
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, PreTrainedModel, PreTrainedTokenizerBase
import pandas as pd
from datasets import load_dataset

# Initialize Hugging Face Hub login (if needed)
from huggingface_hub import login

with open("token_file.txt", "r") as token_file:
    hf_token = token_file.read().strip()
login(token=hf_token)

# Logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def parse_arguments():
    parser = argparse.ArgumentParser(description="Generate paraphrases and evaluate models.")
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.1-8B", help="Base model path.")
    parser.add_argument("--etpc_dir", type=str, default="out/gen-models/llama-3.1-8b-etpc", help="ETPC adapter directory.")
    parser.add_argument("--dpo_dir", type=str, default="out/gen-models/dpo_out-gen-models-llama-3.1-8b-etpc_sigmoid", help="DPO adapter directory.")
    parser.add_argument("--ipo_dir", type=str, default="out/gen-models/dpo_out-gen-models-llama-3.1-8b-etpc_ipo", help="DPO adapter directory.")
    
    args = parser.parse_args()

    # Convert the string "None" to actual None
    if args.etpc_dir == "None":
        args.etpc_dir = None
    if args.dpo_dir == "None":
        args.dpo_dir = None
    if args.ipo_dir == "None":
        args.ipo_dir = None

    return args

def create_etpc_prompts(data):
    prompts = []
    references = []
    for instance in data:
        if not instance["paraphrase_types"]:
            continue

        prompt = (
            "Given the following sentence, generate a"
            " paraphrase with the following types. Sentence:"
            f" {instance['sentence1']} Paraphrase Types:"
            f" {', '.join(instance['paraphrase_types'])}. Generated Paraphrase: "
        )
        prompts.append(prompt)
        references.append(instance["sentence2"])

    return prompts, references

def read_sentences_by_type(data_dir: str) -> dict[str, list[str]]:

    sentences_by_type = {}

    for file_name in os.listdir(data_dir):
        if file_name.endswith(".txt"):
            with open(os.path.join(data_dir, file_name), "r", encoding="utf-8-sig") as file:
                paraphrase_type = next(file).strip()
                sentences = [line.strip() for line in file if line.strip()]
                sentences_by_type[paraphrase_type] = sentences[:10]

    return sentences_by_type

def load_tokenizer(model_name: str) -> tuple[PreTrainedTokenizerBase, PreTrainedModel]:
    tokenizer = AutoTokenizer.from_pretrained(model_name, 
                                              padding_side="left", 
                                              use_fast=True)

    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "<pad>"})
        
    return tokenizer

def tokenize_data(tokenizer, prompts: List[str]) -> Dict[str, torch.Tensor]:

    return tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
    )
    
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
    """Generate paraphrases using the provided model and tokenizer."""
    paraphrases = []
    device = model.device
    input_ids = inputs["input_ids"].to(device)
    attention_mask = inputs["attention_mask"].to(device)
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
                do_sample=True,
                pad_token_id=tokenizer.pad_token_id,
            )
        for output in outputs:
            generated_tokens = output[input_token_length:]
            paraphrase = tokenizer.decode(generated_tokens, skip_special_tokens=True).strip()
            paraphrases.append(paraphrase)
        progress_bar.update(1)
    progress_bar.close()
    return paraphrases


def save_metrics_to_csv(metrics: List[Dict[str, float]], output_csv: str) -> None:
    """
    Save evaluation metrics to a CSV file.

    Args:
        metrics (List[Dict[str, float]]): List of dictionaries containing evaluation metrics.
        output_csv (str): Path to the output CSV file.
    """
    fieldnames = ["Model", "Adapter", "ROUGE-1", "ROUGE-2", "ROUGE-L", "BLEU"]

    # Write the metrics to a CSV file
    pd.DataFrame(metrics, columns=fieldnames).to_csv(output_csv, index=False)

    logging.info(f"Evaluation metrics saved to {output_csv}")

def save_paraphrases_to_json(paraphrases_list, output_file_path):
    """
    Save the generated paraphrases to a JSON file in a human-readable format.

    Args:
        paraphrases_list (list): List of dictionaries containing the generated paraphrases.
        output_file_path (str): Path to the output JSON file.
    """
    with open(output_file_path, 'w', encoding='utf-8') as output_file:
        json.dump(paraphrases_list, output_file, ensure_ascii=False, indent=4)

def evaluate_individual_paraphrases(paraphrases: List[str], references: List[str]) -> List[Dict[str, float]]:
    """
    Evaluate paraphrases using ROUGE and BLEU scores, returning individual scores per sentence pair.

    Args:
        paraphrases (List[str]): The generated paraphrases.
        references (List[str]): The reference texts.

    Returns:
        List[Dict[str, float]]: A list containing the evaluation scores for each paraphrase-reference pair.
    """
    rouge = Rouge()
    scores = []

    for paraphrase, reference in zip(paraphrases, references):
        try:
            rouge_score = rouge.get_scores(paraphrase, reference)[0]
            smoothie = SmoothingFunction().method4
            bleu_score = sentence_bleu([reference.split()], paraphrase.split(), smoothing_function=smoothie)
            scores.append({
                "rouge-1": rouge_score["rouge-1"]["f"],
                "rouge-2": rouge_score["rouge-2"]["f"],
                "rouge-l": rouge_score["rouge-l"]["f"],
                "bleu": bleu_score
            })
        except ValueError:
            scores.append({
                "rouge-1": 0.0,
                "rouge-2": 0.0,
                "rouge-l": 0.0,
                "bleu": 0.0
            })

    return scores
 
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
        
        # Extract the part between "Sentence: " and " Paraphrase Types:"
        start = original_sentence.find("Sentence: ") + len("Sentence: ")
        end = original_sentence.find(" Paraphrase Types:")
        if start != -1 and end != -1:
            cleaned_sentence = original_sentence[start:end].strip()
        else:
            cleaned_sentence = original_sentence
            
        # Extract the part between "Paraphrase Types:" and " Generated Paraphrase:"
        apt_start = original_sentence.find("Paraphrase Types:") + len("Paraphrase Types:")
        apt_end = original_sentence.find(" Generated Paraphrase:")
        if apt_start != -1 and apt_end != -1:
            apt = original_sentence[apt_start:apt_end].strip()
        else:
            apt = ""
        
        evaluation = evaluate_individual_paraphrases([paraphrase], [reference])[0]
        if cleaned_sentence not in grouped_paraphrases:
            grouped_paraphrases[cleaned_sentence] = {
                "id": hash(cleaned_sentence) % 1000,
                "original": cleaned_sentence,
                "APT": apt,
                "reference": reference,
                "dataset": "ETPC",
                "List": []
            }
        grouped_paraphrases[cleaned_sentence]["List"].append({
            "id": len(grouped_paraphrases[cleaned_sentence]["List"]),
            "paraphrase": paraphrase,
            "evaluation": evaluation,
            "model": model_suffix
        })
    return grouped_paraphrases

def generate_and_evaluate(
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
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,  # Load in 4-bit precision
        bnb_4bit_compute_dtype=torch.bfloat16,  # Use bfloat16 for computations
    )

    for model_name, adapter_dir, model_suffix in models:
        
        if adapter_dir:
            if any(file.endswith(".safetensors") for file in os.listdir(adapter_dir)):
                logging.info(f"Loading merged model {model_suffix} from {adapter_dir}")                
                model = AutoModelForCausalLM.from_pretrained(adapter_dir, 
                                                             torch_dtype=torch.bfloat16, 
                                                             low_cpu_mem_usage=True,
                                                             quantization_config=bnb_config)

            else:
                logging.info(f"Loading base model and adding adapter {model_suffix} from {adapter_dir}")
                model = AutoModelForCausalLM.from_pretrained(
                        model_name,
                        quantization_config=bnb_config,
                        torch_dtype=torch.bfloat16,  # Load in bfloat16 precision
                        low_cpu_mem_usage=True,
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
            )
                
        model.resize_token_embeddings(len(tokenizer))
        model.config.pad_token_id = tokenizer.pad_token_id or model.config.eos_token_id
        
        model.eval()

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
        
        del model
        torch.cuda.empty_cache()
    
    json_paraphrases = [
        {
            "id": details["id"],
            "original": details["original"],
            "APT": details["APT"],
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
    
    apty_data = read_sentences_by_type("out/basesentences")    
    etpc_data = (
        load_dataset("jpwahle/etpc", split="train")
        .filter(lambda x: x["etpc_label"] == 1)
        .shuffle(seed=42)
        .select(range(num_examples)) 
    )

    # Create prompts and references from the test set
    etpc_prompts, etpc_references = create_etpc_prompts(etpc_data)

    tokenizer = load_tokenizer(args.model_name)
    
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
       # (args.model_name, None, "base_model"),
        (args.model_name, args.etpc_dir, "etpc_model"),
        #(args.model_name, args.dpo_dir, "dpo_model"),    # DPO adapter 
        #(args.model_name, args.ipo_dir, "ipo_model"),    # IPO adapter  
    ]

    generate_and_evaluate(tokenizer, models, tokenized_apty_data, tokenized_etpc_data, 
        etpc_references, output_csv, output_json, batch_size
    )
if __name__ == "__main__":
    main()


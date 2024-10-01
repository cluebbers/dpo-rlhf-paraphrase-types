'''
python3 src/eval_dpo_gen.py \
    --model_name=facebook/bart-large \
    --etpc_dir=out/gen-models/facebook/bart-large_paraphrase-type-generation \
    --dpo_dir=out/gen-models/facebook/bart-large_paraphrase-type-generation_sigmoid \
    --ipo_dir=out/gen-models/facebook/bart-large_paraphrase-type-generation_ipo
'''
import os
import torch
import argparse

import pandas as pd

from tqdm import tqdm
from datasets import load_dataset
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from rouge import Rouge
from transformers import AutoTokenizer, BartForConditionalGeneration
from parascore import ParaScorer 

class ParaphraseDataset(torch.utils.data.Dataset):
    """A dataset class for paraphrase generation.

    Args:
        data (list): The dataset.
        tokenizer (Tokenizer): The tokenizer to use.

    Example:
        ```python
        dataset = ParaphraseDataset(data, tokenizer)
        ```
    """

    def __init__(self, data, tokenizer):
        self.data = data
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data[idx]
        sentence1 = row["question1"]
        sentence2 = row["question2"]

        # Tokenize sentence1 for input_ids
        input_data = self.tokenizer(
            sentence1,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=512,
        )

        # Tokenize sentence2 for labels
        label_data = self.tokenizer(
            sentence2,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=512,
        )

        return {
            "input_ids": input_data["input_ids"].squeeze(0),
            "attention_mask": input_data["attention_mask"].squeeze(0),
            "labels": label_data["input_ids"].squeeze(0),
        }

class ParaphraseTypeDataset(torch.utils.data.Dataset):
    """A dataset class for paraphrase type generation.

    Args:
        data (list): The dataset.
        tokenizer (Tokenizer): The tokenizer to use.

    Example:
        ```python
        dataset = ParaphraseTypeDataset(data, tokenizer)
        ```
    """

    def __init__(self, data, tokenizer):
        self.data = data
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data[idx]
        sentence1 = self.add_paraphrase_type_tags(
            row["sentence1_tokenized"],
            row["sentence1_segment_location_indices"],
            row["paraphrase_type_ids"],
        )
        sentence2 = row["sentence2"]

        # Tokenize sentence1 for input_ids
        input_data = self.tokenizer(
            sentence1,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=512,
        )

        # Tokenize sentence2 for labels
        label_data = self.tokenizer(
            sentence2,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=512,
        )

        return {
            "input_ids": input_data["input_ids"].squeeze(0),
            "attention_mask": input_data["attention_mask"].squeeze(0),
            "labels": label_data["input_ids"].squeeze(0),
        }
        
    def add_paraphrase_type_tags(self, sentence, segment_indices, type_ids):
        """Adds paraphrase type tags to specific indices in a sentence.

        Args:
            sentence (list): The input sentence as a list of tokens.
            segment_indices (list): The indices of the segments to tag.
            type_ids (list): The corresponding type IDs for each segment.

        Returns:
            str: The modified sentence with paraphrase type tags added.

        Example:
            ```python
            sentence = ["This", "is", "a", "sentence", "."]
            segment_indices = [[0, 3]]
            type_ids = [1]
            modified_sentence = add_paraphrase_type_tags(sentence, segment_indices, type_ids)
            print(modified_sentence)
            ```
        """
        for indices, type_id in zip(segment_indices, type_ids):
            for index in indices:
                sentence[index] = f"<type-{type_id}>{sentence[index]}"
        return " ".join(sentence)
    
def evaluate_on_datasets(model, tokenizer, eval_datasets, parascore_scorer):
    """
    Evaluate the model on different datasets.

    Args:
        model: The trained model.
        tokenizer: The tokenizer used with the model.
        eval_datasets (dict): Dictionary of datasets to evaluate on.

    Returns:
        dict: Dictionary containing evaluation metrics for each dataset.
    """
    results = {}
    for dataset_name, dataset in eval_datasets.items():
        val_loader = torch.utils.data.DataLoader(dataset, batch_size=8)
        metrics = eval_loop(val_loader, model, tokenizer, parascore_scorer)
        results[dataset_name] = metrics
    return results

def eval_loop(data_loader, model, tokenizer, parascore_scorer, max_new_tokens=50):
    """Performs evaluation on a given data loader using a pre-trained model and tokenizer."""
    model.eval()
    avg_scores = {"bleu": [], "rouge-1": [], "rouge-2": [], "rouge-l": [], "free_score": [], "base_score": []}

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating"):
            inputs = batch["input_ids"].to(model.device)
            attention_masks = batch["attention_mask"].to(model.device)
            outputs = model.generate(inputs, attention_mask=attention_masks, max_new_tokens=max_new_tokens)

            pred_texts = [
                tokenizer.decode(output, skip_special_tokens=True, clean_up_tokenization_spaces=True) 
                for output in outputs
            ]
            target_texts = [
                tokenizer.decode(target, skip_special_tokens=True, clean_up_tokenization_spaces=True)
                for target in batch["labels"]
            ]
            source_texts = [
                tokenizer.decode(inputs[i], skip_special_tokens=True) 
                for i in range(len(inputs))
            ] 

            scores = evaluate(pred_texts, target_texts)
            for key, value in scores.items():
                avg_scores[key].append(value)

            # Perform Parascore evaluation
            parascore_results = parascore_evaluation(pred_texts, source_texts, target_texts, parascore_scorer, batch_size=len(pred_texts))
            for key, value in parascore_results.items():
                if isinstance(value, torch.Tensor):
                    # Calculate the mean of the tensor
                    avg_scores[key].append(value.mean().item())  # Get the scalar value
                else:
                    avg_scores[key].append(value)

    # Compute averages
    for key, value in avg_scores.items():
        avg_scores[key] = sum(value) / len(value) if value else 0.0  # Ensure no division by zero

    return avg_scores

def parascore_evaluation(paraphrases: list, sources: list, references: list, parascore_scorer, batch_size=64):
    """Evaluate generated paraphrases using Parascore."""
    
    # Ensure inputs are lists
    paraphrase_list = [paraphrases[0]]  # Single candidate wrapped in a list
    source_list = [sources[0]]           # Single source wrapped in a list
    reference_list = [references[0]]     # Single reference wrapped in a list

    free_scores = parascore_scorer.free_score(cands=paraphrase_list, sources=source_list, batch_size=batch_size)
    base_scores = parascore_scorer.base_score(cands=paraphrase_list, sources=source_list, refs=reference_list, batch_size=batch_size)

    # Extract the first element and convert tensor to float for free score
    avg_free_score = free_scores[0].item() if isinstance(free_scores[0], torch.Tensor) else float(free_scores[0])
    avg_base_score = base_scores[0] if isinstance(base_scores, list) else float(base_scores)
    
    return {
        "free_score": avg_free_score,  
        "base_score": avg_base_score,   
    }

def calculate_bleu(reference, candidate):
    """
    Calculates the BLEU score between a reference sentence and a candidate sentence.

    Args:
        reference (list): The reference sentences.
        candidate (str): The candidate sentence.

    Returns:
        float: The BLEU score."""

    smoothing = (
        SmoothingFunction().method1
    )  # Using SmoothingFunction's method1 for avoiding division by zero
    return sentence_bleu([reference], candidate, smoothing_function=smoothing)

def evaluate(predictions, targets):
    """
    Evaluates the predictions against the target references using BLEU and ROUGE scores.

    Args:
        predictions (list): The predicted sentences.
        targets (list): The target reference sentences.

    Returns:
        dict: A dictionary containing the BLEU score, ROUGE-1 score, ROUGE-2 score,
        and ROUGE-L score.

    Example:
        ```python
        predictions = ["This is a predicted sentence."]
        targets = ["This is a target sentence."]

        evaluation_results = evaluate(predictions, targets)
        print(evaluation_results)
        ```"""

    bleu_score = 0.0
    rouge_scores = {
        "rouge-1": {"f": 0.0},
        "rouge-2": {"f": 0.0},
        "rouge-l": {"f": 0.0},
    }

    if predictions and len(predictions) > 0:
        # BLEU Score
        for taget, prediction in zip(targets, predictions):
            b_sc = calculate_bleu(taget, prediction)
            bleu_score += b_sc
        bleu_score /= len(predictions)

        # ROUGE Scores
        rouge_calculator = Rouge()
        rouge_scores = rouge_calculator.get_scores(predictions, targets, avg=True)

    return {
        "bleu": bleu_score,
        "rouge-1": rouge_scores["rouge-1"]["f"],
        "rouge-2": rouge_scores["rouge-2"]["f"],
        "rouge-l": rouge_scores["rouge-l"]["f"],
    }

def parse_args():
    """
    Parses the command line arguments and returns the parsed arguments.

    Returns:
        argparse.Namespace: The parsed command line arguments.

    Example:
        ```python
        args = parse_args()
        print(args.model_name)
        ```"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="facebook/bart-large")
    parser.add_argument("--etpc_dir", type=str, default="out/gen-models/facebook/bart-large_paraphrase-type-generation", help="ETPC adapter directory.")
    parser.add_argument("--dpo_dir", type=str, default="out/gen-models/facebook/bart-large_paraphrase-type-generation_sigmoid", help="DPO adapter directory.")
    parser.add_argument("--ipo_dir", type=str, default="out/gen-models/facebook/bart-large_paraphrase-type-generation_ipo", help="DPO adapter directory.")
    return parser.parse_args()

def save_metrics_to_csv(metrics, output_csv):
    """
    Save evaluation metrics to a CSV file.

    Args:
        metrics (list): List of dictionaries containing evaluation metrics.
        output_csv (str): Path to the output CSV file.
    """
    fieldnames = ["Model", "Adapter", "dataset_name", "rouge-1", "rouge-2", "rouge-l", "bleu", "free_score", "base_score"]

    # Use pandas to write the CSV file
    pd.DataFrame(metrics).to_csv(output_csv, index=False, header=fieldnames)
    
def evaluate_models(models, eval_datasets, output_csv):
    """
    Loop over models and adapters, perform evaluation, and save metrics to CSV.
    
    Args:
        models (list): List of tuples containing model name, adapter directory, and model type.
        eval_datasets (dict): Dictionary of datasets to evaluate on.
        output_csv (str): Output CSV file path.
    """
    metrics = []

    for model_name, adapter_dir, model_type in models:
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Find the checkpoint directory if adapter_dir is provided
        if adapter_dir:
            checkpoint_dir = find_checkpoint_dir(adapter_dir)
            if checkpoint_dir:
                model = BartForConditionalGeneration.from_pretrained(checkpoint_dir)
            else:
                model = BartForConditionalGeneration.from_pretrained(model_name)
        else:
            # Load base model if no adapter directory
            model = BartForConditionalGeneration.from_pretrained(model_name)
        
        model.eval()
        model.to("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize ParaScorer
        parascore_scorer = ParaScorer(model_type=model_name, lang='en')

        # Evaluate on all datasets (ETPC, QQP, etc.)
        for dataset_name, dataset in eval_datasets.items():
            # Evaluate on the specific dataset
            eval_results = evaluate_on_datasets(model, tokenizer, {dataset_name: dataset}, parascore_scorer)
            
            # Collect results with additional model type info and dataset name
            for ds_name, scores in eval_results.items():
                metrics_dict = {
                    "Model": model_name,
                    "Adapter": adapter_dir or 'base',
                    "dataset_name": ds_name,  # Add the dataset name to the dictionary
                    "rouge-1": scores.get("rouge-1", 0.0),
                    "rouge-2": scores.get("rouge-2", 0.0),
                    "rouge-l": scores.get("rouge-l", 0.0),
                    "bleu": scores.get("bleu", 0.0),
                    "free_score": scores.get("free_score", 0.0),
                    "base_score": scores.get("base_score", 0.0),
                }
                metrics.append(metrics_dict)
    
    # Save all metrics to CSV
    save_metrics_to_csv(metrics, output_csv)
    
def find_checkpoint_dir(adapter_dir):
    """
    Find the latest checkpoint directory within the given adapter directory.
    
    Args:
        adapter_dir (str): The base adapter directory where checkpoints are stored.
        
    Returns:
        str: Path to the latest checkpoint directory or None if no checkpoints are found.
    """
    checkpoint_dir = None
    if os.path.exists(adapter_dir) and os.listdir(adapter_dir):
        checkpoint_dirs = [f for f in os.listdir(adapter_dir) if f.startswith('checkpoint')]
        if checkpoint_dirs:
            # Get the latest checkpoint based on the creation time
            checkpoint_dir = os.path.join(adapter_dir, max(checkpoint_dirs, key=lambda x: os.path.getctime(os.path.join(adapter_dir, x))))
            print(f"Loading from fine-tuned checkpoint: {checkpoint_dir}")
        else:
            print("No checkpoint found in adapter directory, loading base model instead.")
    else:
        print(f"Adapter directory does not exist or is empty: {adapter_dir}. Loading base model instead.")
    
    return checkpoint_dir

def main():    
    args = parse_args()
    
    # Define output files
    output_csv = f"out/eval_{args.model_name.split('/')[-1]}.csv"
    output_json = f"out/generated_paraphrases_{args.model_name.split('/')[-1]}.json"   
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    
    # Prepare evaluation datasets (ETPC and QQP)
    etpc_dataset = load_dataset("jpwahle/etpc").filter(lambda x: x["etpc_label"] == 1)
    etpc_dataset = etpc_dataset["train"].train_test_split(test_size=0.2)  # Split for evaluation
    qqp_dataset = load_dataset("glue", "qqp")["validation"]
    
    eval_datasets = {
        "ETPC": ParaphraseTypeDataset(etpc_dataset["test"], tokenizer),  # Use test split for evaluation
        "QQP": ParaphraseDataset(qqp_dataset, tokenizer),
        #TODO APTY
    }  
    
    # List of models to evaluate: base model and adapters
    models = [
        (args.model_name, None, "base_model"),           # Base model (no adapter)
        (args.model_name, args.etpc_dir, "etpc_model"),  # ETPC adapter
        (args.model_name, args.dpo_dir, "dpo_model"),    # DPO adapter 
        (args.model_name, args.ipo_dir, "ipo_model"),    # IPO adapter       
    ]
    
    # Run the evaluation
    evaluate_models(models, eval_datasets, output_csv)
    
    #TODO save paraphrases to JSON

if __name__ == "__main__":
    main()
import os
import torch
import argparse
from tqdm import tqdm


from datasets import load_dataset
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from rouge import Rouge
from transformers import (
    AutoTokenizer,
    BartForConditionalGeneration,
    PegasusForConditionalGeneration,
    )

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
    parser.add_argument(
        "--task_name",
        type=str,
        default="paraphrase-type-generation",
        help="Name of the task to use",
    )

    return parser.parse_args()

def main():    
    args = parse_args()
    
    # Check if CUDA is available
    if torch.cuda.is_available():
        device = "cuda"
        torch.cuda.empty_cache()  # Clear GPU cache before starting    
    
    # Find the latest checkpoint from the fine-tuned model directory
    fine_tuned_model_dir = f"./out/gen-models/{args.model_name}_paraphrase-type-generation"  # Path to the fine-tuned model
    
    checkpoint_dir = None
    if os.path.exists(fine_tuned_model_dir) and os.listdir(fine_tuned_model_dir):
        checkpoint_dirs = [f for f in os.listdir(fine_tuned_model_dir) if f.startswith('checkpoint')]
        if checkpoint_dirs:
            checkpoint_dir = os.path.join(fine_tuned_model_dir, max(checkpoint_dirs, key=lambda x: os.path.getctime(os.path.join(fine_tuned_model_dir, x))))
            print(f"Loading from fine-tuned checkpoint: {checkpoint_dir}")
        else:
            print("No checkpoint found in fine-tuned model directory, loading base model instead.")
    else:
        print(f"Fine-tuned model directory does not exist or is empty: {fine_tuned_model_dir}. Loading base model instead.")
    
    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large")
    model = (
        BartForConditionalGeneration.from_pretrained(checkpoint_dir) if checkpoint_dir else BartForConditionalGeneration.from_pretrained("facebook/bart-large")
        if "bart" in args.model_name 
        else PegasusForConditionalGeneration.from_pretrained(checkpoint_dir) if checkpoint_dir else PegasusForConditionalGeneration.from_pretrained("google/pegasus-large")
    )
    model = model.to(device)  # Move model to GPU
    
    # Prepare evaluation datasets (ETPC and QQP)
    etpc_dataset = load_dataset("jpwahle/etpc").filter(lambda x: x["etpc_label"] == 1)
    etpc_dataset = etpc_dataset["train"].train_test_split(test_size=0.2)  # Split for evaluation
    eval_datasets = {
        "ETPC": ParaphraseTypeDataset(etpc_dataset["test"], tokenizer),  # Use test split for evaluation
        #"QQP": ParaphraseDataset(load_dataset("glue", "qqp")["validation"], tokenizer),
    }

    # Evaluate the model on different datasets
    # Initialize ParaScorer
    parascore_scorer = ParaScorer(model_type=args.model_name, lang='en')
    eval_results = evaluate_on_datasets(model, tokenizer, eval_datasets, parascore_scorer)

    # Print evaluation results
    print("#" * 20)
    print(f"Model: {args.model_name}")
    print(f"Task: {args.task_name}")
    for dataset_name, metrics in eval_results.items():
        print(f"{dataset_name} evaluation results:", metrics)            
    print("#" * 20)

if __name__ == "__main__":
    main()
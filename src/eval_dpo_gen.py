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
import json

class ParaphraseDataset(torch.utils.data.Dataset):
    """A dataset class for paraphrase generation."""

    def __init__(self, data, tokenizer):
        self.data = data
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data[idx]
        sentence1 = row["question1"]
        sentence2 = row["question2"]

        input_data = self.tokenizer(
            sentence1,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=512,
        )

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
    """A dataset class for paraphrase type generation."""

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

        input_data = self.tokenizer(
            sentence1,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
            max_length=512,
        )

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
        for indices, type_id in zip(segment_indices, type_ids):
            for index in indices:
                sentence[index] = f"<type-{type_id}>{sentence[index]}"
        return " ".join(sentence)

def eval_loop(data_loader, model, tokenizer, parascore_scorer, max_new_tokens=50):
    """
    Performs evaluation on a given data loader using a pre-trained model and tokenizer.

    Returns:
        list: A list of dictionaries containing the paraphrase, source, target, and associated metrics.
    """
    model.eval()
    paraphrases = []  # Store paraphrases for JSON output

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating"):
            inputs = batch["input_ids"].to(model.device)
            attention_masks = batch["attention_mask"].to(model.device)

            outputs = model.generate(inputs, attention_mask=attention_masks, max_new_tokens=max_new_tokens)

            pred_texts = [
                tokenizer.decode(output, skip_special_tokens=True, clean_up_tokenization_spaces=False)
                for output in outputs
            ]
            target_texts = [
                tokenizer.decode(target, skip_special_tokens=True, clean_up_tokenization_spaces=False)
                for target in batch["labels"]
            ]
            source_texts = [
                tokenizer.decode(inputs[i], skip_special_tokens=True,clean_up_tokenization_spaces=False)
                for i in range(len(inputs))
            ]

            # Evaluate each paraphrase with its corresponding source and target
            for source, target, pred in zip(source_texts, target_texts, pred_texts):
                metrics = evaluate_single(pred, target)
                parascore_metrics = parascore_evaluation_single(pred, source, target, parascore_scorer)

                combined_metrics = {**metrics, **parascore_metrics}

                paraphrases.append({
                    "source": source,
                    "target": target,
                    "paraphrase": pred,
                    "metrics": combined_metrics
                })

    return paraphrases  # Return the list of paraphrases with metrics

def parascore_evaluation_single(paraphrase, source, reference, parascore_scorer):
    """Evaluate generated paraphrase using Parascore."""
    free_scores = parascore_scorer.free_score(cands=[paraphrase], sources=[source], batch_size=1)
    base_scores = parascore_scorer.base_score(cands=[paraphrase], sources=[source], refs=[reference], batch_size=1)

    return {
        "free_score": free_scores[0].item(),
        "base_score": base_scores[0] if isinstance(base_scores, list) else float(base_scores)
    }

def evaluate_single(prediction, target):
    """Evaluates a single prediction against the target reference using BLEU and ROUGE scores."""
    rouge_calculator = Rouge()
    rouge_scores = rouge_calculator.get_scores([prediction], [target], avg=False)[0]

    smoothie = SmoothingFunction().method1
    bleu_score = sentence_bleu([target.split()], prediction.split(), smoothing_function=smoothie)

    return {
        "rouge-1": rouge_scores["rouge-1"]["f"],
        "rouge-2": rouge_scores["rouge-2"]["f"],
        "rouge-l": rouge_scores["rouge-l"]["f"],
        "bleu": bleu_score,
    }

def evaluate_on_datasets(model, tokenizer, eval_datasets, parascore_scorer):
    """
    Evaluate the model on different datasets.

    Args:
        model: The trained model.
        tokenizer: The tokenizer used with the model.
        eval_datasets (dict): Dictionary of datasets to evaluate on.

    Returns:
        dict: Dictionary containing evaluation metrics and paraphrases for each dataset.
    """
    results = {}
    for dataset_name, dataset in eval_datasets.items():
        val_loader = torch.utils.data.DataLoader(dataset, batch_size=32)
        paraphrases_with_metrics = eval_loop(val_loader, model, tokenizer, parascore_scorer)
        results[dataset_name] = paraphrases_with_metrics
    return results

def save_paraphrases_to_json(paraphrases, output_file):
    """
    Saves the generated paraphrases to a JSON file in a human-readable format.

    Args:
        paraphrases (list): List of dictionaries containing the generated paraphrases.
        output_file (str): Path to the output JSON file.
    """
    with open(output_file, 'w', encoding='utf-8') as file:
        json.dump(paraphrases, file, ensure_ascii=False, indent=4)
    print(f"Paraphrases saved to {output_file}")

def save_metrics_to_csv(metrics, output_csv):
    """
    Save evaluation metrics to a CSV file.

    Args:
        metrics (list): List of dictionaries containing evaluation metrics.
        output_csv (str): Path to the output CSV file.
    """
    fieldnames = ["Model", "Adapter", "dataset_name", "rouge-1", "rouge-2", "rouge-l", "bleu", "free_score", "base_score"]
    pd.DataFrame(metrics).to_csv(output_csv, index=False, header=fieldnames)
    print(f"Evaluation metrics saved to {output_csv}")

def evaluate_models(models, eval_datasets, tokenizer, output_csv, output_json):
    """
    Loop over models and adapters, perform evaluation, save metrics to CSV, and save paraphrases to JSON.

    Args:
        models (list): List of tuples containing model name, adapter directory, and model type.
        eval_datasets (dict): Dictionary of datasets to evaluate on.
        output_csv (str): Output CSV file path.
        output_json (str): Output JSON file path.
    """
    metrics = []
    all_paraphrases = []

    for model_name, adapter_dir, model_type in models:

        if adapter_dir:
            checkpoint_dir = find_checkpoint_dir(adapter_dir)
            if checkpoint_dir:
                model = BartForConditionalGeneration.from_pretrained(checkpoint_dir)
            else:
                model = BartForConditionalGeneration.from_pretrained(model_name)
        else:
            model = BartForConditionalGeneration.from_pretrained(model_name)
        
        model.resize_token_embeddings(len(tokenizer))  # Adjust model embeddings to handle new token

        model.eval()
        model.to("cuda" if torch.cuda.is_available() else "cpu")
        parascore_scorer = ParaScorer(model_type=model_name, lang='en')

        for dataset_name, dataset in eval_datasets.items():
            paraphrases_with_metrics = evaluate_on_datasets(model, tokenizer, {dataset_name: dataset}, parascore_scorer)
            all_paraphrases.extend(paraphrases_with_metrics[dataset_name])

            for entry in paraphrases_with_metrics[dataset_name]:
                metrics_dict = {
                    "Model": model_name,
                    "Adapter": adapter_dir or 'base',
                    "dataset_name": dataset_name,
                    **entry['metrics']
                }
                metrics.append(metrics_dict)

    save_metrics_to_csv(metrics, output_csv)
    save_paraphrases_to_json(all_paraphrases, output_json)

def find_checkpoint_dir(adapter_dir):
    checkpoint_dir = None
    if os.path.exists(adapter_dir) and os.listdir(adapter_dir):
        checkpoint_dirs = [f for f in os.listdir(adapter_dir) if f.startswith('checkpoint')]
        if checkpoint_dirs:
            checkpoint_dir = os.path.join(adapter_dir, max(checkpoint_dirs, key=lambda x: os.path.getctime(os.path.join(adapter_dir, x))))
            print(f"Loading from fine-tuned checkpoint: {checkpoint_dir}")
        else:
            print("No checkpoint found in adapter directory, loading base model instead.")
    else:
        print(f"Adapter directory does not exist or is empty: {adapter_dir}. Loading base model instead.")
    
    return checkpoint_dir

def parse_args():
    """
    Parses the command line arguments and returns the parsed arguments.

    Returns:
        argparse.Namespace: The parsed command line arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="facebook/bart-large", help="Base model name or path.")
    parser.add_argument("--etpc_dir", type=str, default="out/gen-models/facebook/bart-large_paraphrase-type-generation", help="ETPC adapter directory.")
    parser.add_argument("--dpo_dir", type=str, default="out/gen-models/facebook/bart-large_paraphrase-type-generation_sigmoid", help="DPO adapter directory.")
    parser.add_argument("--ipo_dir", type=str, default="out/gen-models/facebook/bart-large_paraphrase-type-generation_ipo", help="IPO adapter directory.")
    return parser.parse_args()


def main():    
    args = parse_args()
    output_csv = f"out/eval_{args.model_name.split('/')[-1]}.csv"
    output_json = f"out/generated_paraphrases_{args.model_name.split('/')[-1]}.json"
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.add_tokens([f"<type-{i}>" for i in range(1, 30)])  # Add custom type tokens
    


    etpc_dataset = load_dataset("jpwahle/etpc").filter(lambda x: x["etpc_label"] == 1)
    etpc_dataset = etpc_dataset["train"].train_test_split(test_size=0.2)
    etpc_test_dataset = etpc_dataset["test"].select(range(10)) 
    qqp_dataset = load_dataset("glue", "qqp")["validation"].filter(lambda x: x["label"] == 1).select(range(10))

    eval_datasets = {
        "ETPC": ParaphraseTypeDataset(etpc_test_dataset, tokenizer),
        "QQP": ParaphraseDataset(qqp_dataset, tokenizer),
        #TODO APTY
    }  

    models = [
        (args.model_name, None, "base_model"),
        (args.model_name, args.etpc_dir, "etpc_model"),
        (args.model_name, args.dpo_dir, "dpo_model"),
        (args.model_name, args.ipo_dir, "ipo_model"),
    ]

    evaluate_models(models, eval_datasets, tokenizer, output_csv, output_json)

if __name__ == "__main__":
    main()

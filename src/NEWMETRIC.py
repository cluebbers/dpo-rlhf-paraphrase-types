import torch
import json
import numpy as np
import pandas as pd
from typing import List, Tuple, Dict
from sklearn.metrics import f1_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification


TOP_10_PARAPHRASE_TYPES = [
    "addition/deletion", "change of order", "derivational changes", "inflectional changes",
    "punctuation changes", "same polarity substitution (contextual)", "semantic based",
    "spelling changes", "subordination and nesting changes", "synthetic/analytic substitution"
]


class ParaphraseTypeEvaluator:
    def __init__(self, model_name: str, top_k: int = 10):
        self.model_name = model_name
        self.top_k = top_k
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, clean_up_tokenization_spaces=True)
        self.model.eval()
        self.normalized_top_10_types = [ptype.strip().lower() for ptype in TOP_10_PARAPHRASE_TYPES]

    def preprocess_apt(self, apt: str) -> List[str]:
        apt_types = apt.lower().split(", ")
        filtered_types = [
            ptype.strip() for ptype in apt_types if ptype.strip() in self.normalized_top_10_types
        ]
        return filtered_types

    def predict_paraphrase_types(self, reference: str, paraphrase: str) -> List[str]:
        inputs = self.tokenizer(reference, paraphrase, truncation=True, max_length=256, return_tensors="pt")
        with torch.no_grad():
            logits = self.model(**inputs).logits
        probs = torch.sigmoid(logits).squeeze().cpu().numpy()
        
        top_k_indices = np.argsort(probs)[-self.top_k:][::-1]  # Descending order
        predicted_types = [TOP_10_PARAPHRASE_TYPES[idx] for idx in top_k_indices if probs[idx] > 0.5]
        return predicted_types

    def evaluate_and_update_json(self, json_file_path: str, output_file_path: str) -> Dict[Tuple[str, str], float]:
        with open(json_file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        
        model_dataset_scores = {}

        for entry in data:
            reference = entry["reference"] if entry["reference"] else entry["original"]
            true_types = self.preprocess_apt(entry["APT"])
            dataset_name = entry.get("dataset", "Unknown")
            
            for paraphrase_entry in entry["List"]:
                paraphrase = paraphrase_entry["paraphrase"]
                model_name = paraphrase_entry["model"]
                true_labels = [1 if ptype in true_types else 0 for ptype in TOP_10_PARAPHRASE_TYPES]
                predicted_types = self.predict_paraphrase_types(reference, paraphrase)
                pred_labels = [1 if ptype in predicted_types else 0 for ptype in TOP_10_PARAPHRASE_TYPES]
                f1 = f1_score(true_labels, pred_labels, average="macro", zero_division=0)

                if "evaluation" not in paraphrase_entry:
                    paraphrase_entry["evaluation"] = {}
                paraphrase_entry["evaluation"]["newmetric-F1"] = f1
                paraphrase_entry["evaluation"]["newmetric-apt"] = predicted_types

                # Store scores by (Model, Dataset)
                key = (model_name, dataset_name)
                if key not in model_dataset_scores:
                    model_dataset_scores[key] = []
                model_dataset_scores[key].append(f1)

        # Save the updated JSON file
        with open(output_file_path, 'w', encoding='utf-8') as outfile:
            json.dump(data, outfile, indent=4, ensure_ascii=False)
        
        # Calculate the average F1 score for each model-dataset combination
        avg_model_scores = {key: np.mean(scores) for key, scores in model_dataset_scores.items()}
        print(f"Updated JSON with newmetric-F1 and newmetric-apt saved to {output_file_path}")
        return avg_model_scores

    def update_csv_with_f1(self, csv_file_path: str, output_csv_path: str, avg_model_scores: Dict[Tuple[str, str], float]) -> None:
        df = pd.read_csv(csv_file_path)
        
        # Debugging: Print out the keys of avg_model_scores to verify what keys are present
        print("Available avg_model_scores keys:", list(avg_model_scores.keys()))

        # Extract the adapter name (e.g., "base_model", "etpc_model") for matching
        df['Adapter'] = df['Adapter'].str.strip()  # Ensure no leading/trailing spaces
        
        # Add new columns for F1 scores for each dataset
        df["newmetric-F1-APTY"] = df.apply(
            lambda row: avg_model_scores.get((row["Adapter"], "APTY"), None), axis=1
        )
        df["newmetric-F1-ETPC"] = df.apply(
            lambda row: avg_model_scores.get((row["Adapter"], "ETPC"), None), axis=1
        )

        # Debugging: Print out the DataFrame head to verify the updates
        print("Updated DataFrame head:\n", df.head())

        # Save the updated DataFrame to a new CSV file
        df.to_csv(output_csv_path, index=False)
        print(f"Updated CSV with newmetric-F1-APTY and newmetric-F1-ETPC saved to {output_csv_path}")

# Example usage:
evaluator = ParaphraseTypeEvaluator(model_name="out/cls-models/deberta-base_qqp_pd_etpc_ptd/run-16/checkpoint-184")
avg_model_scores = evaluator.evaluate_and_update_json(
    json_file_path="out/gen-models/generated_paraphrases_Llama-3.1-8B.json",
    output_file_path="out/gen-models/generated_paraphrases_Llama-3.1-8B_F1.json"
)
evaluator.update_csv_with_f1(
    csv_file_path="out/gen-models/eval_Llama-3.1-8B.csv",
    output_csv_path="out/gen-models/eval_Llama-3.1-8B_F1.csv",
    avg_model_scores=avg_model_scores
    
)

#TODO apty and etpxc score column
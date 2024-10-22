import torch
import json
import numpy as np
import pandas as pd
from typing import List, Dict
from sklearn.metrics import f1_score
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import re
import random

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

    def extract_paraphrase(self, paraphrase_list: List[str]) -> str:
        """Extracts the actual paraphrase text from the given list entry."""
        # Ensure the list has the expected structure and process the second element.
        if len(paraphrase_list) >= 2:
            paraphrase_entry = paraphrase_list[1]
            # Extract text after 'Generated Paraphrase:' and clean it up.
            match = re.search(r"Generated Paraphrase:(.*)", paraphrase_entry, re.IGNORECASE)
            if match:
                # Extract everything after 'Generated Paraphrase:'
                cleaned_paraphrase = match.group(1).strip()
                
                # If there are multiple numbered paraphrases, extract only the first one.
                # Look for "1." and "2."
                numbered_match = re.search(r"1\.\s*(.*?)(?=\s*2\.)", cleaned_paraphrase, re.DOTALL)
                if numbered_match:
                    return numbered_match.group(1).strip()
                
                return cleaned_paraphrase
        return ""

    def predict_paraphrase_types(self, reference: str, paraphrase: str) -> List[str]:
        inputs = self.tokenizer(reference, paraphrase, truncation=True, max_length=256, return_tensors="pt")
        with torch.no_grad():
            logits = self.model(**inputs).logits
        probs = torch.sigmoid(logits).squeeze().cpu().numpy()
        
        top_k_indices = np.argsort(probs)[-self.top_k:][::-1]  # Descending order
        predicted_types = [TOP_10_PARAPHRASE_TYPES[idx] for idx in top_k_indices if probs[idx] > 0.5]
        return predicted_types

    def evaluate(self, json_file_path: str) -> pd.DataFrame:
        with open(json_file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        
        model_scores = {}
        single_accuracies = {}
        paraphrase_samples = []

        for entry in data:
            reference = entry["data"]["Original"]
            true_types = self.preprocess_apt(entry["data"]["APT"])
            model_name = entry["data"]["Kind"]
            
            # Extract and clean the paraphrase text from the list.
            paraphrase = self.extract_paraphrase(entry["data"]["Paraphrase"])
            paraphrase_samples.append(paraphrase)  # Store for debugging
            true_labels = [1 if ptype in true_types else 0 for ptype in TOP_10_PARAPHRASE_TYPES]
            predicted_types = self.predict_paraphrase_types(reference, paraphrase)
            pred_labels = [1 if ptype in predicted_types else 0 for ptype in TOP_10_PARAPHRASE_TYPES]
            f1 = f1_score(true_labels, pred_labels, average="macro", zero_division=0)

            # Calculate single accuracy: if any true type is in predicted types.
            correct_prediction = any(ptype in predicted_types for ptype in true_types)
            if model_name not in single_accuracies:
                single_accuracies[model_name] = {"correct": 0, "total": 0}
            single_accuracies[model_name]["correct"] += int(correct_prediction)
            single_accuracies[model_name]["total"] += 1

            # Store F1 scores by model
            if model_name not in model_scores:
                model_scores[model_name] = []
            model_scores[model_name].append(f1)

        # Calculate the average F1 score for each model
        avg_model_scores = {
            model: np.mean(scores) for model, scores in model_scores.items()
        }

        # Calculate single accuracy for each model
        model_single_accuracies = {
            model: (counts["correct"] / counts["total"]) * 100 if counts["total"] > 0 else 0
            for model, counts in single_accuracies.items()
        }

        # Create a DataFrame from the average scores and single accuracies
        df = pd.DataFrame(
            [
                {"Model": model, "Average F1 Score": avg_model_scores[model], "Single Accuracy": model_single_accuracies[model]}
                for model in avg_model_scores
            ]
        )
        
        return df

# Example usage:
evaluator = ParaphraseTypeEvaluator(model_name="out/cls-models/deberta-base_qqp_pd_etpc_ptd/run-16/checkpoint-184")


df_results = evaluator.evaluate(json_file_path="/home/slim/dpo-rhlf-paraphrase-types/out/gen-models/project-5-at-2024-10-21-12-25-522ea966.json")

# Display the results DataFrame
print(df_results)

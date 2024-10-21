import torch
import json
import numpy as np
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
        print(f"Loading model from {model_name}...")
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, clean_up_tokenization_spaces=True)
        self.model.eval()
        print("Model loaded successfully.")

    def predict_paraphrase_types(self, reference: str, paraphrase: str) -> List[str]:
        """Predict paraphrase types for a given reference-paraphrase pair."""
        inputs = self.tokenizer(reference, paraphrase, truncation=True, max_length=256, return_tensors="pt")
        with torch.no_grad():
            logits = self.model(**inputs).logits
        probs = torch.sigmoid(logits).squeeze().cpu().numpy()
        
        # Get the top-k paraphrase types based on probabilities
        top_k_indices = np.argsort(probs)[-self.top_k:][::-1]  # Descending order
        predicted_types = [TOP_10_PARAPHRASE_TYPES[idx] for idx in top_k_indices if probs[idx] > 0.5]
        return predicted_types

    def evaluate_and_update_json(self, json_file_path: str, output_file_path: str) -> None:
        """
        Read the JSON file, compute F1 scores, update the JSON with "newmetric-F1",
        and save it to a new file.
        """
        # Read the content of the JSON file
        with open(json_file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        
        # Iterate over the entries and compute F1 scores for each paraphrase
        for entry in data:
            reference = entry["reference"] if entry["reference"] else entry["original"]
            true_types = entry["APT"].split(", ")
            for paraphrase_entry in entry["List"]:
                paraphrase = paraphrase_entry["paraphrase"]
                model_name = paraphrase_entry["model"]
                
                # Compute true labels for the given APT
                true_labels = [1 if ptype in true_types else 0 for ptype in TOP_10_PARAPHRASE_TYPES]
                
                # Predict types and get predicted labels
                predicted_types = self.predict_paraphrase_types(reference, paraphrase)
                pred_labels = [1 if ptype in predicted_types else 0 for ptype in TOP_10_PARAPHRASE_TYPES]
                
                # Calculate the F1 score for this pair
                f1 = f1_score(true_labels, pred_labels, average="macro", zero_division=0)

                # Add the new F1 score to the paraphrase's evaluation
                if "evaluation" not in paraphrase_entry:
                    paraphrase_entry["evaluation"] = {}
                paraphrase_entry["evaluation"]["newmetric-F1"] = f1

        # Save the updated data to the output file
        with open(output_file_path, 'w', encoding='utf-8') as outfile:
            json.dump(data, outfile, indent=4, ensure_ascii=False)

        print(f"Updated JSON with newmetric-F1 saved to {output_file_path}")






evaluator = ParaphraseTypeEvaluator(model_name="out/cls-models/deberta-base_qqp_pd_etpc_ptd/run-16/checkpoint-184")

evaluator.evaluate_and_update_json(
    json_file_path="out/gen-models/generated_paraphrases_Llama-3.1-8B.json",
    output_file_path="out/gen-models/generated_paraphrases_with_F1.json"
)


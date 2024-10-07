import argparse
import xml.etree.ElementTree as ET

import evaluate
import numpy as np
import pandas as pd
import requests
import spacy
from datasets import concatenate_datasets, load_dataset
from transformers import (
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)
import torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification

nlp = spacy.load("en_core_web_sm")


# Groups (for paraphrase types)
grouped_types = {
    "Morphology-based changes": [
        "Inflectional changes",
        "Modal verb changes",
        "Derivational changes",
    ],
    "Lexicon-based changes": [
        "Spelling changes",
        "Change of format",
        "Same Polarity Substitution (contextual)",
        "Same Polarity Substitution (habitual)",
        "Same Polarity Substitution (named ent.)",
    ],
    "Lexico-syntactic based changes": [
        "Converse substitution",
        "Opposite polarity substitution (contextual)",
        "Opposite polarity substitution (habitual)",
        "Synthetic/analytic substitution",
    ],
    "Syntax-based changes": [
        "Coordination changes",
        "Diathesis alternation",
        "Ellipsis",
        "Negation switching",
        "Subordination and nesting changes",
    ],
    "Discourse-based changes": [
        "Direct/indirect style alternations",
        "Punctuation changes",
        "Syntax/discourse structure changes",
    ],
    "Extremes": ["Entailment", "Identity", "Non-paraphrase"],
    "Others": ["Addition/Deletion", "Change of order", "Semantic-based"],
}


# Custom model that incorporates token information for sequence classification
class SequenceClassificationWithTokenInfo(torch.nn.Module):
    def __init__(self, model_name, num_labels):
        super(SequenceClassificationWithTokenInfo, self).__init__()
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
        self.dropout = torch.nn.Dropout(0.1)
        self.classifier = torch.nn.Linear(self.model.config.hidden_size * 2, num_labels)

    def forward(self, input_ids, attention_mask=None, labels=None):
        # Forward pass through the base model
        outputs = self.model.base_model(input_ids=input_ids, attention_mask=attention_mask, output_hidden_states=True)
        
        # Get the hidden states of all tokens (last layer's hidden states)
        token_embeddings = outputs.hidden_states[-1]
        
        # Mean pooling of token embeddings to get a sequence-level representation
        pooled_token_embeddings = torch.mean(token_embeddings, dim=1)
        
        # Concatenate [CLS] token and pooled token embeddings
        cls_embedding = outputs.last_hidden_state[:, 0, :]
        combined_embeddings = torch.cat([cls_embedding, pooled_token_embeddings], dim=-1)
        
        # Apply dropout and pass through classifier
        logits = self.classifier(self.dropout(combined_embeddings))
        
        if labels is not None:
            loss = F.cross_entropy(logits, labels)
            return loss, logits
        return logits


# Tokenize function for sequence classification
def tokenize_fn(examples, sentence1_key, sentence2_key, tokenizer):
    """
    Tokenizes input examples using a tokenizer.

    Args:
        examples (dict): The input examples.
        sentence1_key (str): The key for the first sentence in the examples.
        sentence2_key (str, optional): The key for the second sentence in the examples.
        tokenizer (Tokenizer): The tokenizer to use for tokenization.

    Returns:
        dict: The tokenized inputs.
    """
    # Tokenize the inputs (single sentence or sentence pair)
    tokenized_inputs = tokenizer(
        *(
            (examples[sentence1_key],)
            if sentence2_key is None
            else (examples[sentence1_key], examples[sentence2_key])
        ),
        padding="max_length",
        truncation=True,
        return_tensors="pt"
    )

    # Set the classification label
    tokenized_inputs["labels"] = examples["etpc_label"]

    return tokenized_inputs


# Split dataset into training and testing based on binary labels (for paraphrase detection)
def split_dataset_binary(dataset, seed=42):
    """
    Splits a dataset into train and test sets based on binary labels.

    Args:
        dataset: The dataset to split.
        seed: The random seed for shuffling.
    """
    num_positive = len(dataset.filter(lambda example: example["labels"] == 1))
    num_negative = len(dataset.filter(lambda example: example["labels"] == 0))
    train_negatives = (
        dataset.filter(lambda example: example["labels"] == 0)
        .shuffle(seed=seed)
        .select(range(int(num_negative * 0.7)))
    )
    train_positives = (
        dataset.filter(lambda example: example["labels"] == 1)
        .shuffle(seed=seed)
        .select(range(int(num_positive * 0.7)))
    )
    train = concatenate_datasets([train_negatives, train_positives])
    test_negatives = (
        dataset.filter(lambda example: example["labels"] == 0)
        .shuffle(seed=seed)
        .select(range(int(num_negative * 0.7), num_negative))
    )
    test_positives = (
        dataset.filter(lambda example: example["labels"] == 1)
        .shuffle(seed=seed)
        .select(range(int(num_positive * 0.7), num_positive))
    )
    test = concatenate_datasets([test_negatives, test_positives])
    return train, test


# Compute metrics function for classification (accuracy)
def compute_metrics(p):
    predictions, labels = p
    preds = np.argmax(predictions, axis=1)
    accuracy = (preds == labels).mean()
    return {"accuracy": accuracy}


# Main function to execute training and evaluation
def main():
    args = parse_args()

    # Load the dataset
    dataset = load_dataset(args.dataset_name)

    # Create tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, add_prefix_space=True)

    if "etpc" in args.dataset_name:
        # ETPC dataset keys
        sentence1_key = "sentence1"
        sentence2_key = "sentence2"
        dataset = dataset["train"]
    elif "qqp" in args.dataset_name:
        # QQP dataset keys
        sentence1_key = "question1"
        sentence2_key = "question2"

    # Split the dataset into train and test
    train, test = split_dataset_binary(dataset)

    # Tokenize the dataset
    dataset_tokenized = dataset.map(
        tokenize_fn,
        batched=True,
        fn_kwargs={
            "sentence1_key": sentence1_key,
            "sentence2_key": sentence2_key,
            "tokenizer": tokenizer,
        },
    )

    # Instantiate the model with token-level information
    model = SequenceClassificationWithTokenInfo(args.model_name, num_labels=27).to(args.device)

    # Replace '/' with '_' to avoid directory creation issues
    sanitized_dataset_name = args.dataset_name.replace('/', '-')
    sanitized_model_name = args.model_name.replace('/', '-')
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=f"./out/cls-models/{sanitized_model_name}_{sanitized_dataset_name}_{args.task_name}",
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        gradient_accumulation_steps=4,
        fp16=True,
        evaluation_strategy="epoch",  # Add evaluation per epoch
    )

    # Trainer object
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train,
        eval_dataset=test,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,  # Updated compute_metrics for accuracy
    )

    # Training
    print("Starting training")
    trainer.train()

    # Evaluation
    print("Starting evaluation")
    results = trainer.evaluate()
    print("#" * 20)
    print(args.model_name)
    print(args.task_name)
    print(results)
    print("#" * 20)

    # Store results
    results_df = pd.DataFrame([results])
    results_df.to_csv(f"./out/{sanitized_model_name}_{sanitized_dataset_name}_{args.task_name}_seq_results.csv", index=False)


# Argument parser
def parse_args():
    """
    Parses command line arguments.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        type=str,
        default="bert-large-uncased",
        help="Name of the model to use",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="jpwahle/etpc",
        help="Name of the dataset to use",
    )
    parser.add_argument(
        "--task_name",
        type=str,
        default="paraphrase-type-detection",
        help="Name of the task to use",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Hardware to use, either of 'cpu', 'cuda', or 'mps'",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()

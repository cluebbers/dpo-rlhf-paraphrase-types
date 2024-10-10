import argparse
import os
import torch
import random
import csv
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import defaultdict
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, hamming_loss, multilabel_confusion_matrix
from datasets import load_dataset, Dataset
from transformers import (
    AutoConfig,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
)

# Top 10 paraphrase types
top_10_types = [
    "addition/deletion", "change of order", "derivational changes",
    "inflectional changes", "punctuation changes",
    "same polarity substitution (contextual)", "semantic-based", 
    "spelling changes", "subordination and nesting changes", "synthetic/analytic substitution"
]

# Focal loss implementation
class FocalLossWithWeights(nn.Module):
    def __init__(self, alpha=None, gamma=1, class_weights=None):
        super(FocalLossWithWeights, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.class_weights = class_weights

    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)  # Prevents NaNs when probability is 0
        F_loss = (self.alpha * (1 - pt) ** self.gamma * BCE_loss).mean(dim=1)
        if self.class_weights is not None:
            class_weights = self.class_weights.unsqueeze(0)  # Make sure it's broadcastable
            F_loss = F_loss * class_weights
        return F_loss.mean()

# Model preparation and class weight calculation
def prepare_model_and_class_weights(model_name, train_dataset, paraphrase_type2cls_id):
    num_labels = len(paraphrase_type2cls_id)  # Determine the number of labels from paraphrase_type2cls_id
    config = AutoConfig.from_pretrained(
        model_name,
        num_labels=num_labels,
        problem_type="multi_label_classification"
    )
    
    model = AutoModelForSequenceClassification.from_pretrained(model_name, config=config)
    class_weights = recalculate_class_weights(train_dataset, num_labels=num_labels, scale_factor=100.0)
    model.classifier.loss_fct = FocalLossWithWeights(gamma=1, class_weights=class_weights)
    model = model.to("cuda" if torch.cuda.is_available() else "cpu")
    return model, class_weights

# Confusion matrix calculation
def compute_confusion_matrix(labels, preds):
    conf_matrix = multilabel_confusion_matrix(labels, preds)
    print(f"Confusion matrix for each class:\n {conf_matrix}")
    return conf_matrix

# Main function that integrates everything
def main():
    args = parse_args()

    dataset, tokenizer = load_and_prepare_dataset(args.model_name)
    paraphrase_type2cls_id, paraphrase_id2cls_type, paraphrase_type_to_category = create_label_maps()
    dataset_tokenized = tokenize_and_process_dataset(
        dataset, tokenizer, paraphrase_type2cls_id
    )

    inspect_labels(dataset_tokenized)
    train_dataset, test_dataset = split_datasets(dataset_tokenized, paraphrase_type2cls_id)
    model, class_weights = prepare_model_and_class_weights(
        args.model_name, train_dataset, paraphrase_type2cls_id
    )
    
    trainer = create_trainer(
        model, train_dataset, test_dataset, tokenizer, class_weights, paraphrase_id2cls_type
    )
    
    train_and_evaluate(trainer, args.model_name)

# Argument parser function
def parse_args():
    parser = argparse.ArgumentParser(description="Paraphrase Type Detection Training")
    parser.add_argument("--model_name", type=str, default="microsoft/deberta-base", help="Pretrained model name")
    return parser.parse_args()

# Dataset loading and preparation
def load_and_prepare_dataset(model_name):
    dataset = load_dataset("jpwahle/etpc").filter(lambda x: x["etpc_label"] == 1)
    tokenizer = AutoTokenizer.from_pretrained(model_name, add_prefix_space=True, clean_up_tokenization_spaces=True)
    dataset = dataset["train"].map(lowercase_labels)
    return dataset, tokenizer

# Function to lowercase labels
def lowercase_labels(examples):
    examples["paraphrase_types"] = [ptype.strip().lower() for ptype in examples["paraphrase_types"]]
    return examples

# Function to create label maps
def create_label_maps():
    paraphrase_types = [
        "inflectional changes", "modal verb changes", "derivational changes", "spelling changes",
        "same polarity substitution (habitual)", "same polarity substitution (contextual)",
        "same polarity substitution (named ent.)", "change of format", "opposite polarity substitution (habitual)",
        "opposite polarity substitution (contextual)", "synthetic/analytic substitution", "converse substitution",
        "diathesis alternation", "negation switching", "ellipsis", "coordination changes",
        "subordination and nesting changes", "punctuation changes", "direct/indirect style alternations",
        "sentence modality changes", "syntax/discourse structure changes", "addition/deletion", "change of order",
        "semantic based", "identity", "non-paraphrase", "entailment", "synthetic/analytic substitution (named ent.)", 
        "negation"
    ]

    paraphrase_type_ids = list(range(len(paraphrase_types)))
    paraphrase_type_categories = ["category_placeholder"] * len(paraphrase_types)
    top_10_types_lower = [ptype.lower() for ptype in top_10_types]
    
    paraphrase_type2cls_id = {ptype: idx for idx, ptype in enumerate(top_10_types_lower)}
    paraphrase_id2cls_type = {v: k for k, v in paraphrase_type2cls_id.items()}
    paraphrase_type_to_category = {ptype: "category_placeholder" for ptype in top_10_types_lower}

    return paraphrase_type2cls_id, paraphrase_id2cls_type, paraphrase_type_to_category

# Function to tokenize and process dataset
def tokenize_and_process_dataset(dataset, tokenizer, paraphrase_type2cls_id):
    dataset_tokenized = dataset.map(
        tokenize_fn,
        batched=True,
        fn_kwargs={
            "sentence1_key": "sentence1",
            "sentence2_key": "sentence2",
            "tokenizer": tokenizer,
            "paraphrase_type2cls_id": paraphrase_type2cls_id,
            "top_10_types_lower": list(paraphrase_type2cls_id.keys()),
        },
    )
    return dataset_tokenized

# Function to tokenize data
def tokenize_fn(examples, sentence1_key, sentence2_key, tokenizer, paraphrase_type2cls_id, top_10_types_lower):
    max_length = 512
    tokenized_inputs = tokenizer(
        examples[sentence1_key],
        examples[sentence2_key] if sentence2_key else None,
        truncation=True,
        max_length=max_length
    )

    num_labels = len(paraphrase_type2cls_id)
    standardized_paraphrase_type2cls_id = {k.strip().lower(): v for k, v in paraphrase_type2cls_id.items()}

    labels = []
    for paraphrase_type_list in examples["paraphrase_types"]:
        binary_labels = [0] * num_labels  # Initialize all to 0

        # Assign the correct paraphrase types to each label
        for paraphrase_type in paraphrase_type_list:
            paraphrase_type_standardized = paraphrase_type.strip().lower()

            # Check if the paraphrase type is in the top 10 types
            if paraphrase_type_standardized in top_10_types_lower:
                cls_id = standardized_paraphrase_type2cls_id.get(paraphrase_type_standardized)
                if cls_id is not None:
                    binary_labels[cls_id] = 1  # Set label to 1 for applicable paraphrase type

        labels.append(binary_labels)

    tokenized_inputs["labels"] = torch.tensor(labels, dtype=torch.float32)
    return tokenized_inputs

# Dataset splitting into training and testing
def split_datasets(dataset_tokenized, paraphrase_type2cls_id):
    train_dataset, test_dataset = split_dataset_by_type(dataset_tokenized, train_percent=0.8)
    train_dataset = balance_dataset(train_dataset, paraphrase_type2cls_id, max_size=1000)
    return train_dataset, test_dataset

# Split dataset based on paraphrase type
def split_dataset_by_type(dataset, train_percent=0.8):
    paraphrase_type_to_examples = {}
    for example in dataset:
        paraphrase_types = example["paraphrase_types"]
        for ptype in paraphrase_types:
            ptype = ptype.strip().lower()
            if ptype not in paraphrase_type_to_examples:
                paraphrase_type_to_examples[ptype] = []
            paraphrase_type_to_examples[ptype].append(example)

    added_examples = set()
    train, test = [], []
    for ptype, examples in paraphrase_type_to_examples.items():
        num_examples = len(examples)
        split_idx = int(train_percent * num_examples)
        random.shuffle(examples)

        for example in examples[:split_idx]:
            example_id = id(example)
            if example_id not in added_examples:
                train.append(example)
                added_examples.add(example_id)

        for example in examples[split_idx:]:
            example_id = id(example)
            if example_id not in added_examples:
                test.append(example)
                added_examples.add(example_id)

    train_dataset = Dataset.from_list(train)
    test_dataset = Dataset.from_list(test)
    return train_dataset, test_dataset

# Class weights recalculation
def recalculate_class_weights(train_dataset, num_labels, scale_factor=1.0):
    label_counts = np.zeros(num_labels)
    for labels in train_dataset["labels"]:
        label_counts += np.array(labels)
    print(f"Class counts: {label_counts}")

    class_weights = 1.0 / (label_counts + np.finfo(np.float32).eps)
    class_weights = (class_weights / np.sum(class_weights)) * scale_factor
    class_weights = torch.tensor(class_weights, dtype=torch.float32).to("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Class weights (scaled by {scale_factor}): {class_weights}")
    return class_weights

# Balancing the dataset (handling imbalance)
def balance_dataset(train_dataset, paraphrase_type2cls_id, max_size=1000, min_size=10):
    class_to_examples = defaultdict(list)
    for example in train_dataset:
        labels = example["labels"]
        if isinstance(labels, torch.Tensor):
            labels = labels.tolist()
        for idx, label in enumerate(labels):
            if label == 1:
                class_to_examples[idx].append(example)

    low_count_classes = {cls for cls, examples in class_to_examples.items() if len(examples) < min_size}
    high_count_classes = {cls for cls, examples in class_to_examples.items() if len(examples) > max_size}
    
    balanced_examples = []

    for paraphrase_class, examples in class_to_examples.items():
        if paraphrase_class in high_count_classes:
            filtered_examples = [ex for ex in examples if not any(
                ex["labels"][low_cls] == 1 for low_cls in low_count_classes)]
            examples_to_add = random.sample(filtered_examples, max_size) if len(filtered_examples) > max_size else filtered_examples
        else:
            examples_to_add = examples  
        
        for example in examples_to_add:
            if example not in balanced_examples:
                balanced_examples.append(example)

    for paraphrase_class, examples in class_to_examples.items():
        if paraphrase_class in low_count_classes:
            for example in examples:
                if example not in balanced_examples:
                    balanced_examples.append(example)
            num_to_add = min_size - len(examples)
            if num_to_add > 0:
                oversampled_examples = random.choices(examples, k=num_to_add)
                balanced_examples.extend(oversampled_examples)

    for paraphrase_class in range(len(paraphrase_type2cls_id)):
        current_class_examples = [ex for ex in balanced_examples if ex["labels"][paraphrase_class] == 1]
        if len(current_class_examples) == 0 and paraphrase_class in class_to_examples:
            balanced_examples.extend(random.sample(class_to_examples[paraphrase_class], min_size))

    return Dataset.from_list(balanced_examples)

if __name__ == "__main__":
    main()

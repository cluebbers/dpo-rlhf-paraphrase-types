"""
python src/unsloth2.py \
    --model_name=meta-llama/Llama-2-7b-hf \
    --adapter_dir=out/gen-models/llama-2-7b-etpc \
    --loss_type=ipo
 
python src/unsloth2.py \
    --model_name=meta-llama/Llama-3.1-8B \
    --adapter_dir=out/gen-models/llama-3.1-8b-etpc \
    --loss_type=ipo
"""

import argparse

import torch
from datasets import DatasetDict, load_dataset
from huggingface_hub import login
from peft import PeftConfig, PeftModel
from trl import DPOConfig, DPOTrainer
from unsloth import FastLanguageModel, PatchDPOTrainer, is_bfloat16_supported

login(new_session=False)


def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Run DPO training")
    parser.add_argument(
        "--model_name",
        type=str,
        default="meta-llama/Llama-2-7b-hf",
        help="Path to the model",
    )
    parser.add_argument(
        "--adapter_dir",
        type=str,
        default="out/gen-models/llama-2-7b-etpc",
        help="Directory of the PEFT adapter",
    )
    parser.add_argument(
        "--loss_type",
        type=str,
        default="sigmoid",
        choices=["sigmoid", "ipo"],
        help="Loss type",
    )
    return parser.parse_args()


def load_and_prepare_model(model_name, adapter_dir):
    """Load the base model, tokenizer, and apply PEFT adapters."""
    max_seq_length = 2048
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=model_name,
        max_seq_length=max_seq_length,
        dtype=None,
        load_in_4bit=True,
    )

    peft_config = PeftConfig.from_pretrained(adapter_dir)
    print("PeftModel:")
    model = PeftModel.from_pretrained(model, adapter_dir)

    model.load_adapter(adapter_dir, adapter_name="reference")
    # print("get_peft_model:")
    # model = FastLanguageModel.get_peft_model(model, **vars(peft_config))

    return model, tokenizer


def load_datasets(train_json_path, eval_json_path):
    """Load the datasets from JSONL files."""
    train_dataset = load_dataset("json", data_files={"train": train_json_path})["train"]
    validation_dataset = load_dataset(
        "json", data_files={"validation": eval_json_path}
    )["validation"]
    return DatasetDict(
        {
            "train": train_dataset,
            "validation": validation_dataset,
        }
    )


def main():
    # Initialize everything
    args = parse_arguments()
    output_dir = f"./out/gen-models/{args.model_name.split('/')[-1]}-paraphrase-type-generation-etpc-apty-{args.loss_type}-unsloth"

    PatchDPOTrainer()  # Patch the DPOTrainer with Unsloth optimizations

    # Load model and tokenizer
    model, tokenizer = load_and_prepare_model(args.model_name, args.adapter_dir)

    # Load dataset from JSONL files
    train_json_path = "out/generation_apty_ranked_train.jsonl"
    eval_json_path = "out/generation_apty_ranked_test.jsonl"
    ds = load_datasets(train_json_path, eval_json_path)

    # Prepare datasets for training
    train_dataset = ds["train"]
    eval_dataset = ds["validation"]

    args = DPOConfig(
        per_device_train_batch_size=2,  # 4 for llama-2
        gradient_accumulation_steps=4,  # 2 for llama-2
        warmup_ratio=0.1,
        num_train_epochs=3,
        fp16=not is_bfloat16_supported(),
        bf16=is_bfloat16_supported(),
        loss_type=args.loss_type,
        learning_rate=5e-5,
        eval_strategy="epoch",
        optim="adamw_8bit",
        output_dir=output_dir,
    )

    # Setup trainer
    trainer = DPOTrainer(
        model,
        args=args,
        beta=0.1,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        tokenizer=tokenizer,
        max_length=1024,
        max_prompt_length=512,
    )

    print("Training started...")  # Start training
    torch.cuda.empty_cache()
    trainer.train()

    # # Save the trained model
    trainer.save_model(output_dir)


if __name__ == "__main__":
    main()

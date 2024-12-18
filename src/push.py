import logging

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    BitsAndBytesConfig,
)

from common import login_to_huggingface


def main():
    
    logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
    
    login_to_huggingface()

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )

    logging.info("microsoft/deberta-base")
    model_name = "microsoft/deberta-base"
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, quantization_config=bnb_config
    )
    del model
    
    logging.info("cluebbers/deberta-base-paraphrase-detection-qqp")
    model_name = "cluebbers/deberta-base-paraphrase-detection-qqp"
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, quantization_config=bnb_config
    )
    del model

    logging.info("cluebbers/deberta-base-paraphrase-type-detection-etpc")
    model_name = "cluebbers/deberta-base-paraphrase-type-detection-etpc"
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, quantization_config=bnb_config
    )
    del model

    logging.info("meta-llama/Llama-3.1-8B")
    model_name = "meta-llama/Llama-3.1-8B"
    model = AutoModelForCausalLM.from_pretrained(
        model_name, quantization_config=bnb_config
    )
    del model

    logging.info("cluebbers/Llama-3.1-8B-paraphrase-type-generation-apty-sigmoid")
    model_name = "cluebbers/Llama-3.1-8B-paraphrase-type-generation-apty-sigmoid"
    model = AutoModelForCausalLM.from_pretrained(
        model_name, quantization_config=bnb_config
    )
    del model

    logging.info("cluebbers/Llama-3.1-8B-paraphrase-type-generation-etpc-apty-reward")
    model_name = "cluebbers/Llama-3.1-8B-paraphrase-type-generation-etpc-apty-reward"
    model = AutoModelForCausalLM.from_pretrained(
        model_name, quantization_config=bnb_config
    )
    del model

    logging.info("cluebbers/Llama-3.1-8B-paraphrase-type-generation-etpc")
    model_name = "cluebbers/Llama-3.1-8B-paraphrase-type-generation-etpc"
    model = AutoModelForCausalLM.from_pretrained(
        model_name, quantization_config=bnb_config
    )
    del model

    logging.info("cluebbers/Llama-3.1-8B-paraphrase-type-generation-apty-ipo")
    model_name = "cluebbers/Llama-3.1-8B-paraphrase-type-generation-apty-ipo"
    model = AutoModelForCausalLM.from_pretrained(
        model_name, quantization_config=bnb_config
    )
    del model


if __name__ == "__main__":
    main()

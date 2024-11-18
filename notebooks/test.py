"""
test for Open LLM Leaderboard
"""
import logging
from transformers import AutoConfig, AutoModel, AutoTokenizer

model_etpc ="cluebbers/Llama-3.1-8B-paraphrase-type-generation-etpc"
model_dpo = "cluebbers/Llama-3.1-8B-paraphrase-type-generation-apty-sigmoid"
model_ipo = "cluebbers/Llama-3.1-8B-paraphrase-type-generation-apty-ipo"

revision = None
revision_etpc = "8785393c6eaf7633cfefcbc7aff17b88c5581d2c"
revision_dpo = "fa38dce3477b915bfc2a7ed687a689547ec0b383"
revision_ipo = "1eae340c39532ea46b1a51a5646c4192e6db5938"

logging.info(f"Loading model {model_etpc} revision {revision}")
config = AutoConfig.from_pretrained(model_etpc, revision=revision)
model = AutoModel.from_pretrained(model_etpc, revision=revision)
tokenizer = AutoTokenizer.from_pretrained(model_etpc, revision=revision)

del config, model, tokenizer

logging.info(f"Loading model {model_dpo} revision {revision}")
config = AutoConfig.from_pretrained(model_dpo, revision=revision)
model = AutoModel.from_pretrained(model_dpo, revision=revision)
tokenizer = AutoTokenizer.from_pretrained(model_dpo, revision=revision)

del config, model, tokenizer

logging.info(f"Loading model {model_ipo} revision {revision}")
config = AutoConfig.from_pretrained(model_ipo, revision=revision)
model = AutoModel.from_pretrained(model_ipo, revision=revision)
tokenizer = AutoTokenizer.from_pretrained(model_ipo, revision=revision)

del config, model, tokenizer

logging.info(f"Loading model {model_etpc} revision {revision_etpc}")
config = AutoConfig.from_pretrained(model_etpc, revision=revision_etpc)
model = AutoModel.from_pretrained(model_etpc, revision=revision_etpc)
tokenizer = AutoTokenizer.from_pretrained(model_etpc, revision=revision_etpc)

del config, model, tokenizer

logging.info(f"Loading model {model_etpc} revision {revision_etpc}")
config = AutoConfig.from_pretrained(model_dpo, revision=revision_dpo)
model = AutoModel.from_pretrained(model_dpo, revision=revision_dpo)
tokenizer = AutoTokenizer.from_pretrained(model_dpo, revision=revision_dpo)

del config, model, tokenizer

logging.info(f"Loading model {model_etpc} revision {revision_etpc}")
config = AutoConfig.from_pretrained(model_ipo, revision=revision_ipo)
model = AutoModel.from_pretrained(model_ipo, revision=revision_ipo)
tokenizer = AutoTokenizer.from_pretrained(model_ipo, revision=revision_ipo)
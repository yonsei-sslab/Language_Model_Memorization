import torch
from easydict import EasyDict
from transformers import AutoTokenizer, AutoModel, AutoModelForSeq2SeqLM, AutoModelForCausalLM
from utils import load_config

CFG = load_config()

# https://github.com/openai/gpt-2/blob/master/model_card.md#gpt-2-model-card
# -small(124 million), -medium(355 million), -large(774 million), -xl(1.5 billion)
GPT_DICT = EasyDict({"small": "gpt2", "middle": "gpt2-medium", "baseline": "gpt2-xl"})

# https://huggingface.co/docs/transformers/model_doc/t5
# T5v1.1: T5v1.1 is an improved version of T5 with some architectural tweaks, and is pre-trained on C4 only without mixing in the supervised tasks.
# https://github.com/google-research/text-to-text-transfer-transformer/blob/main/released_checkpoints.md#experimental-t5-pre-trained-model-checkpoints
# T5 LM Adapt: pretrained on both the denoising and language modeling objectives.
# -small(77 million), -base(250 million), -large(800 million), -3b(3 billion), -11b(11 billion)
T5_DICT = EasyDict({"small": "t5-base", "middle": "t5-large", "baseline": "t5-3b"})

decoder_candidates = ["gpt2", "ctrl", "transfoxl"]
seq2seq_candidates = ["t5", "bart", "pegasus", "marian"]


def load_tokenizer():
    if CFG.model_type.lower() in seq2seq_candidates:
        model_name = T5_DICT["baseline"]
    elif CFG.model_type.lower() in decoder_candidates:
        model_name = GPT_DICT["baseline"]
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token if "gpt" in model_name else tokenizer.pad_token
    print("loaded", CFG.model_type.lower(), "tokenizer")
    return tokenizer


def tokenize_fn(examples):
    tokenizer = load_tokenizer()
    return tokenizer(examples["text"], max_length=CFG.max_prefix_length, truncation=True)


def load_generation_model(model_size):
    if CFG.model_type.lower() in seq2seq_candidates:
        model_name = T5_DICT[model_size]
        model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    elif CFG.model_type.lower() in decoder_candidates:
        model_name = GPT_DICT[model_size]
        model = AutoModelForCausalLM.from_pretrained(model_name)
        model.config.pad_token_id = model.config.eos_token_id
    print("loaded", model.__class__.__name__, model_name, "model")
    return model

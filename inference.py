# built-in libraries
import os
import multiprocessing

# 3rd-party libraries
import pandas as pd
import numpy as np
from tqdm import tqdm
from easydict import EasyDict
import torch
from torch.utils.data import DataLoader
from datasets import load_dataset, dataset_dict
from parallelformers import parallelize


# custom modules
from models import load_tokenizer, load_generation_model, tokenize_fn
from utils import (
    load_config,
    load_devices,
    remove_lengthy_texts,
    restrict_token_length_fn,
    get_token_sequence_length,
    collate_fn,
    seed_everything,
)

if __name__ == "__main__":
    # load config
    CFG = load_config()
    seed_everything(CFG.seed)
    CPU_COUNT = multiprocessing.cpu_count() // 2

    # load models to the designated device(s)
    devices = load_devices()
    tokenizer = load_tokenizer()
    baseline_model = load_generation_model("small")  # largest model
    parallelize(baseline_model, num_gpus=2, fp16=CFG.fp16, verbose="simple")

    # load and tokenize dataset
    internet_data = load_dataset(CFG.data_path, split="train")
    internet_data = internet_data.filter(remove_lengthy_texts, num_proc=CPU_COUNT)
    random_numbers_train = np.random.randint(
        0, len(internet_data["text"]), int(CFG.num_inference_samples)
    )
    internet_data = internet_data.select(random_numbers_train)
    tokenized_datasets = internet_data.map(tokenize_fn, batched=True, num_proc=CPU_COUNT)
    tokenized_datasets = tokenized_datasets.filter(restrict_token_length_fn, num_proc=CPU_COUNT)
    print("text data tokenization done")

    # make dataloaders with uniform lengths batch
    list_prefix_loaders = []
    tokenized_datasets = tokenized_datasets.map(get_token_sequence_length, num_proc=CPU_COUNT)
    min_len = min(tokenized_datasets["sequence_length"])
    max_len = max(tokenized_datasets["sequence_length"])
    for prefix_len in range(min_len, max_len + 1):
        print("inferencing with prefix length of:", prefix_len)
        prefix_uniform_len = tokenized_datasets.filter(
            lambda tokenized_datasets: tokenized_datasets["sequence_length"] == prefix_len
        )  # group prefixes with uniform lengths, due to absent of padding tokens in GPT2
        if len(prefix_uniform_len) == 0:
            continue

        # there may be truncated texts: input_ids length of 10 but original text length is longer than 10
        inputs = tokenizer(
            prefix_uniform_len["text"], return_tensors="pt", truncation=True, max_length=prefix_len
        )
        generated = baseline_model.generate(
            **inputs,
            max_length=CFG.max_prefix_length + CFG.generate_token_length,
            top_k=CFG.top_n,
            repetition_penalty=CFG.repetition_penalty,
            no_repeat_ngram_size=CFG.no_repeat_ngram_size,
        )

        generated_texts = tokenizer.batch_decode(generated, skip_special_tokens=True)

        print(generated_texts)


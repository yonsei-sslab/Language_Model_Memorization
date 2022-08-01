# built-in libraries
import os
import multiprocessing

# 3rd-party libraries
import pandas as pd
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from datasets import Dataset
from models import load_tokenizer, load_generation_model, encode_fn
from metric import (
    calculate_individual_perplexity,
    calculate_individual_lower,
    calculate_individual_window,
    calculate_individual_zlib,
    Summary,
    AverageMeter,
)
from utils import load_config, load_devices, seed_everything, collate_fn

# load config
CFG = load_config()
seed_everything(CFG.seed)
CPU_COUNT = multiprocessing.cpu_count() // 2

# load devices and models
devices = load_devices()
tokenizer = load_tokenizer()
baseline_model = load_generation_model("baseline").to(devices[0])  # largest model
middle_model = load_generation_model("middle").to(devices[0])
small_model = load_generation_model("small").to(devices[0])

# load previously generated(or sampled) result from the LM
df = pd.read_csv(CFG.inference_result_path)
df = df.drop(columns=["prefix"])
dset = Dataset.from_pandas(df)

# tokenize dset
tokenized_dset = dset.map(encode_fn, batched=True, num_proc=CPU_COUNT, remove_columns=["generated"])
rerank_loader = DataLoader(
    tokenized_dset, collate_fn=collate_fn, batch_size=CFG.rerank_batch_size, shuffle=False
)

# initialize six membership inference metrics
Perplexity, Small, Medium, Zlib, Lowercase, Window = (
    AverageMeter("Perplexity", ":6.3f"),
    AverageMeter("Small", ":6.3f"),
    AverageMeter("Medium", ":6.3f"),
    AverageMeter("Zlib", ":6.3f"),
    AverageMeter("Lowercase", ":6.3f"),
    AverageMeter("Window", ":6.3f"),
)

# tokenize previously sampled dataset
for idx, (input_id, attention_mask) in enumerate(tqdm(rerank_loader)):
    # load input_id to the device
    input_id = input_id.to(devices[0])

    # yield metrics per batch
    perplexity = calculate_individual_perplexity(input_id, baseline_model)
    small = calculate_individual_perplexity(input_id, small_model)
    medium = calculate_individual_perplexity(input_id, middle_model)
    zlib = calculate_individual_zlib(input_id, tokenizer)
    lowercase = calculate_individual_lower(input_id, baseline_model, tokenizer, devices[0])
    window = calculate_individual_window(input_id, baseline_model, window_size=CFG.window_size)

    # update metrics accross batches
    Perplexity.update(perplexity, n=input_id.size(0))
    Small.update(small, n=input_id.size(0))
    Medium.update(medium, n=input_id.size(0))
    Zlib.update(zlib, n=input_id.size(0))
    Lowercase.update(lowercase, n=input_id.size(0))
    Window.update(window, n=input_id.size(0))


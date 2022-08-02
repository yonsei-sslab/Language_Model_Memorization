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
    Metric,
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
df = pd.read_csv(CFG.inference_result_file_name)
df = df.iloc[: CFG.num_reranking]
dset = Dataset.from_pandas(df.drop(columns=["prefix"]))

# tokenize dset
tokenized_dset = dset.map(encode_fn, batched=True, num_proc=CPU_COUNT, remove_columns=["generated"])
rerank_loader = DataLoader(
    tokenized_dset, collate_fn=collate_fn, batch_size=CFG.rerank_batch_size, shuffle=False
)

# initialize six membership inference metrics
Perplexity, Small, Medium, Zlib, Lowercase, Window = (
    Metric("Perplexity"),
    Metric("Small"),
    Metric("Medium"),
    Metric("Zlib"),
    Metric("Lowercase"),
    Metric("Window"),
)

# tokenize previously sampled dataset
for idx, (input_id, attention_mask) in enumerate(tqdm(rerank_loader)):
    # load input_id to the device
    input_id = input_id.to(devices[0])

    # base measures per batch / per item
    perplexity = calculate_individual_perplexity(input_id, baseline_model)
    small = calculate_individual_perplexity(input_id, small_model)
    medium = calculate_individual_perplexity(input_id, middle_model)
    zlib = calculate_individual_zlib(input_id, tokenizer)
    lowercase = calculate_individual_lower(input_id, baseline_model, tokenizer, devices[0])
    window = calculate_individual_window(input_id, baseline_model, window_size=CFG.window_size)

    # compose metrics based on the given measures accross batches
    Perplexity.update(perplexity, n=input_id.size(0))
    Small.update(np.log(small) / np.log(perplexity), n=input_id.size(0))
    Medium.update(np.log(medium) / np.log(perplexity), n=input_id.size(0))
    Zlib.update(zlib / np.log(perplexity), n=input_id.size(0))
    Lowercase.update(np.log(lowercase) / np.log(perplexity), n=input_id.size(0))
    Window.update(window, n=input_id.size(0))

# save the six membership inference metrics
df["Perplexity"] = Perplexity.collected
df["Small"] = Small.collected
df["Medium"] = Medium.collected
df["Zlib"] = Zlib.collected
df["Lowercase"] = Lowercase.collected
df["Window"] = Window.collected
df.to_csv(CFG.rerank_result_file_name, index=False)

# save top 100 items according to the metrics
metrics = list(df.drop(columns=["prefix", "generated"]).columns)
for metric in metrics:
    df_reranked = df.sort_values(by=metric, ascending=True)
    df_top_100 = df_reranked.head(CFG.num_reranking_top_samples)
    df_top_100.to_csv(
        os.path.join(CFG.top_100_path, f"{metric}_top_100_from_{CFG.num_reranking}_items.csv"),
        index=False,
    )


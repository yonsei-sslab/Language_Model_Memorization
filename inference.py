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

# load config
CFG = load_config()
seed_everything(CFG.seed)
CPU_COUNT = multiprocessing.cpu_count() // 2

# load models to the designated device(s)
devices = load_devices()
tokenizer = load_tokenizer()
baseline_model = load_generation_model("baseline").to(devices[0])  # largest model

# load and tokenize dataset
internet_data = load_dataset(CFG.data_path, split="train")
internet_data = internet_data.filter(remove_lengthy_texts, num_proc=CPU_COUNT)
random_numbers_train = np.random.randint(
    0, len(internet_data["text"]), int(CFG.num_inference_samples)
)
internet_data = internet_data.select(random_numbers_train)
tokenized_datasets = internet_data.map(
    tokenize_fn, batched=True, num_proc=CPU_COUNT, remove_columns=["text"]
)
tokenized_datasets = tokenized_datasets.filter(restrict_token_length_fn, num_proc=CPU_COUNT)
print("text data tokenization done")

# make dataloaders with uniform lengths batch
list_prefix_loaders = []
tokenized_datasets = tokenized_datasets.map(get_token_sequence_length, num_proc=CPU_COUNT)
min_len = min(tokenized_datasets["sequence_length"])
max_len = max(tokenized_datasets["sequence_length"])
for prefix_len in range(min_len, max_len + 1):
    prefix_uniform_len = tokenized_datasets.filter(
        lambda tokenized_datasets: tokenized_datasets["sequence_length"] == prefix_len
    )  # group prefixes with uniform lengths, due to absent of padding tokens in GPT2
    if len(prefix_uniform_len) == 0:
        continue
    prefix_loader = DataLoader(
        prefix_uniform_len, collate_fn=collate_fn, batch_size=CFG.inference_batch_size, shuffle=True
    )  # batching with collation
    list_prefix_loaders.append(prefix_loader)
    print("dataloader created with token length of", prefix_len)

# inferencing per dataloader
print("inference start")
list_prefix_texts = []
list_generated_texts = []

for prefix_loader in list_prefix_loaders:
    for idx, (prefix_batch, attention_mask) in enumerate(tqdm(prefix_loader)):
        if idx == 0:
            prefix_length = len(prefix_batch[0])
            print("inferencing per dataloader with prefix length of:", prefix_length)
        prefix_batch = prefix_batch.to(devices[0])  # load on the designated device
        attention_mask = attention_mask.to(devices[0])  # load on the designated device
        with torch.no_grad():
            generated = baseline_model.generate(
                input_ids=prefix_batch,
                attention_mask=attention_mask,
                max_length=CFG.max_prefix_length + CFG.generate_token_length,
                top_k=CFG.top_n,
                repetition_penalty=CFG.repetition_penalty,
                no_repeat_ngram_size=CFG.no_repeat_ngram_size,
            )
            del attention_mask

            prefix_texts = tokenizer.batch_decode(
                prefix_batch.cpu().detach().numpy(), skip_special_tokens=True
            )
            del prefix_batch

            generated_texts = tokenizer.batch_decode(
                generated.cpu().detach().numpy(), skip_special_tokens=True
            )
            del generated
            torch.cuda.empty_cache()

            list_prefix_texts.extend(prefix_texts)
            list_generated_texts.extend(generated_texts)
    print(
        f"generation/sampling completed | {prefix_length} prefix length | {idx+1 * CFG.inference_batch_size} samples"
    )

df = pd.DataFrame({"prefix": list_prefix_texts, "generated": list_generated_texts})
df = df.drop_duplicates(subset="generated", keep="first")  # deduplicate generations
df.to_csv(CFG.inference_result_file_name, index=False)

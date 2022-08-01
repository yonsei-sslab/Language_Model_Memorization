import os
import numpy as np
import random
import torch
from easydict import EasyDict
import yaml


def load_config(config_path="./config.yaml"):
    # Read config.yaml file
    with open(config_path) as infile:
        SAVED_CFG = yaml.load(infile, Loader=yaml.FullLoader)
        CFG = EasyDict(SAVED_CFG["CFG"])
    return CFG


def load_devices():
    CFG = load_config()
    device_ids = CFG["device_ids"]
    list_devices = []
    # cpu
    if device_ids == -1 and not torch.cuda.is_available():
        list_devices.append(torch.device("cpu"))
    # single-gpu
    elif device_ids != -1 and type(device_ids) == int and torch.cuda.is_available():
        return torch.device("cuda:" + str(device_ids))
    # multiple-gpu
    elif device_ids != -1 and type(device_ids) == list and torch.cuda.is_available():
        for device_index in device_ids:
            list_devices.append(torch.device(f"cuda:{device_index}"))
    print("working on", list_devices)
    return list_devices


def seed_everything(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    np.random.default_rng(seed)
    random.seed(seed)


def make_sequence_length(examples):
    examples["sequence_length"] = len(examples["input_ids"])
    return examples


def remove_lengthy_texts(examples):
    MAX_LENGTH = 4000
    if len(examples["text"]) >= MAX_LENGTH:
        return False
    else:
        return True


def restrict_token_length_fn(examples):

    CFG = load_config()
    if (
        CFG.min_prefix_length <= len(examples["input_ids"])
        and len(examples["input_ids"]) <= CFG.max_prefix_length
    ):
        return True
    else:
        return False


def get_token_sequence_length(examples):
    examples["sequence_length"] = len(examples["input_ids"])
    return examples


def collate_fn(batch):
    return (
        torch.tensor([item["input_ids"] for item in batch], dtype=torch.long),
        torch.tensor([item["attention_mask"] for item in batch], dtype=torch.long),
    )


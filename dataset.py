import pandas as pd
from transformers import PreTrainedTokenizerFast, DataCollatorForSeq2Seq, 
from datasets import load_dataset, Dataset, DatasetDict
from utils import load_config


def parse_commoncrawl(wet_file):
    """
    Prefix candidates for the GPT-2 model generations.
    Parses of a WET file and port to huggingface dataset
    Tested for the May 2021 crawl.
    @shreyansh26
    """
    dset_list = []
    with open(wet_file) as f:
        lines = f.readlines()

    start_idxs = [i for i in range(len(lines)) if "WARC/1.0" in lines[i]]

    count_eng = 0
    for i in range(len(start_idxs) - 1):
        start = start_idxs[i]
        end = start_idxs[i + 1]
        if "WARC-Identified-Content-Language: eng" in lines[start + 7]:
            count_eng += 1
            for j in range(start + 10, end):
                dset_list.append(lines[j])

    return dset_list


def remove_line_break(input_list: list):
    """
    removes \n from all the items in a list.
    """
    return [item.replace("\n", "") for item in input_list]


def remove_duplicates(input_list: list):
    """
    Delete duplicates from a list.
    """
    return list(set(input_list))


def remove_blank_items(input_list: list):
    """
    Delete blank items from a list.
    """
    return [item for item in input_list if item != ""]


def remove_short_items(input_list: list, min_length: int = 5):
    CFG = load_config()
    tokenizer = PreTrainedTokenizerFast.from_pretrained(CFG.model_name)
    return [item for item in input_list if len(tokenizer.tokenize(item)) >= min_length]


def upload_huggingface_hub(dset: list):
    # package to huggingface dataset and push to hub
    df = pd.DataFrame(dset, columns=["text"])
    dataset = Dataset.from_pandas(df)
    dataset_dict = DatasetDict({"train": dataset})
    dataset_dict.push_to_hub("snoop2head/common_crawl")
    pass


def package_openwebtext():
    """ GPT train dataset: https://huggingface.co/datasets/openwebtext """
    pass


def package_c4():
    """ T5 train dataset: https://huggingface.co/datasets/c4 """
    pass


def __main__():

    # read and parse
    dset = parse_commoncrawl("./commoncrawl.warc.wet")
    print(len(dset))

    # preprocess
    dset = remove_duplicates(dset)
    dset = remove_line_break(dset)
    dset = remove_blank_items(dset)
    dset = remove_short_items(dset)
    print(len(dset))

    # upload to huggingface hub
    upload_huggingface_hub(dset)


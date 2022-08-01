import zlib
import torch
import numpy as np
from enum import Enum


def calculate_individual_perplexity(input_ids, model):
    """ perplexity defined as the exponential of the model's loss """
    model.eval()
    with torch.no_grad():
        output = model(input_ids, labels=input_ids)
    perplexity = torch.exp(output.loss)
    del output, input_ids
    return perplexity.cpu().detach().numpy()


def calculate_individual_lower(input_ids, model, tokenizer, device):
    # if input_ids is nested sequence, lower the dimension
    if len(input_ids.size()) != 1:
        input_ids = input_ids.squeeze()
    text = "".join(tokenizer.decode(input_ids.cpu().detach().numpy()))
    text = text.lower()
    input_ids = tokenizer.encode(text, return_tensors="pt")
    input_ids = input_ids.to(device)
    return calculate_individual_perplexity(input_ids, model)


def calculate_individual_window(input_ids, model, window_size=50):
    """ 
    Sometimes a model is not confident when the sample 
    contains one memorized substring surrounded by a 
    block of non-memorized (and high perplexity) text. 
    To handle this, we use the minimum perplexity when 
    averaged over a sliding window of 50 tokens.
    """
    model.eval()

    # if input_ids is nested sequence, lower the dimension
    if len(input_ids.size()) != 1:
        input_ids = input_ids.squeeze()

    # if not sliding window unavailable, then return mere perplexity
    if input_ids.size(0) < window_size:
        return calculate_individual_perplexity(input_ids, model)

    # make tensors for the sliding window
    sliding_windows = input_ids.unfold(0, window_size, 1)
    min_perplexity = np.inf

    # yield the lowest perplexity score out of given sliding window
    with torch.no_grad():
        for tensor in sliding_windows:
            perplexity = calculate_individual_perplexity(tensor, model)
            del tensor
            min_perplexity = min(min_perplexity, perplexity.sum())

    del input_ids
    return min_perplexity


def calculate_individual_zlib(input_ids, tokenizer):
    """
    As a simple baseline method, we compute the zlib entropy of the text: 
    the number of bits of entropy when the sequence is compressed with zlib compression.
    Although text compressors are simple, they can identify many of the 
    examples of trivial memorization and repeated patterns described above 
    (e.g., they are excellent at modeling repeated substrings).
    """
    # if input_ids is nested sequence, lower the dimension
    if len(input_ids.size()) != 1:
        input_ids = input_ids.squeeze()
    text = "".join(tokenizer.decode(input_ids.cpu().detach().numpy()))
    text = text.lower()
    return len(zlib.compress(bytes(text, "utf-8")))


class Summary(Enum):
    NONE = 0
    AVERAGE = 1
    SUM = 2
    COUNT = 3


class AverageMeter(object):
    """Computes and stores the average accross the given batches"""

    def __init__(self, name, fmt=":f", summary_type=Summary.AVERAGE):
        self.name = name
        self.fmt = fmt
        self.summary_type = summary_type
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)

    def summary(self):
        fmtstr = ""
        if self.summary_type is Summary.NONE:
            fmtstr = ""
        elif self.summary_type is Summary.AVERAGE:
            fmtstr = "{name} {avg:.3f}"
        elif self.summary_type is Summary.SUM:
            fmtstr = "{name} {sum:.3f}"
        elif self.summary_type is Summary.COUNT:
            fmtstr = "{name} {count:.3f}"
        else:
            raise ValueError("invalid summary type %r" % self.summary_type)

        return fmtstr.format(**self.__dict__)


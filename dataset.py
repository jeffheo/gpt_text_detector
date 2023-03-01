"""
TODO: Data Processing Goes Here!

Reference:
    - https://github.com/openai/gpt-2-output-dataset/blob/2c102400c7e4e698acd3f0e51d3b6cf1c637c0fe/detector/dataset.py#L32
    - https://github.com/eric-mitchell/detect-gpt/blob/main/custom_datasets.py
"""
import numpy as np
from typing import List

import torch
from torch.utils.data import Dataset, DataLoader, DistributedSampler, RandomSampler
from tqdm import tqdm
from transformers import PreTrainedTokenizer

from stat_extractor import StatFeatureExtractor
import torch.distributed as dist
from csv import reader
import csv
import math


def load_csv(path):
    rows = []
    with open(path, 'r') as read_obj:
        csv_reader = reader(read_obj)
        # Iterate over each row in the csv using reader object
        for row in csv_reader:
            # row variable is a list that represents a row in csv
            rows.append(row)
    return rows

def write_csv(path, data):
    with open(path, "w+") as f:
        writer = csv.writer(f)
        writer.writerows(data)


class Corpus:
    def __init__(self, name, data_dir='data', skip_train=False):
        self.name = name
        self.train = load_csv(f"{data_dir}/{name}.train.csv")
        self.test = load_csv(f"{data_dir}/{name}.test.csv")
        self.valid = load_csv(f"{data_dir}/{name}.val.csv")


class EncodedDataset(Dataset):
    """
    EncodedDataset from https://github.com/openai/gpt-2-output-dataset/blob/2c102400c7e4e698acd3f0e51d3b6cf1c637c0fe/detector/dataset.py
    """
    def __init__(self, real_texts: List[str], fake_texts: List[str], tokenizer: PreTrainedTokenizer,
                 stat_extractor: StatFeatureExtractor, max_sequence_length: int = None, min_sequence_length: int = None,
                 epoch_size: int = None, token_dropout: float = None, seed: int = None, **kwargs):
        self.real_texts = real_texts
        self.fake_texts = fake_texts
        self.tokenizer = tokenizer
        self.stat_extractor = stat_extractor
        self.max_sequence_length = max_sequence_length
        self.min_sequence_length = min_sequence_length
        self.epoch_size = epoch_size
        self.token_dropout = token_dropout
        self.random = np.random.RandomState(seed)

    def __len__(self):
        return self.epoch_size or len(self.real_texts) + len(self.fake_texts)

    def __getitem__(self, index):
        if self.epoch_size is not None:
            label = self.random.randint(2)
            texts = [self.fake_texts, self.real_texts][label]
            text = texts[self.random.randint(len(texts))]
        else:
            if index < len(self.real_texts):
                text = self.real_texts[index]
                label = 1
            else:
                text = self.fake_texts[index - len(self.real_texts)]
                label = 0

        tokens = self.tokenizer.encode(text[0])

        # TODO: Encode Stat Vec
        stat_vec = self.stat_extractor.encode(text[0])
        print(f'Stat Vector: {stat_vec}')

        if self.max_sequence_length is None:
            tokens = tokens[:self.tokenizer.max_len - 2]
        else:
            output_length = min(len(tokens), self.max_sequence_length)
            if self.min_sequence_length:
                output_length = self.random.randint(min(self.min_sequence_length, len(tokens)), output_length + 1)
            start_index = 0 if len(tokens) <= output_length else self.random.randint(0, len(tokens) - output_length + 1)
            end_index = start_index + output_length
            tokens = tokens[start_index:end_index]

        if self.token_dropout:
            dropout_mask = self.random.binomial(1, self.token_dropout, len(tokens)).astype(np.bool)
            tokens = np.array(tokens)
            tokens[dropout_mask] = self.tokenizer.unk_token_id
            tokens = tokens.tolist()

        if self.max_sequence_length is None or len(tokens) == self.max_sequence_length:
            mask = torch.ones(len(tokens) + 2)
            return torch.tensor([self.tokenizer.bos_token_id] + tokens + [self.tokenizer.eos_token_id]), mask, label, stat_vec

        padding = [self.tokenizer.pad_token_id] * (self.max_sequence_length - len(tokens))
        tokens = torch.tensor([self.tokenizer.bos_token_id] + tokens + [self.tokenizer.eos_token_id] + padding)
        mask = torch.ones(tokens.shape[0])
        mask[-len(padding):] = 0
        return tokens, mask, label, stat_vec


def preload_data(data_path, output_dir, train_pct, val_pct):
    """
    Takes in the raw .csv file of data and splits it into train val test files.
    """
    rows = load_csv(data_path)[1:] # Drop the title row

    real = []
    fake = []
    for row in rows:
        # Have rows only contain generated and non
        real.append([row[3]]) # wiki intro
        fake.append([row[3]]) # generated intro

    num_examples = len(rows)
    train_idx = math.floor(num_examples * train_pct / 100)
    val_idx = train_idx + math.floor(num_examples * val_pct / 100)

    ## Need to figure out what format we want the data to look like

    write_csv(f"{output_dir}/real.train.csv", real[:train_idx])
    write_csv(f"{output_dir}/real.val.csv", real[train_idx:val_idx])
    write_csv(f"{output_dir}/real.test.csv", real[val_idx:])

    write_csv(f"{output_dir}/fake.train.csv", fake[:train_idx])
    write_csv(f"{output_dir}/fake.val.csv", fake[train_idx:val_idx])
    write_csv(f"{output_dir}/fake.test.csv", fake[val_idx:])


# TODO: Implement DataLoader, using dataset processed by ./dataset.py
def load_datasets(data_dir, real_dataset, fake_dataset, tokenizer, stat_extractor, batch_size,
                  max_sequence_length, random_sequence_length,
                  epoch_size=None, token_dropout=None, seed=None, **kwargs) -> DataLoader:

    real_corpus = Corpus(real_dataset, data_dir=data_dir)
    fake_corpus = Corpus(fake_dataset, data_dir=data_dir)

    real_train, real_valid = real_corpus.train, real_corpus.valid
    fake_train, fake_valid = fake_corpus.train, fake_corpus.valid

    Sampler = DistributedSampler if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1 else RandomSampler

    min_sequence_length = 10 if random_sequence_length else None
    train_dataset = EncodedDataset(real_train, fake_train, tokenizer, stat_extractor, max_sequence_length, min_sequence_length,
                                   epoch_size, token_dropout, seed)
    train_loader = DataLoader(train_dataset, batch_size, sampler=Sampler(train_dataset), num_workers=0)

    validation_dataset = EncodedDataset(real_valid, fake_valid, tokenizer, stat_extractor)
    validation_loader = DataLoader(validation_dataset, batch_size=1, sampler=Sampler(validation_dataset))

    return train_loader, validation_loader



############################
## Download the dataset here: https://huggingface.co/datasets/aadityaubhat/GPT-wiki-intro/blob/main/GPT-wiki-intro.csv.zip

preload_data("GPT-wiki-intro.csv", "data", 80, 10)
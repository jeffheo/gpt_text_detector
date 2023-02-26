import matplotlib.pyplot as plt
import numpy as np
import datasets
import transformers
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, DistributedSampler, RandomSampler
import torch.nn.functional as F
from torch.optim import Adam
import tqdm
import random
from sklearn.metrics import roc_curve, precision_recall_curve, auc
import argparse
import datetime
import os
import json
import functools
from multiprocessing.pool import ThreadPool
import time


def load_dummy_data():
    return json.load(open('./toy_dataset.json'))


# TODO: Implement DataLoader, using dataset processed by ./dataset.py
def load_datasets(data_dir, real_dataset, fake_dataset, batch_size,
                  max_sequence_length, random_sequence_length, epoch_size=None, token_dropout=None, seed=None):
    raise NotImplementedError


def train(optimizer, loader: DataLoader, freeze):
    model.train()

    # freeze roberta layer, ONLY train roberta classification head
    if freeze:
        for param in model.roberta.parameters():
            param.requires_grad = False

    # TODO: Calculate / Report Train Accuracy
    train_accuracy = 0
    train_epoch_size = 1
    train_loss = 0
    with tqdm.tqdm(loader, desc="training") as loop:
        for texts, masks, labels in loop:
            texts, masks, labels = texts.to(device), masks.to(device), labels.to(device)
            batch_size = texts.shape[0]

            optimizer.zero_grad()
            loss, logits = model(texts, attention_mask=masks, labels=labels)
            loss.backward()
            optimizer.step()

            # batch_accuracy = accuracy_sum(logits, labels)
            # train_accuracy += batch_accuracy
            train_epoch_size += batch_size
            train_loss += loss.item() * batch_size

            loop.set_postfix(loss=loss.item(), acc=train_accuracy / train_epoch_size)

    return {
        "train/accuracy": train_accuracy,
        "train/epoch_size": train_epoch_size,
        "train/loss": train_loss
    }


def compute_auroc(real, fake):
    """
    Compute FPR, TPR, and AUROC Metrics.
    Learn more about each here: https://developers.google.com/machine-learning/crash-course/classification/roc-and-auc#:~:text=An%20ROC%20curve%20(receiver%20operating,False%20Positive%20Rate
    :param real: Scores (probabilities) for Real (Human-written) Samples
    :param fake: Scores (probabilities) for Fake (Machine-generated) Samples
    :return:
    """
    fpr, tpr, _ = roc_curve([0] * len(real) + [1] * len(fake), real + fake)
    roc_auc = auc(fpr, tpr)
    return fpr.tolist(), tpr.tolist(), float(roc_auc)


def run_baseline_roberta(batch_size=1, large=False):
    """
    TODO: Maybe add more evaluation metrics? Precision-Recall etc.,
    TODO: Also need to plot useful information. Some swavy matplotlib stuff might look nice.
    Baseline: Just use OpenAI Pretrained RoBERTa.
    Code VERY, VERY heavily influenced from detectGPT code (run.py::eval_supervised())
    """
    detector = transformers.AutoModelForSequenceClassification.from_pretrained(name).to(device)

    # TODO: Replace with actual data loading function
    dataset = load_dummy_data()
    real, fake = dataset['real'], dataset['fake']

    with torch.no_grad():
        real_score = []
        for batch in tqdm.tqdm(range(len(real) // batch_size), "Classification on Real"):
            batch_real = real[batch * batch_size:(batch + 1) * batch_size]
            batch_real = tokenizer(batch_real, padding=True, truncation=True, max_length=512, return_tensors="pt").to(device)
            real_score += detector(**batch_real).logits.softmax(-1)[:, 0].tolist()

        fake_score = []
        for batch in tqdm.tqdm(range(len(fake) // batch_size), "Classification on Fake"):
            batch_fake = fake[batch * batch_size:(batch + 1) * batch_size]
            batch_fake = tokenizer(batch_fake, padding=True, truncation=True, max_length=512, return_tensors="pt").to(device)
            fake_score += detector(**batch_fake).logits.softmax(-1)[:, 0].tolist()

        fpr, tpr, auroc = compute_auroc(real_score, fake_score)
        print(f'FPR: {fpr}\n'
              f'TPR: {tpr}\n'
              f'AUROC: {auroc}')


def run(max_epochs=None,
        batch_size=24,
        max_sequence_length=128,
        random_sequence_length=False,
        epoch_size=None,
        seed=None,
        data_dir='data',
        real_dataset='webtext',
        fake_dataset='xl-1542M-nucleus',
        token_dropout=None,
        learning_rate=2e-5,
        weight_decay=0,
        freeze=False,
        **kwargs):
    optimizer = Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    train_loader = load_datasets(data_dir, real_dataset, fake_dataset, batch_size, max_sequence_length, random_sequence_length, epoch_size, token_dropout, seed)
    train(optimizer, train_loader, freeze)


if __name__ == '__main__':
    device = "cuda" if torch.cuda.is_available() else "cpu"

    parser = argparse.ArgumentParser()
    parser.add_argument('-b', '--baseline', action='store_false')  # run baseline, default False
    parser.add_argument('--baseline_only', action='store_false')  # run baseline ONLY, default False
    parser.add_argument('--large', action='store_true', help='use the roberta-large model instead of roberta-base')

    parser.add_argument('--max-epochs', type=int, default=None)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--max-sequence-length', type=int, default=128)
    parser.add_argument('--random-sequence-length', action='store_true')
    parser.add_argument('--epoch-size', type=int, default=None)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--data-dir', type=str, default='data')
    parser.add_argument('--real-dataset', type=str, default='webtext')
    parser.add_argument('--fake-dataset', type=str, default='xl-1542M-k40')
    parser.add_argument('--token-dropout', type=float, default=None)

    parser.add_argument('--learning-rate', type=float, default=2e-5)
    parser.add_argument('--weight-decay', type=float, default=0)

    parser.add_argument('--freeze', action='store_true')
    args = parser.parse_args()

    name = f'roberta-{"large" if args.large else "base"}-openai-detector'
    tokenizer = transformers.RobertaTokenizer.from_pretrained(name)

    if args.baseline:
        run_baseline_roberta(batch_size=args.batch_size)

    if not args.baseline_only:
        config = transformers.RobertaConfig.from_pretrained(name)
        # TODO: Modify configuration hidden size, e.g. config.hidden_size = ???
        model = transformers.RobertaForSequenceClassification.from_pretrained(name, config)
        run(**vars(args))
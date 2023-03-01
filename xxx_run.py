# import matplotlib.pyplot as plt
# import numpy as np
# import datasets
# import transformers
# import torch
# import torch.nn as nn
# from torch.utils.data import DataLoader, DistributedSampler, RandomSampler
# import torch.nn.functional as F
# from torch.optim import Adam
# import tqdm
# import random
# from sklearn.metrics import roc_curve, precision_recall_curve, auc
# import argparse
# import datetime
# import os
# import json
# import functools
# from multiprocessing.pool import ThreadPool
# import time
#
#
# def load_dummy_data():
#     return json.load(open('./toy_dataset.json'))
#
#
# def compute_auroc(real, fake):
#     """
#     Compute FPR, TPR, and AUROC Metrics.
#     Learn more about each here: https://developers.google.com/machine-learning/crash-course/classification/roc-and-auc#:~:text=An%20ROC%20curve%20(receiver%20operating,False%20Positive%20Rate
#     :param real: Scores (probabilities) for Real (Human-written) Samples
#     :param fake: Scores (probabilities) for Fake (Machine-generated) Samples
#     :return:
#     """
#     fpr, tpr, _ = roc_curve([0] * len(real) + [1] * len(fake), real + fake)
#     roc_auc = auc(fpr, tpr)
#     return fpr.tolist(), tpr.tolist(), float(roc_auc)
#
#
# def run_baseline_roberta(batch_size=1, large=False):
#     """
#     TODO: Maybe add more evaluation metrics? Precision-Recall etc.,
#     TODO: Also need to plot useful information. Some swavy matplotlib stuff might look nice.
#     Baseline: Just use OpenAI Pretrained RoBERTa.
#     Code VERY, VERY heavily influenced from detectGPT code (xxx_run.py::eval_supervised())
#     """
#     detector = transformers.AutoModelForSequenceClassification.from_pretrained(name).to(device)
#
#     # TODO: Replace with actual data loading function
#     dataset = load_dummy_data()
#     real, fake = dataset['real'], dataset['fake']
#
#     with torch.no_grad():
#         real_score = []
#         for batch in tqdm.tqdm(range(len(real) // batch_size), "Classification on Real"):
#             batch_real = real[batch * batch_size:(batch + 1) * batch_size]
#             batch_real = tokenizer(batch_real, padding=True, truncation=True, max_length=512, return_tensors="pt").to(
#                 device)
#             real_score += detector(**batch_real).logits.softmax(-1)[:, 0].tolist()
#
#         fake_score = []
#         for batch in tqdm.tqdm(range(len(fake) // batch_size), "Classification on Fake"):
#             batch_fake = fake[batch * batch_size:(batch + 1) * batch_size]
#             batch_fake = tokenizer(batch_fake, padding=True, truncation=True, max_length=512, return_tensors="pt").to(
#                 device)
#             fake_score += detector(**batch_fake).logits.softmax(-1)[:, 0].tolist()
#
#         fpr, tpr, auroc = compute_auroc(real_score, fake_score)
#         print(f'FPR: {fpr}\n'
#               f'TPR: {tpr}\n'
#               f'AUROC: {auroc}')
#
#
# if __name__ == '__main__':
#     device = "cuda" if torch.cuda.is_available() else "cpu"
#
#     parser = argparse.ArgumentParser()
#     parser.add_argument('-b', '--baseline', action='store_true')  # run baseline, default False
#     parser.add_argument('--baseline_only', action='store_true')  # run baseline ONLY, default False
#
#     args = parser.parse_args()
#
#     name = f'roberta-{"large" if args.large else "base"}-openai-detector'
#     tokenizer = transformers.RobertaTokenizer.from_pretrained(name)
#
#     if args.baseline:
#         run_baseline_roberta(batch_size=args.batch_size)
#
#     # TODO: Experiment Pipeline for Main Model

import copy
import time
import torch
import torch.nn as nn
import math
from stat_model import StatRobertaModel
from baseline_model import BaselineRobertaModel
from transformers import RobertaModel, RobertaTokenizer

def main():
    model_name = 'roberta-base'
    tokenizer = RobertaTokenizer.from_pretrained(model_name)
    model = RobertaModel.from_pretrained(model_name)
    configs = {}
    configs['criterion'] = nn.BCELoss()
    configs['lr'] = 5.0  # learning rate
    configs['optimizer'] = torch.optim.SGD(model.parameters(), lr=lr)
    configs['scheduler'] = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)
import copy
import time
import torch
import torch.nn as nn
import math
from late.late_fusion_model import StatRobertaModel
from baseline.baseline_model import BaselineRobertaModel
from transformers import RobertaModel, RobertaTokenizer, RobertaConfig

def main():
    model_name = 'roberta-base'
    tokenizer = RobertaTokenizer.from_pretrained(model_name)
    model = RobertaModel.from_pretrained(model_name)
    new_config = RobertaConfig.from_pretrained('roberta-base')
    old_hidden_size = new_config.hidden_size
    new_config.hidden_size = old_hidden_size + stat_embedding_size

    # Load new RoBERTa model with modified input embedding size
    new_model = RobertaModel(new_config)
    embeddings = model.get_input_embeddings()

    configs = {}
    configs['criterion'] = nn.BCELoss()
    configs['lr'] = 5.0  # learning rate
    configs['optimizer'] = torch.optim.SGD(model.parameters(), lr=lr)
    configs['scheduler'] = torch.optim.lr_scheduler.StepLR(configs['optimizer'], 1.0, gamma=0.95)
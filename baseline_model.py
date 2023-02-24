import torch
from transformers import AutoTokenizer, RobertaForSequenceClassification

import torch
import torch.nn as nn
from transformers import RobertaModel

# Define the model architecture
class BaselineRobertaModel(nn.Module):
    def __init__(self, roberta_model):
        super(BaselineRobertaModel, self).__init__()
        self.roberta = roberta_model
        self.dropout = nn.Dropout(0.1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.linear1 = nn.Linear(768, 256)
        self.linear2 = nn.Linear(256, 64)
        self.linear3 = nn.Linear(64, 1) # binary classification

    def forward(self, input_ids, attention_mask):
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output #batch_size * hidden_size
        pooled_output = self.dropout(pooled_output)
        linear1_output = self.linear1(pooled_output)
        linear1_output = self.relu(linear1_output)
        linear2_output = self.linear2(linear1_output)
        linear2_output = self.relu(linear2_output)
        logits = self.linear3(linear2_output)
        logits = self.sigmoid(logits)
        return logits
import torch
from transformers import AutoTokenizer, RobertaForSequenceClassification

import torch
import torch.nn as nn
from transformers import RobertaModel

# Define the model architecture
class CustomRobertaModel(nn.Module):
    def __init__(self, roberta_model):
        super(CustomRobertaModel, self).__init__()
        self.roberta = roberta_model
        self.dropout = nn.Dropout(0.1)
        self.linear1 = nn.Linear(768, 256)
        self.linear2 = nn.Linear(256, 64)
        self.linear3 = nn.Linear(64, 2) # binary classification

    def forward(self, input_ids, attention_mask):
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        linear1_output = self.linear1(pooled_output)
        linear2_output = self.linear2(linear1_output)
        logits = self.linear3(linear2_output)

        return logits

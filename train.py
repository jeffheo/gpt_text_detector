import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
import tqdm
from typing import Optional, Tuple, Union
from transformers.modeling_outputs import SequenceClassifierOutput


class RobertaWrapper(nn.Module):
    """
    Wrapper Module to process Statistical Features Vector
    Applies Linear Transformation and Non-linearity to Stat Vector to compute "Stat Embedding"
    forward() is same as RobertaForSequenceClassification
    """

    def __init__(self, roberta_seq_classifier, stat_vec_size, unfreeze, baseline, early_fusion=True):
        super(RobertaWrapper, self).__init__()
        self.linear = None
        self.relu = None
        if not baseline:
            self.linear = nn.Linear(stat_vec_size, roberta_seq_classifier.config.hidden_size)
            self.relu = nn.ReLU()

        self.roberta_seq_classifier = roberta_seq_classifier
        self.frozen = False
        if not unfreeze:
            for param in self.roberta_seq_classifier.roberta.parameters():
                param.requires_grad = False
            self.frozen = True

        self.is_baseline = baseline
        self.is_early_fusion = early_fusion

    def stat_embeddings(self, stat: torch.Tensor) -> torch.Tensor:
        assert not self.is_baseline, "Stat Embeddings Not Used in Baseline"
        linear_output = self.linear(stat)
        linear_output = self.relu(linear_output)
        return linear_output

    def word_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.roberta_seq_classifier.roberta.embeddings.word_embeddings(input_ids)

    def forward(
            self,
            input_ids: Optional[torch.LongTensor] = None,
            attention_mask: Optional[torch.FloatTensor] = None,
            token_type_ids: Optional[torch.LongTensor] = None,
            position_ids: Optional[torch.LongTensor] = None,
            head_mask: Optional[torch.FloatTensor] = None,
            inputs_embeds: Optional[torch.FloatTensor] = None,
            labels: Optional[torch.LongTensor] = None,
            output_attentions: Optional[bool] = None,
            output_hidden_states: Optional[bool] = None,
            return_dict: Optional[bool] = None,
            stat_embeds: Optional[torch.FloatTensor] = None
    ) -> Union[Tuple[torch.Tensor], SequenceClassifierOutput]:
        if self.is_early_fusion or self.is_baseline:
            return self.roberta_seq_classifier(input_ids, attention_mask, token_type_ids, position_ids, head_mask,
                                               inputs_embeds, labels, output_attentions, output_hidden_states,
                                               return_dict)
        else:
            outputs = self.roberta_seq_classifier.roberta(input_ids, attention_mask, token_type_ids, position_ids,
                                                          head_mask,
                                                          inputs_embeds)
            outputs = outputs[0]
            print("Outputs dim:")
            print(outputs.size())
            print("Stat embeds dim:")
            print(stat_embeds.size())
            outputs += stat_embeds.unsqueeze(1)
            logits = self.roberta_seq_classifier.classifier(outputs)
            print("logits dimension")
            print(logits.size())
            loss_fct = CrossEntropyLoss()
            print("labels dimension:")
            print(labels.size())
            loss = loss_fct(logits, labels)
            return loss, logits
            #output = (logits,) + outputs[2:]
            #return ((loss,) + output) if loss is not None else output


def accuracy_sum(logits: torch.Tensor, labels: torch.Tensor):
    if list(logits.shape) == list(labels.shape) + [2]:
        # 2-d outputs
        classification = (logits[..., 0] < logits[..., 1]).long().flatten()
    else:
        classification = (logits > 0).long().flatten()
    assert classification.shape == labels.shape
    return (classification == labels).float().sum().item()


def train(model: nn.Module, optimizer, loader, device, desc='Training'):
    model.train()

    train_accuracy = 0
    train_epoch_size = 0
    train_loss = 0

    with tqdm.tqdm(loader, desc=desc, position=0, leave=True) as loop:
        for input_ids, masks, labels, stats in loop:
            input_ids, masks, labels, stats = input_ids.to(device), masks.to(device), labels.to(device), \
                                              stats.to(device)
            batch_size = input_ids.shape[0]
            input_embeds = model.word_embeddings(input_ids)
            stat_embeds = None

            if not model.is_baseline:
                stat_embeds = model.stat_embeddings(stats)
                if model.is_early_fusion:
                    input_embeds += stat_embeds[:, None, :]

            optimizer.zero_grad()
            loss, logits = model(inputs_embeds=input_embeds, attention_mask=masks, labels=labels,
                                 stat_embeds=stat_embeds, return_dict=False)
            loss.backward()
            optimizer.step()

            batch_accuracy = accuracy_sum(logits, labels)
            train_accuracy += batch_accuracy
            train_epoch_size += batch_size
            train_loss += loss.item() * batch_size

            loop.set_postfix(loss=loss.item(), acc=train_accuracy / train_epoch_size)

    return {
        "train/accuracy": train_accuracy,
        "train/epoch_size": train_epoch_size,
        "train/loss": train_loss
    }


def validate(model: nn.Module, loader, device, votes=1, desc='Validating'):
    model.eval()

    validation_accuracy = 0
    validation_epoch_size = 0
    validation_loss = 0

    # TODO: I have no idea what this is doing...
    records = [record for v in range(votes) for record in tqdm.tqdm(loader, desc=f'Preloading data ... {v}')]
    records = [[records[v * len(loader) + i] for v in range(votes)] for i in range(len(loader))]

    with tqdm.tqdm(records, desc=desc, position=0, leave=True) as loop, torch.no_grad():
        for example in loop:
            losses = []
            logit_votes = []
            for input_ids, masks, labels, stats in example:
                input_ids, masks, labels, stats = input_ids.to(device), masks.to(device), \
                                                  labels.to(device), stats.to(device)
                batch_size = input_ids.shape[0]
                input_embeds = model.word_embeddings(input_ids)
                stat_embeds = None
                if not model.is_baseline:
                    stat_embeds = model.stat_embeddings(stats)
                    if model.is_early_fusion:
                        input_embeds += stat_embeds[:, None, :]

                loss, logits = model(inputs_embeds=input_embeds, attention_mask=masks, labels=labels,
                                     stat_embeds=stat_embeds, return_dict=False)

                losses.append(loss)
                logit_votes.append(logits)

            # TODO: Again, no idea what this is doing...
            loss = torch.stack(losses).mean(dim=0)
            logits = torch.stack(logit_votes).mean(dim=0)

            batch_accuracy = accuracy_sum(logits, labels)
            validation_accuracy += batch_accuracy
            validation_epoch_size += batch_size
            validation_loss += loss.item() * batch_size

            loop.set_postfix(loss=loss.item(), acc=validation_accuracy / validation_epoch_size)

    return {
        "val/accuracy": validation_accuracy,
        "val/epoch_size": validation_epoch_size,
        "val/loss": validation_loss
    }

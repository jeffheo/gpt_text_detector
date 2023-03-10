import torch
import torch.nn as nn
from torch.nn import BCEWithLogitsLoss
import tqdm
from typing import Optional, Tuple, Union
from transformers.modeling_outputs import SequenceClassifierOutput


class RobertaWrapper(nn.Module):
    """
    Wrapper Module to process Statistical Features Vector
    Applies Linear Transformation and Non-linearity to Stat Vector to compute "Stat Embedding"
    forward() is same as RobertaForSequenceClassification
    """

    def __init__(self, roberta_seq_classifier, stat_vec_size, unfreeze, baseline, early_fusion, rank_embedding):
        super(RobertaWrapper, self).__init__()
        # W: R^{stat_vec_size} --> R^{hidden_size}
        self.linear = None
        self.relu = None
        self.rank_embed_matrix = None
        if not baseline:
            self.linear = nn.Linear(stat_vec_size, roberta_seq_classifier.config.hidden_size)
            self.relu = nn.ReLU()
        if rank_embedding:
            self.rank_embed_matrix = nn.Parameter(torch.randn(6, roberta_seq_classifier.config.hidden_size) * 0.01)

        self.roberta_seq_classifier = roberta_seq_classifier
        # self.roberta_base = roberta_seq_classifier.roberta
        self.frozen = False
        if not unfreeze:
            for param in self.roberta_seq_classifier.roberta.parameters():
                param.requires_grad = False
            self.frozen = True
        # self.classifier_head = roberta_seq_classifier.classifier

        self.is_baseline = baseline
        self.is_rank_embedding = rank_embedding
        self.is_early_fusion = early_fusion

    def stat_embeddings(self, stat: torch.Tensor) -> torch.Tensor:
        assert not self.is_baseline, "Stat Embeddings Not Used in Baseline"
        linear_output = self.linear(stat)
        linear_output = self.relu(linear_output)
        return linear_output
    
    def rank_embeddings(self, rank: torch.Tensor) -> torch.Tensor:
        #TODO implement this
        #rank has dimension batch_size * sentence length
        #embedding matrix has dimension 6 * hidden_size
        #we need to return dimension of batch_size * sentence length * hidden_size
        assert not self.is_baseline and not self.is_early_fusion, "Rank Embeddings Not Used in Baseline or Early Fusion"
        print("Rank Dim:")
        print(rank.size())
        retval = self.rank_embed_matrix[rank]
        print("Return dimension")
        print(retval.size())
        return retval

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
        """TODO: LATE FUSION DOESN'T WORK, PROBABLY DIMENSION ISSUE"""
        if self.is_early_fusion or self.is_baseline or self.is_rank_embedding:
            return self.roberta_seq_classifier(input_ids, attention_mask, token_type_ids, position_ids, head_mask,
                                               inputs_embeds, labels, output_attentions, output_hidden_states,
                                               return_dict)
        else:
            outputs = self.roberta_seq_classifier.roberta(input_ids, attention_mask, token_type_ids, position_ids,
                                                          head_mask,
                                                          inputs_embeds, labels, output_attentions,
                                                          output_hidden_states, return_dict)
            output = outputs[0]
            output += stat_embeds
            logits = self.roberta_seq_classifier.classifier(output)
            loss_fct = BCEWithLogitsLoss()
            loss = loss_fct(logits, labels)
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output


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
        for input_ids, masks, labels, stats, ranks in loop:
            input_ids, masks, labels, stats, ranks = input_ids.to(device), masks.to(device), labels.to(device), \
                                              stats.to(device), ranks.to(device)
            batch_size = input_ids.shape[0]
            input_embeds = model.word_embeddings(input_ids)
            stat_embeds = None
            rank_embeds = None #ranks dim is batch size x sentence length, rank_embeds dim is bs x sl x embeddingdim

            #TODO write code to add rank_embeds to the input_embeds

            if not model.is_baseline:
                stat_embeds = model.stat_embeddings(stats)

                if model.is_early_fusion:
                    # TODO: we need to fix the dimensions
                    input_embeds += stat_embeds[:, None, :]
            #TODO: add rank embeddings
            if model.is_rank_embedding:
                rank_embeds = model.rank_embeddings(ranks)
                print(rank_embeds.size())
                print(input_embeds.size())
                input_embeds += rank_embeds
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
            for input_ids, masks, labels, stats, ranks in example:
                input_ids, masks, labels, stats, ranks = input_ids.to(device), masks.to(device), \
                                                  labels.to(device), stats.to(device), ranks.to(device)
                batch_size = input_ids.shape[0]
                input_embeds = model.word_embeddings(input_ids)
                stat_embeds = None
                rank_embeds = None
                if not model.is_baseline:
                    stat_embeds = model.stat_embeddings(stats)
                    if model.is_early_fusion:
                        input_embeds += stat_embeds[:, None, :]
                if model.is_rank_embedding:
                    rank_embeds = model.rank_embeddings(ranks)
                    input_embeds += rank_embeds
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

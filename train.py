import os
import torch
import torch.nn as nn
from torch.nn import BCEWithLogitsLoss
import transformers
import tqdm
import argparse
from typing import Optional, Tuple, Union
from torch.optim import Adam
from dataset import load_datasets
from transformers.modeling_outputs import SequenceClassifierOutput

from stat_extractor import StatFeatureExtractor


class RobertaWrapper(nn.Module):
    """
    Wrapper Module to process Statistical Features Vector
    Applies Linear Transformation and Non-linearity to Stat Vector to compute "Stat Embedding"
    forward() is same as RobertaForSequenceClassification
    """

    def __init__(self, roberta_seq_classifier, stat_vec_size, unfreeze, baseline, early_fusion):
        super(RobertaWrapper, self).__init__()
        # W: R^{stat_vec_size} --> R^{hidden_size}
        self.linear = None
        self.relu = None
        if not baseline:
            self.linear = nn.Linear(stat_vec_size, roberta_seq_classifier.config.hidden_size)
            self.relu = nn.ReLU()

        self.roberta_seq_classifier = roberta_seq_classifier
        self.roberta_base = roberta_seq_classifier.roberta
        self.frozen = False
        if not unfreeze:
            for param in self.roberta_base.parameters():
                param.requires_grad = False
            self.frozen = True
        self.classifier_head = roberta_seq_classifier.classifier

        self.is_baseline = baseline
        self.is_early_fusion = early_fusion
        # self.early_fusion = early_fusion  # early or late fusion boolean flag

    def stat_embeddings(self, stat: torch.Tensor) -> torch.Tensor:
        linear_output = self.linear(stat)
        linear_output = self.relu(linear_output)
        return linear_output

    def word_embeddings(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.roberta_base.embeddings.word_embeddings(input_ids)

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
        if self.is_early_fusion or self.is_baseline:
            return self.roberta_seq_classifier(input_ids, attention_mask, token_type_ids, position_ids, head_mask,
                                               inputs_embeds, labels, output_attentions, output_hidden_states,
                                               return_dict)
        else:
            outputs = self.roberta_base(input_ids, attention_mask, token_type_ids, position_ids, head_mask,
                                        inputs_embeds, labels, output_attentions, output_hidden_states, return_dict)
            output = outputs[0]
            output += stat_embeds
            logits = self.classifier_head(output)
            # print(logits, labels)
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
        for input_ids, masks, labels, stats in loop:
            input_ids, masks, labels, stats = input_ids.to(device), masks.to(device), labels.to(device), \
                                              stats.to(device)
            batch_size = input_ids.shape[0]
            input_embeds = model.word_embeddings(input_ids)
            stat_embeds = None
            if not model.is_baseline:
                stat_embeds = model.stat_embeddings(stats)
                if model.is_early_fusion:
                    # TODO: we need to fix the dimensions
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
        "accuracy": train_accuracy,
        "epoch_size": train_epoch_size,
        "loss": train_loss
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
        "accuracy": validation_accuracy,
        "epoch_size": validation_epoch_size,
        "loss": validation_loss
    }

# if __name__ == '__main__':
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--device', type=str, default="cuda" if torch.cuda.is_available() else "cpu")
#     parser.add_argument('--large', action='store_true', help='use the roberta-large model instead of roberta-base')
#     parser.add_argument('--max-epochs', type=int, default=None)
#     parser.add_argument('--batch_size', type=int, default=24)
#     parser.add_argument('--max-sequence-length', type=int, default=128)
#     parser.add_argument('--random-sequence-length', action='store_true')
#     parser.add_argument('--epoch-size', type=int, default=None)
#     parser.add_argument('--seed', type=int, default=None)
#     parser.add_argument('--data_dir', type=str, default='data')
#     parser.add_argument('--real-dataset', type=str, default='real')
#     parser.add_argument('--fake-dataset', type=str, default='fake')
#     parser.add_argument('--token-dropout', type=float, default=None)
#     parser.add_argument('--early_fusion', '-f', action="store_true")
#     parser.add_argument('--baseline', action="store_true")
#
#     parser.add_argument('--learning-rate', type=float, default=2e-5)
#     parser.add_argument('--weight-decay', type=float, default=0)
#     parser.add_argument('--no_freeze', action='store_true')
#
#     parser.add_argument('--all', '-a', action="store_true")
#     parser.add_argument('--zipf', '-z', action="store_true")
#     parser.add_argument('--clumpiness', '-l', action="store_true")
#     parser.add_argument('--punctuation', '-p', action="store_true")
#     parser.add_argument('--coreference', '-c', action="store_true")
#     parser.add_argument('--creativity', '-r', action="store_true")
#
#     parser.add_argument('--baseline_only', action='store_true')  # run baseline ONLY, default False
#
#     args = parser.parse_args()
#
#     d = {
#         'zipf': args.all or args.zipf,
#         'clumpiness': args.all or args.clumpiness,
#         'punctuation': args.all or args.punctuation,
#         'coreference': args.all or args.coreference,
#         'creativity': args.all or args.creativity,
#     }
#
#     name = f'roberta-{"large" if args.large else "base"}-openai-detector'
#     tokenizer = transformers.RobertaTokenizer.from_pretrained(name)
#     args.tokenizer = tokenizer
#
#     args.stat_extractor = StatFeatureExtractor(d)
#     stat_size = args.stat_extractor.stat_vec_size
#     args.stat_size = stat_size
#
#     name = f'roberta-{"large" if args.large else "base"}-openai-detector'
#
#     _roberta = transformers.RobertaForSequenceClassification.from_pretrained(name)
#     _model = RobertaWrapper(_roberta, stat_size, args.baseline, args.early_fusion)
#     _model = _model.to(args.device)
#
#     _optimizer = Adam(_model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
#     _loader, _ = load_datasets(**vars(args))
#     param_name = ""
#     if args.baseline:
#         param_name = "baseline"
#     elif args.early_fusion:
#         param_name = "early_fusion"
#     else:
#         param_name = "late_fusion"
#     train(_model, _optimizer, _loader, not args.no_freeze, args.device)
#
#     params_path = os.path.join('gpt_text_detector/params', f'{param_name}_parameters.pth')
#     torch.save(_model.state_dict(), params_path)

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
from run import run_baseline_roberta


class RobertaWrapper(nn.Module):
    """
    Wrapper Module to process Statistical Features Vector
    Applies Linear Transformation and Non-linearity to Stat Vector to compute "Stat Embedding"
    forward() is same as RobertaForSequenceClassification
    """

    def __init__(self, roberta_seq_classifier, stat_vec_size, baseline, early_fusion):
        super(RobertaWrapper, self).__init__()
        # W: R^{stat_vec_size} --> R^{hidden_size}
        self.linear = nn.Linear(stat_vec_size, roberta_seq_classifier.config.hidden_size)
        self.relu = nn.ReLU()
        self.roberta_seq_classifier = roberta_seq_classifier  # the whole thing
        self.base = roberta_seq_classifier.roberta
        self.classifier_head = roberta_seq_classifier.classifier
        self.is_baseline = baseline
        self.early_fusion = early_fusion  # early or late fusion boolean flag

    def stat_embeddings(self, stat):
        linear_output = self.linear(stat)
        linear_output = self.relu(linear_output)
        return linear_output

    def word_embeddings(self, input_ids):
        return self.roberta_seq_classifier.roberta.embeddings.word_embeddings(input_ids)

    def freeze_roberta(self):
        for param in self.roberta_seq_classifier.roberta.parameters():
            param.requires_grad = False

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
        if self.early_fusion or self.is_baseline:
            return self.roberta_seq_classifier(input_ids, attention_mask, token_type_ids, position_ids, head_mask,
                                               inputs_embeds, labels, output_attentions, output_hidden_states,
                                               return_dict)
        else:
            outputs = self.base(input_ids, attention_mask, token_type_ids, position_ids, head_mask,
                                inputs_embeds, labels, output_attentions, output_hidden_states, return_dict)
            output = outputs[0]
            output += stat_embeds
            logits = self.classifier_head(output)
            print(logits, labels)
            loss_fct = BCEWithLogitsLoss()
            loss = loss_fct(logits, labels)
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output


def train(model: nn.Module, optimizer, loader, freeze, device):
    model.train()

    # freeze roberta layer, ONLY train roberta classification head
    if freeze:
        model.freeze_roberta()

    # TODO: Calculate / Report Train Accuracy
    train_accuracy = 0
    train_epoch_size = 1
    train_loss = 0
    with tqdm.tqdm(loader, desc="training") as loop:
        for input_ids, masks, labels, stats in loop:
            stats = torch.tensor(stats).float()
            input_ids, masks, labels, stats = input_ids.to(device), masks.to(device), labels.to(device), stats.to(
                device)
            batch_size = input_ids.shape[0]

            input_embeds = model.word_embeddings(input_ids)
            # print('intput', input_embeds.size())
            stat_embeds = None
            # TODO: Convert Stat Vector to input_embeds size
            if not model.is_baseline:
                stat_embeds = model.stat_embeddings(stats)
                # print('stat', stat_embeds.size())
                # assert input_embeds.size() == stat_embeds.size()
                if model.early_fusion:
                    input_embeds += stat_embeds

            optimizer.zero_grad()
            loss, logits = model(inputs_embeds=input_embeds, attention_mask=masks, labels=labels,
                                 stat_embeds=stat_embeds, return_dict=False)
            # print(loss, logits)
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument('--large', action='store_true', help='use the roberta-large model instead of roberta-base')
    parser.add_argument('--max-epochs', type=int, default=None)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--max-sequence-length', type=int, default=128)
    parser.add_argument('--random-sequence-length', action='store_true')
    parser.add_argument('--epoch-size', type=int, default=None)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--data_dir', type=str, default='data')
    parser.add_argument('--real-dataset', type=str, default='real')
    parser.add_argument('--fake-dataset', type=str, default='fake')
    parser.add_argument('--token-dropout', type=float, default=None)
    parser.add_argument('--early_fusion', '-f', action="store_true")
    parser.add_argument('--baseline', action="store_true")

    parser.add_argument('--learning-rate', type=float, default=2e-5)
    parser.add_argument('--weight-decay', type=float, default=0)
    parser.add_argument('--no_freeze', action='store_true')

    parser.add_argument('--all', '-a', action="store_true")
    parser.add_argument('--zipf', '-z', action="store_true")
    parser.add_argument('--clumpiness', '-l', action="store_true")
    parser.add_argument('--punctuation', '-p', action="store_true")
    parser.add_argument('--coreference', '-c', action="store_true")
    parser.add_argument('--creativity', '-r', action="store_true")

    parser.add_argument('--baseline_only', action='store_true')  # run baseline ONLY, default False

    args = parser.parse_args()

    d = {
        'zipf': args.all or args.zipf,
        'clumpiness': args.all or args.clumpiness,
        'punctuation': args.all or args.punctuation,
        'coreference': args.all or args.coreference,
        'creativity': args.all or args.creativity,
    }

    name = f'roberta-{"large" if args.large else "base"}-openai-detector'
    tokenizer = transformers.RobertaTokenizer.from_pretrained(name)
    args.tokenizer = tokenizer

    args.stat_extractor = StatFeatureExtractor(d)
    stat_size = args.stat_extractor.stat_vec_size
    args.stat_size = stat_size

    name = f'roberta-{"large" if args.large else "base"}-openai-detector'

    _roberta = transformers.RobertaForSequenceClassification.from_pretrained(name)
    _model = RobertaWrapper(_roberta, stat_size, args.baseline, args.early_fusion)
    _optimizer = Adam(_model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    _loader, _ = load_datasets(**vars(args))

    train(_model, _optimizer, _loader, not args.no_freeze, args.device)

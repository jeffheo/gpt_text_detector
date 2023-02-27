import torch
import torch.nn as nn
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
    def __init__(self, roberta_seq_classifier, stat_vec_size):
        super(RobertaWrapper, self).__init__()
        # W: R^{stat_vec_size} --> R^{hidden_size}
        self.linear = nn.Linear(stat_vec_size, roberta_seq_classifier.config.hidden_size)
        self.relu = nn.ReLU()
        self.roberta_seq_classifier = roberta_seq_classifier

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
    ) -> Union[Tuple[torch.Tensor], SequenceClassifierOutput]:
        return self.roberta_seq_classifier(input_ids, attention_mask, token_type_ids, position_ids, head_mask,
                                           inputs_embeds, labels, output_attentions, output_hidden_states, return_dict)


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
            input_ids, masks, labels, stats = input_ids.to(device), masks.to(device), labels.to(device), stats.to(
                device)
            batch_size = input_ids.shape[0]

            input_embeds = model.word_embeddings(input_ids)

            # TODO: Convert Stat Vector to input_embeds size
            stat_embeds = model.stat_embedding(stats)

            assert input_embeds.size() == stat_embeds.size()

            input_embeds += stat_embeds

            optimizer.zero_grad()
            loss, logits = model(input_embeds=input_embeds, attention_mask=masks, labels=labels)
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
    parser.add_argument('--data-dir', type=str, default='data')
    parser.add_argument('--real-dataset', type=str, default='webtext')
    parser.add_argument('--fake-dataset', type=str, default='xl-1542M-k40')
    parser.add_argument('--token-dropout', type=float, default=None)

    parser.add_argument('--learning-rate', type=float, default=2e-5)
    parser.add_argument('--weight-decay', type=float, default=0)
    parser.add_argument('--no_freeze', action='store_true')

    parser.add_argument('--all', '-a', action="store_true")
    parser.add_argument('--zipf', '-z', action="store_true")
    parser.add_argument('--gini', '-g', action="store_true")
    parser.add_argument('--punctuation', '-p', action="store_true")
    parser.add_argument('--coreference', '-c', action="store_true")
    parser.add_argument('--creativity', '-r', action="store_true")

    args = parser.parse_args()

    d = {
        'zipf': args.all or args.zipf,
        'punctuation': args.all or args.punctuation,
        'gini': args.all or args.gini,
        'creativity': args.all or args.creativity,  # TODO
        'coreference': args.all or args.coreference,  # TODO
    }

    args.stat_extractor = StatFeatureExtractor(d)
    stat_size = args.stat_extractor.stat_vec_size

    name = f'roberta-{"large" if args.large else "base"}-openai-detector'

    _roberta = transformers.RobertaForSequenceClassification.from_pretrained(name)
    _model = RobertaWrapper(_roberta, stat_size)
    _optimizer = Adam(_model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    _loader = load_datasets(**vars(args))

    train(_model, _optimizer, _loader, not args.no_freeze, args.device)

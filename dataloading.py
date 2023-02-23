from transformers import RobertaTokenizer
import torch
import math
from torch.utils.data import Dataset, DataLoader

class CustomDataset(Dataset):
    def __init__(self, corpus_file_path, tokenizer):
        self.tokenizer = tokenizer
        self.sentences = []
        self.labels = []
        self.entropies = []
        with open(corpus_file_path, "r") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) != 2:
                    continue
                sentence = parts[0]
                label = int(parts[1])
                self.sentences.append(sentence)
                self.labels.append(label)
                entropy = self.compute_entropy(sentence)
                self.entropies.append(entropy)
        self.max_seq_length = max([len(self.tokenizer.encode(sentence)) for sentence in self.sentences])

    def compute_entropy(self, sentence):
        char_count = {}
        for char in sentence:
            if char not in char_count:
                char_count[char] = 1
            else:
                char_count[char] += 1
        entropy = 0.0
        for char in char_count:
            prob = char_count[char] / len(sentence)
            entropy -= prob * math.log2(prob)
        return entropy

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        sentence = self.sentences[idx]
        label = self.labels[idx]
        entropy = self.entropies[idx]
        input_ids = self.tokenizer.encode(sentence, padding='max_length', max_length=self.max_seq_length)
        attention_mask = [int(token_id > 0) for token_id in input_ids]

        return input_ids, attention_mask, label, entropy

# Initialize the tokenizer and model
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

# Create the dataset and data loader
corpus_file_path = "path/to/corpus.txt"
dataset = CustomDataset(corpus_file_path, tokenizer)
data_loader = DataLoader(dataset, batch_size=32)

# Loop through the batches of data
for batch in data_loader:
    input_ids, attention_mask, labels, entropies = batch
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    # Process the model outputs, labels, and entropies as needed

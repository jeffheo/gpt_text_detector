# import numpy as np
# from typing import List, Tuple
#
# import torch
# from torch.utils.data import Dataset, DataLoader, DistributedSampler, RandomSampler
# from transformers import PreTrainedTokenizer
# from datasets import load_dataset
# import torch.distributed as dist
#
#
# def process(datapoint):
#     def process_string(s):
#         s = s.strip()
#         return s.replace('\r', ' ').replace('\n', ' ')
#
#     datapoint['wiki_intro'] = process_string(datapoint['wiki_intro'])
#     datapoint['generated_intro'] = process_string(datapoint['generated_intro'])
#
#     return datapoint
#
#
# class EncodedDataset(Dataset):
#     """
#     EncodedDataset from https://github.com/openai/gpt-2-output-dataset/blob/2c102400c7e4e698acd3f0e51d3b6cf1c637c0fe/detector/dataset.py
#     """
#
#     def __init__(self, real_texts: List[str], fake_texts: List[str], tokenizer: PreTrainedTokenizer,
#                  stat_extractor=None, max_sequence_length: int = None, min_sequence_length: int = None,
#                  epoch_size: int = None, token_dropout: float = None, seed: int = None, **kwargs):
#         self.real_texts = real_texts
#         self.fake_texts = fake_texts
#         self.tokenizer = tokenizer
#         self.stat_extractor = stat_extractor
#         self.max_sequence_length = max_sequence_length
#         self.min_sequence_length = min_sequence_length
#         self.epoch_size = epoch_size
#         self.token_dropout = token_dropout
#         self.random = np.random.RandomState(seed)
#
#     def __len__(self):
#         return self.epoch_size or len(self.real_texts) + len(self.fake_texts)
#
#     def __getitem__(self, index):
#         if self.epoch_size is not None:
#             label = self.random.randint(2)
#             texts = [self.fake_texts, self.real_texts][label]
#             text = texts[self.random.randint(len(texts))]
#         else:
#             if index < len(self.real_texts):
#                 text = self.real_texts[index]
#                 label = 1
#             else:
#                 text = self.fake_texts[index - len(self.real_texts)]
#                 label = 0
#
#         # This will throw truncation warning, but truncation is explicitly handled below, so should be fine i think...?
#         tokens = self.tokenizer.encode(text)
#         stat_vec = torch.zeros(1)
#         if self.stat_extractor:
#             stat_vec = self.stat_extractor.encode(text)
#             stat_vec = torch.tensor(stat_vec).float()
#
#         # TODO implement this
#         rank_vec = torch.zeros(1)
#         if self.rank_extractor:
#             rank_vec = []
#             payload = self.rank_extractor.check_probabilities(text)
#             real_ranks = payload['real_topk']  # list of ranks
#             for rank in real_ranks:
#                 index = 5
#                 if rank <= 10:
#                     index = 0
#                 elif rank <= 100:
#                     index = 1
#                 elif rank <= 500:
#                     index = 2
#                 elif rank <= 1000:
#                     index = 3
#                 elif rank <= 5000:
#                     index = 4
#                 rank_vec.append(index)
#
#             rank_vec = torch.tensor(rank_vec).long()
#
#         if self.max_sequence_length is None:
#             tokens = tokens[:self.tokenizer.model_max_length - 2]
#
#         else:
#             output_length = min(len(tokens), self.max_sequence_length)
#             if self.min_sequence_length:
#                 output_length = self.random.randint(min(self.min_sequence_length, len(tokens)), output_length + 1)
#             start_index = 0 if len(tokens) <= output_length else self.random.randint(0, len(tokens) - output_length + 1)
#             end_index = start_index + output_length
#             tokens = tokens[start_index:end_index]
#
#         if self.token_dropout:
#             dropout_mask = self.random.binomial(1, self.token_dropout, len(tokens)).astype(np.bool)
#             tokens = np.array(tokens)
#             tokens[dropout_mask] = self.tokenizer.unk_token_id
#             tokens = tokens.tolist()
#
#         if self.max_sequence_length is None or len(tokens) == self.max_sequence_length:
#             mask = torch.ones(len(tokens) + 2)
#
#             return torch.tensor(
#                 [self.tokenizer.bos_token_id] + tokens + [self.tokenizer.eos_token_id]), mask, label, stat_vec, rank_vec
#
#         padding = [self.tokenizer.pad_token_id] * (self.max_sequence_length - len(tokens))
#         tokens = torch.tensor([self.tokenizer.bos_token_id] + tokens + [self.tokenizer.eos_token_id] + padding)
#         mask = torch.ones(tokens.shape[0])
#         mask[-len(padding):] = 0
#         return tokens, mask, label, stat_vec, rank_vec
#
#
# def load_datasets(tokenizer, batch_size, max_sequence_length, random_sequence_length, datatype="wiki_intro",
#                   stat_extractor=None, rank_extractor=None, epoch_size=None, token_dropout=None, seed=None,
#                   num_workers=0, **kwargs) -> \
#         Tuple[DataLoader, DataLoader, DataLoader]:
#
#     corpus = globals()[f'load_{datatype}']()
#
#     Sampler = DistributedSampler if dist.is_available() and dist.is_initialized() and dist.get_world_size() > 1 else RandomSampler
#
#     min_sequence_length = 10 if random_sequence_length else None
#     train_dataset = EncodedDataset(corpus.train_real, corpus.train_fake, tokenizer, stat_extractor, rank_extractor,
#                                    max_sequence_length,
#                                    min_sequence_length,
#                                    epoch_size, token_dropout, seed)
#
#     train_loader = DataLoader(train_dataset, batch_size, sampler=Sampler(train_dataset), num_workers=num_workers)
#
#     validation_dataset = EncodedDataset(corpus.val_real, corpus.val_fake, tokenizer, stat_extractor, rank_extractor,
#                                         max_sequence_length,
#                                         min_sequence_length,
#                                         epoch_size, token_dropout, seed)
#
#     validation_loader = DataLoader(validation_dataset, batch_size, sampler=Sampler(validation_dataset))
#
#     # exactly the same as other datasets, but set dropout explicitly to None
#     test_dataset = EncodedDataset(corpus.test_real, corpus.test_fake, tokenizer, stat_extractor, rank_extractor, max_sequence_length,
#                                   min_sequence_length, epoch_size, token_dropout=None, seed=seed)
#
#     test_loader = DataLoader(test_dataset, batch_size, sampler=Sampler(test_dataset))
#     return train_loader, validation_loader, test_loader
#
#
# ############################
#
# class Corpus:
#     def __init__(self, train_real, train_fake, val_real, val_fake, test_real, test_fake):
#         self.train_real = train_real
#         self.train_fake = train_fake
#         self.val_real = val_real
#         self.val_fake = val_fake
#         self.test_real = test_real
#         self.test_fake = test_fake
#
#
# def load_wiki_intro(min_tokens=100, max_tokens=250):
#     dataset = load_dataset("aadityaubhat/GPT-wiki-intro")
#     # only use generated text >= 100 tokens
#     dataset['train'] = dataset['train'].filter(lambda x: min_tokens <= x['generated_text_tokens'] <= max_tokens)
#     dataset['train'] = dataset['train'].map(process)
#
#     # train 0.8 / val 0.1 / test 0.1
#     train_test = dataset['train'].train_test_split(test_size=0.2, seed=0)
#     test_val = train_test['test'].train_test_split(test_size=0.5, seed=0)
#
#     train_set = train_test['train']
#     val_set = test_val['train']
#     test_set = test_val['test']
#
#     return Corpus(train_set['wiki_intro'], train_set['generated_intro'],
#                   val_set['wiki_intro'], val_set['generated_intro'],
#                   test_set['wiki_intro'], test_set['generated_intro'])
#
#
# def load_pubmed_qa(min_tokens=100, max_tokens=250):
#     """load pubmed qa dataset for testing purposes"""
#     fake = load_dataset("pubmed_qa", "pqa_artificial")
#     real = load_dataset("pubmed_qa", "pqa_unlabeled")
#
#     fake = fake.shuffle(0)
#     real = real.shuffle(0)
#
#     return Corpus(['n'], ['n'], ['n'], ['n'], real['train']['long_answer'][:10000], fake['train']['long_answer'][:10000])
#
# # dataset = load_dataset("aadityaubhat/GPT-wiki-intro")
# # # only use generated text >= 100 tokens
# # dataset['train'] = dataset['train'].filter(lambda x: x['generated_text_tokens'] >= 100)
# # dataset['train'] = dataset['train'].map(process)
# #
# # # train 0.8 / val 0.1 / test 0.1
# # train_test = dataset['train'].train_test_split(test_size=0.2, seed=0)
# # test_val = train_test['test'].train_test_split(test_size=0.5, seed=0)
# #
# # train_set = train_test['train']
# # val_set = test_val['train']
# # test_set = test_val['test']
# #
# # train_real = train_set['wiki_intro']
# # train_fake = train_set['generated_intro']
# #
# # val_real = val_set['wiki_intro']
# # val_fake = val_set['generated_intro']
# #
# # test_real = test_set['wiki_intro']
# # test_fake = test_set['generated_intro']

import string
import numpy as np
from typing import List, Dict
from nltk.stem import WordNetLemmatizer
from scipy.stats import linregress
from collections import Counter
import nltk


nltk.download('wordnet')
nltk.download('omw-1.4')
lemmatizer = WordNetLemmatizer()


def get_frequency_count(text: str):
    tokens = [lemmatizer.lemmatize(w) for w in text]
    counts = Counter(tokens)
    return counts


class StatFeatureExtractor:
    def __init__(self, args: Dict[str, bool]):
        self.features = [globals()[k] for k in args if k in globals() and args[k]]
        # TODO: How to get stat_vec_size based on what stat features we're using?
        self.stat_vec_size = 2082

    def encode(self, text) -> List[int]:
        vec = []
        count = get_frequency_count(text)
        for phi in self.features:
            vec += phi(text, count)
        # print("LENGTH OF FEATURE VECTOR: ", len(vec))
        assert len(vec) == self.stat_vec_size
        return vec


def zipf(_, word_frequency: Counter) -> List[float]:
    """Calculate distribution and information loss relative to Zipf Distribution
    :return: float
    """
    frequencies = []
    for _, count in word_frequency.most_common():
        frequencies.append(np.log10(count))

    rank = np.log10(np.arange(1, len(frequencies) + 1))
    slope = linregress(rank, frequencies)[0]
    return [slope]


def clumpiness(_, word_frequency: Counter) -> List[float]:
    """We compute the clumpiness of text using Gini coefficient.
    The hypothesis is that human-written text is more likely to be clumpy,
    while machine-generated text is smoother (uniform). This is essentially burstiness.
    Inspiration: http://proceedings.mlr.press/v10/sanasam10a/sanasam10a.pdf
    Code: https://stackoverflow.com/a/61154922
    :return:
    """
    x = np.array(list(word_frequency.values()))
    # print("THIS IS X", x)
    diffsum = 0
    for i, xi in enumerate(x[:-1], 1):
        diffsum += np.sum(np.abs(xi - x[i:]))
    return [diffsum / (len(x) ** 2 * np.mean(x))]


def punctuation(text: str, _) -> List[float]:
    """Compute Punctuation Features, Darmon et al. (https://arxiv.org/pdf/1901.00519.pdf)
    TODO: impl. f4, f5, f6,,,but that seems to be a bit unwieldy? Might put too much emphasis on punct. distribution.
    :param text: String corpus
    :return: ndarray of punctuation features [f1,...,f6]
    """
    text = text.strip(' ')
    i2p = list(string.punctuation)
    p2i = {i2p[i]: i for i in range(len(i2p))}

    f1 = np.zeros(len(i2p))
    P = np.zeros((len(i2p), len(i2p)))

    for i in range(len(text)):
        if text[i] in p2i:
            f1[p2i[text[i]]] += 1
            if i + 1 < len(text) and text[i + 1] in p2i:
                P[p2i[text[i]], p2i[text[i + 1]]] += 1
    ttl = np.sum(f1)
    f1 /= ttl
    P = np.divide(P, np.sum(P, axis=1, keepdims=True), out=np.zeros_like(P), where=P != 0)
    P_tilde = P / ttl
    f2 = P.flatten()
    f3 = P_tilde.flatten()
    return np.hstack((f1, f2, f3)).tolist()


def coreference_resolution():
    """Compute coreference resolution feature. Use Huggingface Neuralcoref
    TODO: This might make our model a bit too heavy. Might need to reconsider.
    :return:
    """
    raise NotImplementedError


def creativity():
    """Compute creativity score based on Kuznetsova et al. (https://aclanthology.org/D13-1124.pdf)
    TODO: This requires logits outputted by a model such as GPT-2. This might make our models a bit too heavy.
    :param:
    :return:
    """
    raise NotImplementedError


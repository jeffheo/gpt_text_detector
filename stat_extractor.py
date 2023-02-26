import string
import numpy as np

from typing import List, Dict


class StatFeatureExtractor:
    def __init__(self, args: Dict[str, bool]):
        self.features = [locals()[k] for k in args if args[k]]

    def stat_vec_size(self):
        # TODO: Calculate Stat Vector size based on features used
        return len(self.features) * 10

    def encode(self, text: str) -> List[int]:
        vec = []
        for phi in self.features:
            vec += phi(text)
        return vec


def zipf():
    """Calculate distribution and information loss relative to Zipf Distribution
    :return: float
    """
    raise NotImplementedError


def heaps():
    """Compute distribution and information loss relative to Heap's Law
    :return:
    """
    raise NotImplementedError


def punctuation(document):
    """Compute Punctuation Features, Darmon et al. (https://arxiv.org/pdf/1901.00519.pdf)
    TODO: impl. f4, f5, f6
    :param document: String corpus
    :return: ndarray of punctuation features [f1,...,f6]
    """
    document = document.strip(' ')
    i2p = list(string.punctuation)
    p2i = {i2p[i]: i for i in range(len(i2p))}

    f1 = np.zeros(len(i2p))
    P = np.zeros((len(i2p), len(i2p)))

    for i in range(len(document)):
        if document[i] in p2i:
            f1[p2i[document[i]]] += 1
            if i + 1 < len(document) and document[i + 1] in p2i:
                P[p2i[document[i]], p2i[document[i + 1]]] += 1
    ttl = np.sum(f1)
    f1 /= ttl
    P = np.divide(P, np.sum(P, axis=1, keepdims=True), out=np.zeros_like(P), where=P != 0)
    P_tilde = P / ttl
    f2 = P.flatten()
    f3 = P_tilde.flatten()
    raise NotImplementedError


def coreference_resolution():
    """Compute coreference resolution feature. Use Huggingface Neuralcoref
    :return:
    """
    raise NotImplementedError


def creativity():
    """Compute creativity score based on Kuznetsova et al. (https://aclanthology.org/D13-1124.pdf)
    :param:
    :return:
    """
    raise NotImplementedError


def extract_features(self, document):
    """Compose features extracted using methods above into a single feature vector
    TODO: This feature vector is going to be super long. Should we add linear layer to reshape?
    """
    raise NotImplementedError


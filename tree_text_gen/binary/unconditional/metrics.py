# --- metrics & evaluation
from collections import defaultdict
import torch.nn as nn


class Metrics(object):
    def __init__(self, tok2i, i2tok):
        self._metrics = defaultdict(list)
        self._metrics['n'] = 0
        self._tok2i = tok2i
        self._i2tok = i2tok
        self.loss = nn.CrossEntropyLoss()

    def reset(self):
        self._metrics = defaultdict(list)
        self._metrics['n'] = 0

    def update(self, scores, samples, batch):
        pass

    def report(self, kind='train', round_level=4):
        metrics = {}
        return metrics

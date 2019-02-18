

import random
import pandas as pd
import torch
import pickle

from tqdm import tqdm
from dataclasses import dataclass
from typing import List
from collections import defaultdict, Counter
from itertools import islice, chain

from torch.utils.data import random_split

from . import utils, logger


@dataclass
class Line:

    tokens: List[str]
    label: str
    split: str

    @classmethod
    def from_dict(cls, row):
        """Map in a raw dictionary.
        """
        field_names = cls.__dataclass_fields__.keys()
        return cls(**{fn: row.get(fn) for fn in field_names})

    @classmethod
    def read_json_lines(cls, path):
        """Parse JSON lines, build match objects.
        """
        df = pd.read_json(path, lines=True)

        for row in df.to_dict('records'):
            yield cls.from_dict(row)


class Corpus:

    @classmethod
    def from_json_lines(cls, path, skim=None):
        """Read JSON gz lines.
        """
        lines_iter = islice(Line.read_json_lines(path), skim)

        # Label -> [line1, line1]
        groups = defaultdict(list)
        for line in tqdm(lines_iter):
            groups[line.label].append(line)

        return cls(groups)

    @classmethod
    def load(cls, path):
        with open(path, 'rb') as fh:
            return pickle.load(fh)

    def save(self, path):
        with open(path, 'wb') as fh:
            pickle.dump(self, fh)

    def __init__(self, groups, test_frac=0.1):
        self.groups = groups
        self.test_frac = test_frac
        self.set_splits()

    def lines_iter(self):
        for group in self.groups.values():
            yield from group

    def labels(self):
        return sorted(list(self.groups))

    def min_label_count(self):
        return min([len(v) for v in self.groups.values()])

    def set_splits(self):
        """Balance classes, fix train/val/test splits.
        """
        min_count = self.min_label_count()

        pairs = list(chain(*[
            [(line, label) for line in random.sample(lines, min_count)]
            for label, lines in self.groups.items()
        ]))

        test_size = round(len(pairs) * self.test_frac)
        train_size = len(pairs) - (test_size * 2)
        sizes = (train_size, test_size, test_size)

        splits = random_split(pairs, sizes)

        # TODO: Store under 'splits' dict?
        for split, name in zip(splits, ('train', 'val', 'test')):

            # Set split on corpus.
            setattr(self, name, split)

            # Set `split` field on individual lines.
            for line, _ in split:
                line.split = name

    def token_counts(self):
        """Collect all token -> count.
        """
        logger.info('Gathering token counts.')

        counts = Counter()
        for line in tqdm(self.lines_iter()):
            counts.update(line.tokens)

        return counts

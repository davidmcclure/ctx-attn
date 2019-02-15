

import torch
import numpy as np
import string

from itertools import chain

from torch import nn
from torchtext.vocab import Vocab, Vectors
from torch.nn.utils import rnn
from torch.nn import functional as F

from . import utils


DEVICE = (torch.device('cuda')
    if torch.cuda.is_available()
    else torch.device('cpu'))


class PretrainedTokenEmbedding(nn.Module):

    def __init__(self, token_counts, vector_file='glove.840B.300d.txt',
        vocab_size=10000, freeze=False):
        """Load pretrained embeddings.
        """
        super().__init__()

        self.vocab = Vocab(
            token_counts,
            vectors=Vectors(vector_file),
            max_size=vocab_size,
        )

        self.embed = nn.Embedding.from_pretrained(self.vocab.vectors, freeze)

        self.out_dim = self.embed.weight.shape[1]

    def forward(self, tokens):
        """Map to token embeddings.
        """
        x = [self.vocab.stoi[t] for t in tokens]
        x = torch.LongTensor(x).to(DEVICE)

        return self.embed(x)


# CharCNN params from https://arxiv.org/abs/1508.06615

class CharEmbedding(nn.Embedding):

    def __init__(self, embed_dim=15):
        """Set vocab, map s->i.
        """
        self.vocab = (
            string.ascii_letters +
            string.digits +
            string.punctuation
        )

        # <PAD> -> 0, <UNK> -> 1
        self._ctoi = {s: i+2 for i, s in enumerate(self.vocab)}

        super().__init__(len(self.vocab)+2, embed_dim)

    def ctoi(self, c):
        return self._ctoi.get(c, 1)

    def chars_to_idxs(self, chars, max_size=20):
        """Map characters to embedding indexes.
        """
        # Truncate super long tokens, to prevent CUDA OOMs.
        chars = chars[:max_size]

        idxs = [self.ctoi(c) for c in chars]

        return torch.LongTensor(idxs).to(DEVICE)

    def forward(self, tokens, min_size=7):
        """Batch-embed token chars.

        Args:
            tokens (list<str>)
        """
        # Map chars -> indexes.
        xs = [self.chars_to_idxs(t) for t in tokens]

        pad_size = max(min_size, max(map(len, xs)))

        # Pad + stack index tensors.
        x = torch.stack([
            F.pad(x, (0, pad_size-len(x)))
            for x in xs
        ])

        return super().forward(x)


class CharCNN(nn.Module):

    def __init__(self, widths=range(1, 7), fpn=25, out_dim=512):
        """Conv layers + linear projection.
        """
        super().__init__()

        self.embed = CharEmbedding()

        self.widths = widths

        self.convs = nn.ModuleList([
            nn.Conv2d(1, w*fpn, (w, self.embed.weight.shape[1]))
            for w in self.widths
        ])

        conv_dims = sum([c.out_channels for c in self.convs])

        self.out = nn.Linear(conv_dims, out_dim)

        self.out_dim = out_dim

    def forward(self, tokens):
        """Convolve, max pool, linear projection.
        """
        x = self.embed(tokens, max(self.widths))

        # 1x input channel.
        x = x.unsqueeze(1)

        # Convolve, max pool.
        x = [torch.tanh(conv(x)).squeeze(3) for conv in self.convs]
        x = [F.max_pool1d(c, c.size(2)).squeeze(2) for c in x]

        # Cat filter maps.
        x = torch.cat(x, 1)

        return self.out(x)


class TokenEmbedding(nn.Module):

    def __init__(self, token_counts):
        """Initialize token + char embeddings
        """
        super().__init__()

        self.embed_t = PretrainedTokenEmbedding(token_counts)
        self.embed_c = CharCNN()

        self.out_dim = self.embed_t.out_dim + self.embed_c.out_dim

    def forward(self, tokens):
        """Map to token embeddings, cat with character convolutions.
        """
        # Token embeddings.
        xt = self.embed_t(tokens)

        # Char embeddings.
        xc = self.embed_c(tokens)
        x = torch.cat([xt, xc], dim=1)

        return x


LSTM_HIDDEN_SIZE = 512
LSTM_NUM_LAYERS = 1


class TokenLSTM(nn.Module):

    def __init__(self, input_size, hidden_size=LSTM_HIDDEN_SIZE,
        num_layers=LSTM_NUM_LAYERS):
        """Initialize LSTM.
        """
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
        )

        self.out_dim = self.lstm.hidden_size * 2

    def forward(self, xs):
        """Sort, pack, encode, reorder.

        Args:
            xs (list<Tensor>): Variable-length embedding tensors.

        Returns:
            x (Tensor)
            states (list<Tensor): LSTM states per input.
        """
        sizes = list(map(len, xs))

        # Indexes to sort descending.
        sort_idxs = np.argsort(sizes)[::-1]

        # Indexes to restore original order.
        unsort_idxs = torch.from_numpy(np.argsort(sort_idxs)).to(DEVICE)

        # Sort by size descending.
        xs = [xs[i] for i in sort_idxs]

        # Pad + pack, LSTM.
        x = rnn.pack_sequence(xs)
        states, (hn, _) = self.lstm(x)

        # Unpack + unpad states.
        states, _ = rnn.pad_packed_sequence(states, batch_first=True)
        states = [t[:size] for t, size in zip(states[unsort_idxs], sizes)]

        # Cat forward + backward hidden layers.
        x = torch.cat([hn[0,:,:], hn[1,:,:]], dim=1)
        x = x[unsort_idxs]

        return x, states


CLF_EMBED_DIM = 512


class Classifier(nn.Module):

    def __init__(self, labels, token_counts, embed_dim=CLF_EMBED_DIM):
        """Initialize encoders + clf.
        """
        super().__init__()

        self.labels = labels

        self.ltoi = {label: i for i, label in enumerate(labels)}

        self.embed_tokens = TokenEmbedding(token_counts)
        self.encode_lines = TokenLSTM(self.embed_tokens.out_dim)

        self.merge = nn.Linear(self.encode_lines.out_dim, embed_dim)

        self.dropout = nn.Dropout()

        self.predict = nn.Sequential(
            nn.Linear(embed_dim, len(labels)),
            nn.LogSoftmax(1),
        )

    def embed(self, lines):
        """Embed lines.
        """
        tokens = [line.tokens for line in lines]

        # Line lengths.
        sizes = [len(ts) for ts in tokens]

        # Embed tokens, regroup by line.
        x = self.embed_tokens(list(chain(*tokens)))
        xs = utils.group_by_sizes(x, sizes)

        # Embed lines.
        x, states = self.encode_lines(xs)

        # Blend encoder outputs, dropout.
        x = self.merge(x)
        x = self.dropout(x)

        return x

    def forward(self, lines):
        return self.predict(self.embed(lines))

    def collate_batch(self, batch):
        """Labels -> indexes.
        """
        lines, labels = list(zip(*batch))

        yt_idx = [self.ltoi[label] for label in labels]
        yt = torch.LongTensor(yt_idx).to(DEVICE)

        return lines, yt

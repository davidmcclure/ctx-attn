

import torch
import pandas as pd
import click

from tqdm import tqdm
from torch.utils.data import DataLoader
from torch import optim, nn
from sklearn import metrics
from itertools import chain

from ctx_attn import logger
from ctx_attn.model import Classifier, DEVICE
from ctx_attn.corpus import Corpus


def train_epoch(model, optimizer, loss_func, split):
    """Train a single epoch, return batch losses.
    """
    loader = DataLoader(
        split,
        collate_fn=model.collate_batch,
        batch_size=50,
    )

    losses = []
    with tqdm(total=len(split)) as bar:
        for lines, yt in loader:

            model.train()
            optimizer.zero_grad()

            yp = model(lines)

            loss = loss_func(yp, yt)
            loss.backward()

            optimizer.step()

            losses.append(loss.item())
            bar.update(len(lines))

    return losses


def predict(model, split):
    """Predict inputs in a split.
    """
    model.eval()

    loader = DataLoader(
        split,
        collate_fn=model.collate_batch,
        batch_size=50,
    )

    yt, yp = [], []
    with tqdm(total=len(split)) as bar:
        for lines, yti in loader:
            yp += model(lines).tolist()
            yt += yti.tolist()
            bar.update(len(lines))

    yt = torch.LongTensor(yt)
    yp = torch.FloatTensor(yp)

    return yt, yp


def evaluate(model, loss_func, split, log_acc=True):
    """Predict matches in split, log accuracy, return loss.
    """
    yt, yp = predict(model, split)

    if log_acc:
        acc = metrics.accuracy_score(yt, yp.argmax(1))
        logger.info(f'Accuracy: {acc}')

    return loss_func(yp, yt)


def train_model(model, optimizer, loss_func, corpus, max_epochs, es_wait):
    """Train for N epochs, or stop early.
    """
    losses = []
    for i in range(max_epochs):

        logger.info(f'Epoch {i+1}')
        train_epoch(model, optimizer, loss_func, corpus.train)

        loss = evaluate(model, loss_func, corpus.val)
        losses.append(loss)

        logger.info(loss.item())

        # Stop early.
        if len(losses) > es_wait and losses[-1] > losses[-es_wait]:
            break

    return model


def predict_df_rows(model, split):
    """Predict all lines, dump DF.
    """
    model.eval()

    loader = DataLoader(
        split,
        collate_fn=model.collate_batch,
        batch_size=50,
    )

    rows = []
    with tqdm(total=len(split)) as bar:
        for lines, yt in loader:

            x, (dists, ctx_dists) = model.embed(lines)
            yp = model.predict(x).exp()

            for line, ypi, ad, acd in zip(lines, yp, dists, ctx_dists):

                preds = dict(zip(model.labels, ypi.tolist()))

                rows.append(dict(
                    **line.__dict__,
                    preds=preds,
                    attn_dist=ad.tolist(),
                    attn_ctx_dist=acd.tolist(),
                ))

            bar.update(len(lines))

    return rows


def predict_df(model, corpus):
    """Predict all splits,
    """
    rows = chain(
        predict_df_rows(model, corpus.train),
        predict_df_rows(model, corpus.val),
        predict_df_rows(model, corpus.test),
    )

    return pd.DataFrame(rows)


@click.group()
def cli():
    pass


@cli.command()
@click.argument('src', type=click.Path())
@click.argument('dst', type=click.Path())
@click.option('--skim', type=int, default=None)
def build_corpus(src, dst, skim):
    """Freeze off train/dev/test splits.
    """
    corpus = Corpus.from_json_lines(src, skim)
    corpus.save(dst)


@cli.command()
@click.argument('src', type=click.Path())
@click.argument('dst', type=click.Path())
@click.option('--max_epochs', type=int, default=100)
@click.option('--es_wait', type=int, default=5)
def train(src, dst, max_epochs, es_wait):
    """Train, dump model.
    """
    corpus = Corpus.load(src)

    labels = corpus.labels()
    counts = corpus.token_counts()

    model = Classifier(labels, counts).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    loss_func = nn.NLLLoss()

    model = train_model(model, optimizer, loss_func, corpus,
        max_epochs, es_wait)

    df = predict_df(model, corpus)
    df.to_json(dst, orient='records', lines=True)


if __name__ == '__main__':
    cli()

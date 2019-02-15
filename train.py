

import torch
import click

from tqdm import tqdm

from torch.utils.data import DataLoader
from torch import optim, nn
from sklearn import metrics

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
        for matches, yt in loader:

            model.train()
            optimizer.zero_grad()

            yp = model(matches)

            loss = loss_func(yp, yt)
            loss.backward()

            optimizer.step()

            losses.append(loss.item())
            bar.update(len(matches))

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
        for matches, yti in loader:
            yp += model(matches).tolist()
            yt += yti.tolist()
            bar.update(len(matches))

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
@click.option('--es_wait', type=int, default=5)
@click.option('--max_epochs', type=int, default=100)
def train(src, es_wait, max_epochs):
    """Train, dump model.
    """
    corpus = Corpus.load(src)

    labels = corpus.labels()
    counts = corpus.token_counts()

    model = Classifier(labels, counts).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    loss_func = nn.NLLLoss()

    # Train.
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


if __name__ == '__main__':
    cli()

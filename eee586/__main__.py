import click
import torch

from eee586.pretrain import pretrain_bert_model
from eee586.word_embedding import embed_sentence


@click.group()
def cli():
    pass


@cli.command()
@click.option(
    "--dataset-sample-size",
    default=1000,
    help="Number of samples to use.",
    type=click.INT,
)
def pretrain(dataset_sample_size):
    pretrain_bert_model(
        dataset_sample_size=dataset_sample_size,
    )


@cli.command()
@click.option(
    "--sentence",
    default="Replace me by any text you'd like.",
    help="Sentence to embed.",
    type=click.STRING,
)
def embed(sentence):
    output = embed_sentence(sentence)
    print(output[0].shape)


@cli.command()
def check_torch_cuda():
    print(torch.cuda.is_available())


cli()

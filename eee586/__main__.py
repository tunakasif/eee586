import click
import torch

from eee586.pretrain import pretrain_bert_model
from eee586.word_embedding import get_token_encodings


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
def embed():
    train_toke_enc_dict = get_token_encodings("train")
    print(train_toke_enc_dict["input_ids"][0])
    print(train_toke_enc_dict["labels"][0])


@cli.command()
def check_torch_cuda():
    print(torch.cuda.is_available())


cli()

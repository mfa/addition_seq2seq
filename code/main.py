from dataclasses import dataclass

import click
from utils import set_seed
from train import train


@dataclass
class Config:
    train_data_path: str
    val_data_path: str
    #
    cuda: bool
    batch_size: int = 1024
    epochs: int = 100
    learning_rate: float = 0.001
    use_teacher_forcing_perc: float = 0.5
    seed: int = 42
    #
    num_layers: int = 2
    rnn_size: int = 200
    bidirectional_encoder: bool = True
    attn_method: str = "disabled"
    vocab_size: int = 15  # 0-9 + s,e,pad + "+-"


@click.command()
@click.option("--train-data-path", help="train filename", required=True)
@click.option("--val-data-path", help="validation filename", required=True)
@click.option("--cuda", default=False)
@click.option(
    "--attn_method",
    default="disabled",
    type=click.Choice(["disabled", "concat", "dot", "global"]),
)
@click.option("--batch-size", default=1024)
@click.option("--epochs", default=10)
def main(**kwargs):
    config = Config(**kwargs)
    set_seed(config)
    print(config)

    train(config)


if __name__ == "__main__":
    main()

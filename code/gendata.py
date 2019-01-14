import click
import csv
import random

import numpy as np
import pandas as pd
import tqdm


def generate_dataset(size, low, high, seed=None):
    if seed is not None:
        np.random.seed(seed)

    num1_data = np.random.randint(low=low, high=high, size=size)
    num2_data = np.random.randint(low=low, high=high, size=size)
    df_dict = {"q": [], "a": []}
    for i, j in tqdm.tqdm(zip(num1_data, num2_data), mininterval=1.0):
        q = f"{i}+{j}"
        # no duplicates
        if q in df_dict["q"]:
            continue
        df_dict["q"].append(q)
        df_dict["a"].append(i + j)
    return pd.DataFrame(df_dict)


def save_dataset(df, filepath):
    df.to_csv(filepath, index=False, header=False, quoting=csv.QUOTE_NONNUMERIC)


@click.command()
@click.option("-o", "--output", help="Filename", required=True)
@click.option("-s", "--size", help="Number of samples", required=True, type=int)
@click.option("--high", default=1000, help="Highest number", type=int)
def main(output, size, high):
    dataset = generate_dataset(size=size, low=0, high=high, seed=42)
    save_dataset(df=dataset, filepath=output)


if __name__ == "__main__":
    main()

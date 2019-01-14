import numpy as np
import pandas as pd
import torch.utils.data
from torch.autograd import Variable

BOS = "<s>"
EOS = "<e>"
PAD = "<>"
PAD_IDX = 0
BOS_IDX = 1
EOS_IDX = 2

NUMBER_MAPPING = [PAD, BOS, EOS] + ["-", "+"] + [str(i) for i in range(10)]


class NumberDataset(torch.utils.data.Dataset):
    def __init__(self, data_list):
        self.index2token = dict(zip(range(len(NUMBER_MAPPING)), NUMBER_MAPPING))
        self.token2index = dict(zip(NUMBER_MAPPING, range(len(NUMBER_MAPPING))))

        self.data_list = data_list
        self.data_index_list = list(map(self.to_index, data_list))

    def __getitem__(self, index):
        return self.data_index_list[index]

    def __len__(self):
        return len(self.data_index_list)

    def to_index(self, string):
        return np.array([BOS_IDX] + list(map(self.token2index.get, string)) + [EOS_IDX])

    def to_string(self, index_list):
        return "".join([self.index2token.get(i) for i in index_list])

    def batch_tensor_to_string(self, batch_tensor):
        if isinstance(batch_tensor, Variable):
            batch_tensor = batch_tensor.data

        return [
            self.to_string(tensor_line) for tensor_line in batch_tensor.cpu().numpy()
        ]


class SequenceDataset(torch.utils.data.Dataset):
    def __init__(self, input_sequence, output_sequence):
        assert len(input_sequence) == len(output_sequence)
        self.input_sequence = input_sequence
        self.output_sequence = output_sequence

    def __getitem__(self, index):
        return (self.input_sequence[index], self.output_sequence[index])

    def __len__(self):
        return len(self.input_sequence)


def collate_fn(batch):
    # process batches; add padding; add ordering for shuffling later
    padded_x_ls = []
    padded_y_ls = []
    ordered_x_len_ls = []
    ordered_y_len_ls = []

    x_len_ls = [len(_[0]) for _ in batch]
    y_len_ls = [len(_[1]) for _ in batch]
    max_x_len = max(x_len_ls)
    max_y_len = max(y_len_ls)

    ordered_x = sorted(zip(x_len_ls, range(len(x_len_ls))), reverse=True)

    for _, i in ordered_x:
        (x, y) = batch[i]
        padded_x_ls.append(
            np.pad(x, (0, max_x_len - len(x)), "constant", constant_values=PAD_IDX)
        )
        padded_y_ls.append(
            np.pad(y, (0, max_y_len - len(y)), "constant", constant_values=PAD_IDX)
        )
        ordered_x_len_ls.append(len(x))
        ordered_y_len_ls.append(len(y))

    padded_x = np.vstack(padded_x_ls)
    padded_y = np.vstack(padded_y_ls)

    if padded_y.dtype != np.dtype("int64"):
        # TODO: Better UNK handling
        raise Exception("Unknown token encountered")

    x_tensor = torch.LongTensor(padded_x)
    y_tensor = torch.LongTensor(padded_y)

    return ((x_tensor, ordered_x_len_ls), (y_tensor, ordered_y_len_ls))


def get_data_loader(datafile_path, config):
    # Load files
    data_df = pd.read_csv(datafile_path, header=None, dtype={1: str})

    (_, quest_column), (_, answer_column) = data_df.iteritems()
    quest_data = NumberDataset(data_list=quest_column)
    answer_data = NumberDataset(data_list=answer_column)

    dataset = SequenceDataset(input_sequence=quest_data, output_sequence=answer_data)

    return torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=config.batch_size,
        shuffle=True,
        pin_memory=config.cuda,
        collate_fn=collate_fn,
    )

import torch
import numpy as np


def set_seed(config, seed=None):
    """Set seed for Torch, CUDA and NumPy"""
    seed = config.seed
    torch.manual_seed(seed)
    if config.cuda:
        torch.cuda.manual_seed(seed)
    np.random.seed(seed)


def random_draw_use_teaching_forcing(use_teacher_forcing_perc):
    return np.random.uniform(0, 1) < use_teacher_forcing_perc


def maybe_cuda(x, cuda):
    """Helper for converting to a Variable"""
    if cuda:
        x = x.cuda()
    return x

from typing import Tuple

import torch
from einops import reduce


def add_noise(data: torch.Tensor, noise_std: float) -> torch.Tensor:
    noise = torch.normal(0, noise_std, size=data.shape)
    return data + noise


def get_action_dim_maxmin(data: torch.Tensor) -> Tuple:
    r"""
    Get the maximum and minimum values of a data array along the last dim.
    Parameters
    ----------
    data : torch.Tensor
        Array with shape (batch, seq, dim).

    Returns
    -------
    tuple of torch.Tensor
        Tuple containing two arrays, each with shape (dim,).
        The first array contains the maximum values along the last dim,
        and the second array contains the minimum values along the last dim.
    """
    dim_max = reduce(data, "batch seq dim -> dim", "max")
    dim_min = reduce(data, "batch seq dim -> dim", "min")
    return dim_max, dim_min


def normalize_action(
    data: torch.Tensor, max_array: torch.Tensor, min_array: torch.Tensor
) -> torch.Tensor:
    r"""
    Normalize PyTorch tensor with given maximum and minimum values.

    Parameters
    ----------
    data : torch.Tensor
        The array-like object to be normalized.
        It must have the same shape as `max_array` and `min_array`.
    max_array : torch.Tensor
        The array-like object with maximum values to be used for normalization.
    min_array : torch.Tensor
        The array-like object with minimum values to be used for normalization.
    Returns
    -------
    torch.Tensor
        The normalized array-like object with the same type as the input.
    """
    copy_data = data.detach().clone()
    _, _, dim = copy_data.shape
    for d in range(dim):
        copy_data[:, :, d] -= min_array[d]
        copy_data[:, :, d] /= max_array[d] - min_array[d]
        copy_data[:, :, d] *= 2.0
        copy_data[:, :, d] -= 1.0
    return copy_data


def sliding_window_stacking(
    iterable: torch.Tensor, window_size: int, window_stride: int
) -> torch.Tensor:
    r"""Stack data into batch according to window_size and stride.

    Parameters
    ----------
    iterable : list of Tensor or Tensor
    window_size : int
        Window size.
    window_stride : int
        Width of window movement.
    Returns
    -------
    torch.Tensor
    """

    splits = []
    for data in iterable:
        state_batch_size = data.shape[0] - window_size + 1
        for slice_start in range(0, state_batch_size, window_stride):
            slice_end = slice_start + window_size
            splits.append(data[slice_start:slice_end])
    return torch.stack(splits, dim=0)

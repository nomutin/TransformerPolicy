import torch


def add_noise(data: torch.Tensor, noise_std: float) -> torch.Tensor:
    noise = torch.normal(0, noise_std, size=data.shape)
    return data + noise


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

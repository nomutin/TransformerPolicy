import matplotlib.pyplot as plt
import torch


def save_time_series_prediction(
    target: torch.Tensor, prediction: torch.Tensor, save_path: str
) -> None:
    """
    Save a plot of the time series multi-dimensional data.

    Parameters
    ----------
    target : torch.Tensor
        A 2D tensor of shape (batch_size, dims) representing the true outputs.
    prediction : torch.Tensor
        A 2D tensor of the same shape as target representing the prediction.
    save_path : str
        A string representing the location where to save the plot.

    Returns
    -------
    None
    """
    assert target.ndim == prediction.ndim == 2

    _, dim = target.shape
    fig, axes = plt.subplots(dim, 1, tight_layout=True)

    for i, axe in enumerate(axes):
        axe.plot(target[:, i], color="gray")
        axe.plot(prediction[:, i], color="blue")
        axe.set_title(f"Dim {dim}")

    fig.savefig(save_path)
    plt.clf()
    plt.close()
    print(f"Saved time series data on {save_path}")

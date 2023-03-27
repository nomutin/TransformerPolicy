import torch


def add_noise(data: torch.Tensor, noise_std: float) -> torch.Tensor:
    noise = torch.normal(0, noise_std, size=data.shape)
    return data + noise

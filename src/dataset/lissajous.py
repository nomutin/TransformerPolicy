import math
from typing import Tuple

import pytorch_lightning as pl
import torch
from einops import repeat
from omegaconf import DictConfig
from torch.utils.data import DataLoader, Dataset

from .process_data import add_noise


def create_lissajous_data(
    data_num: int, points_num: int, round_times: int
) -> torch.Tensor:
    t = torch.linspace(0, 2 * math.pi * round_times, points_num * round_times)
    x, y = torch.cos(t), torch.sin(2 * t)
    data = torch.stack([x, y], dim=-1)
    return repeat(data, "t d -> b t d", b=data_num)


class LissajousDataset(Dataset):
    def __init__(self, data: torch.Tensor) -> None:
        super().__init__()
        self.data = data
        self.input = self.data[:, :-1]
        self.target = self.data[:, 1:]

    def __len__(self) -> int:
        return len(self.input)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        input_data = add_noise(self.input[idx], 0.1)
        target_data = self.target[idx]
        return input_data, target_data


class LissajousDataModule(pl.LightningDataModule):
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()
        self.cfg = cfg

    def setup(self, stage: str = "train") -> None:
        self.data = create_lissajous_data(
            data_num=self.cfg.data_num,
            points_num=self.cfg.points_num,
            round_times=self.cfg.round_times,
        )
        noised_data = add_noise(self.data, self.cfg.noise_std)
        self.dataset = LissajousDataset(noised_data)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.dataset, self.cfg.batch_size, shuffle=True)

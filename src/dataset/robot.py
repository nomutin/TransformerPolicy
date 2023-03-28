import os
from typing import Tuple

import pytorch_lightning as pl
import requests
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader, Dataset

from .process_data import add_noise, sliding_window_stacking

DRIVE_FILES = {
    "fuji_put_ball_lv4": "19qVl1HiRzviqT414SLl-pKP392nyoJLx",
}


def download_small_file_from_drive(file_id: str, file_path: str) -> None:
    url = "https://drive.google.com/uc?export=download"
    chunk_size = 32768
    session = requests.Session()
    response = session.get(url, params={"id": file_id}, stream=True)
    with open(file_path, "wb") as f:
        for chunk in response.iter_content(chunk_size):
            if chunk:
                f.write(chunk)


def get_robot_data(data_name: str) -> torch.Tensor:
    file_id = DRIVE_FILES[data_name]
    path = os.path.join("data", data_name) + ".pt"
    if not os.path.isfile(path):
        download_small_file_from_drive(file_id, path)
    return torch.load(path)


class RobotDataset(Dataset):
    def __init__(
        self, data: torch.Tensor, window_size: int, window_stride: int
    ) -> None:
        super().__init__()
        self.data = data
        stacked_data = sliding_window_stacking(
            iterable=self.data,
            window_size=window_size,
            window_stride=window_stride,
        )
        self.input = stacked_data[:, :-1]
        self.target = stacked_data[:, 1:]

    def __len__(self) -> int:
        return len(self.input)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        input_data = add_noise(self.input[idx], 0.3)
        target_data = self.target[idx]
        return input_data, target_data


class RobotDataModule(pl.LightningDataModule):
    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()
        self.cfg = cfg

    def setup(self, stage: str = "train") -> None:
        self.data = get_robot_data(self.cfg.data_name)
        self.dataset = RobotDataset(
            data=self.data,
            window_size=self.cfg.window_size,
            window_stride=self.cfg.window_stride,
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.dataset, self.cfg.batch_size, shuffle=True)

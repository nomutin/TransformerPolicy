from typing import Dict, List

import torch
from omegaconf import DictConfig
from pytorch_lightning.utilities.types import STEP_OUTPUT
from torch import nn

from .base import PolicyBase
from .distributions import GMM


class RNNPolicy(PolicyBase):
    def __init__(self, cfg: DictConfig) -> None:
        """Recurrent Neural Network (GRU) Policy

        GRU + fc * num_hidden_layers + gmm

        Parameters
        ----------
        cfg : DictConfig
        """
        super().__init__(cfg=cfg)
        self.cfg = cfg
        self.rnn = self.build_rnn()

    def build_rnn(self) -> nn.GRU:
        return nn.GRU(
            input_size=self.cfg.input_size,
            hidden_size=self.cfg.hidden_size,
            batch_first=True,
        )

    def init_hidden(self, batch_size: int, device: torch.device) -> None:
        """Initialize hidden state

        References
        ----------
        * https://pytorch.org/docs/stable/generated/torch.nn.GRU.html

        Parameters
        ----------
        batch_size : int
        device : torch.device
        """
        size = [1, batch_size, self.cfg.hidden_size]
        self.hidden_state = torch.zeros(size, device=device)

    def forward(self, in_state: torch.Tensor) -> GMM:
        hidden, self.hidden_state = self.rnn(in_state, self.hidden_state)
        gaussian_mixture: GMM = self.hidden_to_gmm(hidden)
        return gaussian_mixture

    def training_step(self, batch: List, **kwargs: Dict) -> STEP_OUTPUT:
        inputs, targets = batch
        self.init_hidden(inputs.shape[0], device=self.device)  # type: ignore
        outputs = self.forward(inputs)
        loss_dict = self.calc_loss(outputs, targets)
        self.log("loss", loss_dict["loss"])
        return loss_dict

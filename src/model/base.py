from typing import List

import pytorch_lightning as pl
import torch
from omegaconf import DictConfig
from torch import nn

from .distributions import GMM


class MLP(nn.Module):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        hidden_size: int,
        activation: str,
        out_activation: str,
        num_hidden_layers: int,
    ) -> None:
        """Multi-Layer Perceptron

        Parameters
        ----------
        input_size : int
            Model input size
        output_size : int
            Model output size
        hidden_size : int
            MLP hidden size
        activation : str
            Non-linear activation function
        out_activation : str
            Model output activation function
        num_hidden_layers : int
            Number of hidden layers.
        """

        super().__init__()
        self.in_size = input_size
        self.out_size = output_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.activation = getattr(nn, str(activation))
        self.out_activation = getattr(nn, str(out_activation), nn.Identity)
        self.model = self._build_model()

    def _build_model(self) -> nn.Sequential:
        seq: List[nn.Module] = [nn.Linear(self.in_size, self.hidden_size)]
        seq += [self.activation()]
        for _ in range(self.num_hidden_layers):
            seq += self._build_hidden_layer()
        seq += [nn.Linear(self.hidden_size, self.out_size)]
        seq += [self.out_activation()]
        return nn.Sequential(*seq)

    def _build_hidden_layer(self) -> List:
        layer: List[nn.Module] = [
            nn.Linear(self.hidden_size, self.hidden_size)
        ]
        layer += [self.activation()]
        return layer

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.model(features)


class PolicyBase(pl.LightningModule):
    """
    Base class for pytorch-lightning.LightningModule
    """

    def __init__(self, cfg: DictConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.mlp = self.build_mlp()
        self.training_step_outputs: List = []

    def configure_optimizers(self) -> torch.optim.Optimizer:
        optimizer = getattr(torch.optim, self.cfg.optimizer)
        return optimizer(self.parameters(), lr=self.cfg.lr)

    def calc_loss(self, outputs: GMM, targets: torch.Tensor) -> torch.Tensor:
        return -outputs.log_prob(targets).mean()

    def on_train_epoch_end(self) -> None:
        epoch_loss = torch.stack(self.training_step_outputs).mean()
        self.log("step", self.current_epoch)
        self.log("loss", epoch_loss)
        self.training_step_outputs.clear()

    def build_mlp(self) -> MLP:
        return MLP(
            input_size=self.cfg.hidden_size,
            output_size=self.cfg.output_size * self.cfg.num_mix * 3,
            hidden_size=self.cfg.hidden_size,
            num_hidden_layers=self.cfg.num_hidden_layers,
            activation=self.cfg.activation,
            out_activation=self.cfg.out_activation,
        )

    def hidden_to_gmm(self, hidden: torch.Tensor) -> GMM:
        mlp_output = self.mlp(hidden)
        weighting, mean, std = torch.chunk(mlp_output, dim=-1, chunks=3)
        mixed_shape = [*weighting.shape[:-1], -1, self.cfg.num_mix]
        probs = torch.softmax(weighting.reshape(mixed_shape), dim=-1)
        mean = torch.tanh(mean.reshape(mixed_shape))
        std = nn.functional.softplus(std.reshape(mixed_shape)) + 0.1
        return GMM(probs=probs, mean=mean, std=std)

    def load(self, model_name: str) -> "PolicyBase":
        path = f"models/{model_name}/{self.__class__.__name__}.ckpt"
        state_dict = torch.load(path)["state_dict"]
        self.load_state_dict(state_dict)
        return self

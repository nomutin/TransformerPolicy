import math
from typing import Dict, List

import torch
from omegaconf import DictConfig
from pytorch_lightning.utilities.types import STEP_OUTPUT
from torch import nn

from .base import PolicyBase


class PositionalEncoder(nn.Module):
    """
    Embed location information in input

    Unlike RNNs, Transformers cannot retain position information,
    so trigonometric functions are used to embed position information.
    [Vaswani+ 2017]

    References
    ----------
    * https://pytorch.org/tutorials/beginner/transformer_tutorial.html
    * https://qiita.com/shiba_inu_/items/197f0f48587ed12e591f
    """

    def __init__(
        self,
        d_model: int,
        max_seq_len: int,
        dropout: float = 0.0,
    ):
        """
        Parameters
        ----------
        d_model: int
            The dimension of the output of sub-layers (Vaswani et al, 2017)
        max_seq_len: int
            the maximum length of the input sequences
        dropout: float
            the dropout rate
        """

        super().__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len
        self.dropout = nn.Dropout(p=dropout)
        self.build_positional_encoding()

    def build_positional_encoding(self) -> None:
        position = torch.arange(self.max_seq_len).unsqueeze(1)
        mul_term = -math.log(10000.0) / self.d_model
        div_term = (torch.arange(0, self.d_model, 2) * mul_term).exp()
        pe = torch.zeros(self.max_seq_len, 1, self.d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.transpose(0, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[: x.size(1)]  # type: ignore
        return self.dropout(x)


class GPTLayer(nn.Module):
    """
    Single Layer of GPT (Generative Pre-Trained transformer).

    This class is practically equivalent to `torch.nn.TransformerDecoderLayer`,
    which has no memory argument & always uses a causal mask.
    Note that this is Nomura's own modification of
    `torch/nn/modules/transformer.py/TransformerDecoderLayer()`
    and may contain errors.

    References
    ----------
    * Attention is All You Need [Vaswani+ 2017]
    * Improving Language Understanding by Generative Pre-Train [Radford+ 2018]
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        activation: str = "ReLU",
    ) -> None:
        """

        Parameters
        ----------
        d_model : int
            the number of expected features in the input (required).
        nhead : int
            the number of heads in the multiheadattention models (required).
        dim_feedforward : int, optional
            the dimension of the feedforward network model, by default 2048
        dropout : float, optional
            the dropout value, by default 0.1
        activation : str, optional
            the activation of the intermediate layer, by default "ReLU".
        """
        super().__init__()
        self.self_attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True,
        )
        self.activation = getattr(nn, activation)
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            self.activation(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout),
        )
        self.layer_norms = nn.ModuleList(
            [nn.LayerNorm(d_model) for _ in range(2)]
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layer_norms[0](x + self.sa_block(x))
        x = self.layer_norms[1](x + self.ff_block(x))
        return x

    def sa_block(self, x: torch.Tensor) -> torch.Tensor:
        x, *_ = self.self_attention(
            query=x, key=x, value=x, is_causal=True, need_weights=False
        )
        return self.dropout(x)

    def ff_block(self, x: torch.Tensor) -> torch.Tensor:
        return self.feed_forward(x)


class TransformerPolicy(PolicyBase):
    """
    Policy with Transformer Encoder and GMM (`a_t ~ π (a_t ~ | a_{1:t-1})`).
    """

    def __init__(self, cfg: DictConfig) -> None:
        super().__init__(cfg=cfg)
        self.input_embedding_layer = nn.Linear(
            in_features=self.cfg.input_size, out_features=self.cfg.hidden_size
        )
        self.positional_encoder = PositionalEncoder(
            d_model=self.cfg.hidden_size, max_seq_len=self.cfg.max_seq_len
        )

    def training_step(self, batch: List, **kwargs: Dict) -> STEP_OUTPUT:
        inputs, targets = batch
        outputs = self.forward(inputs)
        loss = self.calc_loss(outputs, targets)
        self.training_step_outputs.append(loss)
        return loss

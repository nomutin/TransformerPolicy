import math
from typing import Dict, List

import torch
from omegaconf import DictConfig
from pytorch_lightning.utilities.types import STEP_OUTPUT
from torch import nn

from .base import PolicyBase
from .distributions import GMM


class PositionalEncoder(nn.Module):
    r"""
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

    def build_positional_encoding(self) -> None:
        position = torch.arange(self.max_seq_len).unsqueeze(1)
        mul_term = -math.log(10000.0) / self.d_model
        div_term = (torch.arange(0, self.d_model, 2) * mul_term).exp()
        pe = torch.zeros(self.max_seq_len, 1, self.d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.transpose(0, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[: x.size(1)]
        return self.dropout(x)


def generate_square_subsequent_mask(dim1: int, dim2: int) -> torch.Tensor:
    """
    Generates an upper-triangular matrix of -inf, with zeros on diag.

    Implementation of `look-ahead masking` such as BERT or
    causal transformer, to avoid referring to future information.

    References
    ----------
    * https://pytorch.org/tutorials/beginner/transformer_tutorial.html
    * https://a16mixx.com/企画職向けdl論文解説：transformerを使った強化学習：decision-transfo/
    * https://arxiv.org/pdf/1810.04805.pdf

    Parameters
    ----------
    dim1: int
        This must be target sequence length
    dim2: int
        This must be encoder sequence length
        (i.e. the length of the input sequence to the model),

    Returns
    -------
    torch.Tensor
        A Tensor of shape [dim1, dim2]
    """
    return torch.triu(torch.ones(dim1, dim2) * float("-inf"), diagonal=1)


class TransformerPolicy(PolicyBase):
    """
    Policy with Transformer Encoder and GMM (`a_t ~ π (a_t ~ | a_{1:t-1})`).

    References
    ----------
    * https://pytorch.org/docs/stable/generated/torch.nn.TransformerEncoder
    """

    def __init__(self, cfg: DictConfig) -> None:
        super().__init__(cfg=cfg)
        self.input_embedding_layer = nn.Linear(
            in_features=self.cfg.input_size, out_features=self.cfg.hidden_size
        )
        self.positional_encoder = PositionalEncoder(
            d_model=self.cfg.hidden_size, max_seq_len=self.cfg.max_seq_len
        )
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=self.cfg.hidden_size,
            nhead=self.cfg.n_heads,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=self.cfg.num_transformer_layers,
            norm=None,
        )

    def forward(self, in_state: torch.Tensor) -> GMM:
        seq_len = in_state.shape[1]
        embed = self.input_embedding_layer(in_state)
        pos_embed = self.positional_encoder(embed)
        mask = generate_square_subsequent_mask(seq_len, seq_len)
        hidden = self.encoder(pos_embed, mask=mask)
        gaussian_mixture = self.hidden_to_gmm(hidden)
        return gaussian_mixture

    def training_step(self, batch: List, **kwargs: Dict) -> STEP_OUTPUT:
        inputs, targets = batch
        outputs = self.forward(inputs)
        loss_dict = self.calc_loss(outputs, targets)
        self.log("loss", loss_dict["loss"])
        return loss_dict

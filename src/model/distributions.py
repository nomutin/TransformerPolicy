import torch
import torch.distributions as td


class GMM:
    r"""
    Gaussian Mixture Model Implementation

    References
    ----------
    * https://geostatisticslessons.com/lessons/gmm
    """

    def __init__(
        self, probs: torch.Tensor, mean: torch.Tensor, std: torch.Tensor
    ) -> None:
        r"""Gaussian Mixture Model

        Parameters
        ----------
        probs : torch.Tensor
            [batch_size, seq_len, dim, num_mix] or [batch_size, dim, num_mix]
        mean : torch.Tensor
            same shape as probs and scale
        std : torch.Tensor
            same shape as probs and mean
        """
        self.probs = probs
        self.mean = mean
        self.std = std
        self.dist = self._mixture_of_gaussian()
        self.validate_parameters()

    def validate_parameters(self) -> None:
        assert self.probs.shape == self.mean.shape == self.std.shape
        assert self.probs.ndim in [3, 4]

    def sample(self) -> torch.Tensor:
        r"""Sample from GMM

        Parameters
        ----------
        size: None

        Returns
        -------
        torch.Tensor
        """
        return self.dist.sample()

    def log_prob(self, value: torch.Tensor) -> torch.Tensor:
        return self.dist.log_prob(value)

    def _mixture_of_gaussian(self) -> td.Independent:
        gaussian = td.Normal(self.mean, self.std)
        categorical_dist = td.Categorical(probs=self.probs)
        mixture_dist = td.MixtureSameFamily(
            mixture_distribution=categorical_dist,
            component_distribution=gaussian,
        )
        return td.Independent(
            base_distribution=mixture_dist, reinterpreted_batch_ndims=1
        )

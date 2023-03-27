# TransformerPolicy

[![black](https://img.shields.io/badge/code%20style-black-black.svg)](https://github.com/psf/black)
[![flake8](https://img.shields.io/badge/code%20style-flake8-black)](https://github.com/PyCQA/flake8)
[![isort](https://img.shields.io/badge/imports-isort-blue)](https://pycqa.github.io/isort/)
[![mypy](https://img.shields.io/badge/typing-mypy-blue)](https://github.com/python/mypy)

Transformer + GMM の方策（[Mimicplay](https://mimic-play.github.io)と同じもの）を作りたい

## Usage

### Install

```shell
poetry install
```

### Training

```shell
poetry run python src/train.py <model name>
```

### Test

```shell
poetry run python src/test.py <model name>
```

## References

- [Attention is All You Need [Vaswani+ 2017]](https://arxiv.org/pdf/1706.03762.pdf)
- [Improving Language Understanding by Generative Pre-Training [Radford+ 2018]](https://s3-us-west-2.amazonaws.com/openai-assets/research-covers/language-unsupervised/language_understanding_paper.pdf)
- [Mimicplay (web)](https://mimic-play.github.io)
- [Mimicplay (paper)](https://arxiv.org/pdf/2302.12422.pdf)
- [Pytorch - Transformer](https://pytorch.org/docs/stable/generated/torch.nn.Transformer.html)
- [Pytorch - TransformerDecoder](https://pytorch.org/docs/stable/generated/torch.nn.TransformerDecoder.html)
- [Examining the Transformer Architecture](https://towardsdatascience.com/examining-the-transformer-architecture-part-1-the-openai-gpt-2-controversy-feceda4363bb)
- [How to make a Transformer for time series forecasting with PyTorch](https://towardsdatascience.com/how-to-make-a-pytorch-transformer-for-time-series-forecasting-69e073d4061e)

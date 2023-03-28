import argparse

import pytorch_lightning as pl
import torch
import tqdm
from omegaconf import OmegaConf

import dataset
import model
from util import save_time_series_prediction


@torch.inference_mode()
def main(model_name: str) -> None:
    pl.seed_everything(42)
    open_loop_test(model_name)
    closed_loop_test(model_name)


def open_loop_test(model_name: str) -> None:
    cfg = OmegaConf.load(f"models/{model_name}/config.yaml")
    model_obj = getattr(model, cfg.model)(cfg.model_config).load(model_name)
    datamodule = getattr(dataset, cfg.datamodule)(cfg.dataset_config)
    datamodule.setup()
    inputs = datamodule.dataset.input

    if hasattr(model_obj, "init_hidden"):
        batch_size = inputs.shape[0]
        model_obj.init_hidden(batch_size, device="cpu")

    output = model_obj.forward(inputs)
    prediction = output.sample()

    for idx in cfg.test.open.batch_indices:
        save_path = f"reports/{model_name}/open_loop_test_{idx}.png"
        save_time_series_prediction(
            target=datamodule.dataset.target[idx],
            prediction=prediction[idx],
            save_path=save_path,
        )


def closed_loop_test(model_name: str) -> None:
    cfg = OmegaConf.load(f"models/{model_name}/config.yaml")
    model_obj = getattr(model, cfg.model)(cfg.model_config).load(model_name)
    datamodule = getattr(dataset, cfg.datamodule)(cfg.dataset_config)
    datamodule.setup()
    inputs = datamodule.dataset.input[:, 0:1, :]

    if isinstance(model_obj, model.RNNPolicy):
        predictions = rnn_closed_loop(
            model_obj=model_obj,
            inputs=inputs,
            generation_length=cfg.test.closed.generation_length,
        )
    elif isinstance(model_obj, model.TransformerPolicy):
        predictions = transformer_closed_loop(
            model_obj=model_obj,
            inputs=inputs,
            generation_length=cfg.test.closed.generation_length,
            block_size=cfg.model_config.block_size,
        )
    else:
        raise NotImplementedError

    for idx in cfg.test.closed.batch_indices:
        save_path = f"reports/{model_name}/closed_loop_test_{idx}.png"
        save_time_series_prediction(
            target=datamodule.dataset.target[idx],
            prediction=predictions[idx],
            save_path=save_path,
        )


def rnn_closed_loop(
    model_obj: model.RNNPolicy, inputs: torch.Tensor, generation_length: int
) -> torch.Tensor:
    batch_size = inputs.shape[0]
    model_obj.init_hidden(batch_size, device=torch.device("cpu"))
    prediction_list = [inputs]
    for _ in range(generation_length):
        output = model_obj.forward(prediction_list[-1])
        prediction_list.append(output.sample())
    predictions = torch.cat(prediction_list, dim=1)
    return predictions


def transformer_closed_loop(
    model_obj: model.TransformerPolicy,
    inputs: torch.Tensor,
    generation_length: int,
    block_size: int,
) -> torch.Tensor:
    for _ in tqdm.tqdm(range(generation_length)):
        model_input = inputs[:, -block_size:]
        output = model_obj.forward(model_input)
        prediction = output.sample()[:, -1:, :]
        inputs = torch.cat([inputs, prediction], dim=1)
    return inputs


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model_name", type=str)
    args = parser.parse_args()
    main(model_name=args.model_name)

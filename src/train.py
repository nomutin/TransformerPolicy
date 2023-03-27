import argparse

import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

import dataset
import model
from util import down_tensorboard_port, up_tensorboard_port


def main(model_name: str) -> None:
    pl.seed_everything(42)
    cfg = OmegaConf.load(f"models/{model_name}/config.yaml")
    try:
        up_tensorboard_port(model_name)
        training(
            model_name=model_name,
            model_object=getattr(model, cfg.model),
            cfg=cfg,  # type: ignore
            datamodule_object=getattr(dataset, cfg.datamodule),
        )
    finally:
        down_tensorboard_port()


def training(
    model_name: str,
    model_object: type,
    cfg: DictConfig,
    datamodule_object: type,
) -> None:
    model = model_object(cfg.model_config)
    datamodule = datamodule_object(cfg=cfg.dataset_config)
    save_top = ModelCheckpoint(
        save_weights_only=True,
        dirpath=f"models/{model_name}",
        filename=model.__class__.__name__,
        monitor="loss",
    )
    early_stopping = EarlyStopping(
        verbose=True,
        monitor="loss",
        patience=cfg.patience,
    )
    trainer = pl.Trainer(
        logger=TensorBoardLogger(f"reports/{model_name}"),
        max_epochs=cfg.epoch,
        deterministic=True,
        accelerator=cfg.accelerator,
        devices=cfg.devices,
        callbacks=[save_top, early_stopping],
        log_every_n_steps=5,
    )
    trainer.fit(model=model, datamodule=datamodule)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("model_name", type=str)
    args = parser.parse_args()
    main(model_name=args.model_name)

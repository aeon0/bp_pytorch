import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import ModelCheckpoint

from configs.config import TrainCfg
from data_module import DataModule
from model import SimpleClassifier


@hydra.main(config_path="configs", config_name="train", version_base=None)
def main(cfg_dict: DictConfig):
    cfg = TrainCfg.model_validate(cfg_dict)
    print("\n" + OmegaConf.to_yaml(cfg_dict) + "\n")

    checkpoint_cb = ModelCheckpoint(
        dirpath="checkpoints",
        filename="latest",
        save_last=True,
        save_top_k=0,
        every_n_epochs=1,
    )

    data_module = DataModule(
        batch_size=cfg.batch_size,
        num_workers=cfg.batch_size,
    )
    model = SimpleClassifier(lr=cfg.model.lr)
    trainer = pl.Trainer(
        max_epochs=cfg.max_epochs,
        accelerator=cfg.accelerator,
        devices=cfg.devices,
        callbacks=[checkpoint_cb],
    )
    trainer.fit(model, datamodule=data_module)


if __name__ == "__main__":
    main()

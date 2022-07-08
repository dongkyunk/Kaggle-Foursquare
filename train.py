import sys
import hydra
import pandas as pd
import pytorch_lightning as pl

from omegaconf import OmegaConf
from pytorch_lightning.plugins import DDPPlugin
from pytorch_lightning.loggers.neptune import NeptuneLogger
from pytorch_lightning.utilities.distributed import rank_zero_info
from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint

from dataset.four_square_datamodule import FourSquareDataModule
from model.four_square_model import FourSquareModel
from config import register_configs, Config


@hydra.main(config_path=None, config_name="config")
def train(cfg: Config) -> None:
    pl.seed_everything(cfg.trainer_cfg.seed)
    rank_zero_info(OmegaConf.to_yaml(cfg=cfg, resolve=True))
    pd.options.mode.chained_assignment = None 

    datamodule = FourSquareDataModule(cfg=cfg)
    num_of_classes = datamodule.get_num_of_classes()
    # model = FourSquareModel(cfg=cfg, num_of_classes=num_of_classes)
    model = FourSquareModel.load_from_checkpoint(cfg.path_cfg.lm_model_path, cfg=cfg, num_of_classes=num_of_classes)

    checkpoint_callback = ModelCheckpoint(
        filename='{epoch:02d}-{val_loss:.4f}',
        save_top_k=5,
        monitor='val_loss',
    )

    trainer_args = dict(
        gpus=cfg.trainer_cfg.gpus,
        val_check_interval=cfg.trainer_cfg.val_check_interval,
        num_sanity_val_steps=1,
        max_epochs=cfg.trainer_cfg.epoch,
        callbacks=[checkpoint_callback],
        strategy=DDPPlugin(find_unused_parameters=False),
        profiler="simple",
        precision=16,
    )    
    if cfg.neptune_cfg.use_neptune:
        logger = NeptuneLogger(
            project=cfg.neptune_cfg.project_name,
            name=cfg.neptune_cfg.exp_name,
            tags=list(cfg.neptune_cfg.tags),
            api_key='eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vYXBwLm5lcHR1bmUuYWkiLCJhcGlfdXJsIjoiaHR0cHM6Ly9hcHAubmVwdHVuZS5haSIsImFwaV9rZXkiOiIxOWY4YTFhZS00NGU5LTQxOTUtOGI5NC04ZjgwOTJkMDFmNjYifQ==',
            log_model_checkpoints=False
        )
        logger.log_hyperparams(params=cfg.trainer_cfg.__dict__)
        trainer_args['logger'] = logger

    trainer = pl.Trainer(**trainer_args)
    trainer.fit(model, datamodule)


if __name__ == "__main__":
    sys.argv.append(f'hydra.run.dir={Config.path_cfg.save_dir}')
    register_configs()
    train()

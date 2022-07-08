import os
import hydra
from typing import Optional
from dataclasses import dataclass, field
from hydra.core.config_store import ConfigStore


@dataclass
class NeptuneConfig:
    use_neptune: Optional[bool] = None
    project_name: str = "dongkyuk/four-square"
    exp_name: Optional[str] = "initial experiment"
    tags: Optional[tuple] = ('xlm-roberta-base', 'arcface')


@dataclass
class TrainerConfig:
    gpus: tuple = (0,1)
    num_workers: int = 4 * len(gpus)
    seed: int = 42
    pin_memory: bool = True
    persistent_workers: bool = True
    val_check_interval: float = 0.1

    epoch: int = 1000
    lr: float = 1e-5
    train_batch_size: int = 32
    val_batch_size: int = 96

    max_length: int = 256
    embd_dim: int = 224
    target_col: str = 'point_of_interest'
    model_name: str = 'xlm-roberta-base'
    # model_name: str = 'xlm-roberta-large'


@dataclass
class PathConfig:
    data_dir: str = "/home/dongkyun/Desktop/Other/Kaggle-Foursquare/data"
    lm_train_parquet_path: str = os.path.join(data_dir, "train.parquet")
    ml_train_parquet_path: str = os.path.join(data_dir, "train_0.parquet")
    ml_train_pair_parquet_path: str = os.path.join(data_dir, "train_0_pair.parquet")
    save_dir: Optional[str] = "save"
    train_1_lm_model_path: str = '/home/dongkyun/Desktop/Other/Kaggle-Foursquare/save/initial experiment/.neptune/initial experiment/FOUR-16/checkpoints/epoch=17-val_loss=3.2474.ckpt'
    train_0_lm_model_path: str = '/home/dongkyun/Desktop/Other/Kaggle-Foursquare/save/initial experiment/.neptune/initial experiment/FOUR-9/checkpoints/epoch=12-val_loss=3.6144.ckpt'
    lm_model_path: str = '/home/dongkyun/Desktop/Other/Kaggle-Foursquare/save/initial experiment/.neptune/initial experiment/FOUR-17/checkpoints/epoch=15-val_loss=4.0110.ckpt'

@dataclass
class Config:
    neptune_cfg: NeptuneConfig = NeptuneConfig()
    trainer_cfg: TrainerConfig = TrainerConfig()
    path_cfg: PathConfig = PathConfig()
    path_cfg.save_dir = os.path.join(path_cfg.save_dir, neptune_cfg.exp_name)
    os.makedirs(path_cfg.save_dir, exist_ok=True)


def register_configs() -> None:
    cs = ConfigStore.instance()
    cs.store(name="config", node=Config)

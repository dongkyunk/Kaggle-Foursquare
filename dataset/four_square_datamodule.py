import pandas as pd
import pytorch_lightning as pl

from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from sklearn.preprocessing import LabelEncoder

from config import Config
from dataset.four_square_dataset import FourSquareDataset
from utils.utils import get_n_sample_from_each_class_with_minimum_m


class FourSquareDataModule(pl.LightningDataModule):
    def __init__(self, cfg: Config):
        super(FourSquareDataModule, self).__init__()
        self.cfg = cfg
        self.df = pd.read_parquet(self.cfg.path_cfg.lm_train_parquet_path)
        self.tokenizer = AutoTokenizer.from_pretrained(self.cfg.trainer_cfg.model_name)

    def get_num_of_classes(self):
        return self.df.point_of_interest.nunique()

    def _preprocess(self):
        self.df = self.df.fillna('')
        encoder = LabelEncoder()
        self.df['point_of_interest'] = encoder.fit_transform(self.df['point_of_interest'])

    def setup(self, stage):
        self._preprocess()
        val_df = get_n_sample_from_each_class_with_minimum_m(1, 3, self.df, 'point_of_interest', 42)
        train_df = self.df.drop(val_df.index)

        print(f"train len: {len(train_df)}")
        print(f"val len: {len(val_df)}")
        print(f"total len: {len(self.df)}")

        self.train_dataset = FourSquareDataset(self.cfg, train_df, self.tokenizer)
        self.val_dataset = FourSquareDataset(self.cfg, val_df, self.tokenizer)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.cfg.trainer_cfg.train_batch_size, num_workers=self.cfg.trainer_cfg.num_workers,
                          shuffle=True, pin_memory=self.cfg.trainer_cfg.pin_memory, persistent_workers=self.cfg.trainer_cfg.persistent_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.cfg.trainer_cfg.val_batch_size, num_workers=self.cfg.trainer_cfg.num_workers,
                          shuffle=False, pin_memory=self.cfg.trainer_cfg.pin_memory, persistent_workers=self.cfg.trainer_cfg.persistent_workers)

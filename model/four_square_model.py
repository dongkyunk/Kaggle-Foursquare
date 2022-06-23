import torch.nn as nn
import pytorch_lightning as pl
from transformers import AutoModelForSequenceClassification, AutoConfig
from transformers.optimization import get_cosine_schedule_with_warmup
from torch_optimizer import MADGRAD
from model.arcface import ArcFace


class FourSquareModel(pl.LightningModule):
    def __init__(self, cfg, num_of_classes):
        super(FourSquareModel, self).__init__()
        self.cfg = cfg
        self.lr = self.cfg.trainer_cfg.lr

        self.transformer = AutoModelForSequenceClassification.from_pretrained(cfg.trainer_cfg.model_name, num_labels=cfg.trainer_cfg.embd_dim)
        self.criterion = ArcFace(
            in_features=cfg.trainer_cfg.embd_dim,
            out_features=num_of_classes,
            scale_factor=30,
            margin=0.15,
            criterion=nn.CrossEntropyLoss()
        )

    def forward(self, ids, mask):
        embd = self.transformer(input_ids=ids.squeeze(), attention_mask=mask.squeeze()).logits
        return embd

    def shared_step(self, batch, batch_idx):
        embd = self(batch['ids'], batch['mask'])
        loss, _ = self.criterion(embd, batch["label"])
        return loss
    
    def training_step(self, batch, batch_idx):
        return self.shared_step(batch, batch_idx)
        
    def validation_step(self, batch, batch_idx):
        loss = self.shared_step(batch, batch_idx)
        self.log_dict({'val_loss': loss.detach()}, prog_bar=True)

    def configure_optimizers(self):
        optimizer = MADGRAD(self.parameters(), lr=self.lr)
        scheduler = get_cosine_schedule_with_warmup(optimizer, 2000, 1000000)
        return [optimizer], [scheduler]
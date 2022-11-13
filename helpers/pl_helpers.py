import torch
import torch.nn as nn
import pytorch_lightning as pl
import os
import monai.data as m_data

from ModelFusion.helpers.utils import load_yml_file
from monai.metrics import DiceMetric
from torchmetrics import MetricCollection, Accuracy
from monai.losses import DiceCELoss
from monai.optimizers import Novograd
from monai.transforms import (
    Compose,
    AsDiscrete
)
from torch.optim.lr_scheduler import ReduceLROnPlateau
from pytorch_lightning.loggers import TensorBoardLogger
from monai.utils import CommonKeys
from datetime import datetime


dirname = os.path.dirname(os.path.dirname(__file__))
TRAINER_CONFIGS = load_yml_file(os.path.join(dirname, "configs/trainer_configs.yml"))


def get_logger(save_dir=".", name="lightning_logs"):
    time_stamp = datetime.now().strftime("%Y_%m_%d_%H_%M_%S_%f")
    logger = TensorBoardLogger(save_dir=save_dir, name=name, version=time_stamp)

    return logger, time_stamp


class TrainSeg(pl.LightningModule):
    def __init__(self, ds_dict, model):
        """
        ds_dict: keys: train, val, test
        """
        super(TrainSeg, self).__init__()
        self.model = model
        self.ds_dict = ds_dict
        self.batch_size = TRAINER_CONFIGS.Segmentation.batch_size
        self.lr = TRAINER_CONFIGS.Segmentation.lr
        self.num_epochs = TRAINER_CONFIGS.Segmentation.num_epochs
        self.loss_fn = DiceCELoss(
            include_background=False,
            to_onehot_y=True,
            softmax=True,
            squared_pred=True,
            lambda_dice=0.5,
            lambda_ce=0.5
        )
        self.dice_metric = DiceMetric(
            include_background=False
        )
        self.post_processing_img = Compose(
            AsDiscrete(argmax=True, to_onehot=TRAINER_CONFIGS.Segmentation.num_cls)
        )
        self.post_processing_label = Compose(
            AsDiscrete(to_onehot=TRAINER_CONFIGS.Segmentation.num_cls)
        )

    def training_step(self, batch, batch_idx):
        # (B, C, H, W), (B, 1, H, W)
        img, label = batch[CommonKeys.IMAGE], batch[CommonKeys.LABEL]
        pred = self.model(img)  # (B, num_cls, H, W)
        loss = self.loss_fn(pred, label)

        out_dict = {"loss": loss}
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)

        return out_dict

    def validation_step(self, batch, batch_idx):
        # (B, C, H, W), (B, 1, H, W)
        img, label = batch[CommonKeys.IMAGE], batch[CommonKeys.LABEL]
        pred = self.model(img)  # (B, num_cls, H, W)
        loss = self.loss_fn(pred, label)
        self.log("val_loss", loss, prog_bar=True, on_epoch=True)

        # [(num_cls, H, W)] for both
        pred_decollated = [self.post_processing_img(item_iter) for item_iter in m_data.decollate_batch(pred)]
        label_decollated = [self.post_processing_label(item_iter) for item_iter in m_data.decollate_batch(label)]
        self.dice_metric(y_pred=pred_decollated, y=label_decollated)

    def validation_epoch_end(self, outputs):
        dice_metric = self.dice_metric.aggregate().item()
        self.dice_metric.reset()
        self.log("val_dsc", dice_metric, prog_bar=True, on_epoch=True)

    def train_dataloader(self):
        ds_key = "train"
        ds = self.ds_dict[ds_key]
        loader = m_data.DataLoader(ds, batch_size=self.batch_size,
                                   num_workers=TRAINER_CONFIGS.Segmentation.num_workers, shuffle=True)

        return loader

    def val_dataloader(self):
        ds_key = "val"
        ds = self.ds_dict[ds_key]
        loader = m_data.DataLoader(ds, batch_size=self.batch_size,
                                   num_workers=TRAINER_CONFIGS.Segmentation.num_workers)

        return loader

    def configure_optimizers(self):
        opt = Novograd(self.model.parameters(), lr=self.lr)
        scheduler = ReduceLROnPlateau(opt, mode="max", patience=5, factor=0.5, min_lr=1e-6)
        opt_config = {
            "optimizer": opt,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_dsc"
            }
        }

        return opt_config


class TrainClf(pl.LightningModule):
    def __init__(self, ds_dict, model):
        """
        ds_dict: keys: train, val, test
        """
        super(TrainClf, self).__init__()
        self.model = model
        self.ds_dict = ds_dict
        self.batch_size = TRAINER_CONFIGS.Classification.batch_size
        self.lr = TRAINER_CONFIGS.Classification.lr
        self.num_epochs = TRAINER_CONFIGS.Classification.num_epochs
        self.loss_fn = nn.CrossEntropyLoss()
        metrics = MetricCollection([Accuracy()])
        self.train_metrics = metrics.clone(prefix="train_")
        self.val_metrics = metrics.clone(prefix="val_")

    def training_step(self, batch, batch_idx):
        # (B, C, H, W), (B,)
        img, label = batch
        pred, _ = self.model(img)  # (B, num_cls)
        loss = self.loss_fn(pred, label)

        out_dict = {"loss": loss}
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        metrics = self.train_metrics(pred, label)
        self.log_dict(metrics, prog_bar=True, on_step=True, on_epoch=True)

        return out_dict

    def validation_step(self, batch, batch_idx):
        # (B, C, H, W), (B, 1, H, W)
        img, label = batch
        pred, _ = self.model(img)  # (B, num_cls, H, W)
        loss = self.loss_fn(pred, label)
        self.val_metrics.update(pred, label)
        self.log("val_loss", loss, prog_bar=True, on_epoch=True)

    def training_epoch_end(self, outputs):
        metrics = self.train_metrics.compute()
        self.train_metrics.reset()
        self.log_dict(metrics, prog_bar=True, on_epoch=True)

    def validation_epoch_end(self, outputs):
        metrics = self.val_metrics.compute()
        self.val_metrics.reset()
        self.log_dict(metrics, prog_bar=True, on_epoch=True)

    def train_dataloader(self):
        ds_key = "train"
        ds = self.ds_dict[ds_key]
        loader = m_data.DataLoader(ds, batch_size=self.batch_size,
                                   num_workers=TRAINER_CONFIGS.Classification.num_workers, shuffle=True)

        return loader

    def val_dataloader(self):
        ds_key = "val"
        ds = self.ds_dict[ds_key]
        loader = m_data.DataLoader(ds, batch_size=self.batch_size,
                                   num_workers=TRAINER_CONFIGS.Classification.num_workers)

        return loader

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        scheduler = ReduceLROnPlateau(opt, mode="max", patience=5, factor=0.5, min_lr=1e-6)
        opt_config = {
            "optimizer": opt,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_Accuracy"
            }
        }

        return opt_config

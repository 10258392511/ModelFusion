import numpy as np
import torch
import pytorch_lightning as pl
import ModelFusion.helpers.pytorch_utils as ptu

from pytorch_lightning.callbacks import Callback
from monai.utils import CommonKeys


class ValVisualizationSeg(Callback):
    def __init__(self, save_interval=1):
        super(ValVisualizationSeg, self).__init__()
        self.epochs = 0
        self.save_interval = save_interval

    def on_validation_epoch_end(self, trainer: "pl.Trainer", pl_module: "pl.LightningModule") -> None:
        if not self.epochs % self.save_interval == 0:
            self.epochs += 1

            return

        idx = np.random.randint(len(pl_module.ds_dict["val"]))
        data = pl_module.ds_dict["val"][idx]
        # (C, H, W), (1, H, W)
        img, label = data[CommonKeys.IMAGE], data[CommonKeys.LABEL]
        img_input = img.unsqueeze(0).to(ptu.DEVICE)  # (1, C, H, W)
        model = pl_module.model
        pred = model(img_input).argmax(dim=1, keepdim=True).squeeze(0)  # (1, num_cls, H, W) -> (1, 1, H, W) -> (1, H, W)
        label = label.float() / label.max()
        pred = pred.detach().float()
        pred /= pred.max()
        for tag_name_iter, img_iter in zip(["img", "label", "pred"], [img, label, pred]):
            pl_module.logger.experiment.add_image(tag_name_iter, img_iter, global_step=self.epochs)

        self.epochs += 1

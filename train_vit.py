import sys
import os

# path = os.getcwd()
# for _ in range(2):
#     path = os.path.dirname(path)

# if not path in sys.path:
#     sys.path.append(path)

sys.path.append('/cluster/project/infk/cvg/students/junwang/')


import torch
import pytorch_lightning as pl
import os
import ModelFusion.helpers.pytorch_utils as ptu

from ModelFusion.helpers.load_data import load_data
from ModelFusion.helpers.load_model import load_model, reload_model
from ModelFusion.helpers.pl_helpers import TrainClf, get_logger
from pytorch_lightning.callbacks import ModelCheckpoint
from ModelFusion.helpers.utils import vis_images

ds_name = "CIFAR10"
model_name = "ViT"
save_dir = "../"
log_name = "clf_logs"
if_debug = False
accelerator = "gpu" if torch.cuda.is_available() else "cpu"

train_ds = load_data(ds_name, "train")
val_ds = load_data(ds_name, "val")


model = load_model(model_name)

ds_dict = {
    "train": train_ds,
    "val": val_ds
}
model = load_model(model_name)
lit_model = TrainClf(ds_dict, model)


logger, time_stamp = get_logger(save_dir, log_name)
callbacks = [
    ModelCheckpoint(os.path.join(save_dir, log_name, time_stamp, "checkpoints/"), monitor="val_Accuracy", mode="max")
]

if_debug = False
if if_debug:
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=1,
        fast_dev_run=2
    )
else:
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=1,
        logger=logger,
        callbacks=callbacks,
        num_sanity_val_steps=-1,
#         precision=16,
#         max_epochs=lit_model.num_epochs,
        max_epochs=200
    )

trainer.fit(lit_model)
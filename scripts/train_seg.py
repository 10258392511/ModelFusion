import sys
import os

# This script should be run at the project root: */ModelFusion

path = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

if not path in sys.path:
    sys.path.append(path)


import argparse
import torch
import pytorch_lightning as pl
import os

from ModelFusion.helpers.load_data import load_data
from ModelFusion.helpers.load_model import load_model
from ModelFusion.helpers.pl_helpers import TrainSeg, get_logger
from ModelFusion.helpers.pl_callbacks import ValVisualizationSeg
from pytorch_lightning.callbacks import ModelCheckpoint
from monai.utils import CommonKeys


if __name__ == '__main__':
    """
    python scripts/train_seg.py --vendor A --save_dir "./" --model_name "UNet"
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--vendor", required=True, choices=["A", "B"])
    parser.add_argument("--save_dir", default="./")
    parser.add_argument("--model_name", required=True, choices=["UNet", "SwinUNETR"])
    parser.add_argument("--if_debug", action="store_true")

    args = parser.parse_args()
    args = vars(args)
    args["ds_name"] = "MNMS"
    args["log_name"] = "seg_logs"
    args["accelerator"] = "gpu" if torch.cuda.is_available() else "cpu"

    train_ds = load_data(args["ds_name"], "train", vendor=args["vendor"])
    val_ds = load_data(args["ds_name"], "val", vendor=args["vendor"])

    ds_dict = {
        "train": train_ds,
        "val": val_ds
    }
    model = load_model(args["model_name"])
    lit_model = TrainSeg(ds_dict, model)
    logger, time_stamp = get_logger(args["save_dir"], args["log_name"])
    callbacks = [
        ValVisualizationSeg(save_interval=1),
        ModelCheckpoint(os.path.join(args["save_dir"], args["log_name"], time_stamp, "checkpoints/"),
                        monitor="val_dsc", mode="max")
    ]

    desc_dict_keys = ["ds_name", "vendor", "model_name"]

    log_dir = os.path.join(args["save_dir"], args["log_name"], time_stamp)
    if not os.path.isdir(log_dir):
        os.makedirs(log_dir)
    with open(os.path.join(log_dir, "desc.txt"), "w") as wf:
        for key in desc_dict_keys:
            wf.write(f"{key}: {args[key]}\n")

    if args["if_debug"]:
        trainer = pl.Trainer(
            accelerator="gpu",
            devices=1,
            fast_dev_run=2
        )
    else:
        trainer = pl.Trainer(
            accelerator=args["accelerator"],
            devices=1,
            logger=logger,
            callbacks=callbacks,
            num_sanity_val_steps=-1,
            # precision=16,
            max_epochs=lit_model.num_epochs
            # max_epochs=20
        )

    trainer.fit(lit_model)

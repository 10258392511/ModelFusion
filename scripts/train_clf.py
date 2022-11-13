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
import ModelFusion.helpers.pytorch_utils as ptu

from ModelFusion.helpers.load_data import load_data
from ModelFusion.helpers.load_model import load_model, reload_model
from ModelFusion.helpers.pl_helpers import TrainClf, get_logger
from pytorch_lightning.callbacks import ModelCheckpoint


if __name__ == '__main__':
    """
   python scripts/train_clf.py --save_dir "./" --model_name "ViT"
   """
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_dir", default="./")
    parser.add_argument("--model_name", required=True, choices=["ViT"])
    parser.add_argument("--if_debug", action="store_true")

    args = parser.parse_args()
    args = vars(args)
    args["ds_name"] = "CIFAR10"
    args["log_name"] = "clf_logs"
    args["accelerator"] = "gpu" if torch.cuda.is_available() else "cpu"

    train_ds = load_data(args["ds_name"], "train")
    val_ds = load_data(args["ds_name"], "val")

    ds_dict = {
        "train": train_ds,
        "val": val_ds
    }
    model = load_model(args["model_name"])
    lit_model = TrainClf(ds_dict, model)
    logger, time_stamp = get_logger(args["save_dir"], args["log_name"])
    callbacks = [
        ModelCheckpoint(os.path.join(args["save_dir"], args["log_name"], time_stamp, "checkpoints/"),
                        monitor="val_Accuracy", mode="max")
    ]

    desc_dict_keys = ["ds_name", "model_name"]

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
            # max_epochs=5
        )

    trainer.fit(lit_model)

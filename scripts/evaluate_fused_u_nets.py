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

from torch.utils.data import Subset
from ModelFusion.fusion.fuse_u_net import fuse_u_nets
from ModelFusion.helpers.load_data import load_data
from ModelFusion.helpers.load_model import reload_model
from ModelFusion.helpers.pl_helpers import TrainSeg, get_logger
from ModelFusion.helpers.pl_callbacks import ValVisualizationSeg
from pytorch_lightning.callbacks import ModelCheckpoint
from monai.utils import CommonKeys


if __name__ == '__main__':
    """
    python scripts/evaluate_fused_u_nets.py --vendor A --save_dir "./" --num_retrain_samples 5 --num_retrain_epochs 1 --model1_path "./seg_logs/2022_11_22_02_40_15_864295/" --model2_path "./seg_logs/2022_11_22_02_48_26_706789/"
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--vendor", required=True, choices=["A", "B"])
    parser.add_argument("--save_dir", default="./")
    parser.add_argument("--num_retrain_samples", type=int, default=0)
    parser.add_argument("--num_retrain_epochs", type=int, default=0)
    parser.add_argument("--model1_path", required=True)
    parser.add_argument("--model2_path", required=True)

    args = parser.parse_args()
    args = vars(args)
    args["ds_name"] = "MNMS"
    args["model_name"] = "UNet"
    args["log_name"] = "u_net_eval_logs"
    args["accelerator"] = "gpu" if torch.cuda.is_available() else "cpu"
    model1_path = args["model1_path"]
    model2_path = args["model2_path"]

    model1 = reload_model(args["model_name"], model1_path)
    model2 = reload_model(args["model_name"], model2_path)
    model = fuse_u_nets(model1_path, model2_path)

    train_ds = load_data(args["ds_name"], "train", vendor=args["vendor"])
    val_ds = load_data(args["ds_name"], "val", vendor=args["vendor"])
    if args["num_retrain_samples"] > 0:
        train_indices = torch.arange(args["num_retrain_samples"])
        train_ds = Subset(train_ds, train_indices)

    ds_dict = {
        "train": train_ds,
        "val": val_ds
    }
    lit_model = TrainSeg(ds_dict, model)
    logger, time_stamp = get_logger(args["save_dir"], args["log_name"])
    callbacks = [
        ValVisualizationSeg(save_interval=1),
        ModelCheckpoint(os.path.join(args["save_dir"], args["log_name"], time_stamp, "checkpoints/"),
                        monitor="val_dsc", mode="max")
    ]

    log_dir = os.path.join(args["save_dir"], args["log_name"], time_stamp)
    if not os.path.isdir(log_dir):
        os.makedirs(log_dir)
    with open(os.path.join(log_dir, "desc.txt"), "w") as wf:
        for key in args:
            wf.write(f"{key}: {args[key]}\n")


    trainer = pl.Trainer(
        accelerator=args["accelerator"],
        devices=1,
        logger=logger,
        callbacks=callbacks,
        num_sanity_val_steps=-1,
        max_epochs=args["num_retrain_epochs"]
    )

    if args["num_retrain_epochs"] > 0:
        trainer.fit(lit_model)

    val_metrics = trainer.validate(lit_model)[0]
    with open(os.path.join(log_dir, "desc.txt"), "a") as wf:
        wf.write("\nval_metrics:\n")
        for key in val_metrics:
            wf.write(f"{key}: {val_metrics[key]}\n")

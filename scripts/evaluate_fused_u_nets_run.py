import sys
import os

# This script should be run at the project root: */ModelFusion

path = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

if not path in sys.path:
    sys.path.append(path)


ROOT_DIR = os.path.dirname(os.path.dirname((__file__)))
GENERAL_CONFIG_PATH = os.path.join(ROOT_DIR, "configs/general_configs.yml")
FUSION_CONFIG_PATH = os.path.join(ROOT_DIR, "configs/fusion_configs.yml")


import argparse
import torch
import pytorch_lightning as pl
import os

from torch.utils.data import Subset
from ModelFusion.fusion.fuse_u_net import fuse_u_nets
from ModelFusion.helpers.load_data import load_data
from ModelFusion.helpers.load_model import reload_model, average_two_models
from ModelFusion.helpers.pl_helpers import TrainSeg, get_logger
from ModelFusion.helpers.pl_callbacks import ValVisualizationSeg
from pytorch_lightning.callbacks import ModelCheckpoint
from monai.utils import CommonKeys
from ModelFusion.helpers.utils import (
    load_yml_file,
    obtain_downloaded_weight_paths,
    namespace2dict,
    dict2namespace
)


if __name__ == '__main__':
    """
    python scripts/evaluate_fused_u_nets_run.py --vendor A --save_dir "./" --exp_name "domain_generalization" --num_retrain_epochs 1 --ensemble_step 0.7 --square_factor "1/5" --retrain_fraction 0.1
    or
    python scripts/evaluate_fused_u_nets_run.py --vendor A --save_dir "./" --exp_name "domain_generalization" --num_retrain_epochs 1 --ensemble_step 0.7 --square_factor "1/5" --retrain_fraction 0.1 --model1_path "./seg_logs/2022_11_25_18_42_15_582775/" --model2_path "./seg_logs/2022_12_11_22_07_06_449724/"
    """
    config = load_yml_file(GENERAL_CONFIG_PATH)
    config_dict = namespace2dict(config)
    fusion_config = load_yml_file(FUSION_CONFIG_PATH)
    # print(config_dict)
    parser = argparse.ArgumentParser()
    parser.add_argument("--vendor", required=True, choices=["A", "B"])
    parser.add_argument("--save_dir", default="./")
    # parser.add_argument("--num_retrain_samples", type=float, default=0)
    parser.add_argument("--num_retrain_epochs", type=int, default=0)
    parser.add_argument("--ensemble_step", type=float, default=0.7)
    parser.add_argument("--square_factor", type=str, default="1/2")
    parser.add_argument("--retrain_fraction", type=float, default=0)
    parser.add_argument("--model1_path", default=None)
    parser.add_argument("--model2_path", default=None)
    parser.add_argument("--exp_name", choices=["domain_generalization",
                                               "data_parallel_A",
                                               "data_parallel_B"])

    args = parser.parse_args()
    args = vars(args)
    args["ds_name"] = "MNMS"
    args["model_name"] = "UNet"
    args["log_name"] = "u_net_eval_logs"
    args["accelerator"] = "gpu" if torch.cuda.is_available() else "cpu"

    if args["model1_path"] is not None and args["model2_path"] is not None:
        model1_path = args["model1_path"]
        model2_path = args["model2_path"]
    else:
        # dict
        model_paths = obtain_downloaded_weight_paths(config_dict["weights"], "UNet", args["exp_name"],
                                                     os.path.join(ROOT_DIR, "results/"))
        model1_path = model_paths["model_1"]
        model2_path = model_paths["model_2"]
        # print(f"{model1_path}, {model2_path}")

    model1 = reload_model(args["model_name"], model1_path)
    model2 = reload_model(args["model_name"], model2_path)
    naive_avg_state_dict = average_two_models(model1, model2, fusion_config.ensemble_step)

    fusion_config.ensemble_step = args["ensemble_step"]
    fusion_config.square_factor = args["square_factor"]
    if fusion_config.ensemble_step < 0.5:
        fusion_config.ensemble_step = 1 - fusion_config.ensemble_step
        model = fuse_u_nets(model2_path, model1_path, fusion_config)
    else:
        model = fuse_u_nets(model1_path, model2_path, fusion_config)

    train_ds_dict, val_ds_dict = {}, {}
    train_ds_dict["A"] = load_data(args["ds_name"], "train", vendor="A")
    val_ds_dict["A"] = load_data(args["ds_name"], "val", vendor="A")
    train_ds_dict["B"] = load_data(args["ds_name"], "train", vendor="B")
    val_ds_dict["B"] = load_data(args["ds_name"], "val", vendor="B")
    train_ds = train_ds_dict[args["vendor"]]
    val_ds = val_ds_dict[args["vendor"]]

    if args["retrain_fraction"] > 0:
        num_retrain_samples = int(args["retrain_fraction"] * len(train_ds))
        train_indices = torch.arange(num_retrain_samples)
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

    print("OT fusion:")
    val_metrics = trainer.validate(lit_model)[0]
    with open(os.path.join(log_dir, "desc.txt"), "a") as wf:
        wf.write("\nval_metrics:\n")
        for key in val_metrics:
            wf.write(f"{key}: {val_metrics[key]}\n")

    # naive average
    model1.load_state_dict(naive_avg_state_dict)
    model1.eval()
    lit_model.model = model1
    print("Naive averaging:")
    val_metrics = trainer.validate(lit_model)[0]
    with open(os.path.join(log_dir, "desc.txt"), "a") as wf:
        wf.write("\nval_metrics for naive average:\n")
        for key in val_metrics:
            wf.write(f"{key}: {val_metrics[key]}\n")

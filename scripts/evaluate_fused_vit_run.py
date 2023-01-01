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
from ModelFusion.fusion.fuse_vit import fuse_vits
from ModelFusion.helpers.load_data import load_data
from ModelFusion.helpers.load_model import reload_model, average_two_models
from ModelFusion.helpers.pl_helpers import TrainClf, get_logger
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
    python scripts/evaluate_fused_vit_run.py --save_dir "./" --exp_name "data_parallel" --num_retrain_epochs 1 --ensemble_step 0.7 --square_factor "1/5" --retrain_fraction 0.1
    or
    python scripts/evaluate_fused_vit_run.py --save_dir "./" --exp_name "data_parallel" --num_retrain_epochs 1 --ensemble_step 0.7 --square_factor "1/5" --retrain_fraction 0.1 --model1_path "./clf_logs/2022_12_13_23_55_00_073244" --model2_path "./clf_logs/2022_12_13_23_55_00_086375"
    """
    config = load_yml_file(GENERAL_CONFIG_PATH)
    config_dict = namespace2dict(config)
    fusion_config = load_yml_file(FUSION_CONFIG_PATH)
    # print(config_dict)
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_dir", default="./")
    # parser.add_argument("--num_retrain_samples", type=float, default=0)
    parser.add_argument("--num_retrain_epochs", type=int, default=0)
    parser.add_argument("--ensemble_step", type=float, default=0.7)
    parser.add_argument("--square_factor", type=str, default="1/2")
    parser.add_argument("--retrain_fraction", type=float, default=0)
    parser.add_argument("--model1_path", default=None)
    parser.add_argument("--model2_path", default=None)
    parser.add_argument("--exp_name", choices=["data_parallel"])

    args = parser.parse_args()
    args = vars(args)
    args["ds_name"] = "CIFAR10"
    args["model_name"] = "ViT"
    args["log_name"] = "vit_eval_logs"
    args["accelerator"] = "gpu" if torch.cuda.is_available() else "cpu"

    if args["model1_path"] is not None and args["model2_path"] is not None:
        model1_path = args["model1_path"]
        model2_path = args["model2_path"]
    else:
        # dict
        model_paths = obtain_downloaded_weight_paths(config_dict["weights"], "ViT", args["exp_name"],
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
        new_state_dict = fuse_vits(model2_path, model1_path, fusion_config)
    else:
        new_state_dict = fuse_vits(model1_path, model2_path, fusion_config)

    model1.load_state_dict(new_state_dict)
    model = model1
    model.eval()

    train_ds = load_data(args["ds_name"], "train")
    val_ds = load_data(args["ds_name"], "val")

    if args["retrain_fraction"] > 0:
        num_retrain_samples = int(args["retrain_fraction"] * len(train_ds))
        train_indices = torch.arange(num_retrain_samples)
        train_ds = Subset(train_ds, train_indices)

    ds_dict = {
        "train": train_ds,
        "val": val_ds
    }
    lit_model = TrainClf(ds_dict, model)
    logger, time_stamp = get_logger(args["save_dir"], args["log_name"])
    callbacks = [
        ModelCheckpoint(os.path.join(args["save_dir"], args["log_name"], time_stamp, "checkpoints/"),
                        monitor="val_Accuracy", mode="max")
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

import argparse
from collections import OrderedDict

import torch.nn as nn
import os
import sys
sys.path.append('/cluster/project/infk/cvg/students/junwang/')
sys.path.append('/cluster/project/infk/cvg/students/junwang/ModelFusion')

# import tensorflow
import torch
import pytorch_lightning as pl
from tqdm import tqdm
from torch.utils.data import Subset


from helpers.load_model import reload_model, load_model
from helpers.load_data import load_data
from helpers.model2graph import export_as_onnx
from wasserstein_ensemble_ViT import ViTFuser
from helpers.utils import load_yml_file, dict2namespace, create_filename, dict2filename

from ModelFusion.helpers.load_data import load_data
from ModelFusion.helpers.load_model import load_model, reload_model
from ModelFusion.helpers.pl_helpers import TrainClf, get_logger
from pytorch_lightning.callbacks import ModelCheckpoint

ROOT_DIR = __file__
for _ in range(2):
    ROOT_DIR = os.path.dirname(ROOT_DIR)

def fuse_vits(model1_path: str, model2_path: str, config):

    model_name = "ViT"

    # read_model
    model_1 = reload_model(model_name, model1_path)
    model_2 = reload_model(model_name, model2_path)

    # model_1 = model_1
    # model_2 = model_2

    print(model_1)

    for name, param in model_1.named_parameters():
        print(f"{name}: {param.shape}, {param.ndim}")

    fuser = ViTFuser([model_1, model_2], config)
    fused_weights = fuser()
    return fused_weights

def load_fused_weight(args):
    model = load_model(args["model_name"])
    model1_path = os.path.join(ROOT_DIR, 'clf_logs/2022_12_13_23_55_00_073244')
    model2_path = os.path.join(ROOT_DIR, 'clf_logs/2022_12_13_23_55_00_086375')
    config = load_yml_file(os.path.join(ROOT_DIR, "configs/fusion_configs.yml"))
    config = vars(config)
    config.update(args)
    config = dict2namespace(config)
    # fuse model
    fused_weights = fuse_vits(model1_path, model2_path, config)
    return fused_weights

def load_average_weight(args):
    model1_path = os.path.join(ROOT_DIR, 'clf_logs/2022_12_13_23_55_00_073244')
    model2_path = os.path.join(ROOT_DIR, 'clf_logs/2022_12_13_23_55_00_086375')
    average_weight = OrderedDict()
    model_1 = reload_model(args["model_name"], model1_path)
    model_2 = reload_model(args["model_name"], model2_path)
    weight_1 = model_1.state_dict()
    weight_2 = model_2.state_dict()
    for name, param in model_1.named_parameters():
        average_weight[name] = weight_1[name] * args['ensemble_step'] + weight_2[name] * (1 - args['ensemble_step'])
        average_weight[name] = average_weight[name]

    return average_weight

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_dir", default="./experiment/vit_retrain_log/", type=str)
    parser.add_argument("--model_name", default="ViT")
    parser.add_argument("--if_debug", action="store_true")
    parser.add_argument("--average", action='store_true')
    parser.add_argument("--ensemble_step", type=float, default=0.7)
    parser.add_argument("--square_factor", type=str, default="1/2")
    parser.add_argument("--retrain_fraction", type=float, default=0.1)

    args = parser.parse_args()
    args = vars(args)

    args["ds_name"] = "CIFAR10"
    args["log_name"] = "clf_logs"
    args["accelerator"] = "gpu" if torch.cuda.is_available() else "cpu"

    train_ds = load_data(args["ds_name"], "train")
    val_ds = load_data(args["ds_name"], "val")

    sample_interval = int(1/args["retrain_fraction"])
    train_ds = Subset(train_ds, range(0, len(train_ds), sample_interval))
    ds_dict = {"train": train_ds, "val": val_ds}

    model = load_model(args["model_name"])
    if not args['average']:
        print("using fused weight")
        fused_weight = load_fused_weight(args)
        args["log_name"] = "fusion" + str(args["retrain_fraction"])
    else:
        print("using average weight")
        fused_weight = load_average_weight(args)
        args["log_name"] = "average" +  str(args["retrain_fraction"])
    lit_model = TrainClf(ds_dict, model)
    print("Loading fused weight")
    lit_model.model.load_state_dict(fused_weight)
    print("Done!")
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

if __name__ == "__main__":
    main()

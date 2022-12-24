import torch.nn as nn
import os
import sys
sys.path.append('/home/junchi/deepl_learning')

import tensorflow
import torch
from helpers.load_model import reload_model
from helpers.model2graph import export_as_onnx
from wasserstein_ensemble_ViT import ViTFuser
from helpers.utils import load_yml_file

ROOT_DIR = __file__
for _ in range(2):
    ROOT_DIR = os.path.dirname(ROOT_DIR)


def fuse_vits(model1_path: str, model2_path: str):

    model_name = "ViT"
    model_paths = [model1_path, model2_path]
    models = []
    onnx_paths = []
    input_shape = (1, 3, 224, 224)

    # read config file
    config = load_yml_file(os.path.join(ROOT_DIR, "configs/fusion_configs.yml"))

    # read_model
    model_1 = reload_model(model_name, model1_path)
    model_2 = reload_model(model_name, model2_path)

    # model_1 = model_1
    # model_2 = model_2

    print(model_1)

    for name, param in model_1.named_parameters():
        print(f"{name}: {param.shape}, {param.ndim}")

    fuser = ViTFuser([model_2, model_1], config)
    fused_weights = fuser()
    # save fused weights
    torch.save(fused_weights, os.path.join(ROOT_DIR, "fused_weights.pth"))

def main():

    """
    model can be downloaded from: https://polybox.ethz.ch/index.php/s/FotPNs3BLC6Byzg
    """

    model1_path = os.path.join(ROOT_DIR, 'clf_logs/2022_12_13_23_55_00_073244')
    model2_path = os.path.join(ROOT_DIR, 'clf_logs/2022_12_13_23_55_00_086375')
    fuse_vits(model1_path, model2_path)

if __name__ == "__main__":
    main()
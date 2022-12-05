import torch.nn as nn
import os

from ModelFusion.helpers.load_model import reload_model
from ModelFusion.helpers.model2graph import export_as_onnx
from ModelFusion.wasserstein_ensemble_u_net import UNetFuser
from ModelFusion.helpers.utils import load_yml_file

ROOT_DIR = __file__
for _ in range(2):
    ROOT_DIR = os.path.dirname(ROOT_DIR)


def fuse_u_nets(model1_path: str, model2_path: str) -> nn.Module:
    """
    model1_path and model2_path should be the directory where the logs are stores.
    E.g. clf_logs/2022_11_13_02_20_43_443389
    """
    # export as .onnx file
    model_name = "UNet"
    model_paths = [model1_path, model2_path]
    models = []
    onnx_paths = []
    input_shape = (1, 1, 256, 256)

    for model_path_iter in [model1_path, model2_path]:
        onnx_path = os.path.join(model_path_iter, "model.onnx")
        onnx_paths.append(onnx_path)
        model = reload_model(model_name, model_path_iter)
        models.append(model)
        export_as_onnx(model, input_shape, onnx_path)

    # using UNetFuser
    configs = load_yml_file(os.path.join(ROOT_DIR, "configs/fusion_configs.yml"))
    fuser = UNetFuser(models, onnx_paths, configs)
    model_out = fuser()

    return model_out

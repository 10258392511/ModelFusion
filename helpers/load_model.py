import yaml
import torch.nn as nn
import os
import ModelFusion.helpers.pytorch_utils as ptu
import glob

from monai.networks.nets import UNet, SwinUNETR, ViT
from ModelFusion.helpers.pl_helpers import TrainSeg, TrainClf

dirname = os.path.dirname(os.path.dirname(__file__))
with open(os.path.join(dirname, "configs/general_configs.yml"), "r") as rf:
    GENERAL_CONFIGS = yaml.load(rf, yaml.Loader)


def load_model(model_name):
    assert model_name in REGISTERED_MODELS.keys(), "Unsupported model name!"
    loader = REGISTERED_MODELS[model_name]
    param_dict = GENERAL_CONFIGS["models"][model_name]
    model = loader(**param_dict)

    turn_off_bias(model)

    return model


def reload_model(model_name, model_log_dir):
    model = load_model(model_name)
    lit_model = REGISTERED_LITMODELS[model_name](None, model)
    ckpt_path = glob.glob(os.path.join(model_log_dir, "checkpoints/*ckpt"))[0]
    lit_model.load_from_checkpoint(ckpt_path, ds_dict=None, model=model)
    model = model.to(ptu.DEVICE)
    model.eval()

    return model


def turn_off_bias(model: nn.Module):
    for module in model.modules():
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
            module.bias = None


REGISTERED_MODELS = {
    "UNet": UNet,
    "SwinUNETR": SwinUNETR,
    "ViT": ViT
}


REGISTERED_LITMODELS = {
    "UNet": TrainSeg,
    "SwinUNETR": TrainSeg,
    "ViT": TrainClf
}

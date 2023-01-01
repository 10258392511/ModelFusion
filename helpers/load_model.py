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
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear) or isinstance(module, nn.ConvTranspose2d):
            module.bias = None


def average_two_models(model_1: nn.Module, model_2: nn.Module, ensemble_step: float):
    new_state_dict = {}
    for (name_1, weight_1), (name_2, weight_2) in zip(model_1.named_parameters(), model_2.named_parameters()):
        assert name_1 == name_2
        new_state_dict[name_1] = (1 - ensemble_step) * weight_1 + ensemble_step * weight_2

    return new_state_dict

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

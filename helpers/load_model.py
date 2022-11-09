import yaml
import torch.nn as nn

from monai.networks.nets import UNet, SwinUNETR, ViT


with open("../configs/general_configs.yml", "r") as rf:
    GENERAL_CONFIGS = yaml.load(rf, yaml.Loader)


def load_model(model_name):
    assert model_name in REGISTERED_MODELS.keys(), "Unsupported model name!"
    loader = REGISTERED_MODELS[model_name]
    param_dict = GENERAL_CONFIGS["models"][model_name]
    model = loader(**param_dict)

    turn_off_bias(model)

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

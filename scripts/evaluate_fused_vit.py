from collections import OrderedDict

import torch.nn as nn
import os
import sys
# sys.path.append('/home/junchi/deepl_learning')
# sys.path.append('/home/junchi/deepl_learning/ModelFusion')
path = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

if not path in sys.path:
    sys.path.append(path)

import tensorflow
import torch
from helpers.load_model import reload_model, load_model
from helpers.load_data import load_data
from helpers.model2graph import export_as_onnx
from wasserstein_ensemble_ViT import ViTFuser
from helpers.utils import load_yml_file, dict2namespace, create_filename, dict2filename
from tqdm import tqdm

ROOT_DIR = __file__
for _ in range(2):
    ROOT_DIR = os.path.dirname(ROOT_DIR)


def evaluate_vit(model, dataloader):
    correct = 0
    with torch.no_grad():
        model.eval()
        for batch_idx, (data, target) in tqdm(enumerate(dataloader), total=len(dataloader)):
            data, target = data.cuda(), target.cuda()
            pred, _ = model(data)
            pred_label = pred.argmax(dim=1, keepdim=True)
            correct += pred_label.eq(target.view_as(pred_label)).sum().item()
    print(f"Accuracy: {correct / len(dataloader.dataset)}")
    return correct / len(dataloader.dataset)


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

def main():
    """
    model can be downloaded from: https://polybox.ethz.ch/index.php/s/FotPNs3BLC6Byzg
    """
    # read args using argparse
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--ensemble_step", type=float, default=0.3)
    parser.add_argument("--square_factor", type=str, default="1/2")
    parser.add_argument("--retrain_fraction", type=float, default=1.0)
    parser.add_argument("--save_dir", type=str, default="./dev/vit_logs")
    args = parser.parse_args()
    args = vars(args)   # dict
    print(args)
    log_filename = dict2filename(args)
    log_filename = create_filename(args["save_dir"], log_filename)
    print(log_filename)
    log_file = open(log_filename, "w")

    # evaluate using cifar10
    ds_name = "CIFAR10"
    val_ds = load_data(ds_name, "val")
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=16, shuffle=False)


    model_name = "ViT"
    model1_path = os.path.join(ROOT_DIR, 'clf_logs/2022_12_13_23_55_00_073244')
    model2_path = os.path.join(ROOT_DIR, 'clf_logs/2022_12_13_23_55_00_086375')
    config = load_yml_file(os.path.join(ROOT_DIR, "configs/fusion_configs.yml"))
    # update config with args
    config = vars(config)
    config.update(args)
    config = dict2namespace(config)

    # average the weights
    average_weight = OrderedDict()
    model_1 = reload_model(model_name, model1_path)
    model_2 = reload_model(model_name, model2_path)
    weight_1 = model_1.state_dict()
    weight_2 = model_2.state_dict()
    for name, param in model_1.named_parameters():
        average_weight[name] = weight_1[name] * config.ensemble_step + weight_2[name] * (1 - config.ensemble_step)
        average_weight[name] = average_weight[name]

    average_model = load_model(model_name)
    average_model.load_state_dict(average_weight)
    average_model.cuda()
    print(average_model)
    avg_acc = evaluate_vit(average_model, val_loader)

    # fuse the weights
    if config.ensemble_step <= 0.5:
        config.ensemble_step = 1 - config.ensemble_step
        fused_weight = fuse_vits(model2_path, model1_path, config)
    else:
        fused_weight = fuse_vits(model1_path, model2_path, config)

    fused_model = load_model(model_name)
    fused_model.load_state_dict(fused_weight)
    fused_model.cuda()
    print(fused_model)
    fused_acc = evaluate_vit(fused_model, val_loader)

    print(f"Average accuracy: {avg_acc}, Fused accuracy: {fused_acc}", file=log_file)
    print("-" * 100)

    log_file.close()


if __name__ == "__main__":
    main()

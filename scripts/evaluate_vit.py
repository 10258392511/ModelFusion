import os
import sys

sys.path.append('/home/junchi/deepl_learning')

import tqdm
import torch
from helpers.load_data import load_data
from helpers.load_model import load_model, reload_model
# from helpers.utils import vis_images


def evaluate_vit(model, dataloader):
    correct = 0
    with torch.no_grad():
        model.eval()
        for batch_idx, (data, target) in tqdm.tqdm(enumerate(dataloader), total=len(dataloader)):
            data, target = data.cuda(), target.cuda()
            pred, _ = model(data)
            pred_label = pred.argmax(dim=1, keepdim=True)
            correct += pred_label.eq(target.view_as(pred_label)).sum().item()
    print(f"Accuracy: {correct / len(dataloader.dataset)}")


def main():
    ds_name = "CIFAR10"
    model_name = "ViT"
    log_dir = "/home/junchi/deepl_learning/ModelFusion/clf_logs/2022_12_13_23_55_00_073244"
    accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    val_ds = load_data(ds_name, "val")
    # model_reload = reload_model(model_name, log_dir)

    model = load_model(model_name)
    # load weights
    weights_path = "/home/junchi/deepl_learning/ModelFusion/fused_weights.pth"
    # weights_path = "/home/junchi/deepl_learning/ModelFusion/clf_logs/2022_12_13_23_55_00_086375/checkpoints/epoch=147-step=231324.ckpt"
    weights = torch.load(weights_path)
    model.load_state_dict(weights)
    model = model.cuda()
    # dataloader
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=16, shuffle=False)

    evaluate_vit(model, val_loader)

if __name__ == "__main__":
    main()
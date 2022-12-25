import numpy as np
import matplotlib.pyplot as plt
import torch
import SimpleITK as sitk
import os
import argparse
import ModelFusion.helpers.pytorch_utils as ptu
import re
import yaml


def load_yml_file(filename: str):
    assert ".yml" in filename
    with open(filename, "r") as rf:
        data = yaml.load(rf, yaml.Loader)

    data = dict2namespace(data)

    return data


def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace

dirname = os.path.dirname(os.path.dirname(__file__))
GENERAL_CONFIGS = load_yml_file(os.path.join(dirname, "configs/general_configs.yml"))


def vis_images(*images, **kwargs):
    """
    kwargs: if_save, save_dir, filename, titles
    """
    num_imgs = len(images)
    fig, axes = plt.subplots(1, num_imgs, figsize=(GENERAL_CONFIGS.figsize_unit * num_imgs,
                                                   GENERAL_CONFIGS.figsize_unit))
    if num_imgs == 1:
        axes = [axes]
    titles = kwargs.get("titles", None)
    if titles is not None:
        assert len(titles) == len(images)
    for i, (img_iter, axis) in enumerate(zip(images, axes)):
        channel = 0
        # channel = 0 if img_iter.shape[0] == 1 else 1
        if isinstance(img_iter, torch.Tensor):
            img_iter = ptu.to_numpy(img_iter)
        if img_iter.shape[0] == 3:
            handle = axis.imshow(img_iter.transpose(1, 2, 0))
        else:
            img_iter = img_iter[channel]
            handle = axis.imshow(img_iter, cmap="gray")
            plt.colorbar(handle)
        if titles is not None:
            axis.set_title(titles[i])

    fig.tight_layout()
    if_save = kwargs.get("if_save", False)
    if if_save:
        save_dir = kwargs.get("save_dir", "./outputs")
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        assert "filename" in kwargs
        filename = kwargs.get("filename")
        filename_full = os.path.join(save_dir, filename)
        fig.savefig(filename_full)
    else:
        plt.show()
    plt.close()


def vis_volume(data):
    # data: (D, H, W) or (D, W, H)
    if isinstance(data, torch.Tensor):
        data = ptu.to_numpy(data)
    img_viewer = sitk.ImageViewer()
    img_sitk = sitk.GetImageFromArray(data)
    img_viewer.Execute(img_sitk)


def extract_patient_id(filename: str):
    pattern = re.compile(r"[A-Z0-9]{6}")
    patient_id = re.findall(pattern, filename)[0]
    assert len(patient_id) == 6

    return patient_id


def create_filename(save_dir: str, filename: str) -> str:
    filename_path = os.path.join(save_dir, filename)
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)

    return filename_path


def dict2filename(args_dict: dict, suffix=".txt") -> str:
    filename = ""
    for key, value in args_dict.items():
        if key == "square_factor":
            value = str(value).replace("/", "_")
        filename += f"{key}_{value}_"

    filename = filename.replace(".", "_")
    filename = filename.replace("/", "")
    filename = filename[:-1] + suffix

    return filename

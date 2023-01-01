import numpy as np
import matplotlib.pyplot as plt
import torch
import SimpleITK as sitk
import os
import argparse
import ModelFusion.helpers.pytorch_utils as ptu
import re
import yaml
import urllib.request
import zipfile
import glob


def load_yml_file(filename: str, return_dict=False):
    assert ".yml" in filename
    with open(filename, "r") as rf:
        data = yaml.load(rf, yaml.Loader)

    if return_dict:
        return data

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


def namespace2dict(config: argparse.Namespace):
    if not isinstance(config, argparse.Namespace):
        return config
    config = vars(config)
    for key, val in config.items():
        # print(f"{key}: {val}")
        config[key] = namespace2dict(val)

    return config



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


def download_from_src(root_dir: str, src_url: str, filename: str):
    print("downloading weights...")
    if not os.path.isdir(root_dir):
        os.makedirs(root_dir)
    file_download = urllib.request.urlopen(src_url)
    file_path = os.path.join(root_dir, filename)
    with open(file_path, "wb") as wf:
        wf.write(file_download.read())
    print("Done!")


def unzip(src_filename: str, tgt_dir: str, if_del_zip=True):
    print("unzipping file...")
    assert ".zip" in src_filename
    if not os.path.isdir(tgt_dir):
        os.makedirs(tgt_dir)
    with zipfile.ZipFile(src_filename, "r") as zip_f:
        zip_f.extractall(tgt_dir)

    if if_del_zip:
        os.remove(src_filename)
    print("Done!")


def find_weight_parent_dir(root_dir: str):
    pattern = os.path.join(root_dir, "**/*.ckpt")
    weight_path = glob.glob(pattern, recursive=True)
    weight_path = weight_path[0]
    parent_dir = weight_path
    for _ in range(2):
        parent_dir = os.path.dirname(parent_dir)

    return parent_dir


def obtain_downloaded_weight_paths(config: dict, model_name: str, exp_name: str, save_root_dir: str):
    model_paths = {
        "model_1": config[model_name][exp_name]["model_1"],
        "model_2": config[model_name][exp_name]["model_2"]
    }
    save_root_dir = os.path.join(save_root_dir, model_name, exp_name)  # e.g weights/UNet/domain_generalization
    for model_name_iter, weights_url_iter in model_paths.items():
        download_from_src(save_root_dir, weights_url_iter, f"{model_name_iter}.zip")
        src_filename = os.path.join(save_root_dir, f"{model_name_iter}.zip")
        save_root_dir_iter = os.path.join(save_root_dir, f"{model_name_iter}/")  # e.g weights/UNet/domain_generalization/model_1
        unzip(src_filename, save_root_dir_iter)
        model_paths[model_name_iter] = find_weight_parent_dir(save_root_dir_iter)

    return model_paths

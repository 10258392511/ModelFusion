import numpy as np
import os
import glob
import torch
import torchvision.transforms as tv_tfm

from ModelFusion.helpers.utils import load_yml_file
from monai.transforms import (
    Compose,
    Transform,
    MapTransform,
    EnsureChannelFirstd,
    ScaleIntensityd,
    CropForegroundd,
    Resized,
    RandRotated,
    RandAdjustContrastd,
    RandGaussianNoised
)
from monai.data import CacheDataset
from torchvision.datasets import CIFAR10
from monai.utils import CommonKeys


dirname = os.path.dirname(os.path.dirname(__file__))
GENERAL_CONFIGS = load_yml_file(os.path.join(dirname, "configs/general_configs.yml"))


def load_individual(filename: str):
    assert ".npz" in filename
    data = np.load(filename)
    img, label = data["image"], data["label"]

    return img, label


class LoadDataNumpyDict(Transform):
    def __call__(self, filename):
        data_out = {
            CommonKeys.IMAGE: None,
            CommonKeys.LABEL: None
        }
        image, label = load_individual(filename)  # (1, H, W), (1, H, W)
        label_out = label
        image = torch.tensor(image).float()
        label_out = torch.tensor(label_out).long()
        data_out[CommonKeys.IMAGE] = image
        data_out[CommonKeys.LABEL] = label_out

        return data_out


def load_data(ds_name, mode, **kwargs):
    assert mode in ["train", "val", "test"], "Invalid mode!"
    assert ds_name in REGISTERED_DS.keys(), "Unsupported dataset name!"
    loader = REGISTERED_DS[ds_name]
    ds = loader(mode, **kwargs)

    return ds


def load_mnms(mode, vendor, if_aug=True, num_workers=4):
    """
    Only apply data augmentation when "mode" is "train" and "if_aug" is True.
    """
    dirname = None
    filename_pattern = None
    if mode == "train":
        dirname = "Trainining"
        filename_pattern = os.path.join(GENERAL_CONFIGS.datasets.MNMS.path, f"{dirname}/**/{vendor}/*.npz")
        assert vendor in ["A", "B", "C"], f"Vendor {vendor} is not in the {dirname} set!"
    else:
        dirname = "Validation"
        filename_pattern = os.path.join(GENERAL_CONFIGS.datasets.MNMS.path, f"{dirname}/{vendor}/*.npz")
        assert vendor in ["A", "B", "C", "D"], f"Vendor {vendor} is not in the {dirname} dataset!"

    all_filenames = glob.glob(filename_pattern)
    all_filenames = sorted(all_filenames)

    keys = [CommonKeys.IMAGE, CommonKeys.LABEL]
    transforms = [
        LoadDataNumpyDict(),
        EnsureChannelFirstd(keys=keys),
        ScaleIntensityd(keys=[CommonKeys.IMAGE]),
        CropForegroundd(keys=keys, source_key=CommonKeys.IMAGE),
    ]

    if mode == "train" and if_aug:
        transforms += [
            RandRotated(keys=keys, range_x=np.deg2rad(GENERAL_CONFIGS.datasets.MNMS.aug_rotate_deg),
                        mode=("bilinear", "nearest"), prob=.5),
            RandAdjustContrastd(keys=CommonKeys.IMAGE, prob=.5),
            RandGaussianNoised(keys=CommonKeys.IMAGE, prob=0.1, std=GENERAL_CONFIGS.datasets.MNMS.aug_noise_std)
        ]

    resize_shape = GENERAL_CONFIGS.datasets.MNMS.resize_shape
    transforms += [
        Resized(keys=keys, spatial_size=(resize_shape, resize_shape), mode=("bilinear", "nearest"))
    ]

    transforms = Compose(transforms)

    ds = CacheDataset(all_filenames, transform=transforms, num_workers=num_workers)

    return ds


def load_cifar10(mode):
    root_dir = GENERAL_CONFIGS.datasets.CIFAR10.path
    if not os.path.isdir(root_dir):
        os.makedirs(root_dir)
    transforms = tv_tfm.Compose([
        tv_tfm.TrivialAugmentWide(interpolation=tv_tfm.InterpolationMode.BILINEAR),
        tv_tfm.RandomHorizontalFlip(),
        tv_tfm.RandomCrop(32, padding=4),
        tv_tfm.PILToTensor(),
        tv_tfm.ConvertImageDtype(torch.float),
        tv_tfm.RandomErasing(p=0.1)
    ])
    if_train = True if mode == "train" else False
    if if_train:
        ds = CIFAR10(root_dir, train=if_train, transform=transforms, download=True)
    else:
        ds = CIFAR10(root_dir, train=if_train, transform=tv_tfm.ToTensor(), download=True)

    return ds



REGISTERED_DS = {
    "MNMS": load_mnms,
    "CIFAR10": load_cifar10
}

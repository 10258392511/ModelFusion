import sys
import os

path = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
if path not in sys.path:
    sys.path.append(path)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import nibabel as nib
import SimpleITK as sitk
import cv2 as cv
import torch
import torchvision
import monai
import pytorch_lightning as pl
import ModelFusion.helpers.pytorch_utils as ptu


if __name__ == '__main__':
    B, C, H, W = 2, 1, 28, 28
    x = torch.randn(B, C, H, W).to(ptu.DEVICE)
    print(f"x: class: {type(x)}, shape: {x.shape}, device: {x.device}")
    x_np = ptu.to_numpy(x)
    print(f"x_np: class: {type(x_np)}, shape: {x_np.shape}")

import sys
import os


path = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
if path not in sys.path:
    sys.path.append(path)

import argparse
import numpy as np
import pandas as pd
import nibabel as nib
import glob
import shutil
import pickle

from collections import Counter
from ModelFusion.helpers.utils import extract_patient_id
from tqdm import tqdm


OUTPUT_DIR = r"E:\Datasets\MnMsSlices"
INPUT_DIR = r"E:\Datasets\MnMs"

OUTPUT_DIR_BASENAME = os.path.basename(OUTPUT_DIR)
INPUT_DIR_BASENAME = os.path.basename(INPUT_DIR)

SHAPE_CNT = Counter()


def save_vol_as_slices(img: np.ndarray, seg: np.ndarray, output_filename: dir, time_step: str, vendor: str):
    # vol: (H, W, D)
    assert time_step in ["ES", "ED"]
    assert vendor in ["A", "B", "C", "D"]
    assert img.shape == seg.shape
    img_max, img_min = img.max(), img.min()
    img = (img - img_min) / (img_max - img_min)
    dirname = os.path.dirname(output_filename)
    dirname = os.path.join(dirname, vendor)
    if not os.path.isdir(dirname):
        os.makedirs(dirname)
    patient_id = extract_patient_id(output_filename)

    for slice_idx in range(img.shape[-1]):
        img_slice = img[..., slice_idx]
        seg_slice = seg[..., slice_idx]
        save_filename = f"{patient_id}_{time_step}_{slice_idx}.npz"
        np.savez(os.path.join(dirname, save_filename), image=img_slice, label=seg_slice)
        SHAPE_CNT[img_slice.shape] += 1


if __name__ == '__main__':
    """
    python scripts/vol2slice.py --output_dir "E:\Datasets\MnMsSlices"
    Each .npz file: keys: image, label
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", default=OUTPUT_DIR)
    args_dict = vars(parser.parse_args())

    if not os.path.isdir(args_dict["output_dir"]):
        os.makedirs(args_dict["output_dir"])

    # copy and read info file
    info_filename = glob.glob(os.path.join(INPUT_DIR, "*.csv"))[0]
    shutil.copy2(info_filename, os.path.join(args_dict["output_dir"], os.path.basename(info_filename)))
    info_df = pd.read_csv(info_filename)

    print("Reading in all filenames...")
    img_paths = glob.glob(os.path.join(INPUT_DIR, "/**/*sa.nii.gz"), recursive=True)
    img_paths = sorted(img_paths)
    seg_paths = glob.glob(os.path.join(INPUT_DIR, "/**/*sa_gt.nii.gz"), recursive=True)
    seg_paths = sorted(seg_paths)
    print("Done reading in all filanems!")

    pbar = tqdm(zip(img_paths, seg_paths), total=len(img_paths))
    for img_path_iter, seg_path_iter in pbar:
        if "Testing" in img_path_iter:
            continue
        patient_id = extract_patient_id(img_path_iter)
        df_row = info_df[info_df["External code"] == patient_id]
        img_path_out = img_path_iter.replace(INPUT_DIR_BASENAME, OUTPUT_DIR_BASENAME)
        seg_path_out = seg_path_iter.replace(INPUT_DIR_BASENAME, OUTPUT_DIR_BASENAME)

        img_arr = nib.load(img_path_iter).get_fdata()  # (H, W, D, T)
        seg_arr = nib.load(seg_path_iter).get_fdata()
        ED = df_row["ED"].values[0]
        ES = df_row["ES"].values[0]
        time_step_mapping = {"ED": ED, "ES": ES}
        vendor = df_row["Vendor"].values[0]

        for time_step_name, time_step in time_step_mapping.items():
            img_vol, seg_vol = img_arr[..., time_step], seg_arr[..., time_step]
            save_vol_as_slices(img_vol, seg_vol, img_path_out, time_step_name, vendor)

    with open(os.path.join(OUTPUT_DIR, "shape_stats.pkl"), "wb") as wf:
        pickle.dump(SHAPE_CNT, wf)

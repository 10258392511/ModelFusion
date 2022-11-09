import numpy as np


def load_individual(filename: str):
    assert ".npz" in filename
    data = np.load(filename)
    img, label = data["image"], data["label"]

    return img, label

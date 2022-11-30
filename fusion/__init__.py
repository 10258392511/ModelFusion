import torch.nn as nn


def evaluate_fused_model(model: nn.Module, val_ds, num_retrain_samples: int = 0):
    """
    Returns the evaluation metric (DSC or ACC). Can also be used to evaluate original models.
    """
    pass

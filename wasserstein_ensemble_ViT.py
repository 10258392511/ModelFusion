import torch
import torch.nn as nn
import ModelFusion.helpers.pytorch_utils as ptu

from .wasserstein_ensemble import get_wassersteinized_layers_modularized
from typing import List


class ViTFuser(object):
    """
    TODO:
    (1). Align the embedding weights (Conv2d), and output T_conv
    (2). Use T_conv to align: cls_token, position_embeddings & first MLP weight
    (3). Process Linear and LayerNorm layers; all by local models in 2 cases:
        (3.1). Linear (768, 3072) -> Linear (3072, 768) -T1-> LN (768,) * 2
        (3.2). Linear (768, 768) -> qkv (2304, 768) -T2-> LN (768,) * 2
               qkv: (768, 768) * 3, so there are 3 local models. Use the last T (v of qkv) from these local models
               as T2.
        Note T1 is used for both LN and next Linear layer, and similarly for T2.
    """
    def __init__(self, models: List[nn.Module]):
        self.models = models

    def __call__(self):
        pass

import torch
import torch.nn as nn
import helpers.pytorch_utils as ptu

from wasserstein_ensemble import get_wassersteinized_layers_modularized
from wasserstein_ensemble_u_net import ModelWrapper
from collections import OrderedDict
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
    def __init__(self, models: List[nn.Module], configs):
        self.models = models
        self.configs = configs
        self.patch_embd_T = None

    def __call__(self):
        non_fc_params = []
        fc_params = []
        for model in self.models:
            non_fc_param_iter, fc_param_iter = self._separate_fc_and_non_fc_layers(model)
            non_fc_params.append(non_fc_param_iter)
            fc_params.append(fc_param_iter)

        # fuse non-fc layers
        aligned_non_fc_params_dict = self._fuse_non_fc_weights(non_fc_params)
        # TODO: comment this out
        print("-" * 100)
        for key, weight in aligned_non_fc_params_dict.items():
            print(f"{key}: {weight.device}, {weight.shape}")
        #########################

        # TODO: fuse fc layers
        aligned_fc_params_dict = self._fuse_fc_weights(fc_params)
        # TODO: load the state_dict of the first model with new weights

        # TODO: return aligned model
        pass

    def _separate_fc_and_non_fc_layers(self, model: nn.Module):
        non_fc_params = OrderedDict()
        fc_params = OrderedDict()

        for name, param in model.named_parameters():
            weight_dim = param.ndim
            if weight_dim <= 2:
                fc_params[name] = param.data
            else:
                non_fc_params[name] = param.data

        return non_fc_params, fc_params

    def _fuse_fc_weights(self, params: List[OrderedDict]):
        """
        Fuse all the weights of the Fully Connected Layer
        the source code of ViT: https://github.com/Project-MONAI/MONAI/blob/dev/monai/networks/nets/vit.py
        Two parts need to be aligned: 
            1. TransformerBlock
            2. Classification Head

        The classification head is much easier: it only contains weights hidden_size*num_classes, to be specific: 768*100

        For the TransformerBlock, there are another two blocks inside, the multi-head-attention block and simple MLP
        Let's start from the easy part, for the MLP, it only contains two layers of FC, the first layer's size of 768*3072,
        the second layer's size if 3072*768

        For the MHA block, it contains serveral parts. The first one is qkv projection layer, which is just a FC layer
        of size 768*(3*768). The next part is dot product layey, but it has no parameters. The third part is out-projection
        layer, which is still a single layer of FC of size 768*768.

        Overall, the work here is to fuse multiple FC layer.
        """
        aligned_weights = OrderedDict()
        key_list = list(params[0].keys())
        # for idx, key in enumerate(params[0]):
        #     model_weight1 = params[0][key]
        #     model_weight2 = params[1][key]
        #     if model_weight1.ndim != 2:
        #         continue
        #     if idx == 0:
        #         # align the first fc layer
        #         aligned_wt = self._manual_align_next_layer(model_weight1, model_weight2, self.patch_embd_T)
        #     else:
        #         cur_key = key
        #         prev_key = key_list[idx - 1]
        #         local_model1 = ModelWrapper([("cur", params[0][cur_key])])
        #         local_model2 = ModelWrapper([("cur", params[1][cur_key])])
        #         avg_aligned_weights, T_list = get_wassersteinized_layers_modularized(self.configs,
        #                                                                              [local_model1, local_model2],
        #                                                                              return_T=True)
        #         pass
        # align the backbone without the last normalization and classification head
        idx = 0
        T_last_layer = None
        while idx < len(params[0])-3:
            cur_key = key_list[idx]
            if idx % 8 == 0:
                if idx == 0:
                    T_last_layer = self.patch_embd_T
                # align the first fc layer
                fc1_weight1 = params[0][cur_key]
                fc1_weight2 = params[1][cur_key]
                # post_multiple with the patch embedding T
                assert T_last_layer is not None, "The patch embedding T should not be None"
                fc1_weight1 = fc1_weight1 @ T_last_layer
                # align the first and second fc layer
                next_key = key_list[idx + 1]
                fc2_weight1 = params[0][next_key]
                fc2_weight2 = params[1][next_key]
                # extract local models
                local_model1 = ModelWrapper([("cur", fc1_weight1), ("next", fc2_weight1)])
                local_model2 = ModelWrapper([("cur", fc1_weight2), ("next", fc2_weight2)])
                # align the first and second fc layer
                avg_aligned_weights, T_list = get_wassersteinized_layers_modularized(self.configs,
                                                                                        [local_model1, local_model2],
                                                                                        return_T=True)
                aligned_weights[cur_key] = avg_aligned_weights[0]
                aligned_weights[next_key] = avg_aligned_weights[1]
                T_last_layer = T_list[-1]
                idx += 2
            elif idx % 8 == 2 or idx % 8 == 6:
                assert "norm" in cur_key, "The current key should be norm*.weight"
                # align the norm1.weight
                norm1_weight1 = params[0][cur_key]
                norm1_weight2 = params[1][cur_key]
                # align manually
                aligned_wt = self._manual_align_next_layer(norm1_weight1, norm1_weight2, T_last_layer)
                aligned_weights[cur_key] = aligned_wt
                idx += 1
            elif idx % 8 == 3 or idx % 8 == 7:
                assert "norm" in cur_key, "The current key should be norm*.bias"
                # align the norm1.bias
                norm1_bias1 = params[0][cur_key]
                norm1_bias2 = params[1][cur_key]
                # align manually
                aligned_wt = self._manual_align_next_layer(norm1_bias1, norm1_bias2, T_last_layer)
                aligned_weights[cur_key] = aligned_wt
                idx += 1
            elif idx % 8 == 4:
                assert "attn.out_proj.weight" in cur_key, "The current key should be attn.out_proj.weight"
                # align the attn.out_proj.weight
                attn_out_proj_weight1 = params[0][cur_key]
                attn_out_proj_weight2 = params[1][cur_key]
                # post_multiple with the patch embedding T
                assert T_last_layer is not None, "The patch embedding T should not be None"
                attn_out_proj_weight1 = attn_out_proj_weight1 @ T_last_layer
                # first align the attn.out_proj.weight
                local_model1 = ModelWrapper([("cur", attn_out_proj_weight1)])
                local_model2 = ModelWrapper([("cur", attn_out_proj_weight2)])
                avg_aligned_weights, T_list = get_wassersteinized_layers_modularized(self.configs,
                                                                                     [local_model1, local_model2],
                                                                                     return_T=True)
                aligned_weights[cur_key] = avg_aligned_weights[0]
                T_last_layer = T_list[-1]
                idx += 1
            elif idx % 8 == 5:
                assert "attn.qkv.weight" in cur_key, "The current key should be attn.qkv.weight"
                # align the weight of q, k, v separately
                aligned_qkv_weight = []
                t_qkv = []
                for i in range(3):
                    # extract the q, k, v weights
                    qkv_weight1 = params[0][cur_key][i*128:(i+1)*128, :]
                    qkv_weight2 = params[1][cur_key][i*128:(i+1)*128, :]

                    qkv_weight1 = qkv_weight1 @ T_last_layer
                    local_model1 = ModelWrapper([("cur", qkv_weight1)])
                    local_model2 = ModelWrapper([("cur", qkv_weight2)])
                    avg_aligned_weights, T_list = get_wassersteinized_layers_modularized(self.configs,
                                                                                            [local_model1, local_model2],
                                                                                            return_T=True)
                    aligned_qkv_weight.append(avg_aligned_weights[0])
                    t_qkv.append(T_list[-1])
                aligned_weights[cur_key] = torch.cat(aligned_qkv_weight, dim=0)
                T_last_layer = t_qkv[-1] # the T of v should be used to align the next layer
                idx += 1

        assert idx == len(params[0])-3, "The idx should be the last normalization layer"
        # align the last normalization layer
        cur_key = key_list[idx]
        norm1_weight1 = params[0][cur_key]
        norm1_weight2 = params[1][cur_key]
        # align manually
        aligned_wt = self._manual_align_next_layer(norm1_weight1, norm1_weight2, T_last_layer)
        aligned_weights[cur_key] = aligned_wt
        idx += 1
        # align the bias
        cur_key = key_list[idx]
        norm1_bias1 = params[0][cur_key]
        norm1_bias2 = params[1][cur_key]
        # align manually
        aligned_wt = self._manual_align_next_layer(norm1_bias1, norm1_bias2, T_last_layer)
        aligned_weights[cur_key] = aligned_wt
        idx += 1
        # align the head
        cur_key = key_list[idx]
        head1 = params[0][cur_key]
        head2 = params[1][cur_key]
        head1 = head1 @ T_last_layer
        local_model1 = ModelWrapper([("cur", head1)])
        local_model2 = ModelWrapper([("cur", head2)])
        avg_aligned_weights = get_wassersteinized_layers_modularized(self.configs,
                                                                                [local_model1, local_model2],
                                                                                return_T=False)
        aligned_weights[cur_key] = avg_aligned_weights[0]
        idx += 1

        assert idx == len(params[0]), "all layers should be aligned"
        print("All layers are aligned")

        return aligned_weights

    def _fuse_non_fc_weights(self, params: List[OrderedDict]):
        """
        Refer to wasserstein_ensemble line 159 for align columns of next weight matrix and
        line 244 for align rows of current weight matrix. (Together: W_{k} <- T_{k}^T @ W_{k} @ T_{k - 1} )

        Returns fused patch_embedding (768, 3, 4, 4), cls_token (1, 1, 768) and position_embeddings (1, 64, 768); and
        transport matrix T for patch_embedding. Note T should be used to align the first fc layer.
        """
        # extract patch_embedding weight
        patch_emb_key = None
        aligned_weights = OrderedDict()
        for key in params[0]:
            if params[0][key].ndim == 4:
                patch_emb_key = key
                break

        patch_emb_key_name = patch_emb_key.replace(".", "_")  # "." not allowed in param name in nn.Module
        local_model1 = ModelWrapper([(patch_emb_key_name, params[0][patch_emb_key])])
        local_model2 = ModelWrapper([(patch_emb_key_name, params[1][patch_emb_key])])
        avg_aligned_weights, T_list = get_wassersteinized_layers_modularized(self.configs, [local_model1, local_model2],
                                                                        return_T=True)
        avg_aligned_weights = avg_aligned_weights[-1]
        T_var = T_list[-1]  # (768, 768)
        self.patch_embd_T = T_var
        aligned_weights[patch_emb_key] = avg_aligned_weights
        for key in params[0]:
            model_weight1 = params[0][key]
            model_weight2 = params[1][key]
            if model_weight1.ndim == 4:
                continue

            aligned_weights[key] = self._manual_align_next_layer(model_weight1, model_weight2, T_var)

        return aligned_weights

    def _manual_align_next_layer(self, weights1: torch.Tensor, weights2: torch.Tensor, T_var: torch.Tensor):
        """
        Works for cls_token, patch_embeddings and LN. These layers have weights of shape (..., 768). We need to do
        weights1 @ T_var and average it with weights2.

        Returns the averaged weights
        """
        assert weights1.shape[-1] == T_var.shape[0]
        args = self.configs
        # align model_weight1 to model_weight2
        aligned_wt = torch.matmul(weights1, T_var)
        # to use the same variable name
        fc_layer1_weight_data = weights2
        t_fc0_model = aligned_wt
        # # Average the weights of aligned first layer
        if torch.allclose(T_var, torch.tensor(0.).to(T_var.device)):
            fc_layer1_weight_data_temp = fc_layer1_weight_data.reshape(fc_layer1_weight_data.shape[0], -1)
            geometric_fc = (t_fc0_model + fc_layer1_weight_data_temp) / 1
            assert torch.allclose(geometric_fc, fc_layer1_weight_data_temp)
        elif args.ensemble_step != 0.5:
            geometric_fc = ((1 - args.ensemble_step) * t_fc0_model + args.ensemble_step * fc_layer1_weight_data)
        else:
            geometric_fc = (t_fc0_model + fc_layer1_weight_data) / 2

        return geometric_fc



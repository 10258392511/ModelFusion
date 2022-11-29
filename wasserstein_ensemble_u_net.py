import torch
import torch.nn as nn
import networkx as nx
import warnings
import ModelFusion.helpers.pytorch_utils as ptu

from .wasserstein_ensemble import get_wassersteinized_layers_modularized
from ModelFusion.helpers.model2graph import model2graph_wrapper
from typing import List, Tuple


class ModelWrapper(nn.Module):
    def __init__(self, param_tuple_list: list[Tuple[str, torch.Tensor]]):
        super(ModelWrapper, self).__init__()
        for param_name, param in param_tuple_list:
            param = nn.Parameter(param.to(ptu.DEVICE))
            self.register_parameter(param_name, param)
            # print(param.data.shape)
            # print(param.data)

    def forward(self, x):
        warnings.warn("Not used!")

        return x


class UNetFuser(object):
    def __init__(self, models: List[nn.Module], models_onnx: List[str], configs):
        self.models = models
        self.models_graph = [model2graph_wrapper(filename) for filename in models_onnx]
        self.configs = configs

    def __call__(self):
        """
        DP following the topological order. Use original fusion function on a local model defined by each node.
        Time complexity: O(|V| + |E|)
        """
        model_graph_ref = self.models_graph[0]
        nodes_topo_sorted = list(nx.topological_sort(model_graph_ref))
        self.models_graph = [self._add_backward_edge(model_graph_iter) for model_graph_iter in self.models_graph]

        # for convenience, store weights as node attr
        for model, model_graph in zip(self.models, self.models_graph):
            self._store_weights(model, model_graph)

        # transpose weights of ConvTranspose2d
        for model_graph in self.models_graph:
            self._transpose_weights(model_graph)

        # directly modify the first model
        out_state_dict_temp = {}
        for node in nodes_topo_sorted:
            avg_weights = self._process_node(node)
            out_state_dict_temp[node] = avg_weights

        # transpose bask weights of ConvTranspose2d
        for node in self.models_graph[0]:
            self.models_graph[0].nodes[node]["weight"] = out_state_dict_temp[node]
        self._transpose_weights(self.models_graph[0])

        # modify weights of the first model
        out_state_dict = self.models[0].state_dict()
        for node, attr in self.models_graph[0].nodes(data=True):
            out_state_dict[node] = attr["weight"]
        out_model = self.models[0]
        out_model.load_state_dict(out_state_dict)

        return out_model

    def _add_backward_edge(self, model_graph: nx.DiGraph):
        adj = dict(model_graph.adjacency())
        for node in model_graph.nodes:
            model_graph.nodes[node]["from"] = set()
        for node in adj:
            neighbors = adj[node]
            for neighbor in neighbors:
                model_graph.nodes[neighbor]["from"].add(node)

        return model_graph

    def _store_weights(self, model: nn.Module, model_graph: nx.DiGraph):
        state_dict_iter = model.state_dict()
        for node in model_graph.nodes:
            model_graph.nodes[node]["weight"] = state_dict_iter[node].clone()

        return model_graph

    def _transpose_weights(self, model_graph: nx.DiGraph):
        # ConvTranspose2d: indegree = 2
        for node, attrs in model_graph.nodes(data=True):
            indegree = len(attrs["from"])
            if indegree == 2:
                weight = model_graph.nodes[node]["weight"]  # (C_in, C_out, kx, ky) for first transpose
                model_graph.nodes[node]["weight"] = weight.permute(1, 0, 2, 3)

        return model_graph

    def _process_node(self, current_node: str):
        """
        Wrapper for _indegree_i for all i = 0, 1, 2
        """
        indeg = len(self.models_graph[0].nodes[current_node]["from"])
        if indeg == 0:
            return self._indegree_0(current_node)
        elif indeg == 1:
            return self._indegree_1(current_node)
        elif indeg == 2:
            return self._indegree_2(current_node)
        else:
            raise NotImplementedError

    def _indegree_0(self, current_node: str):
        model1_node_attr = self.models_graph[0].nodes[current_node]
        model2_node_attr = self.models_graph[1].nodes[current_node]
        local_model1 = ModelWrapper([("current", model1_node_attr["weight"])])
        local_model2 = ModelWrapper([("current", model2_node_attr["weight"])])
        avg_aligned_layers = get_wassersteinized_layers_modularized(self.configs, [local_model1, local_model2])

        return avg_aligned_layers[-1]

    def _indegree_1(self, current_node: str):
        model1_node_attr = self.models_graph[0].nodes[current_node]
        model2_node_attr = self.models_graph[1].nodes[current_node]
        # use model_1_node as ref
        from_node_names = [node for node in model1_node_attr["from"]]
        local_model1 = ModelWrapper([("prev", self.models_graph[0].nodes[from_node_names[0]]["weight"]),
                        ("current", model1_node_attr["weight"])])
        local_model2 = ModelWrapper([("prev", self.models_graph[1].nodes[from_node_names[0]]["weight"]),
                        ("current", model2_node_attr["weight"])])
        avg_aligned_layers = get_wassersteinized_layers_modularized(self.configs, [local_model1, local_model2])

        return avg_aligned_layers[-1]

    def _indegree_2(self, current_node: str):
        model1_node_attr = self.models_graph[0].nodes[current_node]
        model2_node_attr = self.models_graph[1].nodes[current_node]
        # use model_1_node as ref
        from_node_names = [node for node in model1_node_attr["from"]]
        # concatenating order: [down_feature, up_feature]
        # two cases:
        # (16, 8, 3, 3) --> (8, 32, 3, 3)
        #       |                  |
        # (32, 16, 3, 3) --> (16, 96, 3, 3)
        #       |                  |
        #          (64, 32, 3, 3)
        # bottom: sort by topological order (both: indeg = 1)
        # others: first: indeg = 1; second: indeg = 2

        # from_node1 = self.models_graph[0].nodes[from_node_names[0]]
        # from_node2 = self.models_graph[0].nodes[from_node_names[1]]
        from_node1 = from_node_names[0]
        from_node2 = from_node_names[1]
        from_node1_indeg = len(self.models_graph[0].nodes[from_node1]["from"])
        from_node2_indeg = len(self.models_graph[0].nodes[from_node2]["from"])
        # print(f"{from_node1_indeg}, {from_node2_indeg}")
        # ref: model1
        if from_node1_indeg > from_node2_indeg:
            # case 2
            from_node_names = from_node_names[::-1]
        elif from_node1_indeg == from_node2_indeg:
            # case 1
            if from_node1 in self.models_graph[0].nodes[from_node2]["from"]:
                # correct order
                pass
            else:
                from_node_names = from_node_names[::-1]

        out_weights = []
        layer_start_ptr = 0
        for from_node_name_iter in from_node_names:
            # (C_out, C_in, kx, ky)
            prev_weight_shape =  self.models_graph[0].nodes[from_node_name_iter]["weight"].shape
            layer_end_ptr = layer_start_ptr + prev_weight_shape[0]
            local_model1 = ModelWrapper([("prev", self.models_graph[0].nodes[from_node_name_iter]["weight"]),
                                         ("current", model1_node_attr["weight"][:, layer_start_ptr:layer_end_ptr, ...])])
            local_model2 = ModelWrapper([("prev", self.models_graph[1].nodes[from_node_name_iter]["weight"]),
                                         ("current", model2_node_attr["weight"][:, layer_start_ptr:layer_end_ptr, ...])])
            avg_aligned_layers = get_wassersteinized_layers_modularized(self.configs, [local_model1, local_model2])
            out_weights.append(avg_aligned_layers[-1])
            layer_start_ptr = layer_end_ptr

        out_weights = torch.cat(out_weights, dim=1)  # e.g. bottom: [(16, 32, 3, 3), (16, 64, 3, 3)] -> (16, 96, 3, 3)

        return out_weights

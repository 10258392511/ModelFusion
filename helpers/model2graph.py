import networkx as nx
import torch
import torch.nn as nn
import onnx
import onnx.numpy_helper
import ModelFusion.helpers.pytorch_utils as ptu

# Note:
# (1). We only want the topological order of modules with parameters. After obtaining the order, we'll operate on
# model.state_dict(). Thus it's fine that the following operates on CPU.
# (2). model.state_dict() shares the same keys as model_onnx.graph.initializer


def export_as_onnx(model: nn.Module, input_shape: tuple, filename: str, device=None):
    """
    Performed on CPU or GPU
    """
    assert ".onnx" in filename

    if device is None:
        device = ptu.DEVICE
    x_in = torch.randn(input_shape).to(device)
    torch.onnx.export(model, x_in, filename)


def onnx2graph(filename: str):
    assert ".onnx" in filename

    model_onnx = onnx.load(filename)
    graph = nx.DiGraph()
    all_initializer_names = {module_iter.name: module_iter for module_iter in model_onnx.graph.initializer}
    for node in model_onnx.graph.node:
        for input_node in node.input:
            for output_node in node.output:
                graph.add_edge(input_node, node.name)
                graph.add_edge(node.name, output_node)

    for node in graph.nodes:
        # weight = None
        weight_shape = None
        if node in all_initializer_names:
            weight = onnx.numpy_helper.to_array(all_initializer_names[node])
            # weight = torch.tensor(weight)
            weight_shape = weight.shape
        # graph.nodes[node]["weight"] = weight
        graph.nodes[node]["weight_shape"] = weight_shape

    return graph


def contract_initializer_and_layer(graph: nx.DiGraph, onnx_filename: str):
    assert ".onnx" in onnx_filename
    graph_onnx = onnx.load(onnx_filename).graph
    all_initializer_names = [module_iter.name for module_iter in graph_onnx.initializer]
    graph_in = graph.copy()
    adj_dict = dict(graph_in.adjacency())

    for initializer_iter in all_initializer_names:
        neighbors = list(adj_dict[initializer_iter].keys())
        assert len(neighbors) == 1
        graph_in = nx.contracted_edge(graph_in, (initializer_iter, neighbors[0]), self_loops=False)

    return graph_in


def onnx2graph_contracted_wrapper(filename: str):
    graph = onnx2graph(filename)
    graph = contract_initializer_and_layer(graph, filename)

    return graph


def extract_param_graph(graph: nx.DiGraph):
    # DP on G = (V, E)
    # time complexity: O(|V| + |E|)
    # subprob(G, node) -> list[closest nodes representing weights]
    # backtracking: E
    nodes_topo_reversed = list(reversed(list(nx.topological_sort(graph))))
    adj_dict = dict(graph.adjacency())
    subprob_sol_key = "subprob_sol"

    for node in nodes_topo_reversed:
        node_dict = graph.nodes[node]
        if node_dict["weight_shape"] is not None:
            node_dict[subprob_sol_key] = [node]

        else:
            neighbors = adj_dict[node]
            node_dict[subprob_sol_key] = []
            if len(neighbors) == 0:
                pass
            else:
                for neighbor_iter in neighbors:
                    node_dict[subprob_sol_key] += graph.nodes[neighbor_iter][subprob_sol_key]

    # construct the new graph
    graph_out = nx.DiGraph()
    adj_dict = dict(graph.adjacency())
    for node in graph.nodes:
        if graph.nodes[node]["weight_shape"] is None:
            continue
        for neighbor in adj_dict[node]:
            for closest_weight_node in graph.nodes[neighbor][subprob_sol_key]:
                graph_out.add_edge(node, closest_weight_node)

    for node in graph_out.nodes:
        graph_out.nodes[node]["weight_shape"] = graph.nodes[node]["weight_shape"]

    return graph_out


def model2graph_wrapper(filename: str):
    """
    .onnx file to graph.
    """
    graph = onnx2graph_contracted_wrapper(filename)
    param_graph = extract_param_graph(graph)

    return param_graph

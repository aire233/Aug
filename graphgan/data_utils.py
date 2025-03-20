import os
import numpy as np
import networkx as nx
import torch
from torch_geometric.data import Data
from tqdm import tqdm


def load_adjacency_matrix(file_path):
    edges = []
    with open(file_path, "r") as f:
        for line in f:
            node1, node2 = map(int, line.strip().split(","))
            edges.append((node1, node2))
    return np.array(edges, dtype=int)


def load_graph_indicator(file_path):
    return np.loadtxt(file_path, dtype=int)


def load_node_labels(file_path):
    return np.loadtxt(file_path, dtype=int)


def load_graph_labels(file_path):
    return np.loadtxt(file_path, dtype=int)


def build_graphs(edges, graph_indicator, node_labels):
    graphs = {}
    print("Building graphs...")
    for edge in tqdm(edges, desc="Building graphs"):
        node1, node2 = edge
        graph_id = graph_indicator[node1 - 1]  # 假定节点ID从 1 开始
        if graph_id not in graphs:
            graphs[graph_id] = nx.Graph()
        graphs[graph_id].add_edge(node1, node2)
        graphs[graph_id].nodes[node1]["label"] = node_labels[node1 - 1]
        graphs[graph_id].nodes[node2]["label"] = node_labels[node2 - 1]
    return graphs


def convert_to_pyg(graphs, graph_labels):
    """
    将 networkx 格式的图转换为 PyG 的 Data 对象列表，同时返回数据集中最大节点数。
    """
    pyg_data_list = []
    print("Converting graphs to PyG format...")
    all_nodes = [g.number_of_nodes() for g in graphs.values()]
    max_nodes = max(all_nodes)
    print(f"Maximum nodes in dataset: {max_nodes}")

    for graph_id, graph in tqdm(graphs.items(), desc="Converting graphs"):
        nodes = sorted(list(graph.nodes()))
        num_nodes = len(nodes)
        x = torch.zeros((max_nodes, 1), dtype=torch.float)
        x[:num_nodes] = torch.tensor(
            [graph.nodes[node]["label"] for node in nodes]
        ).unsqueeze(1)
        node_to_idx = {node: i for i, node in enumerate(nodes)}
        edge_index_list = []
        for u, v in graph.edges():
            if u in node_to_idx and v in node_to_idx:
                edge_index_list.append([node_to_idx[u], node_to_idx[v]])
                edge_index_list.append([node_to_idx[v], node_to_idx[u]])
        if len(edge_index_list) > 0:
            edge_index = (
                torch.tensor(edge_index_list, dtype=torch.long).t().contiguous()
            )
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)
        y = torch.tensor([graph_labels[graph_id - 1]], dtype=torch.long)
        data = Data(x=x, edge_index=edge_index, y=y)
        pyg_data_list.append(data)
    return pyg_data_list, max_nodes

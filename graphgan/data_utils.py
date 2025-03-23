import os
import numpy as np
import networkx as nx
import torch
from torch_geometric.data import Data
from tqdm import tqdm

def load_node_attrs(file_path):
    """
    加载节点属性矩阵，返回形状 (n, d) 的 numpy 数组，
    每一行为一个节点的属性向量（逗号分隔）
    """
    return np.loadtxt(file_path, delimiter=',')

def load_edges(file_path):
    """
    加载边文件，返回形状 (m, 2) 的 numpy 数组，
    每一行对应 (node_id1, node_id2)，节点编号从 1 开始
    """
    return np.loadtxt(file_path, delimiter=',', dtype=int)

def load_graph_labels(file_path):
    """
    加载图标签，返回一维 numpy 数组（共 N 行）
    """
    return np.loadtxt(file_path, dtype=int)

def load_graph_idx(file_path):
    """
    加载图索引，返回一维 numpy 数组（共 n 行），
    每一行表示对应节点所属图的编号
    """
    return np.loadtxt(file_path, dtype=int)

def build_graphs(edges, graph_idx, node_attrs):
    """
    根据边、图索引和节点属性构建 networkx 图字典。
    
    1. 根据 graph_idx（长度 n）为每个节点（编号1~n）创建节点，
       并设置其属性（key "attr"），即 node_attrs[i] 对应节点 i+1。
    2. 再遍历边文件，按边所在图（由节点所属图决定）添加边。
    
    返回：图字典，key 为图 id，value 为 networkx.Graph 对象
    """
    graphs = {}
    n = len(graph_idx)
    unique_graph_ids = np.unique(graph_idx)
    # 为所有出现的图 id 创建空图
    for gid in unique_graph_ids:
        graphs[gid] = nx.Graph()
    # 添加所有节点（包括孤立节点）
    for i in range(n):
        node_id = i + 1
        gid = graph_idx[i]
        graphs[gid].add_node(node_id, attr=node_attrs[i])
    # 添加边，假设同一条边的两个节点属于同一图（block diagonal结构）
    for edge in edges:
        node1, node2 = edge
        gid = graph_idx[node1 - 1]
        graphs[gid].add_edge(node1, node2)
    return graphs

def convert_to_pyg(graphs, graph_labels):
    """
    将 networkx 图转换为 PyG Data 对象列表。
    每个图的节点特征矩阵大小统一设为 (max_nodes, d)（不足部分补零），
    d 为节点属性维度。图标签取 graph_labels[graph_id-1]。
    
    返回：pyg_data_list, max_nodes（所有图中的最大节点数）
    """
    pyg_data_list = []
    print("Converting graphs to PyG format...")
    all_nodes = [g.number_of_nodes() for g in graphs.values()]
    max_nodes = max(all_nodes)
    print(f"Maximum nodes in dataset: {max_nodes}")

    for graph_id, graph in tqdm(graphs.items(), desc="Converting graphs"):
        nodes = sorted(list(graph.nodes()))
        num_nodes = len(nodes)
        d = len(graph.nodes[nodes[0]]["attr"])  # 假设所有节点属性维度相同
        x = torch.zeros((max_nodes, d), dtype=torch.float)
        # 收集已有节点的属性
        attr_list = [graph.nodes[node]["attr"] for node in nodes]
        x[:num_nodes] = torch.tensor(attr_list, dtype=torch.float)
        # 构造局部节点编号映射
        node_to_idx = {node: i for i, node in enumerate(nodes)}
        edge_index_list = []
        for u, v in graph.edges():
            if u in node_to_idx and v in node_to_idx:
                edge_index_list.append([node_to_idx[u], node_to_idx[v]])
                edge_index_list.append([node_to_idx[v], node_to_idx[u]])
        if len(edge_index_list) > 0:
            edge_index = torch.tensor(edge_index_list, dtype=torch.long).t().contiguous()
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)
        y = torch.tensor([graph_labels[graph_id - 1]], dtype=torch.long)
        data = Data(x=x, edge_index=edge_index, y=y)
        pyg_data_list.append(data)
    return pyg_data_list, max_nodes

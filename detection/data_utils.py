# data_util.py
import os
import numpy as np
import torch
from torch_geometric.data import Data, Dataset
from torch_geometric.utils import add_self_loops

def load_node_attrs(file_path):
    """加载节点属性，每行逗号分隔"""
    return np.loadtxt(file_path, delimiter=",")

def load_txt(file_path, dtype=int):
    with open(file_path, "r") as f:
        return np.array([dtype(line.strip()) for line in f])

def load_edges(file_path):
    """加载边数据，每行 "src,dst"，转换为 0-based 索引"""
    edges = []
    with open(file_path, "r") as f:
        for line in f:
            src, dst = line.strip().split(',')
            edges.append((int(src)-1, int(dst)-1))
    return edges

class SimpleGraphDataset(Dataset):
    """
    加载 COIL-DEL 数据集：
      - {prefix}.node_attrs: 节点属性矩阵
      - {prefix}.graph_idx: 每个节点所属图 id（1-based）
      - {prefix}.edges: 边数据，格式 "src,dst"（1-based）
      - {prefix}.graph_labels: 图的标签（1-based）
    
    参数 known_class_num 指定已知类别数；
    原始标签大于或等于 known_class_num 的视为未知类别，标签统一设为 known_class_num；
    对于已知类别标签，将转换为 0-based（即 orig_label - 1）。
    """
    def __init__(self, data_dir, ds_prefix, known_class_num):
        # 文件路径
        path_node_attrs = os.path.join(data_dir, f"{ds_prefix}.node_attrs")
        path_graph_idx = os.path.join(data_dir, f"{ds_prefix}.graph_idx")
        path_edges = os.path.join(data_dir, f"{ds_prefix}.edges")
        path_graph_labels = os.path.join(data_dir, f"{ds_prefix}.graph_labels")
        
        # 加载数据
        node_attrs = load_node_attrs(path_node_attrs)
        graph_idx = load_txt(path_graph_idx, dtype=int)
        graph_labels = load_txt(path_graph_labels, dtype=int)
        edges = load_edges(path_edges)
        
        self.data_list = []
        self.known_class_num = known_class_num
        for gid in np.unique(graph_idx):
            node_mask = (graph_idx == gid)
            global_nodes = np.where(node_mask)[0]
            if len(global_nodes) == 0:
                continue
            X = node_attrs[global_nodes]
            
            # 构造边（只保留该图内的边，并转换为局部索引）
            edge_list = []
            for src, dst in edges:
                if node_mask[src] and node_mask[dst]:
                    # 找到局部索引
                    local_src = int(np.where(global_nodes == src)[0][0])
                    local_dst = int(np.where(global_nodes == dst)[0][0])
                    edge_list.append([local_src, local_dst])
            if len(edge_list) == 0:
                continue
            edge_index = torch.tensor(edge_list, dtype=torch.long).t().contiguous()
            edge_index, _ = add_self_loops(edge_index)
            
            # 标签处理：原始标签为 1-based，若>=known_class_num 则设为 unknown (即 known_class_num)
            orig_label = graph_labels[gid-1]
            if orig_label >= known_class_num:
                label = known_class_num
            else:
                label = orig_label - 1
            self.data_list.append(Data(
                x=torch.tensor(X, dtype=torch.float),
                edge_index=edge_index,
                y=torch.tensor(label, dtype=torch.long)
            ))
        print(f"Loaded {len(self.data_list)} graphs.")

    def len(self):
        return len(self.data_list)

    def get(self, idx):
        return self.data_list[idx]
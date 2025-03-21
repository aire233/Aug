import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import tqdm
import matplotlib.pyplot as plt

# -------------------------------
# 数据加载函数
# -------------------------------

def load_node_attrs(file_path):
    """
    加载节点属性，每行是逗号分隔的属性向量，返回 shape=(n, feat_dim)
    """
    return np.loadtxt(file_path, delimiter=",")

def load_txt(file_path, dtype=int):
    with open(file_path, "r") as f:
        lines = f.readlines()
    return np.array([dtype(line.strip()) for line in lines])

def load_edges(file_path):
    """
    加载边数据，每行为 "src,dst"，假设节点 id 从 1 开始，因此转换为 0-based
    """
    edges = []
    with open(file_path, "r") as f:
        for line in f:
            src, dst = line.strip().split(',')
            edges.append((int(src) - 1, int(dst) - 1))
    return edges

# -------------------------------
# 数据集类定义
# -------------------------------
class GraphDataset(Dataset):
    """
    加载 COIL-DEL 数据集：
      - {prefix}.node_attrs: 节点属性矩阵 (n 行，每行一个属性向量)
      - {prefix}.graph_idx: 每个节点所属图 id（假设图 id 从 1 开始）
      - {prefix}.edges: 边数据，格式 "row,col"，节点 id 从 1 开始
      - {prefix}.graph_labels: 图的类别标签，每行一个标签，对应图 id (1-indexed)
      - {prefix}.link_labels: 边标签（暂不使用）
      
    对于每个图，构造：
      - X: 节点属性矩阵（局部）
      - A: 邻接矩阵（局部），加入自环并进行对称归一化
      - label: 图类别（转换为从 0 开始的类别索引）
    """
    def __init__(self, data_dir, ds_prefix):
        # 文件路径
        path_node_attrs = os.path.join(data_dir, f"{ds_prefix}.node_attrs")
        path_graph_idx = os.path.join(data_dir, f"{ds_prefix}.graph_idx")
        path_edges = os.path.join(data_dir, f"{ds_prefix}.edges")
        path_graph_labels = os.path.join(data_dir, f"{ds_prefix}.graph_labels")
        path_link_labels = os.path.join(data_dir, f"{ds_prefix}.link_labels")  # 暂不使用
        
        # 加载数据
        node_attrs = load_node_attrs(path_node_attrs)  # shape=(n, feat_dim)
        graph_idx = load_txt(path_graph_idx, dtype=int)  # 长度 n，每个元素为图 id（假设从1开始）
        graph_labels = load_txt(path_graph_labels, dtype=int)  # 长度 N，每个元素为图标签
        edges = load_edges(path_edges)  # 每个元素为 (src, dst)
        # 若需要，也可以加载边标签：link_labels = load_txt(path_link_labels, dtype=int)
        
        n = node_attrs.shape[0]
        # 所有出现的图 id
        unique_graph_ids = np.unique(graph_idx)
        self.graphs = []
        
        # 预处理：将边按所属图分组。由于数据为 block diagonal，边只存在于同一图内
        graph_edges = {gid: [] for gid in unique_graph_ids}
        for src, dst in edges:
            gid = graph_idx[src]  # 假设 src 和 dst 属于同一图
            graph_edges[gid].append((src, dst))
        
        # 构造每个图的样本
        for gid in unique_graph_ids:
            # 得到该图所有节点的全局 id（0-based索引）
            global_node_ids = np.where(graph_idx == gid)[0]
            num_nodes = len(global_node_ids)
            # 节点属性
            X = node_attrs[global_node_ids]
            # 构造邻接矩阵
            A = np.zeros((num_nodes, num_nodes), dtype=np.float32)
            # 建立全局到局部的映射
            global2local = {g: i for i, g in enumerate(global_node_ids)}
            # 取出该图内的边
            for src, dst in graph_edges[gid]:
                if src in global2local and dst in global2local:
                    i, j = global2local[src], global2local[dst]
                    A[i, j] = 1.0
                    A[j, i] = 1.0  # 假设无向图
            # 添加自环并归一化
            A = A + np.eye(num_nodes, dtype=np.float32)
            D = np.sum(A, axis=1)
            D_inv_sqrt = np.diag(1.0 / np.sqrt(D))
            A_norm = D_inv_sqrt @ A @ D_inv_sqrt
            
            # 图标签：graph_labels 的第 (gid-1) 行对应图 gid（假设 graph_labels 是 1-indexed）
            graph_label = int(graph_labels[gid - 1])
            
            self.graphs.append({
                "X": torch.tensor(X, dtype=torch.float),
                "A": torch.tensor(A_norm, dtype=torch.float),
                "label": torch.tensor(graph_label, dtype=torch.long)
            })
            
        # 已知类别数（注意：开放集设置下，输出类别数为已知类别数+1）
        self.known_labels = sorted(list({g["label"].item() for g in self.graphs}))
        self.num_known = len(self.known_labels)
        # 同时保存节点属性维度
        self.input_dim = node_attrs.shape[1]
        print(f"Loaded {len(self.graphs)} graphs, {self.num_known} known classes, input dim {self.input_dim}.")

    def __len__(self):
        return len(self.graphs)
    
    def __getitem__(self, idx):
        return self.graphs[idx]

# -------------------------------
# 模型定义部分
# -------------------------------
class GCNLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(GCNLayer, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        
    def forward(self, X, A):
        out = torch.matmul(A, X)
        out = self.linear(out)
        return out

class GraphFeatureExtractor(nn.Module):
    """
    特征生成网络 G：
      - 两层 GCN + ReLU 激活
      - 均值池化获得图级特征
      - 可选加入噪声用于对抗训练
    """
    def __init__(self, in_features, hidden_dim=128, out_dim=128):
        super(GraphFeatureExtractor, self).__init__()
        self.conv1 = GCNLayer(in_features, hidden_dim)
        self.conv2 = GCNLayer(hidden_dim, out_dim)
        self.relu = nn.ReLU()
        
    def forward(self, X, A, noise=None):
        x = self.conv1(X, A)
        x = self.relu(x)
        x = self.conv2(x, A)
        x = self.relu(x)
        # 均值池化（沿节点维度求平均）
        graph_feat = torch.mean(x, dim=0, keepdim=True)
        if noise is not None:
            graph_feat = graph_feat + noise
        return graph_feat

class Classifier(nn.Module):
    """
    分类器网络 C：
      - 简单 MLP，输出维度为 (num_known + 1)
      - 前 num_known 个对应已知类别，最后一个对应开放类别
    """
    def __init__(self, in_dim=128, hidden_dim=64, num_classes=3):
        super(Classifier, self).__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, num_classes)
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        logits = self.fc2(x)
        return logits

# -------------------------------
# 训练过程
# -------------------------------
def train(data_dir, ds_prefix, num_epochs=50, lr=1e-3, noise_std=0.1, lambda_adv=1.0):
    dataset = GraphDataset(data_dir, ds_prefix)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 模型：输入维度根据数据集，输出类别 = known classes + 1（开放类别）
    G_model = GraphFeatureExtractor(in_features=dataset.input_dim, hidden_dim=128, out_dim=128).to(device)
    C_model = Classifier(in_dim=128, hidden_dim=64, num_classes=dataset.num_known + 1).to(device)
    
    optimizer_G = optim.Adam(G_model.parameters(), lr=lr)
    optimizer_C = optim.Adam(C_model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(num_epochs):
        G_model.train()
        C_model.train()
        total_loss_C = 0.0
        total_loss_G = 0.0
        
        for data in dataloader:
            # 提取数据，并确保标签为一维 tensor，且类型为 long
            X = data["X"].to(device)
            A = data["A"].to(device)
            label = data["label"]
            label = label.view(-1).to(device)  # 保证形状为 [batch_size]
            
            # 真实样本：经过 G_model 提取特征，并使用分类器输出 logits
            f_real = G_model(X, A)
            logits_real = C_model(f_real)
            loss_real = criterion(logits_real, label)
            
            # 对抗样本：在图级特征上加入随机噪声
            noise = torch.randn_like(f_real) * noise_std
            f_fake = G_model(X, A, noise=noise)
            logits_fake = C_model(f_fake)
            # 开放类别目标索引为 dataset.num_known（例如已知类别为0和1，则开放类别为2）
            open_target = torch.full(label.size(), dataset.num_known, dtype=torch.long, device=device)
            loss_fake = criterion(logits_fake, open_target)
            
            # 更新分类器 C：既要正确分类真实样本，也要将对抗样本判定为开放类别
            optimizer_C.zero_grad()
            loss_C = loss_real + loss_fake
            loss_C.backward()
            optimizer_C.step()
            
            # 更新特征生成网络 G：对抗目标为让加噪声后的特征被判为真实类别
            optimizer_G.zero_grad()
            noise = torch.randn_like(f_real) * noise_std
            f_fake = G_model(X, A, noise=noise)
            logits_fake_forG = C_model(f_fake)
            loss_G_adv = criterion(logits_fake_forG, label)
            # 同时保持真实样本的判别能力
            loss_G_sup = criterion(C_model(f_real), label)
            loss_G_total = loss_G_sup + lambda_adv * loss_G_adv
            loss_G_total.backward()
            optimizer_G.step()
            
            total_loss_C += loss_C.item()
            total_loss_G += loss_G_total.item()
        
        print(f"Epoch {epoch+1}/{num_epochs} | Loss_C: {total_loss_C/len(dataloader):.4f} | Loss_G: {total_loss_G/len(dataloader):.4f}")

if __name__ == "__main__":
    # 根据实际情况设置数据所在目录和数据集前缀，例如 "COIL-DEL"
    data_directory = "../COIL-DEL"   # 数据文件所在的目录
    ds_prefix = "COIL-DEL"        # 数据集前缀，文件名如 COIL-DEL.node_attrs, COIL-DEL.graph_idx, 等
    train(data_directory, ds_prefix, num_epochs=50)

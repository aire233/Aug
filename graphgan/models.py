import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool


class Generator(nn.Module):
    def __init__(self, latent_dim=64, node_dim=1, max_nodes=50):
        super(Generator, self).__init__()
        self.max_nodes = max_nodes
        self.latent_dim = latent_dim

        self.node_generator = nn.Sequential(
            nn.Linear(latent_dim, 128), nn.ReLU(), nn.Linear(128, max_nodes * node_dim)
        )

        self.edge_generator = nn.Sequential(
            nn.Linear(2 * node_dim, 64), nn.ReLU(), nn.Linear(64, 1), nn.Sigmoid()
        )

    def forward(self, z):
        batch_size = z.size(0)
        # 生成节点特征，形状为 [B, max_nodes, node_dim]
        nodes = self.node_generator(z).view(batch_size, self.max_nodes, -1)
        # 向量化计算所有节点对的边概率
        idx = torch.triu_indices(self.max_nodes, self.max_nodes, offset=1).to(z.device)
        node_pairs = torch.cat([nodes[:, idx[0]], nodes[:, idx[1]]], dim=2)
        edge_probs = self.edge_generator(node_pairs.view(-1, 2 * nodes.size(-1))).view(
            batch_size, -1
        )
        # 构造对称的邻接矩阵
        adj = torch.zeros(batch_size, self.max_nodes, self.max_nodes, device=z.device)
        adj[:, idx[0], idx[1]] = edge_probs
        adj[:, idx[1], idx[0]] = edge_probs
        return nodes, adj


class Discriminator(nn.Module):
    def __init__(self, node_dim=1):
        super(Discriminator, self).__init__()
        self.conv1 = GCNConv(node_dim, 64)
        self.conv2 = GCNConv(64, 32)
        self.fc = nn.Linear(32, 1)

    def forward(self, data):
        # data 应该包含 x, edge_index 以及 batch 信息
        x, edge_index = data.x, data.edge_index
        device = x.device
        x = F.relu(self.conv1(x.to(device), edge_index.to(device)))
        x = F.relu(self.conv2(x, edge_index))
        if hasattr(data, "batch"):
            batch = data.batch.to(device)
        else:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=device)
        x = global_mean_pool(x, batch)
        return torch.sigmoid(self.fc(x))

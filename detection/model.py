import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool, global_max_pool
from torch_geometric.nn.aggr import SortAggregation


class GraphFeatureExtractor(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        # 三层 GCN 提取节点特征
        self.gnn = nn.ModuleList(
            [GCNConv(in_dim, 256), GCNConv(256, 256), GCNConv(256, 128)]
        )
        # 注意力模块对节点特征进行全局聚合
        self.attention = nn.MultiheadAttention(128, num_heads=4)
        # SortPooling 将节点特征进行排序后聚合
        self.sort_pool = SortAggregation(k=10)
        self.dropout = nn.Dropout(0.5)

    def forward(self, data, noise_std=0.0, class_weights=None):
        x = data.x
        edge_index = data.edge_index

        # 逐层 GCN 与 ReLU + Dropout
        for conv in self.gnn:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = self.dropout(x)

        # 注意力聚合：对节点特征进行自注意力操作
        x, _ = self.attention(x.unsqueeze(0), x.unsqueeze(0), x.unsqueeze(0))
        x = x.squeeze(0)

        # 多尺度池化：均值池化、最大池化和SortPooling拼接
        mean_pool = global_mean_pool(x, data.batch)  # 128维
        max_pool = global_max_pool(x, data.batch)  # 128维
        sort_pool = self.sort_pool(x, data.batch)  # 10*128 = 1280维

        feat = torch.cat([mean_pool, max_pool, sort_pool], dim=1)  # 总共1536维

        # 对抗噪声注入（如果设置了noise_std和class_weights）
        if noise_std > 0 and class_weights is not None:
            weights = class_weights[data.y].view(-1, 1)
            noise = torch.randn_like(feat) * noise_std * (1 + weights)
            feat += noise

        return feat


class DomainAwareClassifier(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_classes):
        super().__init__()
        self.fc1 = nn.Sequential(
            nn.Linear(12 * in_dim, 512),
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.5),
        )
        self.fc2 = nn.Linear(512, num_classes)
        # 用于存储正交正则化损失
        self.orth_loss = 0.0

    def forward(self, x):
        x = self.fc1(x)
        # 正交约束（仅在训练时计算，防止特征过于相关）
        if self.training and x.size(0) > 1:
            x_norm = F.normalize(x, p=2, dim=1)
            gram_matrix = torch.mm(x_norm, x_norm.t())
            self.orth_loss = (
                torch.norm(gram_matrix - torch.eye(x.size(0)).to(x.device)) * 0.01
            )
        return self.fc2(x)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool

class SimpleGraphFeatureExtractor(nn.Module):
    """
    两层 GCN 提取节点特征，并用全局均值池化得到图级特征
    """
    def __init__(self, in_dim, hidden_dim=128, out_dim=128):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, out_dim)
        self.dropout = nn.Dropout(0.5)
    
    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.dropout(x)
        # 全局均值池化
        x = global_mean_pool(x, batch)
        return x

class SimpleClassifier(nn.Module):
    """
    简单的 MLP 分类器：一层隐藏层
    """
    def __init__(self, in_dim, hidden_dim=64, num_classes=81):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
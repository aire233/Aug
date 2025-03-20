import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader


# -------------------------------
# 数据加载部分
# -------------------------------
def load_txt(file_path, dtype=int):
    with open(file_path, "r") as f:
        lines = f.readlines()
    return np.array([dtype(line.strip()) for line in lines])


def one_hot_encode(labels, num_classes):
    return np.eye(num_classes)[labels]


class GraphDataset(Dataset):
    """
    根据提供的 DS_A.txt, DS_graph_indicator.txt, DS_node_labels.txt, DS_graph_labels.txt 构造图数据集。
    每个样本为一个图，包含：
      - X: 节点特征矩阵 (num_nodes x feature_dim)，采用 one-hot 编码（这里 feature_dim=65）
      - A: 邻接矩阵 (num_nodes x num_nodes)，已进行归一化处理（加入自环并归一化）
      - label: 图标签（整数，从 0 到 K-1）
    """

    def __init__(self, data_dir, ds_prefix):
        # 文件路径
        path_A = os.path.join(data_dir, f"{ds_prefix}_A.txt")
        path_indicator = os.path.join(data_dir, f"{ds_prefix}_graph_indicator.txt")
        path_node_labels = os.path.join(data_dir, f"{ds_prefix}_node_labels.txt")
        path_graph_labels = os.path.join(data_dir, f"{ds_prefix}_graph_labels.txt")

        # 读取数据
        # DS_graph_indicator.txt：每行给出节点所属图的 id（注意文件中图 id 从 1 开始）
        graph_indicator = load_txt(path_indicator, dtype=int)
        # DS_node_labels.txt：每行是节点标签（这里标签已经转换为整数，取值范围 0~64）
        node_labels = load_txt(path_node_labels, dtype=int)
        # DS_graph_labels.txt：每行是图的类别标签（图 id 顺序）
        graph_labels = load_txt(path_graph_labels, dtype=int)
        # DS_A.txt：每行表示边 (node_i, node_j)，节点 id 从 1 开始
        edges = []
        with open(path_A, "r") as f:
            for line in f:
                src, dst = line.strip().split(',')
                edges.append((int(src) - 1, int(dst) - 1))  # 转换为 0-based

        # 构建图数据。假设图的编号是连续的，从 1 到 N
        unique_graph_ids = np.unique(graph_indicator)
        self.graphs = []
        # 建立节点 id 到所属图 id 的映射
        node2graph = graph_indicator  # 长度为 n，每个元素是图 id

        # 预先将所有边按所属图划分（由于 DS_A.txt 为 block diagonal, 边只存在于同一图中）
        graph_edges = {gid: [] for gid in unique_graph_ids}
        for src, dst in edges:
            gid = node2graph[src]  # 两个节点属于同一图
            graph_edges[gid].append((src, dst))

        # 按图构造数据（注意：由于各图节点在整体中可能是非连续的，我们需要映射到局部索引）
        for gid in unique_graph_ids:
            # 找出属于该图的全局节点 id 列表
            global_node_ids = np.where(node2graph == gid)[0]
            num_nodes = len(global_node_ids)
            # 构造节点特征：利用节点标签进行 one-hot 编码
            # 假设节点标签的取值范围 0~64，共65类
            node_feats = one_hot_encode(node_labels[global_node_ids], num_classes=65)
            # 构造邻接矩阵（先构造稀疏矩阵，再转为 dense）
            A = np.zeros((num_nodes, num_nodes), dtype=np.float32)
            # 建立全局到局部索引映射
            global2local = {g: i for i, g in enumerate(global_node_ids)}
            # 取出边：只保留该图内部的边
            for src, dst in graph_edges[gid]:
                if src in global2local and dst in global2local:
                    i, j = global2local[src], global2local[dst]
                    A[i, j] = 1.0
                    A[j, i] = 1.0  # 无向图
            # 添加自环并归一化邻接矩阵（简化的对称归一化）
            A = A + np.eye(num_nodes, dtype=np.float32)
            D = np.sum(A, axis=1)
            D_inv_sqrt = np.diag(1.0 / np.sqrt(D))
            A_norm = D_inv_sqrt @ A @ D_inv_sqrt

            # 图标签（注意：graph_labels 文件中行号与 gid 对应，且 gid 从 1 开始）
            # 我们将图标签转换为从 0 开始
            graph_label = int(graph_labels[gid - 1])
            self.graphs.append(
                {
                    "X": torch.tensor(node_feats, dtype=torch.float),
                    "A": torch.tensor(A_norm, dtype=torch.float),
                    "label": torch.tensor(graph_label, dtype=torch.long),
                }
            )

        # 得到已知类别数量 K（开放类别为 K）
        self.known_labels = sorted(list({g["label"].item() for g in self.graphs}))
        self.num_known = len(self.known_labels)
        print(f"Loaded {len(self.graphs)} graphs, {self.num_known} known classes.")

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
        # 简单实现：先做邻接矩阵乘法，再线性变换
        out = torch.matmul(A, X)
        out = self.linear(out)
        return out


class GraphFeatureExtractor(nn.Module):
    """
    特征生成网络 G：
      - 两层图卷积
      - 节点特征池化（均值池化）得到图级表示
      - 提供噪声接口，用于生成对抗样本（在图级特征上加噪声）
    """

    def __init__(self, in_features=65, hidden_dim=128, out_dim=128):
        super(GraphFeatureExtractor, self).__init__()
        self.conv1 = GCNLayer(in_features, hidden_dim)
        self.conv2 = GCNLayer(hidden_dim, out_dim)
        self.relu = nn.ReLU()

    def forward(self, X, A, noise=None):
        x = self.conv1(X, A)
        x = self.relu(x)
        x = self.conv2(x, A)
        x = self.relu(x)
        # 节点池化（均值池化），得到图级表示：尺寸 (1, out_dim)
        graph_feat = torch.mean(x, dim=0, keepdim=True)
        if noise is not None:
            graph_feat = graph_feat + noise
        return graph_feat


class Classifier(nn.Module):
    """
    分类器网络 C：简单 MLP，输出 (K+1) 维 logits，
    前 K 维为已知类别，最后一维为开放类别。
    """

    def __init__(self, in_dim=128, hidden_dim=64, num_classes=7):
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
def train(data_dir, ds_prefix, num_epochs=100, lr=1e-3, noise_std=0.1, lambda_adv=1.0):
    # 加载数据
    dataset = GraphDataset(data_dir, ds_prefix)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)  # 每次一个图

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 模型：注意分类器输出维度 = num_known + 1（开放类别）
    G_model = GraphFeatureExtractor().to(device)
    C_model = Classifier(num_classes=dataset.num_known + 1).to(device)

    optimizer_G = optim.Adam(G_model.parameters(), lr=lr)
    optimizer_C = optim.Adam(C_model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        G_model.train()
        C_model.train()
        total_loss_C = 0.0
        total_loss_G = 0.0
        for data in dataloader:
            X, A, label = (
                data["X"].to(device),
                data["A"].to(device),
                data["label"].to(device),
            )
            # data["X"]: (num_nodes, feature_dim)
            # data["A"]: (num_nodes, num_nodes)
            # label: (1,) 整数，取值范围 0~K-1

            # 计算真实图特征
            f_real = G_model(X, A)  # shape: (1, feature_dim)
            logits_real = C_model(f_real)
            loss_real = criterion(logits_real, label)

            # 生成对抗样本：在图级特征上加入噪声
            noise = torch.randn_like(f_real) * noise_std
            f_fake = G_model(X, A, noise=noise)
            logits_fake = C_model(f_fake)
            # 对于生成样本，分类器的目标是判断为开放类别（标签为 num_known）
            open_target = torch.full(
                (label.size(0),), dataset.num_known, dtype=torch.long, device=device
            )
            loss_fake = criterion(logits_fake, open_target)

            # 更新分类器 C
            optimizer_C.zero_grad()
            loss_C = loss_real + loss_fake
            loss_C.backward()
            optimizer_C.step()

            # 更新特征生成网络 G 对抗性部分：
            # 目标：使得经过加噪声后的 f_fake 被判为真实类别 label（欺骗分类器）
            optimizer_G.zero_grad()
            noise = torch.randn_like(f_real) * noise_std
            f_fake = G_model(X, A, noise=noise)
            logits_fake_forG = C_model(f_fake)
            loss_G_adv = criterion(logits_fake_forG, label)
            # 同时对真实样本保持判别能力（监督损失）
            loss_G_sup = criterion(C_model(f_real), label)
            loss_G_total = loss_G_sup + lambda_adv * loss_G_adv
            loss_G_total.backward()
            optimizer_G.step()

            total_loss_C += loss_C.item()
            total_loss_G += loss_G_total.item()

        print(
            f"Epoch {epoch+1}/{num_epochs} | Loss_C: {total_loss_C/len(dataloader):.4f} | Loss_G: {total_loss_G/len(dataloader):.4f}"
        )


if __name__ == "__main__":
    # 请设置数据所在目录及数据集前缀（例如：如果文件为 DS_A.txt, 则 ds_prefix 为 "DS"）
    data_directory = "./SW-620"  # 数据文件所在的文件夹
    ds_prefix = "SW-620"  # 数据集前缀
    train(data_directory, ds_prefix, num_epochs=50)

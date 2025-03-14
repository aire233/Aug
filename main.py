from tqdm import tqdm
import numpy as np
import networkx as nx
from torch_geometric.data import Data, Dataset
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.loader import DataLoader
from torch_geometric.utils import dense_to_sparse

# 检查是否有可用的 GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using device:", device)

# 加载邻接矩阵
def load_adjacency_matrix(file_path):
    edges = []
    with open(file_path, 'r') as f:
        for line in f:
            node1, node2 = map(int, line.strip().split(','))
            edges.append((node1, node2))
    return np.array(edges, dtype=int)

# 加载图指示器
def load_graph_indicator(file_path):
    graph_indicator = np.loadtxt(file_path, dtype=int)
    return graph_indicator

# 加载节点标签
def load_node_labels(file_path):
    node_labels = np.loadtxt(file_path, dtype=int)
    return node_labels

# 加载图标签
def load_graph_labels(file_path):
    graph_labels = np.loadtxt(file_path, dtype=int)
    return graph_labels

# 构建图结构
def build_graphs(edges, graph_indicator, node_labels):
    graphs = {}
    print("Building graphs...")
    for edge in tqdm(edges, desc="Building graphs"):
        node1, node2 = edge
        graph_id = graph_indicator[node1 - 1]  # 节点 ID 从 1 开始
        if graph_id not in graphs:
            graphs[graph_id] = nx.Graph()
        graphs[graph_id].add_edge(node1, node2)
        graphs[graph_id].nodes[node1]['label'] = node_labels[node1 - 1]
        graphs[graph_id].nodes[node2]['label'] = node_labels[node2 - 1]
    return graphs

# 加载所有数据
edges = load_adjacency_matrix('SW-620/SW-620_A.txt')
graph_indicator = load_graph_indicator('SW-620/SW-620_graph_indicator.txt')
node_labels = load_node_labels('SW-620/SW-620_node_labels.txt')
graph_labels = load_graph_labels('SW-620/SW-620_graph_labels.txt')

# 构建图
graphs = build_graphs(edges, graph_indicator, node_labels)

# 打印图信息
print(f"Total graphs: {len(graphs)}")
print(f"Graph 1 nodes: {graphs[1].number_of_nodes()}, edges: {graphs[1].number_of_edges()}")

# 转换为 PyG 格式，不再包含 dense 邻接矩阵
def convert_to_pyg(graphs, graph_labels, max_nodes=50):
    pyg_data_list = []
    print("Converting graphs to PyG format...")
    # 先确定数据集中最大节点数
    all_nodes = [g.number_of_nodes() for g in graphs.values()]
    max_nodes = max(all_nodes)
    print(f"Maximum nodes in dataset: {max_nodes}")

    for graph_id, graph in tqdm(graphs.items(), desc="Converting graphs"):
        # 排序节点保证顺序一致
        nodes = sorted(list(graph.nodes()))
        num_nodes = len(nodes)
        x = torch.zeros((max_nodes, 1), dtype=torch.float)
        x[:num_nodes] = torch.tensor([graph.nodes[node]['label'] for node in nodes]).unsqueeze(1)
        # 构建节点 id 到索引的映射
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

# 转换为 PyG 格式（不包含 dense 邻接矩阵）
pyg_data_list, max_nodes = convert_to_pyg(graphs, graph_labels)

# 定义 GraphGAN 生成器
class Generator(nn.Module):
    def __init__(self, latent_dim=64, node_dim=1, max_nodes=50):
        super().__init__()
        self.max_nodes = max_nodes
        self.latent_dim = latent_dim

        self.node_generator = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, max_nodes * node_dim)
        )

        self.edge_generator = nn.Sequential(
            nn.Linear(2 * node_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, z):
        batch_size = z.size(0)
        # 生成节点特征 [batch_size, max_nodes, node_dim]
        nodes = self.node_generator(z).view(batch_size, self.max_nodes, -1)
        # 生成对称的邻接矩阵
        adj = torch.zeros(batch_size, self.max_nodes, self.max_nodes).to(device)
        for i in range(self.max_nodes):
            for j in range(i+1, self.max_nodes):
                pair = torch.cat([nodes[:, i, :], nodes[:, j, :]], dim=1)
                edge_prob = self.edge_generator(pair).squeeze()
                adj[:, i, j] = edge_prob
                adj[:, j, i] = edge_prob  # 保证对称
        return nodes, adj

class Discriminator(nn.Module):
    def __init__(self, node_dim=1):
        super().__init__()
        self.conv1 = GCNConv(node_dim, 64)
        self.conv2 = GCNConv(64, 32)
        self.fc = nn.Linear(32, 1)

    def forward(self, data):
        x, edge_index = data.x.to(device), data.edge_index.to(device)
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        # 如果 data 包含 batch 属性，则使用它，否则使用全0张量
        if hasattr(data, 'batch'):
            batch = data.batch.to(device)
        else:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=device)
        x = global_mean_pool(x, batch)
        return torch.sigmoid(self.fc(x))

# 训练参数
EPOCHS = 100
BATCH_SIZE = 32
LATENT_DIM = 64

# 初始化模型
generator = Generator(max_nodes=max_nodes, latent_dim=LATENT_DIM).to(device)
discriminator = Discriminator().to(device)

# 优化器
g_optimizer = torch.optim.Adam(generator.parameters(), lr=0.001)
d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=0.001)

# 数据加载器（真实数据已在CPU上）
loader = DataLoader(pyg_data_list, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)

# 训练循环
for epoch in tqdm(range(EPOCHS), desc="Training epochs"):
    epoch_d_loss = 0.0
    epoch_g_loss = 0.0
    num_batches = 0

    for real_data in tqdm(loader, desc="Batches", leave=False):
        real_data = real_data.to(device)
        batch_size = real_data.num_graphs

        # 训练判别器
        d_optimizer.zero_grad()
        # 真实数据损失
        real_loss = F.binary_cross_entropy(
            discriminator(real_data),
            torch.ones(batch_size, 1, device=device)
        )

        # 生成数据（在训练判别器时阻断生成器梯度）
        z = torch.randn(batch_size, LATENT_DIM, device=device)
        gen_nodes, gen_adj = generator(z)
        gen_nodes = gen_nodes.detach()
        gen_adj = gen_adj.detach()

        fake_data_list = []
        for i in range(batch_size):
            edge_index, _ = dense_to_sparse(gen_adj[i])
            # 将 GPU 上的张量转换到 CPU
            edge_index = edge_index.cpu()
            data = Data(x=gen_nodes[i].cpu(), edge_index=edge_index)
            fake_data_list.append(data)
        fake_data = next(iter(DataLoader(fake_data_list, batch_size=batch_size, pin_memory=True)))
        fake_data = fake_data.to(device)

        fake_loss = F.binary_cross_entropy(
            discriminator(fake_data),
            torch.zeros(batch_size, 1, device=device)
        )

        d_loss = real_loss + fake_loss
        d_loss.backward()
        d_optimizer.step()

        # 训练生成器
        g_optimizer.zero_grad()
        z = torch.randn(batch_size, LATENT_DIM, device=device)
        gen_nodes, gen_adj = generator(z)

        fake_data_list = []
        for i in range(batch_size):
            edge_index, _ = dense_to_sparse(gen_adj[i])
            edge_index = edge_index.cpu()
            data = Data(x=gen_nodes[i].cpu(), edge_index=edge_index)
            fake_data_list.append(data)
        fake_data = next(iter(DataLoader(fake_data_list, batch_size=batch_size, pin_memory=True)))
        fake_data = fake_data.to(device)

        g_loss = F.binary_cross_entropy(
            discriminator(fake_data),
            torch.ones(batch_size, 1, device=device)
        )

        g_loss.backward()
        g_optimizer.step()

        epoch_d_loss += d_loss.item()
        epoch_g_loss += g_loss.item()
        num_batches += 1

    avg_d_loss = epoch_d_loss / num_batches
    avg_g_loss = epoch_g_loss / num_batches
    tqdm.write(f"Epoch [{epoch+1}/{EPOCHS}] | D Loss: {avg_d_loss:.4f} | G Loss: {avg_g_loss:.4f}")

# 图生成示例
def generate_graph(generator, num_graphs=1):
    z = torch.randn(num_graphs, LATENT_DIM, device=device)
    with torch.no_grad():
        nodes, adj = generator(z)
    return nodes.cpu(), adj.cpu()

# 生成新的图
generated_nodes, generated_adj = generate_graph(generator)
print("Generated adjacency matrix example:", generated_adj[0])

from tqdm import tqdm
import numpy as np
import networkx as nx
from torch_geometric.data import Data, Batch
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.loader import DataLoader
from torch_geometric.utils import dense_to_sparse
import os

# 检查是否有可用的 GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using device:", device)

# -------------------- 数据加载与图构建 --------------------
def load_adjacency_matrix(file_path):
    edges = []
    with open(file_path, 'r') as f:
        for line in f:
            node1, node2 = map(int, line.strip().split(','))
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
        graph_id = graph_indicator[node1 - 1]  # 节点 ID 从 1 开始
        if graph_id not in graphs:
            graphs[graph_id] = nx.Graph()
        graphs[graph_id].add_edge(node1, node2)
        graphs[graph_id].nodes[node1]['label'] = node_labels[node1 - 1]
        graphs[graph_id].nodes[node2]['label'] = node_labels[node2 - 1]
    return graphs

# 加载数据
edges = load_adjacency_matrix('SW-620/SW-620_A.txt')
graph_indicator = load_graph_indicator('SW-620/SW-620_graph_indicator.txt')
node_labels = load_node_labels('SW-620/SW-620_node_labels.txt')
graph_labels = load_graph_labels('SW-620/SW-620_graph_labels.txt')
graphs = build_graphs(edges, graph_indicator, node_labels)
print(f"Total graphs: {len(graphs)}")
print(f"Graph 1 nodes: {graphs[1].number_of_nodes()}, edges: {graphs[1].number_of_edges()}")

def convert_to_pyg(graphs, graph_labels):
    pyg_data_list = []
    print("Converting graphs to PyG format...")
    all_nodes = [g.number_of_nodes() for g in graphs.values()]
    max_nodes = max(all_nodes)
    print(f"Maximum nodes in dataset: {max_nodes}")

    for graph_id, graph in tqdm(graphs.items(), desc="Converting graphs"):
        nodes = sorted(list(graph.nodes()))
        num_nodes = len(nodes)
        x = torch.zeros((max_nodes, 1), dtype=torch.float)
        x[:num_nodes] = torch.tensor([graph.nodes[node]['label'] for node in nodes]).unsqueeze(1)
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

pyg_data_list, max_nodes = convert_to_pyg(graphs, graph_labels)

# -------------------- 定义模型 --------------------
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
        # 生成节点特征，形状 [B, max_nodes, node_dim]
        nodes = self.node_generator(z).view(batch_size, self.max_nodes, -1)
        # 利用向量化一次性计算所有节点对的边概率
        idx = torch.triu_indices(self.max_nodes, self.max_nodes, offset=1).to(z.device)
        # 构造所有节点对：[B, num_pairs, 2 * node_dim]
        node_pairs = torch.cat([nodes[:, idx[0]], nodes[:, idx[1]]], dim=2)
        # 一次性计算所有边概率
        edge_probs = self.edge_generator(node_pairs.view(-1, 2 * nodes.size(-1))).view(batch_size, -1)
        # 构造对称的邻接矩阵
        adj = torch.zeros(batch_size, self.max_nodes, self.max_nodes, device=z.device)
        adj[:, idx[0], idx[1]] = edge_probs
        adj[:, idx[1], idx[0]] = edge_probs
        return nodes, adj

class Discriminator(nn.Module):
    def __init__(self, node_dim=1):
        super().__init__()
        self.conv1 = GCNConv(node_dim, 64)
        self.conv2 = GCNConv(64, 32)
        self.fc = nn.Linear(32, 1)

    def forward(self, data):
        # data 应包含 x, edge_index, 以及 batch 信息
        x, edge_index = data.x.to(device), data.edge_index.to(device)
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        batch = data.batch.to(device) if hasattr(data, 'batch') else torch.zeros(x.size(0), dtype=torch.long, device=device)
        x = global_mean_pool(x, batch)
        return torch.sigmoid(self.fc(x))

# -------------------- 批量转换生成图为 PyG 格式 --------------------
def batch_dense_to_pyg(gen_nodes, gen_adj, threshold=0.5):
    """
    将生成器输出的批量假数据转换为一个 PyG Batch 对象
    gen_nodes: [B, N, node_dim]
    gen_adj: [B, N, N]
    """
    B, N, node_dim = gen_nodes.size()
    # 将节点特征展开：[B*N, node_dim]
    x = gen_nodes.view(B * N, node_dim)
    # 构造每个节点所属图的 batch 信息
    batch_indices = torch.arange(B, device=gen_nodes.device).repeat_interleave(N)
    # 利用阈值筛选边（可根据需要调整阈值）
    mask = (gen_adj > threshold)
    idx = mask.nonzero(as_tuple=False)  # [num_edges, 3]，每行为 [b, i, j]
    if idx.size(0) == 0:
        edge_index = torch.empty((2, 0), dtype=torch.long, device=gen_nodes.device)
    else:
        b, i, j = idx.unbind(dim=1)
        # 计算全局节点索引：global_index = b * N + local_index
        global_i = b * N + i
        global_j = b * N + j
        edge_index = torch.stack([global_i, global_j], dim=0)
    return Batch(x=x, edge_index=edge_index, batch=batch_indices)

# -------------------- 训练设置与检查点 --------------------
EPOCHS = 100
BATCH_SIZE = 32
LATENT_DIM = 64

generator = Generator(max_nodes=max_nodes, latent_dim=LATENT_DIM).to(device)
discriminator = Discriminator().to(device)
g_optimizer = torch.optim.Adam(generator.parameters(), lr=0.001)
d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=0.001)

checkpoint_path = './graphgan_checkpoint.pth'
start_epoch = 0
if os.path.exists(checkpoint_path):
    checkpoint = torch.load(checkpoint_path)
    generator.load_state_dict(checkpoint['generator_state_dict'])
    discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
    g_optimizer.load_state_dict(checkpoint['g_optimizer_state_dict'])
    d_optimizer.load_state_dict(checkpoint['d_optimizer_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
    print(f"Checkpoint loaded from {checkpoint_path}, starting from epoch {start_epoch}.")
else:
    print("No checkpoint found, starting training from scratch.")

# 真实数据 DataLoader
real_loader = DataLoader(pyg_data_list, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)

# -------------------- 训练循环 --------------------
if start_epoch < EPOCHS:
    for epoch in tqdm(range(start_epoch, EPOCHS), desc="Training epochs"):
        epoch_d_loss = 0.0
        epoch_g_loss = 0.0
        num_batches = 0

        for real_data in tqdm(real_loader, desc="Batches", leave=False):
            real_data = real_data.to(device)
            batch_size = real_data.num_graphs

            # 训练判别器
            d_optimizer.zero_grad()
            real_loss = F.binary_cross_entropy(
                discriminator(real_data),
                torch.ones(batch_size, 1, device=device)
            )

            z = torch.randn(batch_size, LATENT_DIM, device=device)
            gen_nodes, gen_adj = generator(z)
            # 构造假数据 Batch（detach 防止梯度回传）
            fake_data = batch_dense_to_pyg(gen_nodes.detach(), gen_adj.detach(), threshold=0.5)
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
            fake_data = batch_dense_to_pyg(gen_nodes, gen_adj, threshold=0.5)
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

        # 保存检查点
        torch.save({
            'epoch': epoch,
            'generator_state_dict': generator.state_dict(),
            'discriminator_state_dict': discriminator.state_dict(),
            'g_optimizer_state_dict': g_optimizer.state_dict(),
            'd_optimizer_state_dict': d_optimizer.state_dict(),
        }, checkpoint_path)
        tqdm.write(f"Checkpoint saved to {checkpoint_path}")
else:
    print("Training already completed based on the saved checkpoint.")

# -------------------- 生成图示例 --------------------
def generate_graph(generator, num_graphs=1):
    z = torch.randn(num_graphs, LATENT_DIM, device=device)
    with torch.no_grad():
        nodes, adj_matrices = generator(z)
    for i, adj in enumerate(adj_matrices.cpu().numpy()):  # 转换为 numpy 数组
        filename = f"generated_graph/generated_graph_{i}.txt"      
        with open(filename, "w") as w:
            num_nodes = adj.shape[0]
            for u in range(num_nodes):
                for v in range(num_nodes):
                    if adj[u, v] > 0:  # 只存储存在的边
                        w.write(f"{u} {v}\n")
                        w.write(f"{v} {u}\n")
        print(f"Graph {i} saved to {filename}")
    return nodes.cpu(), adj_matrices.cpu()

generated_nodes, generated_adj = generate_graph(generator)
# print("Generated adjacency matrix example:", generated_adj[0])


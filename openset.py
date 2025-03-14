import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from tqdm import tqdm

#############################
# 数据加载与预处理
#############################
class GraphDataset(torch.utils.data.Dataset):
    """
    加载图数据，要求文件放在 folder_path 下，文件名格式为：
       DS_A.txt, DS_graph_indicator.txt, DS_graph_labels.txt, DS_node_labels.txt
    """
    def __init__(self, folder_path, ds_prefix):
        super(GraphDataset, self).__init__()
        self.folder_path = folder_path
        self.ds_prefix = ds_prefix
        
        # 加载各文件（请根据实际文件格式调整 delimiter）
        self.edges = np.loadtxt(os.path.join(folder_path, f"{ds_prefix}_A.txt"), delimiter=',', dtype=int)
        self.graph_indicator = np.loadtxt(os.path.join(folder_path, f"{ds_prefix}_graph_indicator.txt"), delimiter=',', dtype=int)
        self.graph_labels = np.loadtxt(os.path.join(folder_path, f"{ds_prefix}_graph_labels.txt"), delimiter=',', dtype=int)
        self.node_labels = np.loadtxt(os.path.join(folder_path, f"{ds_prefix}_node_labels.txt"), delimiter=',', dtype=int)
        
        # 构建图ID到节点索引的映射
        self.graph2node = {}
        for i, g_id in enumerate(self.graph_indicator):
            self.graph2node.setdefault(g_id, []).append(i)
        
        # 对节点标签进行 one-hot 编码（编码维度为最大标签值+1）
        num_node_labels = int(self.node_labels.max()) + 1
        node_features = F.one_hot(torch.tensor(self.node_labels, dtype=torch.long),
                                    num_classes=num_node_labels).float()
        
        # 预先将全局边分组到对应的图中，减少后续每个图中遍历所有边的时间开销
        graph_edges = {}
        for edge in self.edges:
            # 注意：文件中节点编号从1开始
            i, j = edge[0]-1, edge[1]-1
            # 利用 graph_indicator 获取节点所属图（假定边的两个端点属于同一图）
            g_id = self.graph_indicator[i]
            # 如果有异常情况，也可以加判断： if self.graph_indicator[i] != self.graph_indicator[j]: continue
            graph_edges.setdefault(g_id, []).append((i, j))
        
        # 构造每个图的数据
        self.graphs = []
        for g_id in tqdm(sorted(self.graph2node.keys()), desc="Constructing graph data"):
            nodes = self.graph2node[g_id]
            x = node_features[nodes]  # 节点特征矩阵
            num_nodes = x.shape[0]
            # 初始化邻接矩阵（无自环）
            adj = np.zeros((num_nodes, num_nodes), dtype=np.float32)
            # 构建全局节点id到局部索引的映射
            nodeid2local = {global_id: local_id for local_id, global_id in enumerate(nodes)}
            # 仅遍历属于当前图的边
            for i, j in graph_edges.get(g_id, []):
                li, lj = nodeid2local[i], nodeid2local[j]
                adj[li, lj] = 1.
                adj[lj, li] = 1.  # 无向图，确保对称
            label = int(self.graph_labels[g_id-1])
            self.graphs.append({'x': x, 'adj': torch.tensor(adj), 'label': label})
            
    def __len__(self):
        return len(self.graphs)
    
    def __getitem__(self, idx):
        return self.graphs[idx]

def collate_fn(batch):
    # 由于每个图大小不同，这里直接返回图列表，后续在模型中单独处理
    return batch

#############################
# 模型定义：GCN层、生成器G与分类器C
#############################
class GCNLayer(nn.Module):
    def __init__(self, in_features, out_features, dropout=0.0):
        super(GCNLayer, self).__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, adj):
        # 添加自环
        I = torch.eye(adj.size(0), device=adj.device)
        A_hat = adj + I
        # 计算 D^(-1/2)
        D = torch.sum(A_hat, dim=1)
        D_inv_sqrt = torch.diag(torch.pow(D, -0.5))
        # 归一化邻接矩阵
        A_norm = D_inv_sqrt @ A_hat @ D_inv_sqrt
        out = A_norm @ x
        out = self.linear(out)
        out = self.dropout(out)
        return out

class GraphFeatureExtractor(nn.Module):
    """
    特征生成网络 G：两层 GCN + ReLU 激活 + 全局均值池化得到图级特征
    """
    def __init__(self, in_features, hidden_dim, out_dim, dropout=0.0):
        super(GraphFeatureExtractor, self).__init__()
        self.gcn1 = GCNLayer(in_features, hidden_dim, dropout)
        self.gcn2 = GCNLayer(hidden_dim, out_dim, dropout)
    
    def forward(self, x, adj):
        out = F.relu(self.gcn1(x, adj))
        out = self.gcn2(out, adj)
        graph_feature = torch.mean(out, dim=0)  # 全局均值池化
        return graph_feature

class Classifier(nn.Module):
    """
    分类器 C：简单的 MLP，输出 K+1 维概率（最后一维为未知类）
    """
    def __init__(self, in_dim, hidden_dim, num_known_classes, dropout=0.0):
        super(Classifier, self).__init__()
        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, num_known_classes + 1)  # +1 表示未知类
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        logits = self.fc2(x)
        return logits

#############################
# 对抗样本生成函数
#############################
def generate_adversarial_feature_for_classifier(f, classifier, epsilon, device):
    """
    针对分类器更新：输入 f 已 detach，梯度只作用于分类器参数。
    """
    f_adv = f.clone().detach().to(device)
    f_adv.requires_grad = True
    # 目标为未知类，其索引为 classifier.fc2.out_features - 1
    target = torch.tensor([classifier.fc2.out_features - 1], dtype=torch.long, device=device)
    logits = classifier(f_adv)
    loss = F.cross_entropy(logits.unsqueeze(0), target)
    grad = torch.autograd.grad(loss, f_adv)[0]
    f_adv = f_adv + epsilon * torch.sign(grad)
    return f_adv.detach()  # 返回时 detach

def generate_adversarial_feature_for_generator(f, classifier, epsilon, device):
    """
    针对生成器更新：冻结分类器参数，使梯度只传递给 f，
    使用 create_graph=True 构造二阶梯度，确保计算图完整。
    """
    # 暂时冻结分类器参数
    classifier_params = list(classifier.parameters())
    orig_grad = [p.requires_grad for p in classifier_params]
    for p in classifier_params:
        p.requires_grad = False

    target = torch.tensor([classifier.fc2.out_features - 1],
                          dtype=torch.long, device=device)
    logits = classifier(f)
    loss = F.cross_entropy(logits.unsqueeze(0), target)
    # 这里使用 create_graph=True 构造二阶梯度
    grad = torch.autograd.grad(loss, f, create_graph=True)[0]
    f_adv = f + epsilon * torch.sign(grad)

    # 恢复分类器参数状态
    for p, req in zip(classifier_params, orig_grad):
        p.requires_grad = req
    return f_adv

#############################
# 交替训练过程
#############################
def train(model_G, model_C, dataloader, num_known_classes, device,
          num_epochs=50, epsilon=0.1, lambda_adv=1.0, lr=1e-3):
    optimizer_G = optim.Adam(model_G.parameters(), lr=lr)
    optimizer_C = optim.Adam(model_C.parameters(), lr=lr)
    
    model_G.train()
    model_C.train()
    
    for epoch in range(num_epochs):
        total_loss_C = 0.0
        total_loss_G = 0.0
        
        for batch in dataloader:
            loss_C_list = []
            loss_G_list = []
            for data in batch:
                x = data['x'].to(device)       # 节点特征
                adj = data['adj'].to(device)     # 邻接矩阵
                true_label = torch.tensor([data['label']], dtype=torch.long, device=device)
                
                # ===== 分类器更新 =====
                with torch.no_grad():
                    f = model_G(x, adj)
                f_detached = f.detach()
                logits_real = model_C(f_detached)
                loss_cls = F.cross_entropy(logits_real.unsqueeze(0), true_label)
                
                f_adv_cls = generate_adversarial_feature_for_classifier(f_detached, model_C, epsilon, device)
                logits_adv = model_C(f_adv_cls)
                target_unknown = torch.tensor([num_known_classes], dtype=torch.long, device=device)
                loss_adv = F.cross_entropy(logits_adv.unsqueeze(0), target_unknown)
                loss_C = loss_cls + lambda_adv * loss_adv
                loss_C_list.append(loss_C)
                
                # ===== 生成器更新 =====
                f = model_G(x, adj)  # 重新计算 f，保留梯度
                f_adv_gen = generate_adversarial_feature_for_generator(f, model_C, epsilon, device)
                logits_real_gen = model_C(f)
                loss_cls_gen = F.cross_entropy(logits_real_gen.unsqueeze(0), true_label)
                logits_adv_gen = model_C(f_adv_gen)
                loss_adv_gen = F.cross_entropy(logits_adv_gen.unsqueeze(0), target_unknown)
                loss_G = loss_cls_gen - lambda_adv * loss_adv_gen
                loss_G_list.append(loss_G)
            
            loss_C_batch = sum(loss_C_list)
            loss_G_batch = sum(loss_G_list)
            
            optimizer_C.zero_grad()
            loss_C_batch.backward(retain_graph=True)
            optimizer_C.step()
            
            optimizer_G.zero_grad()
            loss_G_batch.backward()
            optimizer_G.step()
            
            total_loss_C += loss_C_batch.item()
            total_loss_G += loss_G_batch.item()
        
        print(f"Epoch {epoch+1}/{num_epochs}, Loss_C: {total_loss_C:.4f}, Loss_G: {total_loss_G:.4f}")




#############################
# 主函数
#############################
def main():
    # 参数设置，根据实际情况调整
    folder_path = "./SW-620"  # 数据文件所在目录
    ds_prefix = "SW-620"        # 数据集前缀，例如 DS
    num_known_classes = 2   # 假设已知类别编号为 0 ~ num_known_classes-1，请根据数据调整
    in_feature_dim = 65     # 节点特征维度（依据 DS_node_labels 的映射，通常为标签最大值+1）
    hidden_dim_G = 32
    out_dim_G = 32        # 图级特征维度
    hidden_dim_C = 32
    num_epochs = 100
    epsilon = 0.1
    lambda_adv = 1.0
    lr = 1e-3
    dropout_rate = 0.5     # dropout 概率，可根据需要调整

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 加载数据集
    dataset = GraphDataset(folder_path, ds_prefix)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=collate_fn)
    
    # 模型初始化
    model_G = GraphFeatureExtractor(in_features=in_feature_dim, hidden_dim=hidden_dim_G,
                                    out_dim=out_dim_G, dropout=dropout_rate).to(device)
    model_C = Classifier(in_dim=out_dim_G, hidden_dim=hidden_dim_C,
                         num_known_classes=num_known_classes, dropout=dropout_rate).to(device)
    
    # 训练
    train(model_G, model_C, dataloader, num_known_classes, device,
          num_epochs=num_epochs, epsilon=epsilon, lambda_adv=lambda_adv, lr=lr)
    
    # 训练结束后保存模型或进一步评估
    torch.save(model_G.state_dict(), "model_G.pth")
    torch.save(model_C.state_dict(), "model_C.pth")
    
if __name__ == "__main__":
    main()

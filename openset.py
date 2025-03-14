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
    def __init__(self, folder_path, ds_prefix):
        super(GraphDataset, self).__init__()
        self.folder_path = folder_path
        self.ds_prefix = ds_prefix
        
        self.edges = np.loadtxt(os.path.join(folder_path, f"{ds_prefix}_A.txt"), delimiter=',', dtype=int)
        self.graph_indicator = np.loadtxt(os.path.join(folder_path, f"{ds_prefix}_graph_indicator.txt"), delimiter=',', dtype=int)
        self.graph_labels = np.loadtxt(os.path.join(folder_path, f"{ds_prefix}_graph_labels.txt"), delimiter=',', dtype=int)
        self.node_labels = np.loadtxt(os.path.join(folder_path, f"{ds_prefix}_node_labels.txt"), delimiter=',', dtype=int)
        
        self.graph2node = {}
        for i, g_id in enumerate(self.graph_indicator):
            if g_id not in self.graph2node:
                self.graph2node[g_id] = []
            self.graph2node[g_id].append(i)
        
        self.graphs = []
        num_node_labels = int(self.node_labels.max()) + 1
        node_features = F.one_hot(torch.tensor(self.node_labels, dtype=torch.long), num_classes=num_node_labels).float()
        
        for g_id in sorted(self.graph2node.keys()):
            nodes = self.graph2node[g_id]
            x = node_features[nodes]
            num_nodes = x.shape[0]
            adj = np.zeros((num_nodes, num_nodes), dtype=np.float32)
            nodeid2local = {global_id: local_id for local_id, global_id in enumerate(nodes)}
            for edge in self.edges:
                i, j = edge[0]-1, edge[1]-1
                if i in nodeid2local and j in nodeid2local:
                    li, lj = nodeid2local[i], nodeid2local[j]
                    adj[li, lj] = 1.
                    adj[lj, li] = 1.
            label = int(self.graph_labels[g_id-1])
            self.graphs.append({
                'x': x, 
                'adj': torch.tensor(adj), 
                'label': label
            })
            
    def __len__(self):
        return len(self.graphs)
    
    def __getitem__(self, idx):
        return self.graphs[idx]

def collate_fn(batch):
    return batch

#############################
# 训练过程（添加进度条）
#############################
def train(model_G, model_C, dataloader, num_known_classes, device,
          num_epochs=50, epsilon=0.1, lambda_adv=1.0, lr=1e-3):
    optimizer_G = optim.Adam(model_G.parameters(), lr=lr)
    optimizer_C = optim.Adam(model_C.parameters(), lr=lr)
    
    model_G.train()
    model_C.train()
    
    for epoch in range(num_epochs):
        total_loss_G = 0.0
        total_loss_C = 0.0
        
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}", leave=False)
        
        for batch in progress_bar:
            loss_G_batch = 0.0
            loss_C_batch = 0.0
            
            for data in batch:
                x = data['x'].to(device)
                adj = data['adj'].to(device)
                true_label = torch.tensor([data['label']], dtype=torch.long, device=device)
                
                f = model_G(x, adj)
                logits_real = model_C(f)
                loss_cls = F.cross_entropy(logits_real.unsqueeze(0), true_label)
                
                f_adv = generate_adversarial_feature(f, model_C, epsilon, device)
                logits_adv = model_C(f_adv)
                target_unknown = torch.tensor([num_known_classes], dtype=torch.long, device=device)
                loss_adv = F.cross_entropy(logits_adv.unsqueeze(0), target_unknown)
                
                loss_C = loss_cls + lambda_adv * loss_adv
                loss_G = loss_cls - lambda_adv * loss_adv
                
                loss_G_batch += loss_G
                loss_C_batch += loss_C
            
            optimizer_G.zero_grad()
            loss_G_batch.backward(retain_graph=True)
            optimizer_G.step()
            
            optimizer_C.zero_grad()
            loss_C_batch.backward()
            optimizer_C.step()
            
            total_loss_G += loss_G_batch.item()
            total_loss_C += loss_C_batch.item()
            
            progress_bar.set_postfix(Loss_G=f"{total_loss_G:.4f}", Loss_C=f"{total_loss_C:.4f}")
        
        print(f"Epoch {epoch+1}/{num_epochs}, Loss_G: {total_loss_G:.4f}, Loss_C: {total_loss_C:.4f}")

#############################
# 主函数
#############################
def main():
    # 参数设置
    folder_path = "./SW-620"  # 数据文件所在文件夹
    ds_prefix = "SW-620"        # 数据集前缀，如 DS
    num_known_classes = 2   # 假设训练时已知类别个数（0 ~ num_known_classes-1），请根据数据调整
    in_feature_dim = 65     # 节点特征维度（依据 DS_node_labels 映射，取决于标签最大值+1）
    hidden_dim_G = 32
    out_dim_G = 32        # 图级特征维度
    hidden_dim_C = 32
    num_epochs = 100
    epsilon = 0.1
    lambda_adv = 1.0
    lr = 1e-3

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    dataset = GraphDataset(folder_path, ds_prefix)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, collate_fn=collate_fn)
    
    model_G = GraphFeatureExtractor(in_features=in_feature_dim, hidden_dim=hidden_dim_G, out_dim=out_dim_G).to(device)
    model_C = Classifier(in_dim=out_dim_G, hidden_dim=hidden_dim_C, num_known_classes=num_known_classes).to(device)
    
    train(model_G, model_C, dataloader, num_known_classes, device,
          num_epochs=num_epochs, epsilon=epsilon, lambda_adv=lambda_adv, lr=lr)
    
    torch.save(model_G.state_dict(), "model_G.pth")
    torch.save(model_C.state_dict(), "model_C.pth")
    
if __name__ == "__main__":
    main()

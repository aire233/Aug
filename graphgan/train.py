import os
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch_geometric.data import Batch
from tqdm import tqdm
import matplotlib.pyplot as plt

from data_utils import load_node_attrs, load_edges, load_graph_labels, load_graph_idx, build_graphs, convert_to_pyg
from models import Generator, Discriminator

def batch_dense_to_pyg(gen_nodes, gen_adj, threshold=0.5):
    """
    将生成器输出的批量假数据转换为一个 PyG Batch 对象
    gen_nodes: [B, N, node_dim]
    gen_adj: [B, N, N]
    """
    B, N, node_dim = gen_nodes.size()
    x = gen_nodes.view(B * N, node_dim)
    batch_indices = torch.arange(B, device=gen_nodes.device).repeat_interleave(N)
    mask = gen_adj > threshold
    idx = mask.nonzero(as_tuple=False)  # 每行 [b, i, j]
    if idx.size(0) == 0:
        edge_index = torch.empty((2, 0), dtype=torch.long, device=gen_nodes.device)
    else:
        b, i, j = idx.unbind(dim=1)
        global_i = b * N + i
        global_j = b * N + j
        edge_index = torch.stack([global_i, global_j], dim=0)
    return Batch(x=x, edge_index=edge_index, batch=batch_indices)

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    print("Using device:", device)
    
    folder_path = args.folder_path
    ds_prefix = args.ds_prefix

    # 加载新数据集各文件
    node_attrs = load_node_attrs(os.path.join(folder_path, f"{ds_prefix}.node_attrs"))
    edges = load_edges(os.path.join(folder_path, f"{ds_prefix}.edges"))
    graph_labels = load_graph_labels(os.path.join(folder_path, f"{ds_prefix}.graph_labels"))
    graph_idx = load_graph_idx(os.path.join(folder_path, f"{ds_prefix}.graph_idx"))

    # 自动更新节点属性维度，确保模型输入与数据一致
    actual_node_dim = node_attrs.shape[1]
    print("Detected node attribute dimension:", actual_node_dim)
    args.node_dim = actual_node_dim
    
    # 构建图，确保所有节点均加入（包括孤立节点）
    graphs = build_graphs(edges, graph_idx, node_attrs)
    print(f"Total graphs: {len(graphs)}")
    sample_gid = list(graphs.keys())[0]
    print(f"Graph {sample_gid} nodes: {graphs[sample_gid].number_of_nodes()}, edges: {graphs[sample_gid].number_of_edges()}")

    # 转换为 PyG 格式
    pyg_data_list, max_nodes = convert_to_pyg(graphs, graph_labels)
    
    real_loader = DataLoader(pyg_data_list, batch_size=args.batch_size, shuffle=True, pin_memory=True)
    
    # 初始化模型，传入 latent_dim、node_dim 与 max_nodes
    generator = Generator(latent_dim=args.latent_dim, node_dim=args.node_dim, max_nodes=max_nodes).to(device)
    discriminator = Discriminator(node_dim=args.node_dim).to(device)
    g_optimizer = torch.optim.Adam(generator.parameters(), lr=args.lr)
    d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=args.lr / 2) # 降低判别器学习率
    
    checkpoint_path = args.checkpoint_path
    start_epoch = 0
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        generator.load_state_dict(checkpoint["generator_state_dict"])
        discriminator.load_state_dict(checkpoint["discriminator_state_dict"])
        g_optimizer.load_state_dict(checkpoint["g_optimizer_state_dict"])
        d_optimizer.load_state_dict(checkpoint["d_optimizer_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        print(f"Checkpoint loaded from {checkpoint_path}, starting from epoch {start_epoch}.")
    else:
        print("No checkpoint found, starting training from scratch.")

    d_losses = []
    g_losses = []

    for epoch in tqdm(range(start_epoch, args.epochs), desc="Training epochs"):
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
                torch.full((batch_size, 1), 0.9, device=device) # 平滑标签
            )

            z = torch.randn(batch_size, args.latent_dim, device=device)
            gen_nodes, gen_adj = generator(z)
            fake_data = batch_dense_to_pyg(gen_nodes.detach(), gen_adj.detach(), threshold=args.threshold)
            fake_loss = F.binary_cross_entropy(
                discriminator(fake_data),
                torch.zeros(batch_size, 1, device=device)
            )
            d_loss = real_loss + fake_loss
            d_loss.backward()
            d_optimizer.step()

            # 训练生成器
            g_optimizer.zero_grad()
            z = torch.randn(batch_size, args.latent_dim, device=device)
            gen_nodes, gen_adj = generator(z)
            fake_data = batch_dense_to_pyg(gen_nodes, gen_adj, threshold=args.threshold)
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
        tqdm.write(f"Epoch [{epoch+1}/{args.epochs}] | D Loss: {avg_d_loss:.4f} | G Loss: {avg_g_loss:.4f}")

        if (epoch+1) % 50 == 0:          
            torch.save({
                "epoch": epoch,
                "generator_state_dict": generator.state_dict(),
                "discriminator_state_dict": discriminator.state_dict(),
                "g_optimizer_state_dict": g_optimizer.state_dict(),
                "d_optimizer_state_dict": d_optimizer.state_dict(),
                "max_nodes": max_nodes,
                "node_dim": args.node_dim
            }, f"graphgan_checkpoints/graphgan_checkpoint_{epoch+1}.pth")

        torch.save({
            "epoch": epoch,
            "generator_state_dict": generator.state_dict(),
            "discriminator_state_dict": discriminator.state_dict(),
            "g_optimizer_state_dict": g_optimizer.state_dict(),
            "d_optimizer_state_dict": d_optimizer.state_dict(),
            "max_nodes": max_nodes,
            "node_dim": args.node_dim
        }, checkpoint_path)
        tqdm.write(f"Checkpoint saved to {checkpoint_path}")

        d_losses.append(avg_d_loss)
        g_losses.append(avg_g_loss)

    plt.figure()
    plt.plot(d_losses, label="Discriminator Loss")
    plt.plot(g_losses, label="Generator Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig("graphgan/training_loss.png")

    print("Training completed.")
    return generator, discriminator

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train GraphGAN on COIL-DEL dataset")
    parser.add_argument("--folder_path", type=str, default="./COIL-DEL", help="数据文件所在目录")
    parser.add_argument("--ds_prefix", type=str, default="COIL-DEL", help="数据集前缀")
    parser.add_argument("--epochs", type=int, default=100, help="训练周期数")
    parser.add_argument("--batch_size", type=int, default=32, help="批量大小")
    parser.add_argument("--latent_dim", type=int, default=64, help="潜在向量维度")
    parser.add_argument("--node_dim", type=int, default=10, help="节点属性维度")
    parser.add_argument("--lr", type=float, default=0.001, help="学习率")
    parser.add_argument("--threshold", type=float, default=0.5, help="边生成阈值")
    parser.add_argument("--checkpoint_path", type=str, default="./graphgan_checkpoint.pth", help="检查点保存路径")
    parser.add_argument("--no_cuda", action="store_true", help="禁用 CUDA")
    args = parser.parse_args()
    
    train(args)

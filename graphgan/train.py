import os
import torch
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch_geometric.data import Batch
from tqdm import tqdm
import matplotlib.pyplot as plt

from data_utils import (
    load_adjacency_matrix,
    load_graph_indicator,
    load_node_labels,
    load_graph_labels,
    build_graphs,
    convert_to_pyg,
)
from models import Generator, Discriminator


def batch_dense_to_pyg(gen_nodes, gen_adj, threshold=0.5):
    """
    将生成器输出的批量假数据转换为一个 PyG Batch 对象
    gen_nodes: [B, N, node_dim]
    gen_adj: [B, N, N]
    """
    B, N, node_dim = gen_nodes.size()
    # 展平节点特征：[B*N, node_dim]
    x = gen_nodes.view(B * N, node_dim)
    # 构造每个节点所属图的 batch 信息
    batch_indices = torch.arange(B, device=gen_nodes.device).repeat_interleave(N)
    # 根据阈值筛选边
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
    device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"
    )
    print("Using device:", device)

    # 数据文件路径
    folder_path = args.folder_path
    ds_prefix = args.ds_prefix

    # 加载数据并构建图
    edges = load_adjacency_matrix(os.path.join(folder_path, f"{ds_prefix}_A.txt"))
    graph_indicator = load_graph_indicator(
        os.path.join(folder_path, f"{ds_prefix}_graph_indicator.txt")
    )
    node_labels = load_node_labels(
        os.path.join(folder_path, f"{ds_prefix}_node_labels.txt")
    )
    graph_labels = load_graph_labels(
        os.path.join(folder_path, f"{ds_prefix}_graph_labels.txt")
    )
    graphs = build_graphs(edges, graph_indicator, node_labels)
    print(f"Total graphs: {len(graphs)}")
    print(
        f"Graph 1 nodes: {graphs[1].number_of_nodes()}, edges: {graphs[1].number_of_edges()}"
    )

    # 转换为 PyG 格式
    pyg_data_list, max_nodes = convert_to_pyg(graphs, graph_labels)

    # 创建真实数据 DataLoader
    real_loader = DataLoader(
        pyg_data_list, batch_size=args.batch_size, shuffle=True, pin_memory=True
    )

    # 初始化模型
    generator = Generator(latent_dim=args.latent_dim, max_nodes=max_nodes).to(device)
    discriminator = Discriminator().to(device)
    g_optimizer = torch.optim.Adam(generator.parameters(), lr=args.lr)
    d_optimizer = torch.optim.Adam(discriminator.parameters(), lr=args.lr / 2)  # 0.0005

    # 检查点路径与加载
    checkpoint_path = args.checkpoint_path
    start_epoch = 0
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        generator.load_state_dict(checkpoint["generator_state_dict"])
        discriminator.load_state_dict(checkpoint["discriminator_state_dict"])
        g_optimizer.load_state_dict(checkpoint["g_optimizer_state_dict"])
        d_optimizer.load_state_dict(checkpoint["d_optimizer_state_dict"])
        start_epoch = checkpoint["epoch"] + 1
        print(
            f"Checkpoint loaded from {checkpoint_path}, starting from epoch {start_epoch}."
        )
    else:
        print("No checkpoint found, starting training from scratch.")

    # 训练循环前添加列表用于记录每个 epoch 的损失
    epoch_d_losses = []
    epoch_g_losses = []

    # 训练循环
    for epoch in tqdm(range(start_epoch, args.epochs), desc="Training epochs"):
        epoch_d_loss = 0.0
        epoch_g_loss = 0.0
        num_batches = 0

        for real_data in tqdm(real_loader, desc="Batches", leave=False):
            real_data = real_data.to(device)
            batch_size = real_data.num_graphs

            # 训练判别器
            d_optimizer.zero_grad()
            # 在计算真实数据损失时，将标签平滑到0.9
            real_loss = F.binary_cross_entropy(
                discriminator(real_data),
                torch.full((batch_size, 1), 0.9, device=device)
            )

            z = torch.randn(batch_size, args.latent_dim, device=device)
            gen_nodes, gen_adj = generator(z)
            # detach 生成器输出构造假数据 Batch
            fake_data = batch_dense_to_pyg(
                gen_nodes.detach(), gen_adj.detach(), threshold=args.threshold
            )
            fake_loss = F.binary_cross_entropy(
                discriminator(fake_data), torch.zeros(batch_size, 1, device=device)
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
                discriminator(fake_data), torch.ones(batch_size, 1, device=device)
            )
            g_loss.backward()
            g_optimizer.step()

            epoch_d_loss += d_loss.item()
            epoch_g_loss += g_loss.item()
            num_batches += 1

        avg_d_loss = epoch_d_loss / num_batches
        avg_g_loss = epoch_g_loss / num_batches
        tqdm.write(
            f"Epoch [{epoch+1}/{args.epochs}] | D Loss: {avg_d_loss:.4f} | G Loss: {avg_g_loss:.4f}"
        )

        # 记录每个 epoch 的损失
        epoch_d_losses.append(avg_d_loss)
        epoch_g_losses.append(avg_g_loss)

        # 保存检查点
        torch.save(
            {
                "epoch": epoch,
                "generator_state_dict": generator.state_dict(),
                "discriminator_state_dict": discriminator.state_dict(),
                "g_optimizer_state_dict": g_optimizer.state_dict(),
                "d_optimizer_state_dict": d_optimizer.state_dict(),
                "max_nodes": max_nodes,   # 保存 max_nodes
            },
            checkpoint_path,
        )
        tqdm.write(f"Checkpoint saved to {checkpoint_path}")

    # 绘制并保存损失图像
    plt.figure()
    epochs_range = range(start_epoch+1, args.epochs+1)
    plt.plot(epochs_range, epoch_d_losses, label="Discriminator Loss")
    plt.plot(epochs_range, epoch_g_losses, label="Generator Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Losses")
    plt.legend()
    plt.savefig("training_losses.png")
    plt.close()
    print("Training loss graph saved as training_losses.png")

    print("Training completed.")
    return generator, discriminator


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Train GraphGAN")
    parser.add_argument(
        "--folder_path", type=str, default="./SW-620", help="数据文件所在目录"
    )
    parser.add_argument("--ds_prefix", type=str, default="SW-620", help="数据集前缀")
    parser.add_argument("--epochs", type=int, default=100, help="训练周期数")
    parser.add_argument("--batch_size", type=int, default=32, help="批量大小")
    parser.add_argument("--latent_dim", type=int, default=64, help="潜在向量维度")
    parser.add_argument("--lr", type=float, default=0.001, help="学习率")
    parser.add_argument("--threshold", type=float, default=0.5, help="生成边的阈值")
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default="./graphgan_checkpoint.pth",
        help="检查点保存路径",
    )
    parser.add_argument("--no_cuda", action="store_true", help="禁用 CUDA")
    args = parser.parse_args()

    train(args)

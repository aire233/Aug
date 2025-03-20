import os
import torch
import argparse
from models import Generator
from torch_geometric.data import Data
from tqdm import tqdm


def generate_graphs(args):
    device = torch.device(
        "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"
    )
    print("Using device:", device)

    # 初始化生成器，并加载检查点
    from models import Generator  # 确保模型与训练时一致

    checkpoint = torch.load(args.checkpoint_path, map_location=device)
    # 假定 max_nodes 信息存储在检查点或另行传入，这里使用默认值
    max_nodes = args.max_nodes
    generator = Generator(latent_dim=args.latent_dim, max_nodes=max_nodes).to(device)
    generator.load_state_dict(checkpoint["generator_state_dict"])
    generator.eval()

    os.makedirs(args.output_folder, exist_ok=True)

    with torch.no_grad():
        z = torch.randn(args.num_graphs, args.latent_dim, device=device)
        _, adj_matrices = generator(z)
        for i, adj in enumerate(tqdm(adj_matrices, desc="Generating graphs")):
            filename = os.path.join(args.output_folder, f"generated_graph_{i}.txt")
            with open(filename, "w") as w:
                num_nodes = adj.shape[0]
                for u in range(num_nodes):
                    for v in range(num_nodes):
                        if adj[u, v] > args.threshold:
                            # 只保存存在边的信息
                            w.write(f"{u} {v}\n")
            print(f"Graph {i} saved to {filename}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate graphs with GraphGAN")
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default="./graphgan_checkpoint.pth",
        help="模型检查点路径",
    )
    parser.add_argument("--num_graphs", type=int, default=5, help="生成图的数量")
    parser.add_argument("--latent_dim", type=int, default=64, help="潜在向量维度")
    parser.add_argument("--max_nodes", type=int, default=50, help="图中最大节点数")
    parser.add_argument("--threshold", type=float, default=0.5, help="边生成阈值")
    parser.add_argument(
        "--output_folder", type=str, default="generated_graph", help="生成图保存目录"
    )
    parser.add_argument("--no_cuda", action="store_true", help="禁用 CUDA")
    args = parser.parse_args()

    generate_graphs(args)

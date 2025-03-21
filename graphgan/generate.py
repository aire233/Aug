import os
import torch
import argparse
import numpy as np
from tqdm import tqdm
from models import Generator

def generate_and_save(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    print("Using device:", device)

    # 加载检查点，并获得 max_nodes
    checkpoint = torch.load(args.checkpoint_path, map_location=device)
    max_nodes = checkpoint.get("max_nodes", args.max_nodes)
    
    # 初始化生成器并加载状态字典
    generator = Generator(latent_dim=args.latent_dim, max_nodes=max_nodes).to(device)
    generator.load_state_dict(checkpoint["generator_state_dict"])
    generator.eval()

    # 生成图：生成的节点特征 shape [num_graphs, max_nodes, node_dim]
    # 生成的邻接矩阵 shape [num_graphs, max_nodes, max_nodes]
    with torch.no_grad():
        z = torch.randn(args.num_graphs, args.latent_dim, device=device)
        gen_nodes, gen_adj = generator(z)
    gen_nodes = gen_nodes.cpu().numpy()    # [G, max_nodes, node_dim]
    gen_adj = gen_adj.cpu().numpy()          # [G, max_nodes, max_nodes]

    # 生成全局文件所需数据
    total_graphs = args.num_graphs
    total_nodes = total_graphs * max_nodes
    edge_lines = []          # 存储 DS_A.txt 每一行（全局边信息）
    graph_indicator_lines = []  # 存储 DS_graph_indicator.txt 每行为所属图 id
    graph_labels_lines = []     # 存储 DS_graph_labels.txt 每行为图标签（这里统一设为 0）
    node_labels_lines = []      # 存储 DS_node_labels.txt 每行为节点标签

    global_node_offset = 0
    for g in range(total_graphs):
        # 对于每个图，节点标签取生成器输出（取第一维度值并四舍五入为整数）
        # 注意：如果 node_dim>1，可以根据需要进行处理；这里假定 node_dim==1
        nodes_g = np.rint(gen_nodes[g, :, 0]).astype(int)
        # 记录本图所有节点对应的全局节点编号与所属图标
        for i in range(max_nodes):
            graph_indicator_lines.append(str(g + 1))
            node_labels_lines.append(str(nodes_g[i]))
        # 对于邻接矩阵，遍历上三角部分（只保存一遍边）
        adj_g = gen_adj[g]
        for u in range(max_nodes):
            for v in range(u + 1, max_nodes):
                if adj_g[u, v] > args.threshold:
                    # 计算全局节点编号（注意：输入文件中节点编号从1开始）
                    global_u = global_node_offset + u + 1
                    global_v = global_node_offset + v + 1
                    edge_lines.append(f"{global_u},{global_v}")
        # 图标签统一设为0（你也可以根据需要生成其他标签）
        graph_labels_lines.append("0")
        global_node_offset += max_nodes

    # 确保输出文件夹存在
    os.makedirs(args.output_folder, exist_ok=True)

    # 保存到文件
    output_A = os.path.join(args.output_folder, "DS_A.txt")
    output_indicator = os.path.join(args.output_folder, "DS_graph_indicator.txt")
    output_graph_labels = os.path.join(args.output_folder, "DS_graph_labels.txt")
    output_node_labels = os.path.join(args.output_folder, "DS_node_labels.txt")

    with open(output_A, "w") as f:
        f.write("\n".join(edge_lines))
    with open(output_indicator, "w") as f:
        f.write("\n".join(graph_indicator_lines))
    with open(output_graph_labels, "w") as f:
        f.write("\n".join(graph_labels_lines))
    with open(output_node_labels, "w") as f:
        f.write("\n".join(node_labels_lines))

    print(f"Generated {total_graphs} graphs with {max_nodes} nodes each.")
    print(f"Saved DS_A.txt with {len(edge_lines)} edges.")
    print(f"Saved DS_graph_indicator.txt with {len(graph_indicator_lines)} lines.")
    print(f"Saved DS_graph_labels.txt with {len(graph_labels_lines)} lines.")
    print(f"Saved DS_node_labels.txt with {len(node_labels_lines)} lines.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate graphs with GraphGAN and save in dataset format")
    parser.add_argument("--checkpoint_path", type=str, default="./graphgan_checkpoint.pth", help="模型检查点路径")
    parser.add_argument("--num_graphs", type=int, default=5, help="生成图的数量")
    parser.add_argument("--latent_dim", type=int, default=64, help="潜在向量维度")
    parser.add_argument("--max_nodes", type=int, default=50, help="图中最大节点数（若检查点未保存则采用此值）")
    parser.add_argument("--threshold", type=float, default=0.5, help="边生成阈值")
    parser.add_argument("--output_folder", type=str, default="generated_graph", help="生成图保存目录")
    parser.add_argument("--no_cuda", action="store_true", help="禁用 CUDA")
    args = parser.parse_args()
    
    generate_and_save(args)

import os
import torch
import argparse
import numpy as np
from tqdm import tqdm
from models import Generator

def generate_and_save(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    print("Using device:", device)

    # 加载检查点，读取 max_nodes 与 node_dim
    checkpoint = torch.load(args.checkpoint_path, map_location=device)
    max_nodes = checkpoint.get("max_nodes", args.max_nodes)
    node_dim = checkpoint.get("node_dim", args.node_dim)
    
    # 初始化生成器并加载模型参数
    generator = Generator(latent_dim=args.latent_dim, node_dim=node_dim, max_nodes=max_nodes).to(device)
    generator.load_state_dict(checkpoint["generator_state_dict"])
    generator.eval()

    with torch.no_grad():
        z = torch.randn(args.num_graphs, args.latent_dim, device=device)
        gen_nodes, gen_adj = generator(z)
    gen_nodes = gen_nodes.cpu().numpy()    # shape [G, max_nodes, node_dim]
    gen_adj = gen_adj.cpu().numpy()          # shape [G, max_nodes, max_nodes]

    total_graphs = args.num_graphs
    total_nodes = total_graphs * max_nodes

    # 构造输出文件内容（COIL-DEL 格式）
    edges_lines = []           # 保存 COIL-DEL.edges，每行 "node_id1,node_id2"
    graph_idx_lines = []       # 保存 COIL-DEL.graph_idx，每行节点所属图 id
    node_attrs_lines = []      # 保存 COIL-DEL.node_attrs，每行节点属性（逗号分隔）
    graph_labels_lines = []    # 保存 COIL-DEL.graph_labels，每行图标签（这里统一设为 0，可根据需要调整）

    global_node_offset = 0
    for g in range(total_graphs):
        nodes_g = gen_nodes[g]   # shape (max_nodes, node_dim)
        # 每个节点属性与所属图记录
        for i in range(max_nodes):
            attr_str = ",".join(map(str, nodes_g[i]))
            node_attrs_lines.append(attr_str)
            graph_idx_lines.append(str(g + 1))  # 图 id 从1开始
        # 遍历邻接矩阵（只保存上三角边）
        adj_g = gen_adj[g]
        for u in range(max_nodes):
            for v in range(u + 1, max_nodes):
                if adj_g[u, v] > args.threshold:
                    global_u = global_node_offset + u + 1
                    global_v = global_node_offset + v + 1
                    edges_lines.append(f"{global_u},{global_v}")
        graph_labels_lines.append("0")  # 图标签统一设为 0
        global_node_offset += max_nodes

    os.makedirs(args.output_folder, exist_ok=True)
    output_edges = os.path.join(args.output_folder, f"{args.ds_prefix}.edges")
    output_graph_idx = os.path.join(args.output_folder, f"{args.ds_prefix}.graph_idx")
    output_node_attrs = os.path.join(args.output_folder, f"{args.ds_prefix}.node_attrs")
    output_graph_labels = os.path.join(args.output_folder, f"{args.ds_prefix}.graph_labels")

    with open(output_edges, "w") as f:
        f.write("\n".join(edges_lines))
    with open(output_graph_idx, "w") as f:
        f.write("\n".join(graph_idx_lines))
    with open(output_node_attrs, "w") as f:
        f.write("\n".join(node_attrs_lines))
    with open(output_graph_labels, "w") as f:
        f.write("\n".join(graph_labels_lines))

    print(f"Generated {total_graphs} graphs with {max_nodes} nodes each.")
    print(f"Saved {output_edges} with {len(edges_lines)} edges.")
    print(f"Saved {output_graph_idx} with {len(graph_idx_lines)} lines.")
    print(f"Saved {output_node_attrs} with {len(node_attrs_lines)} lines.")
    print(f"Saved {output_graph_labels} with {len(graph_labels_lines)} lines.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate graphs with GraphGAN for COIL-DEL")
    parser.add_argument("--checkpoint_path", type=str, default="./graphgan_checkpoint.pth", help="模型检查点路径")
    parser.add_argument("--num_graphs", type=int, default=5, help="生成图的数量")
    parser.add_argument("--latent_dim", type=int, default=64, help="潜在向量维度")
    parser.add_argument("--max_nodes", type=int, default=50, help="图中最大节点数（若检查点未保存则采用此值）")
    parser.add_argument("--node_dim", type=int, default=10, help="节点属性维度")
    parser.add_argument("--threshold", type=float, default=0.5, help="边生成阈值")
    parser.add_argument("--output_folder", type=str, default="generated_graph", help="生成图保存目录")
    parser.add_argument("--ds_prefix", type=str, default="COIL-DEL", help="数据集前缀")
    parser.add_argument("--no_cuda", action="store_true", help="禁用 CUDA")
    args = parser.parse_args()
    
    generate_and_save(args)

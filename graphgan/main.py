import argparse
from train import train
from generate import generate_and_save

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GraphGAN: Train or Generate")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["train", "generate"],
        default="train",
        help="运行模式：train 或 generate",
    )
    # 共享参数
    parser.add_argument("--no_cuda", action="store_true", help="禁用 CUDA")

    # 训练相关参数
    parser.add_argument(
        "--folder_path", type=str, default="./SW-620", help="数据文件所在目录"
    )
    parser.add_argument("--ds_prefix", type=str, default="SW-620", help="数据集前缀")
    parser.add_argument("--epochs", type=int, default=50, help="训练周期数")
    parser.add_argument("--batch_size", type=int, default=32, help="批量大小")
    parser.add_argument("--latent_dim", type=int, default=64, help="潜在向量维度")
    parser.add_argument("--lr", type=float, default=0.001, help="学习率")
    parser.add_argument("--threshold", type=float, default=0.5, help="边生成阈值")
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default="./graphgan_checkpoint.pth",
        help="检查点保存路径",
    )

    # 生成相关参数
    parser.add_argument("--num_graphs", type=int, default=5, help="生成图的数量")
    parser.add_argument("--max_nodes", type=int, default=50, help="图中最大节点数")
    parser.add_argument(
        "--output_folder", type=str, default="generated_graph", help="生成图保存目录"
    )

    args = parser.parse_args()

    if args.mode == "train":
        train(args)
    elif args.mode == "generate":
        generate_and_save(args)

import argparse
from train_val import train

def main():
    parser = argparse.ArgumentParser(description="Simplified Graph Classification Training")
    parser.add_argument("--data_dir", type=str, required=True, help="数据文件目录，例如 ./COIL-DEL")
    parser.add_argument("--ds_prefix", type=str, required=True, help="数据集前缀，例如 COIL-DEL")
    parser.add_argument("--known_class_num", type=int, default=80, help="已知类别数量，超过该值视为未知")
    parser.add_argument("--batch_size", type=int, default=64, help="批次大小")
    parser.add_argument("--epochs", type=int, default=100, help="训练轮数")
    parser.add_argument("--lr", type=float, default=1e-3, help="学习率")
    parser.add_argument("--val_ratio", type=float, default=0.2, help="验证集比例")
    args = parser.parse_args()
    
    train(args)

if __name__ == "__main__":
    main()
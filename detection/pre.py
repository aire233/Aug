import os
import numpy as np

def load_graph_labels(file_path):
    """
    读取图标签文件，每一行一个标签，返回 numpy 数组
    """
    with open(file_path, "r") as f:
        lines = f.readlines()
    labels = np.array([int(line.strip()) for line in lines])
    return labels

def save_graph_labels(labels, file_path):
    """
    将标签数组保存到文件，每行一个标签
    """
    with open(file_path, "w") as f:
        for label in labels:
            f.write(f"{label}\n")

def rewrite_graph_labels(data_dir, ds_prefix, num_known=80, seed=42):
    """
    读取 COIL-DEL.graph_labels 文件，将原始100个类别划分为：
      - 已知类：随机选取 num_known 个类别（重新映射为 0 ~ num_known-1）
      - 未知类：其余类别统一改为 num_known（表示开放类别）
    最后保存到新文件 ds_prefix + ".graph_labels.new"
    """
    # 构造文件路径
    file_in = os.path.join(data_dir, f"{ds_prefix}.graph_labels")
    file_out = os.path.join(data_dir, f"{ds_prefix}.graph_labels.new")
    
    # 读取原始图标签
    original_labels = load_graph_labels(file_in)
    
    # 得到所有类别（假设原始标签中有 100 个类别）
    all_classes = sorted(np.unique(original_labels).tolist())
    print("原始类别数：", len(all_classes))
    if len(all_classes) != 100:
        print("警告：原始类别数不为100，而为", len(all_classes))
    
    # 固定随机种子，随机选择 num_known 个类别作为已知类
    np.random.seed(seed)
    known_classes = np.random.choice(all_classes, size=num_known, replace=False)
    known_classes = sorted(known_classes.tolist())
    unknown_classes = [c for c in all_classes if c not in known_classes]
    
    print("已知类：", known_classes)
    print("未知类：", unknown_classes)
    
    # 对于每个标签，如果属于已知类，则重映射为其在 known_classes 中的索引，否则标记为 num_known（开放类别）
    new_labels = []
    for label in original_labels:
        if label in known_classes:
            new_label = known_classes.index(label)
        else:
            new_label = num_known  # 未知类统一改为 num_known（例如 90）
        new_labels.append(new_label)
    new_labels = np.array(new_labels)
    
    # 保存新标签到文件
    save_graph_labels(new_labels, file_out)
    print(f"新图标签已保存到 {file_out}")

if __name__ == "__main__":
    # 设置数据所在目录和数据集前缀
    data_directory = "./COIL-DEL"   # 请根据实际情况修改数据路径
    ds_prefix = "COIL-DEL"        # 数据集前缀，对应文件名如 COIL-DEL.graph_labels
    rewrite_graph_labels(data_directory, ds_prefix, num_known=80, seed=42)

# train.py
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.loader import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
from data_utils import SimpleGraphDataset
from model import SimpleGraphFeatureExtractor, SimpleClassifier

def validate(model_G, model_C, loader, device):
    model_G.eval()
    model_C.eval()
    total = 0
    correct = 0
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            feat = model_G(batch)
            logits = model_C(feat)
            pred = logits.argmax(dim=1)
            total += batch.y.size(0)
            correct += (pred == batch.y).sum().item()
    return correct / total

def train(args):
    dataset = SimpleGraphDataset(args.data_dir, args.ds_prefix, args.known_class_num)
    
    # 将数据集随机拆分为训练和验证集
    total_num = len(dataset.data_list)
    val_num = int(args.val_ratio * total_num)
    train_num = total_num - val_num
    indices = torch.randperm(total_num)
    train_data = [dataset.data_list[i] for i in indices[:train_num]]
    val_data = [dataset.data_list[i] for i in indices[train_num:]]
    
    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    in_dim = dataset.data_list[0].x.shape[1]
    
    model_G = SimpleGraphFeatureExtractor(in_dim, hidden_dim=128, out_dim=128).to(device)
    model_C = SimpleClassifier(in_dim=128, hidden_dim=64, num_classes=args.known_class_num+1).to(device)
    
    optimizer = optim.Adam(list(model_G.parameters()) + list(model_C.parameters()), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    losses = []
    accs = []
    
    best_acc = 0.0
    for epoch in tqdm(range(args.epochs)):
        model_G.train()
        model_C.train()
        total_loss = 0.0
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            feat = model_G(batch)
            logits = model_C(feat)
            loss = criterion(logits, batch.y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / len(train_loader)
        val_acc = validate(model_G, model_C, val_loader, device)
        print(f"Epoch {epoch+1}/{args.epochs}, Loss: {avg_loss:.4f}, Val Acc: {val_acc:.4f}")
        losses.append(avg_loss)
        accs.append(val_acc)
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model_G.state_dict(), "best_model_G.pth")
            torch.save(model_C.state_dict(), "best_model_C.pth")
    print("Training completed. Best Val Acc: {:.4f}".format(best_acc))

    plt.figure(figsize=(10, 4))
    plt.subplot(111)
    plt.plot(losses, label="Loss")
    plt.plot(accs, label="Val Acc")
    plt.xlabel("Epoch")
    plt.legend()
    plt.savefig("train_curve.png")
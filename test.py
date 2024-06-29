import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

from swin_transformer import swin_t  # 假设 swin_transformer.py 与本文件在同一目录下
from data import get_cifar100_loaders  # 数据加载函数

def evaluate(model, criterion, data_loader, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for inputs, labels in tqdm(data_loader, desc="Evaluating", leave=False):
            inputs, labels = inputs.to(device), labels.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    epoch_loss = running_loss / total
    epoch_acc = correct / total

    return epoch_loss, epoch_acc


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, test_loader = get_cifar100_loaders()

    model = swin_t(hidden_dim=96, layers=(2, 2, 6, 2), heads=(3, 6, 12, 24), num_classes=100).to(device)

    # 使用DataParallel进行多GPU并行
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()

    # 加载模型权重
    model.load_state_dict(torch.load("best_swin_model.pth"))

    val_loss, val_acc = evaluate(model, criterion, test_loader, device)

    print(f"Validation Loss: {val_loss:.4f}")
    print(f"Validation Accuracy: {val_acc:.4f}")

if __name__ == "__main__":
    main()

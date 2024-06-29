import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import argparse
from data import get_cifar100_loaders
from model import SwinTransformer
from utils import EarlyStopping, save_model

# 确保脚本可以找到其他模块
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train(model, train_loader, criterion, optimizer, epoch):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for inputs, targets in tqdm(train_loader, desc=f'Epoch {epoch} [Training]'):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    epoch_loss = running_loss / len(train_loader.dataset)
    epoch_acc = correct / total
    print(f'Epoch {epoch}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}')
    return epoch_loss, epoch_acc

def validate(model, test_loader, criterion, epoch):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in tqdm(test_loader, desc=f'Epoch {epoch} [Validation]'):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    epoch_loss = running_loss / len(test_loader.dataset)
    epoch_acc = correct / total
    print(f'Validation Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}')
    return epoch_loss, epoch_acc

def main(pretrained=None):
    train_loader, test_loader = get_cifar100_loaders()

    model = SwinTransformer(num_classes=100).to(device)  # CIFAR-100 有 100 个类别
    if pretrained:
        state_dict = torch.load(pretrained, map_location=device)
        model_dict = model.state_dict()

        # 过滤掉不匹配的层
        filtered_dict = {k: v for k, v in state_dict.items() if k in model_dict and v.size() == model_dict[k].size()}
        model_dict.update(filtered_dict)
        model.load_state_dict(model_dict)

    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs for training.")
        model = nn.DataParallel(model)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)

    num_epochs = 200
    early_stopping = EarlyStopping(patience=5, verbose=True)
    best_val_acc = 0

    for epoch in range(num_epochs):
        train_loss, train_acc = train(model, train_loader, criterion, optimizer, epoch)
        val_loss, val_acc = validate(model, test_loader, criterion, epoch)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_model(model.module if hasattr(model, 'module') else model, path='output/best_model.pth')

        early_stopping(val_loss, model.module if hasattr(model, 'module') else model)
        if early_stopping.early_stop:
            print("Early stopping")
            break

    save_model(model.module if hasattr(model, 'module') else model, path='output/final_model.pth')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train Swin Transformer on CIFAR-100')
    parser.add_argument('--pretrained', type=str, default=None, help='Path to pretrained weights')
    args = parser.parse_args()

    main(pretrained=args.pretrained)

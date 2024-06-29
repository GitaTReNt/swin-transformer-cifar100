from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

import sys
import os

# 确保脚本可以找到其他模块
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import argparse
from imagenetdata import get_imagenet_loaders
from model import SwinTransformer
from utils import save_model

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


def validate(model, val_loader, criterion, epoch):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in tqdm(val_loader, desc=f'Epoch {epoch} [Validation]'):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    epoch_loss = running_loss / len(val_loader.dataset)
    epoch_acc = correct / total
    print(f'Validation Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}')
    return epoch_loss, epoch_acc


def main(pretrained=None):
    train_dir = "autodl-tmp/imagenet/train"
    val_dir = "autodl-tmp/imagenet/val"

    if not os.path.exists(train_dir) or not os.path.exists(val_dir):
        raise FileNotFoundError(f"Couldn't find train or val directory. Please check the paths.")

    train_loader, val_loader = get_imagenet_loaders(train_dir, val_dir, batch_size=32)  # 将 batch size 减小到 32

    model = SwinTransformer(num_classes=1000).to(device)  # ImageNet-1K 有 1000 个类别
    if pretrained:
        model.load_state_dict(torch.load(pretrained, map_location=device))

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 1  # 通常在 ImageNet 上训练 90 个 epoch
    best_val_acc = 0

    for epoch in range(num_epochs):
        train_loss, train_acc = train(model, train_loader, criterion, optimizer, epoch)
        val_loss, val_acc = validate(model, val_loader, criterion, epoch)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_model(model, path='pretrained/swin_transformer_pretrained.pth')

    save_model(model, path='pretrained/final_swin_transformer_pretrained.pth')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Pretrain Swin Transformer on ImageNet')
    parser.add_argument('--pretrained', type=str, default=None, help='Path to pretrained weights')
    args = parser.parse_args()

    main(pretrained=args.pretrained)

#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import time
import torch
from torch import nn
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, WeightedRandomSampler

from models import AutoDriveNet_Reg
from datasets import AutoDriveDataset
from utils import *


def build_weighted_sampler(train_dataset):
    """
    根据转向角构造 WeightedRandomSampler
    核心思想：
    - 直行样本（|angle|较小）权重低
    - 转弯样本（|angle|较大）权重高
    """

    angles = train_dataset.get_all_angles()
    sample_weights = []

    for angle in angles:
        abs_angle = abs(angle)

        # 你可以按自己数据分布进一步微调阈值
        if abs_angle < 0.05:
            weight = 1.0      # 直行
        elif abs_angle < 0.15:
            weight = 2.0      # 小转弯
        else:
            weight = 4.0      # 大转弯

        sample_weights.append(weight)

    sample_weights = torch.DoubleTensor(sample_weights)

    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )

    return sampler


def main():
    start_time = time.time()

    data_folder = 'data/aug2'
    checkpoint = None

    batch_size = 32
    start_epoch = 1
    epochs = 40
    lr = 1e-4

    patience = 8
    early_stop_counter = 0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if torch.cuda.is_available():
        print("使用 GPU:", torch.cuda.get_device_name(0))
    else:
        print("当前使用 CPU")

    cudnn.benchmark = True
    writer = SummaryWriter()

    # ----------------------
    # 模型
    # ----------------------
    model = AutoDriveNet_Reg().to(device)

    optimizer = torch.optim.Adam(
        params=filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr
    )

    criterion = nn.MSELoss().to(device)

    if checkpoint is not None:
        checkpoint = torch.load(checkpoint, map_location=device)
        start_epoch = checkpoint['epoch'] + 1
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])

    # ----------------------
    # 预处理
    # Dataset里已经做了 HSV + 亮度归一化
    # 这里仅做 Resize + ToTensor
    # ----------------------
    transformations = transforms.Compose([
        transforms.Resize((120, 160)),
        transforms.ToTensor(),
    ])

    # ----------------------
    # Dataset
    # ----------------------
    train_dataset = AutoDriveDataset(
        data_folder=data_folder,
        mode='train',
        transform=transformations
    )

    val_dataset = AutoDriveDataset(
        data_folder=data_folder,
        mode='val',
        transform=transformations
    )

    # ----------------------
    # Balanced Sampler
    # ----------------------
    train_sampler = build_weighted_sampler(train_dataset)

    # ----------------------
    # DataLoader
    # ----------------------
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=0,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )

    print("Train size:", len(train_dataset))
    print("Val size:", len(val_dataset))

    best_loss = float('inf')
    best_epoch = -1

    # =========================================================
    # 训练循环
    # =========================================================
    for epoch in range(start_epoch, epochs + 1):
        print("\nEpoch:", epoch)

        # ----------------------
        # Train
        # ----------------------
        model.train()
        train_loss_meter = AverageMeter()

        for imgs, labels in train_loader:
            imgs = imgs.to(device)
            labels = labels.to(device)

            pre_labels = model(imgs)
            loss = criterion(pre_labels, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss_meter.update(loss.item(), imgs.size(0))

        train_loss = train_loss_meter.avg

        # ----------------------
        # Validation
        # ----------------------
        model.eval()
        val_loss_meter = AverageMeter()

        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs = imgs.to(device)
                labels = labels.to(device)

                pre_labels = model(imgs)
                loss = criterion(pre_labels, labels)

                val_loss_meter.update(loss.item(), imgs.size(0))

        val_loss = val_loss_meter.avg

        # ----------------------
        # TensorBoard
        # ----------------------
        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Loss/val", val_loss, epoch)

        print(f"Train Loss: {train_loss:.6f}")
        print(f"Val Loss:   {val_loss:.6f}")

        # ----------------------
        # 保存模型
        # ----------------------
        model_state_dict = model.state_dict()

        if val_loss < best_loss:
            best_loss = val_loss
            best_epoch = epoch
            early_stop_counter = 0

            torch.save({
                'epoch': epoch,
                'model': model_state_dict,
                'optimizer': optimizer.state_dict(),
            }, 'model_best.pth')

            print("🌟 保存新的最优模型")

        else:
            early_stop_counter += 1
            print(f"EarlyStopping: {early_stop_counter}/{patience}")

            if early_stop_counter >= patience:
                print("⛔ Early stopping triggered")
                break

        # 保存最后一轮
        if epoch == epochs:
            torch.save({
                'epoch': epoch,
                'model': model_state_dict,
                'optimizer': optimizer.state_dict(),
            }, 'model_last.pth')

            print("✅ 保存最终模型")

    total_time = time.time() - start_time

    print("\n训练完成")
    print(f"训练时间: {total_time / 60:.2f} 分钟")
    print(f"Best Epoch: {best_epoch}")
    print(f"Best Val Loss: {best_loss:.6f}")

    writer.close()


if __name__ == '__main__':
    main()
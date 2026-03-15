#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@文件        :utils.py
@说明        :工具脚本,存放自定义函数和类
@时间        :2022/03/01 11:03:41
@作者        :钱彬
@版本        :1.0
'''
import torch
from torch.utils.data import DataLoader, Subset
import numpy as np

class AverageMeter(object):
    '''
    平均器类,用于计算平均值、总和
    '''
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def evaluate_on_segments(model, dataset, segment_indices, device, batch_size=32):
    """
    在特定样本子集上评估模型性能（如困难路段），返回 MSE 损失。

    :param model: 已训练的模型
    :param dataset: 完整数据集（AutoDriveDataset）
    :param segment_indices: 一个包含困难路段样本索引的列表
    :param device: 设备（cuda 或 cpu）
    :param batch_size: 批大小
    :return: MSE 损失
    """
    model.eval()
    criterion = torch.nn.MSELoss().to(device)
    subset = Subset(dataset, segment_indices)
    loader = DataLoader(subset, batch_size=batch_size, shuffle=False)

    total_loss = 0.0
    total_samples = 0

    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            labels = labels.to(device)

            outputs = model(imgs)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * imgs.size(0)
            total_samples += imgs.size(0)

    avg_mse = total_loss / total_samples
    return avg_mse
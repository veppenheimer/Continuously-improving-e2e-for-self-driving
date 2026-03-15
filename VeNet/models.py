#!/usr/bin/env python
# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F


class AutoDriveNet_Reg(nn.Module):
    def __init__(self):
        """
        正则化增强版 AutoDriveNet
        包含 BatchNorm、双 Dropout，用于小样本防过拟合。
        """
        super(AutoDriveNet_Reg, self).__init__()

        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 24, 5, stride=2),
            nn.BatchNorm2d(24),
            nn.ELU(),

            nn.Conv2d(24, 36, 5, stride=2),
            nn.BatchNorm2d(36),
            nn.ELU(),

            nn.Conv2d(36, 48, 5, stride=2),
            nn.BatchNorm2d(48),
            nn.ELU(),

            nn.Conv2d(48, 64, 3),
            nn.BatchNorm2d(64),
            nn.ELU(),

            nn.Conv2d(64, 64, 3),
            nn.BatchNorm2d(64),
            nn.ELU(),

            nn.Dropout(0.5)  # 第一处Dropout
        )

        self.linear_layers = nn.Sequential(
            nn.Linear(64 * 8 * 13, 100),
            nn.ELU(),
            nn.Dropout(0.5),  # 第二处Dropout
            nn.Linear(100, 50),
            nn.ELU(),
            nn.Linear(50, 10),
            nn.Linear(10, 1)
        )

    def forward(self, x):
        """
        前向传播
        """
        x = x.view(x.size(0), 3, 120, 160)
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.linear_layers(x)
        return x

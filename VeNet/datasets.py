#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import os
import numpy as np
import cv2
from PIL import Image

import torch
from torch.utils.data import Dataset


class AutoDriveDataset(Dataset):
    """
    数据集加载器
    """

    def __init__(self, data_folder, mode, transform=None):
        """
        :param data_folder: 数据文件所在文件夹根路径(train.txt/val.txt/test.txt所在目录)
        :param mode: 'train' / 'val' / 'test'
        :param transform: torchvision transform
        """
        self.data_folder = data_folder
        self.mode = mode.lower()
        self.transform = transform

        assert self.mode in {'train', 'val', 'test'}

        if self.mode == 'train':
            file_path = os.path.join(data_folder, 'train.txt')
        elif self.mode == 'val':
            file_path = os.path.join(data_folder, 'val.txt')
        else:
            file_path = os.path.join(data_folder, 'test.txt')

        self.file_list = []
        with open(file_path, 'r') as f:
            files = f.readlines()
            for file in files:
                line = file.strip()
                if not line:
                    continue

                parts = line.split(' ')
                img_path = parts[0]
                angle = float(parts[1])

                self.file_list.append([img_path, angle])

    def __getitem__(self, i):
        """
        :param i: 图像索引
        :return: (img, label)
        """
        img_path, label = self.file_list[i]

        # 读取图像
        img = cv2.imread(img_path)
        if img is None:
            raise ValueError(f"读取图像失败: {img_path}")

        # BGR -> HSV
        img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # 亮度归一化：对 V 通道做直方图均衡化
        img[:, :, 2] = cv2.equalizeHist(img[:, :, 2])

        # 转为 PIL，交给 transform 做 Resize / ToTensor
        img = Image.fromarray(img)

        if self.transform:
            img = self.transform(img)

        label = torch.from_numpy(np.array([label], dtype=np.float32))

        return img, label

    def __len__(self):
        return len(self.file_list)

    def get_all_angles(self):
        """
        返回当前数据集全部角度，用于构造 sampler 权重
        """
        return [item[1] for item in self.file_list]
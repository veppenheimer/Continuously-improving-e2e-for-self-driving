#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import time
import torch
from torch import nn
import torchvision.transforms as transforms

from datasets import AutoDriveDataset
from models import AutoDriveNet_Reg
from utils import *


def main():
    data_folder = "./data/aug2"
    checkpoint_path = "model_best.pth"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(checkpoint_path, map_location=device)

    model = AutoDriveNet_Reg().to(device)
    model.load_state_dict(checkpoint['model'])
    model.eval()

    transformations = transforms.Compose([
        transforms.Resize((120, 160)),
        transforms.ToTensor(),
    ])

    test_dataset = AutoDriveDataset(
        data_folder=data_folder,
        mode='test',
        transform=transformations
    )

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=True
    )

    criterion = nn.MSELoss().to(device)
    MSEs = AverageMeter()

    start = time.time()

    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs = imgs.to(device)
            labels = labels.to(device)

            pre_labels = model(imgs)
            loss = criterion(pre_labels, labels)

            MSEs.update(loss.item(), imgs.size(0))

    total_time = time.time() - start
    fps = len(test_dataset) / total_time if total_time > 0 else 0.0

    print('Test MSE {mses.avg:.6f}'.format(mses=MSEs))
    print('平均单张样本用时 {:.6f} 秒'.format(total_time / len(test_dataset)))
    print('FPS {:.2f}'.format(fps))


if __name__ == '__main__':
    main()
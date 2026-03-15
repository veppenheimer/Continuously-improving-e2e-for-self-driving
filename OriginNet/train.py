#!/usr/bin/env python
# -*- encoding: utf-8 -*-


import torch.backends.cudnn as cudnn
import torch
from torch import nn
import torchvision.transforms as transforms
from torch.utils.tensorboard import SummaryWriter

from models import AutoDriveNet
from datasets import AutoDriveDataset
from utils import *

import time


def main():
    """
    训练.
    """
    start_time = time.time()

    data_folder = 'data/aug2'
    checkpoint = None
    batch_size = 32
    start_epoch = 1
    epochs = 40
    lr = 1e-4

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        print("使用 GPU:", torch.cuda.get_device_name(0))
    else:
        print("当前使用 CPU")
    ngpu = 1
    cudnn.benchmark = True
    writer = SummaryWriter()

    model = AutoDriveNet()

    optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad,
                                               model.parameters()), lr=lr)

    model = model.to(device)
    criterion = nn.MSELoss().to(device)

    if checkpoint is not None:
        checkpoint = torch.load(checkpoint)
        start_epoch = checkpoint['epoch'] + 1
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])

    if torch.cuda.is_available() and ngpu > 1:
        model = nn.DataParallel(model, device_ids=list(range(ngpu)))

    transformations = transforms.Compose([
        transforms.Resize((120, 160)),
        transforms.ToTensor(),
    ])

    train_dataset = AutoDriveDataset(data_folder, mode='train', transform=transformations)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                               shuffle=True, num_workers=0, pin_memory=True)

    best_loss = float('inf')
    best_epoch = -1

    for epoch in range(start_epoch, epochs + 1):
        model.train()
        loss_epoch = AverageMeter()

        for i, (imgs, labels) in enumerate(train_loader):
            print(f"Batch {i}: imgs.shape = {imgs.shape}")
            imgs = imgs.to(device)
            labels = labels.to(device)

            pre_labels = model(imgs)
            loss = criterion(pre_labels, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_epoch.update(loss.item(), imgs.size(0))
            print(f"第 {i} 个batch训练结束")

        del imgs, labels, pre_labels

        writer.add_scalar('MSE_Loss', loss_epoch.avg, epoch)
        print(f'epoch: {epoch}  MSE_Loss: {loss_epoch.avg:.6f}')

        model_state_dict = model.module.state_dict() if hasattr(model, "module") else model.state_dict()

        # 保存最后一轮
        if epoch == epochs:
            torch.save({
                'epoch': epoch,
                'model': model_state_dict,
                'optimizer': optimizer.state_dict(),
            }, '0606.pth')
            print(f"✅ 最后一轮模型保存为 model_final.pth")

        # 保存最优权重
        if loss_epoch.avg < best_loss:
            best_loss = loss_epoch.avg
            best_epoch = epoch
            torch.save({
                'epoch': epoch,
                'model': model_state_dict,
                'optimizer': optimizer.state_dict(),
            }, '0606x.pth')
            print(f"🌟 当前为最优模型，已保存为 model_best.pth")

    end_time = time.time()
    total_time = end_time - start_time

    print(f"\n训练总耗时：{total_time / 60:.2f} 分钟（{total_time:.2f} 秒）")
    print(f"📌 最低损失率为 {best_loss:.6f}，出现在第 {best_epoch} 轮，并已保存为 model_best.pth")

    writer.close()


if __name__ == '__main__':
    '''
    程序入口
    '''
    main()

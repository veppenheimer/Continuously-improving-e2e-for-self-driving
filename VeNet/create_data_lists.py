#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import os
import random
import matplotlib.pyplot as plt


def create_data_list(dataset_path, file_list, mode='train'):
    """
    生成 train / val / test txt
    """

    save_path = os.path.join(dataset_path, mode + ".txt")

    with open(save_path, 'w') as f:
        for (imgpath, angle) in file_list:
            f.write(imgpath + ' ' + angle + '\n')

    print(mode + ".txt 已生成, 样本数:", len(file_list))


def getFileList(dir, Filelist, ext=None):

    if os.path.isfile(dir):

        if ext is None:
            Filelist.append(dir)

        else:
            if ext in dir[-3:]:
                Filelist.append(dir)

    elif os.path.isdir(dir):

        for s in os.listdir(dir):
            newDir = os.path.join(dir, s)
            getFileList(newDir, Filelist, ext)

    return Filelist


def plot_angle_hist(angle_list, title):

    plt.figure()

    plt.hist(angle_list, bins=50)

    plt.title(title)

    plt.xlabel("Steering Angle")

    plt.ylabel("Sample Count")

    plt.show()


def main():

    # ==========================
    # 数据路径
    # ==========================

    org_img_folder = './data/1_1'

    # ==========================
    # 数据集比例
    # ==========================

    train_ratio = 0.7
    val_ratio = 0.15
    test_ratio = 0.15

    # ==========================
    # 读取图片
    # ==========================

    jpglist = getFileList(org_img_folder, [], 'jpg')

    print('检索到 {} 个 jpg 文件\n'.format(len(jpglist)))

    file_list = []
    angle_list = []

    # ==========================
    # 解析角度
    # ==========================

    for jpgpath in jpglist:

        basename = os.path.basename(jpgpath)

        angle = (basename[:-4]).split('_')[-1]

        imgPath = jpgpath.replace("\\", "/")

        file_list.append((imgPath, angle))

        angle_list.append(float(angle))

    # ==========================
    # 绘制优化前角度分布
    # ==========================

    plot_angle_hist(angle_list, "Angle Distribution Before Balancing")

    # ==========================
    # 解决直行样本过多
    # ==========================

    balanced_list = []
    balanced_angles = []

    for imgpath, angle in file_list:

        angle_float = float(angle)

        if abs(angle_float) < 0.05:

            # 直行样本只保留30%
            if random.random() < 0.3:
                balanced_list.append((imgpath, angle))
                balanced_angles.append(angle_float)

        else:

            balanced_list.append((imgpath, angle))
            balanced_angles.append(angle_float)

    print("原始样本数:", len(file_list))
    print("平衡后样本数:", len(balanced_list))

    # ==========================
    # 绘制优化后角度分布
    # ==========================

    plot_angle_hist(balanced_angles, "Angle Distribution After Balancing")

    # ==========================
    # 随机打乱
    # ==========================

    random.seed(256)
    random.shuffle(balanced_list)

    total_num = len(balanced_list)

    train_num = int(total_num * train_ratio)
    val_num = int(total_num * val_ratio)

    train_list = balanced_list[:train_num]
    val_list = balanced_list[train_num:train_num + val_num]
    test_list = balanced_list[train_num + val_num:]

    # ==========================
    # 生成 txt
    # ==========================

    create_data_list(org_img_folder, train_list, 'train')
    create_data_list(org_img_folder, val_list, 'val')
    create_data_list(org_img_folder, test_list, 'test')

    print("\n数据集划分完成")

    print("train:", len(train_list))
    print("val:", len(val_list))
    print("test:", len(test_list))


if __name__ == "__main__":
    main()
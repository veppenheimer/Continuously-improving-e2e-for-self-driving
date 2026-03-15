#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import os
import random
import matplotlib.pyplot as plt


def create_data_list(dataset_path, file_list, mode='train'):
    """
    生成 train / val / test txt
    每行格式:
    image_path angle
    """
    save_path = os.path.join(dataset_path, mode + ".txt")

    with open(save_path, 'w', encoding='utf-8') as f:
        for (imgpath, angle) in file_list:
            f.write(imgpath + ' ' + angle + '\n')

    print("{} 已生成, 样本数: {}".format(mode + ".txt", len(file_list)))


def get_file_list(root_dir, file_list, ext=None):
    """
    递归获取文件夹及其子文件夹中的文件列表
    :param root_dir: 根目录
    :param file_list: 文件列表
    :param ext: 扩展名, 例如 'jpg'
    """
    if os.path.isfile(root_dir):
        if ext is None:
            file_list.append(root_dir)
        else:
            if root_dir.lower().endswith('.' + ext.lower()):
                file_list.append(root_dir)

    elif os.path.isdir(root_dir):
        for name in os.listdir(root_dir):
            new_path = os.path.join(root_dir, name)
            get_file_list(new_path, file_list, ext)

    return file_list


def plot_angle_hist(angle_list, title, save_path=None, bins=50, show=True):
    """
    绘制转向角分布直方图
    :param angle_list: 转向角列表(float)
    :param title: 图标题
    :param save_path: 图片保存路径
    :param bins: 直方图分箱数
    :param show: 是否显示图像
    """
    plt.figure(figsize=(8, 5))
    plt.hist(angle_list, bins=bins)
    plt.title(title)
    plt.xlabel("Steering Angle")
    plt.ylabel("Sample Count")
    plt.grid(True, linestyle='--', alpha=0.3)

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print("直方图已保存:", save_path)

    if show:
        plt.show()

    plt.close()


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

    # 检查比例是否合理
    ratio_sum = train_ratio + val_ratio + test_ratio
    assert abs(ratio_sum - 1.0) < 1e-8, "train_ratio + val_ratio + test_ratio 必须等于 1"

    # ==========================
    # 随机种子
    # ==========================
    random.seed(256)

    # ==========================
    # 读取图片
    # ==========================
    jpg_list = get_file_list(org_img_folder, [], 'jpg')
    print('检索到 {} 个 jpg 文件\n'.format(len(jpg_list)))

    if len(jpg_list) == 0:
        raise ValueError("未找到 jpg 文件，请检查数据路径: {}".format(org_img_folder))

    file_list = []
    angle_list = []

    # ==========================
    # 解析角度
    # 文件名格式默认形如:
    # xxx_xxx_angle.jpg
    # angle 位于最后一个下划线之后
    # ==========================
    for jpg_path in jpg_list:
        basename = os.path.basename(jpg_path)

        try:
            angle = (basename[:-4]).split('_')[-1]
            angle_float = float(angle)
        except Exception:
            raise ValueError("文件名无法正确解析角度，请检查: {}".format(basename))

        img_path = jpg_path.replace("\\", "/")

        file_list.append((img_path, angle))
        angle_list.append(angle_float)

    print("原始样本数:", len(file_list))
    print("角度最小值: {:.6f}".format(min(angle_list)))
    print("角度最大值: {:.6f}".format(max(angle_list)))
    print("角度均值:   {:.6f}".format(sum(angle_list) / len(angle_list)))

    # ==========================
    # 绘制原始角度分布
    # 注意: 这里不再做 balancing 删除样本
    # 因为训练阶段将采用 WeightedRandomSampler
    # ==========================
    hist_save_path = os.path.join(org_img_folder, 'angle_distribution_before_sampling.png')
    plot_angle_hist(
        angle_list=angle_list,
        title="Angle Distribution (Original Dataset)",
        save_path=hist_save_path,
        bins=50,
        show=True
    )

    # ==========================
    # 随机打乱
    # ==========================
    random.shuffle(file_list)

    total_num = len(file_list)

    train_num = int(total_num * train_ratio)
    val_num = int(total_num * val_ratio)
    test_num = total_num - train_num - val_num

    train_list = file_list[:train_num]
    val_list = file_list[train_num:train_num + val_num]
    test_list = file_list[train_num + val_num:]

    # ==========================
    # 安全检查
    # ==========================
    assert len(train_list) == train_num
    assert len(val_list) == val_num
    assert len(test_list) == test_num
    assert len(train_list) + len(val_list) + len(test_list) == total_num

    # ==========================
    # 生成 txt
    # ==========================
    create_data_list(org_img_folder, train_list, 'train')
    create_data_list(org_img_folder, val_list, 'val')
    create_data_list(org_img_folder, test_list, 'test')

    print("\n数据集划分完成")
    print("train: {}".format(len(train_list)))
    print("val:   {}".format(len(val_list)))
    print("test:  {}".format(len(test_list)))

    print("\n说明:")
    print("1. 本脚本不再删除直行样本")
    print("2. 样本均衡由 train.py 中的 WeightedRandomSampler 完成")
    print("3. 角度分布图已保存，可直接用于实验分析或论文插图")


if __name__ == "__main__":
    main()
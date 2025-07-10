#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
data_loader.py

这个文件创建一个自定义的数据集加载器 (Dataset Loader)，专门用于处理遥感图像分割任务。

Author: he.cl
Date: 2025-06-23
"""

import os
import glob
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import torchvision.transforms as T
import albumentations as A
from albumentations.pytorch import ToTensorV2

from config import global_config


def get_train_transforms():
    """
    定义训练集使用的数据增强流程：增强过程在内存中进行、随机地应用到每个批次的数据上。

    """
    return A.Compose(
        [
            # --- 几何变换 ---
            # 50%的概率进行水平翻转
            A.HorizontalFlip(p=0.5),
            # 50%的概率进行垂直翻转
            A.VerticalFlip(p=0.5),
            # 50%的概率进行90度随机旋转（0, 90, 180, 270度）
            A.RandomRotate90(p=0.5),
            # --- 色彩变换 ---
            # 随机调整亮度、对比度、饱和度和色相，模拟不同光照和季节条件
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.8),
            # 随机高斯噪声
            A.GaussNoise(p=0.2),
            # --- 核心变换 ---
            # 归一化，使用ImageNet的均值和标准差
            A.Normalize(mean=global_config["vars"]["mean"], std=global_config["vars"]["std"]),
            # 将图像和掩码转换为PyTorch张量
            ToTensorV2(),
        ]
    )


def get_val_transforms():
    """
    定义验证集/测试集使用的变换流程：通常只包含归一化和转换为张量，不进行随机增强，以保证评估结果的一致性。

    """
    return A.Compose(
        [
            A.Normalize(mean=global_config["vars"]["mean"], std=global_config["vars"]["std"]),
            ToTensorV2(),
        ]
    )


# 创建RemoteSensingDataset类，继承了torch.utils.data 的 Dataset类
class RemoteSensingDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        """
        Args:
            data_dir (string): 包含 images 和 masks 文件夹的目录。
            transform (callable, optional): 应用于样本的可选变换。
        """
        self.image_dir = os.path.join(data_dir, "images")
        self.mask_dir = os.path.join(data_dir, "masks")
        self.transform = transform

        # 1. 找到所有的图像文件路径
        image_pattern = os.path.join(self.image_dir, "*", "*.tif")
        self.image_files = sorted(glob.glob(image_pattern))

        # 2. 然后根据图像文件名，构造出对应的掩码文件路径
        self.mask_files = []
        for img_path in self.image_files:
            base_filename = os.path.basename(img_path)
            mask_path = os.path.join(self.mask_dir, base_filename)

            if not os.path.exists(mask_path):
                print(f"警告：找不到图像 '{img_path}' 对应的掩码 '{mask_path}'")

            self.mask_files.append(mask_path)

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        mask_path = self.mask_files[idx]

        # 读取图像和掩码
        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        # 将掩码转换为 numpy 数组，并将像素值归一化到 0 或 1
        mask = np.array(mask)
        mask[mask == 255] = 1

        # 应用数据增强和转换
        if self.transform:
            image = self.transform(image)

        # 将 mask 转换为 Tensor
        mask = torch.from_numpy(mask).long().unsqueeze(0)

        return image, mask

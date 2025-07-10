#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
data_loader.py

自定义的数据集加载器 (Dataset Loader)
Author he.cl
Date 2025-06-25
"""

import os
import glob
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_train_transforms(config):
    """
    定义训练集使用的数据增强流程
    Args:
        config 配置字典
    """
    return A.Compose(
        [
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.8),
            A.GaussNoise(p=0.2),
            A.Normalize(mean=config["vars"]["mean"], std=config["vars"]["std"]),
            ToTensorV2(),
        ]
    )


def get_val_transforms(config):
    """
    定义验证集/测试集使用的变换流程。
    Args:
        config 配置字典
    """
    return A.Compose(
        [
            A.Normalize(mean=config["vars"]["mean"], std=config["vars"]["std"]),
            ToTensorV2(),
        ]
    )


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

        # 查找所有图像文件路径, 兼容嵌套和非嵌套两种情况
        image_files = glob.glob(os.path.join(self.image_dir, "*.tif"))
        if not image_files:
            image_files.extend(glob.glob(os.path.join(self.image_dir, "*", "*.tif")))
        self.image_files = sorted(image_files)

        if not self.image_files:
            print(f"警告: 在目录 '{self.image_dir}' 中没有找到 .tif 格式的图像文件。")

        # 根据图像文件名，构造出对应的掩码文件路径
        self.mask_files = []
        for img_path in self.image_files:
            relative_path = os.path.relpath(img_path, self.image_dir)
            mask_path = os.path.join(self.mask_dir, relative_path)

            if not os.path.exists(mask_path):
                print(f"警告：找不到图像 '{img_path}' 对应的掩码 '{mask_path}'")

            self.mask_files.append(mask_path)

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        mask_path = self.mask_files[idx]

        # 使用 numpy array 读取，以配合 albumentations
        image = np.array(Image.open(img_path).convert("RGB"))
        mask = np.array(Image.open(mask_path).convert("L"))

        # 将掩码值从 [0, 255] 映射到 [0, 1]
        mask[mask == 255] = 1

        if self.transform:
            # 同时对图像和掩码应用变换
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]

        # ToTensorV2 已经处理了图像的维度顺序 (H, W, C) -> (C, H, W)
        # 掩码需要增加一个通道维度以匹配模型输出 (H, W) -> (1, H, W)
        # 对于 BCEWithLogitsLoss，mask 需要是 float 类型
        return image, torch.from_numpy(mask).unsqueeze(0).float()

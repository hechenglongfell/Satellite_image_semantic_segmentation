#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
data_loader.py

自定义数据集加载器

Author he.cl
Date 2025-06-25
"""

import os
import glob
import json
import torch
from tqdm import tqdm
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
import rasterio


class DatasetStatistics:
    """
    计算训练集(Training Set)的统计信息,适用于任意波段数的训练集样本:
    均值:每个波段的平均值,如:mean: [0.485, 0.456, 0.406]
    标准差:如:std: [0.229, 0.224, 0.225]
    类别权重:针对训练集的掩码，计算分割类别目标与背景值的比例

    通过比对文件列表和修改时间来判断是否需要重新计算，避免重复计算

    """

    def __init__(self, data_dir, num_classes, num_bands=3, cache_file="dataset_stats.json"):
        """
        Args:
            data_dir (str): 训练集根目录，应包含 'images' 和 'masks' 子文件夹。
            num_classes (int): 数据集中的类别总数。
            num_bands (int): 影像的波段数。
            cache_file (str): 用于存储统计信息的缓存文件名。
        """
        self.data_dir = data_dir
        self.num_classes = num_classes
        self.num_bands = num_bands
        self.cache_path = os.path.join(data_dir, cache_file)
        self.image_dir = os.path.join(data_dir, "images")
        self.mask_dir = os.path.join(data_dir, "masks")

    def _get_current_dataset_fingerprint(self):
        """
        生成当前数据集的指纹（文件列表及修改时间），用于判断数据集是否变化。
        """
        image_files = glob.glob(os.path.join(self.image_dir, "**", "*.tif"), recursive=True)
        image_files.extend(glob.glob(os.path.join(self.image_dir, "**", "*.png"), recursive=True))
        image_files.extend(glob.glob(os.path.join(self.image_dir, "**", "*.jpg"), recursive=True))

        if not image_files:
            return None, []

        # 指纹包含文件名和最后修改时间，排序以保证一致性
        fingerprint = sorted([(os.path.basename(p), os.path.getmtime(p)) for p in image_files])
        return image_files, fingerprint

    def get_stats(self):
        """
        获取统计信息
        Returns:
            dict: 包含 'mean', 'std', 'class_weights' 的字典
        """
        image_files, current_fingerprint = self._get_current_dataset_fingerprint()

        if os.path.exists(self.cache_path):
            with open(self.cache_path, "r") as f:
                stats = json.load(f)

            # 校验1: 波段数是否匹配
            if stats.get("num_bands") != self.num_bands:
                return self._calculate_and_save_stats(image_files, current_fingerprint)

            # 校验 2: 数据集指纹是否匹配
            cached_fingerprint = stats.get("fingerprint")

            if cached_fingerprint:
                cached_fingerprint = [tuple(item) for item in cached_fingerprint]

            if cached_fingerprint != current_fingerprint:
                return self._calculate_and_save_stats(image_files, current_fingerprint)

            return stats
        else:
            return self._calculate_and_save_stats(image_files, current_fingerprint)

    def _calculate_and_save_stats(self, image_files, fingerprint):
        """
        计算训练集统计信息并保存到文件
        """
        if not image_files:
            raise FileNotFoundError(f"错误: 在 '{self.image_dir}' 中没有找到样本文件！")

        # --- 1. 计算均值和标准差 ---
        channel_sum = np.zeros(self.num_bands)
        channel_sum_sq = np.zeros(self.num_bands)
        pixel_count = 0

        for img_path in tqdm(image_files, desc="计算 Mean/Std"):
            try:
                with rasterio.open(img_path) as src:
                    if src.count != self.num_bands:
                        print(
                            f"警告: 图像 '{img_path}' 的波段数 ({src.count}) 与预设值 ({self.num_bands}) 不等，跳过此文件。"
                        )
                        continue
                    img_np = src.read().transpose((1, 2, 0))

                img_np = img_np.astype(np.float64)
                h, w, c = img_np.shape
                pixel_count += h * w
                channel_sum += np.sum(img_np, axis=(0, 1))
                channel_sum_sq += np.sum(np.square(img_np), axis=(0, 1))
            except Exception as e:
                print(f"警告: 读取或处理图像 '{img_path}' 失败: {e}")
                continue

        mean = channel_sum / pixel_count
        std = np.sqrt(channel_sum_sq / pixel_count - np.square(mean))

        # --- 2. 计算类别权重 ---
        class_pixel_counts = np.zeros(self.num_classes)
        for img_path in tqdm(image_files, desc="计算 Class Weights"):
            relative_path = os.path.relpath(img_path, self.image_dir)
            base_mask_path = os.path.join(self.mask_dir, os.path.splitext(relative_path)[0])
            mask_path = None
            for ext in [".png", ".tif", ".jpg"]:
                if os.path.exists(base_mask_path + ext):
                    mask_path = base_mask_path + ext
                    break

            if not mask_path:
                print(f"警告: 找不到图像 '{img_path}' 对应的掩码")
                continue

            try:
                mask = Image.open(mask_path).convert("L")
                mask_np = np.array(mask)
                mask_np[mask_np == 255] = 1

                for i in range(self.num_classes):
                    class_pixel_counts[i] += np.sum(mask_np == i)
            except Exception as e:
                print(f"警告: 读取或处理掩码 '{mask_path}' 失败: {e}")
                continue

        total_pixels = np.sum(class_pixel_counts)
        class_frequencies = class_pixel_counts / total_pixels
        class_weights = 1 / np.log(1.02 + class_frequencies)

        stats = {
            "num_bands": self.num_bands,
            "mean": mean.tolist(),
            "std": std.tolist(),
            "class_weights": class_weights.tolist(),
            "fingerprint": fingerprint,
        }

        print(f"计算完成，结果已保存到 '{self.cache_path}'")
        with open(self.cache_path, "w") as f:
            json.dump(stats, f, indent=4)

        return stats


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
            # A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.8),
            # A.GaussNoise(p=0.2),
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
    def __init__(self, data_dir, num_bands=3, transform=None):
        """
        Args:
            data_dir (string): 包含 images 和 masks 文件夹的目录。
            transform (callable, optional): 应用于样本的可选变换。
        """
        self.image_dir = os.path.join(data_dir, "images")
        self.mask_dir = os.path.join(data_dir, "masks")
        self.num_bands = num_bands
        self.transform = transform

        # 查找所有图像文件路径
        image_files = glob.glob(os.path.join(self.image_dir, "**", "*.tif"), recursive=True)
        image_files.extend(glob.glob(os.path.join(self.image_dir, "**", "*.png"), recursive=True))
        image_files.extend(glob.glob(os.path.join(self.image_dir, "**", "*.jpg"), recursive=True))
        self.image_files = sorted(image_files)

        if not self.image_files:
            print(f"警告: 在目录 '{self.image_dir}' 中没有找到 .tif 格式的图像文件。")

        # 根据图像文件名，构造出对应的掩码文件路径
        self.mask_files = []
        for img_path in self.image_files:
            relative_path = os.path.relpath(img_path, self.image_dir)
            base_mask_path = os.path.join(self.mask_dir, os.path.splitext(relative_path)[0])
            mask_path = None
            for ext in [".png", ".tif", ".jpg"]:
                if os.path.exists(base_mask_path + ext):
                    mask_path = base_mask_path + ext
                    break

            if not mask_path:
                print(f"警告：找不到图像 '{img_path}' 对应的掩码")
                self.mask_files.append(None)
            else:
                self.mask_files.append(mask_path)

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        mask_path = self.mask_files[idx]

        if not mask_path:
            print(f"错误: 索引 {idx} 对应的掩码文件不存在，无法加载。")
            return None, None

        try:
            with rasterio.open(img_path) as src:
                if src.count != self.num_bands:
                    print(f"错误: 图像 '{img_path}' 的波段数 ({src.count}) 与预设 ({self.num_bands}) 不符。")
                    return None, None
                # 读取所有波段并转换为 (H, W, C) 格式以配合 albumentations
                image = src.read().transpose((1, 2, 0))

            mask = np.array(Image.open(mask_path).convert("L"))
        except Exception as e:
            print(f"错误: 在索引 {idx} 处加载图像或掩码失败: {e}")
            return None, None

        mask[mask == 255] = 1

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"]

        return image, mask.unsqueeze(0).float()

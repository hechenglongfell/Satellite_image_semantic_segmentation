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
from tqdm import tqdm
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2
import rasterio


class MyDataLoader(Dataset):
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


class DatasetStatistics:
    """
    计算训练集(Training Set)的统计信息,适用于任意波段数的训练集样本:
    均值:每个波段的平均值,如:mean: [0.485, 0.456, 0.406]
    标准差:如:std: [0.229, 0.224, 0.225]
    类别权重:针对训练集的掩码，计算分割类别目标与背景值的比例

    通过比对文件列表和修改时间来判断是否需要重新计算，避免重复计算,同时更新配置文件

    """

    def __init__(
        self,
        data_dir: str,
        num_classes: int,
        num_bands: int = 3,
        cache_file: str = "dataset_stats.json",
    ):
        """
        初始训练集统计计算器

        Args:
            data_dir (str): 训练集根目录，应包含 'images' 和 'masks' 子文件夹。
            num_classes (int): 数据集中的类别总数。
            num_bands (int): 影像的波段数。
            cache_file (str): 用于存储统计信息的缓存文件名。
        """
        self.data_dir = data_dir
        self.num_classes = num_classes
        self.num_bands = num_bands
        self.image_dir = os.path.join(data_dir, "images")
        self.mask_dir = os.path.join(data_dir, "masks")
        self.cache_path = os.path.join(data_dir, cache_file)

    def _get_current_dataset_fingerprint(self):
        """
        生成当前训练集的指纹（文件列表及修改时间），用于判断训练集是否变化。
        指纹由所有图像文件的名称和最后修改时间构成。
        """
        image_files = []
        for ext in ["*.tif", "*.png", "*.jpg", "*.jpeg"]:
            image_files.extend(glob.glob(os.path.join(self.image_dir, "**", ext), recursive=True))

        if not image_files:
            return None, []

        # 排序，保证一致性
        fingerprint = sorted([(os.path.basename(p), os.path.getmtime(p)) for p in image_files])
        return image_files, fingerprint

    def get_stats(self):
        """
        获取训练集的统计信息字典
        Returns:
            dict: 一个包含 'mean', 'std', 'class_weights' 等键的统计信息字典
        """
        image_files, current_fingerprint = self._get_current_dataset_fingerprint()

        if os.path.exists(self.cache_path):
            with open(self.cache_path, "r") as f:
                stats = json.load(f)

            if (
                stats.get("num_bands") == self.num_bands
                and stats.get("fingerprint")
                and [tuple(item) for item in stats.get("fingerprint")] == current_fingerprint
            ):

                stats["mean"] = np.array(stats["mean"])
                stats["std"] = np.array(stats["std"])
                stats["class_weights"] = np.array(stats["class_weights"])
                return stats

            return self._calculate_and_save_stats(image_files, current_fingerprint)

    def _calculate_and_save_stats(self, image_files, fingerprint):
        """
        计算训练集统计信息并保存到文件
        """
        if not image_files:
            raise FileNotFoundError(f"错误: 在 '{self.image_dir}' 中没有找到任何训练集文件！")

        # --- 计算均值和标准差 ---
        channel_sum = np.zeros(self.num_bands, dtype=np.float64)
        channel_sum_sq = np.zeros(self.num_bands, dtype=np.float64)
        pixel_count = 0

        for img_path in tqdm(image_files, desc="计算 Mean/Std"):
            try:
                with rasterio.open(img_path) as src:
                    if src.count != self.num_bands:
                        print(
                            f"警告: 图像 '{img_path}' 的波段数 ({src.count}) 与预设值 ({self.num_bands}) 不符，已跳过。"
                        )
                        continue
                    img_np = src.read().transpose((1, 2, 0))

                h, w, _ = img_np.shape
                pixel_count += h * w
                channel_sum += np.sum(img_np, axis=(0, 1))
                channel_sum_sq += np.sum(np.square(img_np.astype(np.float64)), axis=(0, 1))
            except Exception as e:
                print(f"警告: 读取或处理图像 '{img_path}' 失败: {e}")
                continue

        mean = channel_sum / pixel_count
        std = np.sqrt(channel_sum_sq / pixel_count - np.square(mean))

        # --- 计算类别权重 ---
        class_pixel_counts = np.zeros(self.num_classes, dtype=np.int64)
        for img_path in tqdm(image_files, desc="计算 Class Weights"):
            relative_path = os.path.relpath(img_path, self.image_dir)
            mask_filename = os.path.splitext(relative_path)[0]

            mask_path = None
            for ext in [".png", ".tif", ".jpg"]:
                potential_path = os.path.join(self.mask_dir, mask_filename + ext)
                if os.path.exists(potential_path):
                    mask_path = potential_path
                    break

            if not mask_path:
                print(f"警告: 找不到图像 '{img_path}' 对应的掩码文件，已跳过。")
                continue

            try:
                mask = Image.open(mask_path).convert("L")
                mask_np = np.array(mask)
                # 兼容常见的 0/255 标签
                mask_np[mask_np == 255] = 1

                for i in range(self.num_classes):
                    class_pixel_counts[i] += np.sum(mask_np == i)
            except Exception as e:
                print(f"警告: 读取或处理掩码 '{mask_path}' 失败: {e}")
                continue

        # 使用反向对数平滑公式计算权重
        total_pixels = np.sum(class_pixel_counts)
        class_frequencies = class_pixel_counts / total_pixels
        class_weights = 1 / np.log(1.02 + class_frequencies)

        # --- 组装字典并保存到缓存文件 ---
        stats = {
            "num_bands": self.num_bands,
            "mean": mean.tolist(),  # 保存为列表以兼容JSON
            "std": std.tolist(),
            "class_weights": class_weights.tolist(),
            "fingerprint": fingerprint,
        }

        print(f"计算完成，结果已保存到 '{self.cache_path}'")
        with open(self.cache_path, "w") as f:
            json.dump(stats, f, indent=4)

        # --- 返回包含Numpy数组的字典 ---
        stats["mean"] = mean
        stats["std"] = std
        stats["class_weights"] = class_weights

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

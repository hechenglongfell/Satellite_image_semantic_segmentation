#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
offline_augment.py

离线数据增强脚本: 读取源目录中的图像和掩码，对每张图像生成指定数量的增强版本，并将结果保存到一个新的目标目录中。
适用于数据集较小，预先生成所有增强样本的场景。

Author: he.cl
Date: 2025-07-11
"""

import os
import glob
import cv2
import numpy as np
import albumentations as A
from tqdm import tqdm
import argparse


def get_offline_transforms():
    """
    定义离线增强使用的变换流程
    """
    return A.Compose(
        [
            # --- 几何变换 ---
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            # 仿射变换：旋转、缩放、平移和错切
            A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=45, p=0.8),
            # 弹性变形，模拟传感器或地形引起的扭曲
            A.ElasticTransform(p=0.3, alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03),
            # --- 色彩变换 ---
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.8),
            A.GaussNoise(p=0.2),
            # 随机改变图像通道的顺序，例如 RGB -> BGR
            A.ToGray(p=0.1),
        ]
    )


def augment_and_save(source_dir, dest_dir, num_augmentations_per_image=10):
    """
    执行离线数据增强并保存结果。

    Args:
        source_dir (str): 包含 'images' 和 'masks' 子目录的源数据路径。
        dest_dir (str): 用于保存增强后数据的目标路径。
        num_augmentations_per_image (int): 每张原始图像要生成的增强版本数量。
    """
    # 定义源和目标的 images/masks 路径
    source_images_dir = os.path.join(source_dir, "images")
    source_masks_dir = os.path.join(source_dir, "masks")

    dest_images_dir = os.path.join(dest_dir, "images")
    dest_masks_dir = os.path.join(dest_dir, "masks")

    # 创建目标目录
    os.makedirs(dest_images_dir, exist_ok=True)
    os.makedirs(dest_masks_dir, exist_ok=True)

    print(f"源数据目录: {source_dir}")
    print(f"目标数据目录: {dest_dir}")
    print(f"每张图片将生成 {num_augmentations_per_image} 个增强版本。")

    # 获取增强变换流程
    transforms = get_offline_transforms()

    # 查找所有源图像
    image_paths = glob.glob(os.path.join(source_images_dir, "*.tif"))
    if not image_paths:
        image_paths = glob.glob(os.path.join(source_images_dir, "*", "*.tif"))

    for img_path in tqdm(image_paths, desc="Processing Images"):
        # 构建对应的掩码路径
        relative_path = os.path.relpath(img_path, source_images_dir)
        mask_path = os.path.join(source_masks_dir, relative_path)

        if not os.path.exists(mask_path):
            print(f"警告：跳过图像 '{img_path}'，因找不到对应的掩码。")
            continue

        # 使用OpenCV读取图像和掩码
        # cv2.IMREAD_COLOR 读取为BGR格式
        image = cv2.imread(img_path, cv2.IMREAD_COLOR)
        # cv2.IMREAD_UNCHANGED 读取单通道灰度图
        mask = cv2.imread(mask_path, cv2.IMREAD_UNCHANGED)

        # 将图像从BGR转换为RGB，以匹配Pillow和大多数库的习惯
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 获取不带扩展名的原始文件名
        base_filename = os.path.splitext(os.path.basename(img_path))[0]

        # 循环生成指定数量的增强版本
        for i in range(num_augmentations_per_image):
            # 应用增强
            augmented = transforms(image=image, mask=mask)
            aug_image = augmented["image"]
            aug_mask = augmented["mask"]

            # 构建新的文件名
            new_filename = f"{base_filename}_aug_{i+1}.tif"
            dest_img_path = os.path.join(dest_images_dir, new_filename)
            dest_mask_path = os.path.join(dest_masks_dir, new_filename)

            # 保存增强后的图像和掩码
            # 将图像从RGB转回BGR以供OpenCV保存
            cv2.imwrite(dest_img_path, cv2.cvtColor(aug_image, cv2.COLOR_RGB2BGR))
            cv2.imwrite(dest_mask_path, aug_mask)

    print("\n离线数据增强完成！")
    print(f"所有增强后的数据已保存至: {dest_dir}")


def main():
    parser = argparse.ArgumentParser(description="离线数据增强脚本")
    parser.add_argument("--source", type=str, required=True, help="源数据目录的路径 (例如: 'data/train')")
    parser.add_argument("--dest", type=str, required=True, help="目标数据目录的路径 (例如: 'data/train_augmented')")
    parser.add_argument("--num", type=int, default=10, help="每张原始图像生成的增强版本数量")
    args = parser.parse_args()

    augment_and_save(args.source, args.dest, args.num)


if __name__ == "__main__":
    main()

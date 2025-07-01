#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
predict.py

模型预测。

Author: he.cl
Date: 2025-06-24
"""

import torch
import numpy as np
import rasterio
from rasterio.windows import Window
from PIL import Image
import torchvision.transforms as T
from tqdm import tqdm
import os

from networks.model import build_unet

# --- 1. 配置 ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
WEIGHTS_PATH = '../weights/best_model.pth'

# 输入和输出路径
INPUT_IMAGE_PATH = '../data/test/images/L19-11.tif'  # 影像路径
OUTPUT_DIR = '../outputs/prediction_results/'

input_filename = os.path.basename(INPUT_IMAGE_PATH)  # L19-11.tif
output_filename = os.path.splitext(input_filename)[0] + '_predicted_mask.tif'  # L19-11_predicted_mask.tif
OUTPUT_IMAGE_PATH = os.path.join(OUTPUT_DIR, output_filename)

# 分块大小预测参数
PATCH_SIZE = 512  # 每次处理的小块尺寸 (像素)，可根据显存大小调整


def main():
    # 自动生成新的输出文件名，以区分概率图
    input_filename = os.path.basename(INPUT_IMAGE_PATH)
    output_filename = os.path.splitext(input_filename)[0] + '_probability_map.tif'
    OUTPUT_IMAGE_PATH = os.path.join(OUTPUT_DIR, output_filename)

    # 确保输出目录存在
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # --- 2. 加载模型和定义转换 ---
    print(f"正在加载模型: {WEIGHTS_PATH}")
    model = build_unet(device=DEVICE)
    model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=DEVICE))
    model.eval()

    transform = T.Compose([
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # --- 3. 使用 Rasterio 处理大图 ---
    print(f"正在打开影像: {INPUT_IMAGE_PATH}")
    with rasterio.open(INPUT_IMAGE_PATH) as src:
        meta = src.meta
        width, height = src.width, src.height
        print(f"影像尺寸: {width}x{height}")

        meta.update(count=1, dtype='uint8', compress='lzw')

        with rasterio.open(OUTPUT_IMAGE_PATH, 'w', **meta) as dst:
            print(f"正在创建输出文件: {OUTPUT_IMAGE_PATH}")

            num_patches_w = (width + PATCH_SIZE - 1) // PATCH_SIZE
            num_patches_h = (height + PATCH_SIZE - 1) // PATCH_SIZE
            total_patches = num_patches_w * num_patches_h
            progress_bar = tqdm(total=total_patches, desc="生成概率图中")

            for j in range(0, height, PATCH_SIZE):
                for i in range(0, width, PATCH_SIZE):
                    actual_width = min(PATCH_SIZE, width - i)
                    actual_height = min(PATCH_SIZE, height - j)
                    window = Window(i, j, actual_width, actual_height)

                    patch = src.read(window=window)

                    full_patch = np.zeros((src.count, PATCH_SIZE, PATCH_SIZE), dtype=patch.dtype)
                    full_patch[:, :actual_height, :actual_width] = patch

                    patch_rgb = np.transpose(full_patch, (1, 2, 0))
                    pil_img = Image.fromarray(patch_rgb).convert("RGB")

                    input_tensor = transform(pil_img).unsqueeze(0).to(DEVICE)

                    with torch.no_grad():
                        output = model(input_tensor)

                    probs = torch.sigmoid(output)

                    # 1.将概率值从 [0.0, 1.0] 的浮点数范围缩放到 [0, 255] 的整数范围
                    prob_map_uint8 = (probs.cpu().numpy().squeeze() * 255).astype(rasterio.uint8)

                    # 2.只截取预测结果中有效的区域进行写回
                    final_prob_map = prob_map_uint8[:actual_height, :actual_width]

                    # 3.将处理好的概率图小块写入输出文件
                    dst.write(final_prob_map, window=window, indexes=1)

                    progress_bar.update(1)

            progress_bar.close()

    print(f"\n概率图生成完成！结果已保存至: {OUTPUT_IMAGE_PATH}")


if __name__ == '__main__':
    main()

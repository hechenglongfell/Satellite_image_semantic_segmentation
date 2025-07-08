#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
predict.py (改进版)

模型预测，采用重叠分块预测策略以减少边缘效应。

Author: he.cl
Date: 2025-06-26
"""


import torch
import numpy as np
import rasterio
from rasterio.windows import Window
from PIL import Image
import torchvision.transforms as T
from tqdm import tqdm
import os

from utils.logger import setup_logger
from networks.model import build_unet

# --- 配置 ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

WEIGHTS_PATH = os.path.join(PROJECT_ROOT, "weights", "best_model.pth")  # 模型路径
INPUT_IMAGE_PATH = os.path.join(PROJECT_ROOT, "data", "test/images/L19-11.tif")  # 影像路径
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs", "prediction_results/")  # 输出结果路径

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 分块大小及重叠率
PATCH_SIZE = 512  # 每次处理的小块尺寸 (像素)，可根据显存大小调整
OVERLAP_RATE = 0.25  # 滑窗重叠率，建议0.25 (即25%的重叠)

logger = setup_logger(__name__, "predict.log")


def main():

    # 自动生成新的输出文件名
    input_filename = os.path.basename(INPUT_IMAGE_PATH)
    output_filename = os.path.splitext(input_filename)[0] + f"_prob_overlap1_{OVERLAP_RATE}.tif"
    OUTPUT_IMAGE_PATH = os.path.join(OUTPUT_DIR, output_filename)

    # 确保输出目录存在
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # --- 1. 加载模型和定义转换 ---
    logger.info(f"正在加载模型: {WEIGHTS_PATH}")
    model = build_unet(device=DEVICE)
    model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=DEVICE))
    model.eval()

    transform = T.Compose([T.ToTensor(), T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    # --- 2. 使用 Rasterio 和重叠滑窗策略处理大图 ---
    logger.info(f"正在打开影像: {INPUT_IMAGE_PATH}")
    with rasterio.open(INPUT_IMAGE_PATH) as src:
        meta = src.meta
        width, height = src.width, src.height
        logger.info(f"影像尺寸: {width}x{height}")
        # 更新输出文件的元数据
        meta.update(count=1, dtype="uint8", compress="lzw")

        with rasterio.open(OUTPUT_IMAGE_PATH, "w", **meta) as dst:
            logger.info(f"正在创建输出文件: {OUTPUT_IMAGE_PATH}")
            # --- 计算滑窗步长和坐标 ---
            if OVERLAP_RATE < 0 or OVERLAP_RATE >= 0.5:
                raise ValueError("OVERLAP_RATE 必须在 [0, 0.5) 范围内")

            # 计算重叠像素数和步长
            offset = int(PATCH_SIZE * OVERLAP_RATE)
            step = PATCH_SIZE - 2 * offset
            if step <= 0:
                raise ValueError("重叠率过高，导致步长小于或等于0，请降低重叠率")

            # 生成滑窗的X, Y坐标，确保边缘被完整覆盖
            x_coords = list(range(0, width - PATCH_SIZE, step))
            if x_coords[-1] != width - PATCH_SIZE:
                x_coords.append(width - PATCH_SIZE)

            y_coords = list(range(0, height - PATCH_SIZE, step))
            if y_coords[-1] != height - PATCH_SIZE:
                y_coords.append(height - PATCH_SIZE)

            total_patches = len(y_coords) * len(x_coords)
            progress_bar = tqdm(total=total_patches, desc="预测中")

            # --- 使用新的滑窗坐标进行循环 ---
            for y_start in y_coords:
                for x_start in x_coords:
                    # 定义读取窗口
                    read_window = Window(x_start, y_start, PATCH_SIZE, PATCH_SIZE)
                    patch = src.read(window=read_window)

                    # 将读取的 patch 转换为模型输入格式
                    patch_rgb = np.transpose(patch, (1, 2, 0))
                    pil_img = Image.fromarray(patch_rgb).convert("RGB")
                    input_tensor = transform(pil_img).unsqueeze(0).to(DEVICE)

                    # 模型推理
                    with torch.no_grad():
                        output = model(input_tensor)

                    # 获取概率图并转换为 uint8
                    probs = torch.sigmoid(output)
                    prob_map_uint8 = (probs.cpu().numpy().squeeze() * 255).astype(rasterio.uint8)

                    # --- 计算并写入非重叠的中心区域 ---

                    # 定义从预测结果中要截取的区域 (中心部分)
                    src_slice_y = slice(offset, PATCH_SIZE - offset)
                    src_slice_x = slice(offset, PATCH_SIZE - offset)
                    data_to_write = prob_map_uint8[src_slice_y, src_slice_x]

                    # 定义要写入到输出文件的窗口
                    write_window = Window(x_start + offset, y_start + offset, step, step)

                    # 边缘处理：确保写入窗口不会超出图像边界
                    # 计算实际的写入窗口和需要截取的数据
                    clipped_window = write_window.intersection(Window(0, 0, width, height))

                    h_off = clipped_window.row_off - write_window.row_off
                    w_off = clipped_window.col_off - write_window.col_off

                    clipped_data = data_to_write[
                        h_off : h_off + clipped_window.height, w_off : w_off + clipped_window.width
                    ]

                    # 将处理好的概率图小块写入输出文件
                    dst.write(clipped_data, window=clipped_window, indexes=1)

                    progress_bar.update(1)

            progress_bar.close()

    logger.info(f"预测完成！结果已保存至: {OUTPUT_IMAGE_PATH}")


if __name__ == "__main__":
    logger.info("--- 预测任务开始！ ---")
    logger.info(f"模型文件： {INPUT_IMAGE_PATH}")
    # 检查输入文件是否存在
    if not os.path.exists(INPUT_IMAGE_PATH):
        logger.error(f"错误: 模型文件不存在 ->： {INPUT_IMAGE_PATH}")
    else:
        main()
        logger.info("--- 预测任务完成！ ---")

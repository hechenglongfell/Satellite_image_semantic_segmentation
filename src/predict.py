#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
predict.py

模型预测

Author he.cl
Date 2025-07-11
"""

import torch
import numpy as np
import rasterio
from rasterio.windows import Window
from PIL import Image
import torchvision.transforms as T
from tqdm import tqdm
import os
import argparse
import warnings
import time

from config import load_config
from utils.logger import setup_logger
from networks.model import build_unet

logger = setup_logger(__name__, "predict.log")


def _create_gaussian_kernel(size, sigma=1.0):
    """
    创建一个二维高斯核，用于加权。

    Args:
        size (int): 核的大小 (size x size).
        sigma (float): 高斯分布的标准差.

    Returns:
        numpy.ndarray: 归一化到 [0, 1] 的二维高斯核.
    """
    coords = np.arange(size, dtype=np.float32)
    coords -= (size - 1) / 2.0
    g = np.exp(-(coords**2) / (2 * sigma**2))
    g /= g.sum()
    kernel = np.outer(g, g)

    # 归一化到 [0, 1] 范围
    kernel /= kernel.max()
    return kernel


def main(config):
    """ """

    # --- 1. 读取配置参数 ---
    device = torch.device(config["vars"]["device"])
    weights_path = config["vars"]["predict_model"]
    image_path = config["vars"]["predict_image"]
    predict_result_dir = config["paths"]["predict_output"]
    patch_size = config["vars"]["predict_patch_size"]
    overlap_rate = config["vars"]["predict_overlap_rate"]
    strip_height = config["vars"]["predict_strip_height"]
    mean = config["vars"]["mean"]
    std = config["vars"]["std"]

    logger.info("--- 预测任务开始！ ---")
    logger.info(f"使用模型: {weights_path}")
    logger.info(f"待预测影像: {image_path}")
    logger.info(f"条带处理高度: {strip_height} pixels")

    # --- 2. 检查输入文件是否存在 ---
    if not os.path.exists(image_path):
        logger.error(f"错误: 待预测影像不存在 -> {image_path}")
        return
    if not os.path.exists(weights_path):
        logger.error(f"错误: 模型权重文件不存在 -> {weights_path}")
        return

    # --- 3. 准备输出路径 ---
    input_filename = os.path.basename(image_path)
    model_filename = os.path.splitext(os.path.basename(weights_path))[0]
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_filename = f"{os.path.splitext(input_filename)[0]}_pred_by_{model_filename}_{timestamp}.tif"
    output_image_path = os.path.join(predict_result_dir, output_filename)
    os.makedirs(predict_result_dir, exist_ok=True)

    # --- 4. 加载模型和定义转换 ---
    logger.info(f"正在加载模型: {weights_path}")
    # 使用 weights_only=True 更安全
    model = build_unet(device=device, config=config)
    model.load_state_dict(torch.load(weights_path, map_location=device, weights_only=True))
    model.eval()

    transform = T.Compose([T.ToTensor(), T.Normalize(mean=mean, std=std)])

    # --- 5. 使用 Rasterio 和重叠滑窗策略处理大图 ---
    logger.info(f"正在打开影像: {image_path}")
    with rasterio.open(image_path) as src:
        meta = src.meta
        width, height = src.width, src.height
        logger.info(f"影像尺寸: {width}x{height}")
        meta.update(count=1, dtype="uint8", compress="lzw", driver="GTiff")

        # --- 计算滑窗步长和所有可能的坐标 ---
        if not (0 <= overlap_rate < 1.0):
            raise ValueError("重叠率必须在 [0, 1.0) 范围内")
        step = int(patch_size * (1 - overlap_rate))
        if step <= 0:
            raise ValueError("重叠率过高或设置为1，导致步长小于或等于0，请降低重叠率")

        x_coords = list(range(0, width - patch_size, step)) if width > patch_size else [0]
        if width > patch_size and x_coords[-1] != width - patch_size:
            x_coords.append(width - patch_size)

        y_coords = list(range(0, height - patch_size, step)) if height > patch_size else [0]
        if height > patch_size and y_coords[-1] != height - patch_size:
            y_coords.append(height - patch_size)

        gaussian_kernel = _create_gaussian_kernel(patch_size, sigma=patch_size / 6)

        # --- 6. 逐条带处理和写入 ---
        logger.info(f"正在创建输出文件: {output_image_path}")
        try:
            with rasterio.open(output_image_path, "w", **meta) as dst:
                total_patches = len(y_coords) * len(x_coords)
                progress_bar = tqdm(total=total_patches, desc="预测中")

                for y_write_start in range(0, height, strip_height):
                    current_write_height = min(strip_height, height - y_write_start)
                    read_y_start = max(0, y_write_start - patch_size)
                    read_y_end = min(height, y_write_start + current_write_height + patch_size)
                    read_height = read_y_end - read_y_start

                    strip_prediction_acc = np.zeros((read_height, width), dtype=np.float32)
                    strip_weight_acc = np.zeros((read_height, width), dtype=np.float32)

                    relevant_y_coords = [y for y in y_coords if y < read_y_end and y + patch_size > read_y_start]

                    for y_start in relevant_y_coords:
                        for x_start in x_coords:
                            if y_start >= y_write_start and y_start < y_write_start + current_write_height:
                                progress_bar.update(1)

                            win_width = min(patch_size, width - x_start)
                            win_height = min(patch_size, height - y_start)
                            read_window = Window(x_start, y_start, win_width, win_height)

                            patch = src.read(window=read_window)
                            patch_rgb = np.transpose(patch, (1, 2, 0))
                            pil_img = Image.fromarray(patch_rgb).convert("RGB")

                            if pil_img.size != (patch_size, patch_size):
                                padded_img = T.Pad(
                                    padding=(0, 0, patch_size - win_width, patch_size - win_height),
                                    padding_mode="reflect",
                                )(pil_img)
                                input_tensor = transform(padded_img).unsqueeze(0).to(device)
                            else:
                                input_tensor = transform(pil_img).unsqueeze(0).to(device)

                            with torch.no_grad():
                                output = model(input_tensor)

                            probs = torch.sigmoid(output).cpu().numpy().squeeze()

                            y_start_in_strip = y_start - read_y_start

                            y_slice = slice(y_start_in_strip, y_start_in_strip + win_height)
                            x_slice = slice(x_start, x_start + win_width)

                            pred_acc_slice = strip_prediction_acc[y_slice, x_slice]
                            actual_slice_shape = pred_acc_slice.shape
                            probs_to_add = probs[: actual_slice_shape[0], : actual_slice_shape[1]]
                            kernel_to_add = gaussian_kernel[: actual_slice_shape[0], : actual_slice_shape[1]]

                            new_pred_value = pred_acc_slice + probs_to_add * kernel_to_add
                            strip_prediction_acc[y_slice, x_slice] = new_pred_value

                            new_weight_value = strip_weight_acc[y_slice, x_slice] + kernel_to_add
                            strip_weight_acc[y_slice, x_slice] = new_weight_value

                    strip_weight_acc[strip_weight_acc == 0] = 1e-6
                    final_strip = strip_prediction_acc / strip_weight_acc

                    write_slice_y_start = y_write_start - read_y_start
                    output_data_slice = final_strip[write_slice_y_start : write_slice_y_start + current_write_height, :]
                    output_data_uint8 = (output_data_slice * 255).astype(rasterio.uint8)

                    write_window = Window(0, y_write_start, width, current_write_height)
                    dst.write(output_data_uint8, window=write_window, indexes=1)

                progress_bar.close()
        except Exception as e:
            logger.error(f"处理或写入文件时发生错误: {e}", exc_info=True)
            return

    logger.info(f"预测完成！结果已保存至: {output_image_path}")
    logger.info("--- 预测任务完成！ ---")


def get_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="模型预测...")
    parser.add_argument("--config", type=str, default="config.yaml", help="配置文件的路径")
    parser.add_argument("--model_path", type=str, help="模型权重路径")
    parser.add_argument("--image_path", type=str, help="待预测影像路径")
    parser.add_argument("--output_dir", type=str, help="预测结果保存目录")
    parser.add_argument("--patch_size", type=int, help="滑窗大小")
    parser.add_argument("--overlap_rate", type=float, help="滑窗重叠率 (0.0-0.9)")
    parser.add_argument("--strip_height", type=int, help="每次处理的条带高度(像素)")
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()

    # 1. 从 YAML 文件加载基础配置
    config = load_config(args.config)

    # 2. 使用命令行参数覆盖配置
    if args.model_path:
        config["vars"]["predict_model"] = args.model_path
    if args.image_path:
        config["vars"]["predict_image"] = args.image_path
    if args.output_dir:
        config["paths"]["predict_output"] = args.output_dir
    if args.patch_size:
        config["vars"]["predict_patch_size"] = args.patch_size
    if args.overlap_rate is not None:
        config["vars"]["predict_overlap_rate"] = args.overlap_rate
    if args.strip_height is not None:
        config["vars"]["predict_strip_height"] = args.strip_height

    # 3. 运行主预测函数
    main(config)

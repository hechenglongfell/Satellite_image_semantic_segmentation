#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
predict.py

模型预测

Author he.cl
Date 2025-06-25
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

from config import load_config
from utils.logger import setup_logger
from networks.model import build_unet

logger = setup_logger(__name__, "predict.log")


def main(config):
    """
    主预测函数，接收配置字典作为参数。
    """

    # --- 1. 读取配置参数 ---
    device = torch.device(config["vars"]["device"])
    weights_path = config["vars"]["predict_model"]
    image_path = config["vars"]["predict_image"]
    predict_result_dir = config["paths"]["predict_output"]
    patch_size = config["vars"]["predict_patch_size"]
    overlap_rate = config["vars"]["predict_overlap_rate"]
    mean = config["vars"]["mean"]
    std = config["vars"]["std"]

    logger.info("--- 预测任务开始！ ---")
    logger.info(f"使用模型: {weights_path}")
    logger.info(f"待预测影像: {image_path}")

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
    output_filename = f"{os.path.splitext(input_filename)[0]}_pred_by_{model_filename}.tif"
    output_image_path = os.path.join(predict_result_dir, output_filename)
    os.makedirs(predict_result_dir, exist_ok=True)

    # --- 4. 加载模型和定义转换 ---
    logger.info(f"正在加载模型: {weights_path}")
    model = build_unet(device=device, config=config)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()

    transform = T.Compose([T.ToTensor(), T.Normalize(mean=mean, std=std)])

    # --- 5. 使用 Rasterio 和重叠滑窗策略处理大图 ---
    logger.info(f"正在打开影像: {image_path}")
    with rasterio.open(image_path) as src:
        meta = src.meta
        width, height = src.width, src.height
        logger.info(f"影像尺寸: {width}x{height}")
        # 更新输出文件的元数据
        meta.update(count=1, dtype="uint8", compress="lzw", driver="GTiff")

        with rasterio.open(output_image_path, "w", **meta) as dst:
            logger.info(f"正在创建输出文件: {output_image_path}")

            # --- 计算滑窗步长和坐标 ---
            if not (0 <= overlap_rate < 0.5):
                raise ValueError("重叠率必须在 [0, 0.5) 范围内")

            offset = int(patch_size * overlap_rate)
            step = patch_size - 2 * offset
            if step <= 0:
                raise ValueError("重叠率过高，导致步长小于或等于0，请降低重叠率")

            x_coords = list(range(0, width - patch_size, step)) if width > patch_size else [0]
            if width > patch_size and x_coords[-1] != width - patch_size:
                x_coords.append(width - patch_size)

            y_coords = list(range(0, height - patch_size, step)) if height > patch_size else [0]
            if height > patch_size and y_coords[-1] != height - patch_size:
                y_coords.append(height - patch_size)

            total_patches = len(y_coords) * len(x_coords)
            progress_bar = tqdm(total=total_patches, desc="预测中")

            for y_start in y_coords:
                for x_start in x_coords:
                    win_width = min(patch_size, width - x_start)
                    win_height = min(patch_size, height - y_start)
                    read_window = Window(x_start, y_start, win_width, win_height)

                    patch = src.read(window=read_window)
                    patch_rgb = np.transpose(patch, (1, 2, 0))

                    # 如果patch尺寸小于模型输入，需要填充
                    pil_img = Image.fromarray(patch_rgb).convert("RGB")
                    if pil_img.size != (patch_size, patch_size):
                        # 使用反射填充，然后裁剪回原始尺寸
                        padded_img = T.Pad(
                            padding=(0, 0, patch_size - win_width, patch_size - win_height), padding_mode="reflect"
                        )(pil_img)
                        input_tensor = transform(padded_img).unsqueeze(0).to(device=device)
                    else:
                        input_tensor = transform(pil_img).unsqueeze(0).to(device=device)

                    with torch.no_grad():
                        output = model(input_tensor)

                    probs = torch.sigmoid(output)
                    prob_map_uint8 = (probs.cpu().numpy().squeeze() * 255).astype(rasterio.uint8)

                    # 裁剪回原始patch尺寸
                    prob_map_uint8 = prob_map_uint8[:win_height, :win_width]

                    # 此处简化，直接写入整个预测块，更优化的方法是加权融合
                    dst.write(prob_map_uint8, window=read_window, indexes=1)
                    progress_bar.update(1)

            progress_bar.close()

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

    # 3. 运行主预测函数
    main(config)

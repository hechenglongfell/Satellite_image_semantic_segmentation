#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
model.py

定义用于图像分割的模型构建函数。
主要功能是利用 segmentation_models_pytorch 库快速构建一个 U-Net 模型。可以方便地替换骨干网络并加载预训练权重。

Author he.cl
Date 2025-06-25
"""

import torch
import segmentation_models_pytorch as smp
import os
import sys


def build_unet(device, config):
    """
    构建 U-Net 模型。
    使用 segmentation_models_pytorch 库可以轻松替换骨干网络。

    Args:
        device: torch.device, 模型将被移动到的设备 (CPU or GPU)。
        config (dict): 包含模型参数的配置字典。

    Returns:
        torch.nn.Module: 构建好的模型。
    """
    # 从传入的 config 字典中获取参数，而不是全局变量
    model = smp.Unet(
        encoder_name=config["vars"]["encoder"],
        encoder_weights=config["vars"]["encoder_weights"],
        in_channels=config["vars"]["num_channels"],
        classes=config["vars"]["num_classes"],
        activation=config["vars"].get("activation", None),  # 使用 .get() 增加灵活性
    )

    # 将模型移动到指定设备 (CPU or GPU)
    model.to(device)

    return model


if __name__ == "__main__":

    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    SRC_ROOT = os.path.dirname(SCRIPT_DIR)
    sys.path.append(SRC_ROOT)

    from config import load_config

    config_path = os.path.join(SRC_ROOT, "..", "config.yaml")
    config = load_config(config_path)

    device = torch.device(config["vars"]["device"])

    # 调用更新后的函数
    model = build_unet(device=device, config=config)

    dummy_input = torch.randn(
        config["vars"]["img_num"],
        config["vars"]["num_channels"],
        config["vars"]["img_size"],
        config["vars"]["img_size"],
    ).to(device=device)

    output = model(dummy_input)

    print(f"模型构建成功!")
    print(f"输入尺寸: {dummy_input.shape}")
    print(f"输出尺寸: {output.shape}")

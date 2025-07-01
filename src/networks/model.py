#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
model.py

这个文件定义了用于图像分割的模型构建函数。
主要功能是利用 segmentation_models_pytorch 库快速构建一个 U-Net 模型。
可以轻松地替换骨干网络并加载预训练权重。

Author: he.cl
Date: 2025-06-23
"""

import torch
import segmentation_models_pytorch as smp


def build_unet(device):
    """
    构建 U-Net 模型。
    使用 segmentation_models_pytorch 库可以轻松替换骨干网络。
    """
    # 'resnet34' 是一个常用的骨干网络，可以根据情况换成其他的网络。
    # 'imagenet' 表示使用在 ImageNet 上预训练的权重来初始化骨干网络
    model = smp.Unet(
        encoder_name="resnet34",  # 选择主干网络
        encoder_weights="imagenet",  # 使用预训练权重，不要从零开始随机初始化骨干网络的权重，而是直接加载已经在 ImageNet 数据集上训练好的权重
        in_channels=3,  # 输入通道数 (RGB 图像为 3)
        classes=1,  # 类别数 (例如，只分割水体，背景为0，水体为1，则为1)
    )

    # 将模型移动到指定设备 (CPU or GPU)
    model.to(device)

    return model


# 只有当这个脚本被直接运行时，才执行下面的代码。如果这个脚本被其他文件作为模块导入，下面的代码则不会执行。
if __name__ == '__main__':
    # 测试代码，确保模型可以被正确构建
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_unet(device)
    dummy_input = torch.randn(2, 3, 256, 256).to(device)  # (batch_size（表示一次性给模型喂入 2 张图片）, channels, H（图片高）, W（图片宽）)
    output = model(dummy_input)
    print(f"模型构建成功!")
    print(f"输入尺寸: {dummy_input.shape}")
    print(f"输出尺寸: {output.shape}")
    # 对于 classes=1, 输出 shape 为 [2, 1, 256, 256]

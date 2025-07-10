#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
train.py

模型训练。

Author: he.cl
Date: 2025-06-24
"""

import torch
import os
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from datetime import datetime

from config import global_config
from utils.logger import setup_logger

from utils.data_loader import RemoteSensingDataset, get_train_transforms, get_val_transforms
from networks.model import build_unet

logger = setup_logger(__name__, "train.log")


def train_one_epoch(model, dataloader, optimizer, loss_fn, device):
    model.train()  # 设置为训练模式
    epoch_loss = 0.0
    progress_bar = tqdm(dataloader, desc="Training")

    for images, masks in progress_bar:
        images = images.to(device)
        masks = masks.float().to(device)  # loss 函数可能需要 float 类型

        # 前向传播
        optimizer.zero_grad()  # 清空上一轮梯度
        outputs = model(images)  # 模型进行预测

        # 计算损失
        loss = loss_fn(outputs, masks)

        # 反向传播和优化
        loss.backward()  # 计算梯度
        optimizer.step()  # 更新权重

        epoch_loss += loss.item()
        progress_bar.set_postfix(loss=loss.item())

    avg_loss = epoch_loss / len(dataloader)
    logger.info(f"训练集平均损失值: {avg_loss:.4f}")

    return avg_loss


def evaluate(model, dataloader, loss_fn, device):
    """
    在验证集上评估模型。
    Args:
        model: 要评估的模型。
        dataloader: 验证数据加载器。
        loss_fn: 损失函数。
        device: 计算设备 (CPU or GPU)。
    Returns:
        验证集上的平均损失。
    """
    # 1. 设置为评估模式
    # 这会关闭 Dropout 和 BatchNorm 等层的训练行为，确保评估结果的确定性
    model.eval()

    total_loss = 0.0
    progress_bar = tqdm(dataloader, desc="Evaluating")

    # 2. 关闭梯度计算
    # 在评估阶段，我们不需要计算梯度，这样可以节省大量计算和内存资源
    with torch.no_grad():
        for images, masks in progress_bar:
            # 将数据移动到指定设备
            images = images.to(device)
            masks = masks.float().to(device)

            # 前向传播，获取模型输出
            outputs = model(images)

            # 计算损失
            loss = loss_fn(outputs, masks)

            # 累加批次损失
            total_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())

    # 3. 切换回训练模式
    # 在评估结束后，将模型恢复到训练模式，以便下一个 epoch 的训练
    model.train()

    # 计算并返回整个验证集的平均损失
    return total_loss / len(dataloader)


def main():

    # --- 从全局配置加载参数 ---

    device = torch.device(global_config["vars"]["device"])
    learning_rate = global_config["vars"]["learning_rate"]
    batch_size = global_config["vars"]["batch_size"]
    epochs = global_config["vars"]["epochs"]

    # 训练超参数配置
    logger.info("--- 训练超参数配置 ---")
    logger.info(f"计算设备: {device}")
    logger.info(f"初始学习率: {learning_rate}")
    logger.info(f"批次大小: {batch_size}")
    logger.info(f"训练轮次: {epochs}")
    logger.info("-----------------------------")

    # --- 1. 准备数据 ---
    train_dataset = RemoteSensingDataset(data_dir=global_config["paths"]["train_dir"], transform=get_train_transforms())
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

    val_dataset = RemoteSensingDataset(data_dir=global_config["paths"]["val_dir"], transform=get_val_transforms())
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    logger.info(f"训练集样本数: {len(train_dataset)}, 验证集样本数: {len(val_dataset)}")

    # --- 2. 构建模型、损失函数和优化器 ---
    model = build_unet(device=device)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # --- 3. 训练循环 ---
    now = datetime.now()
    timestamp_str = now.strftime("%Y%m%d_%H%M%S")
    save_model = os.path.join(global_config["paths"]["output_dir"], f"best_model_{timestamp_str}.pth")
    os.makedirs(os.path.dirname(save_model), exist_ok=True)
    best_val_loss = float("inf")  # 初始化一个无穷大的最佳损失值

    for epoch in range(epochs):
        logger.info(f"--- 开始第{epoch + 1}/{epochs} epoch 训练！ ---")
        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device)
        val_loss = evaluate(model, val_loader, loss_fn, device)
        logger.info(f"第{epoch + 1} epoch 训练结果: 训练集损失值: {train_loss:.4f}, 验证集损失值: {val_loss:.4f}")

        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), save_model)
            logger.info(f"最佳验证集损失值: {best_val_loss:.4f}。 保存当前模型!")

    # --- 4. 训练结束 ---
    logger.info(f"训练结束! 模型保存位置：{save_model}")


if __name__ == "__main__":
    main()

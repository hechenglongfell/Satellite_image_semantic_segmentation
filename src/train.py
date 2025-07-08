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
from utils.data_loader import RemoteSensingDataset, transform
from networks.model import build_unet

# --- 1. 超参数和配置 ---
LEARNING_RATE = 1e-4  # 学习率初始值，它控制模型每次更新权重时的“步长”。太大会导致模型不稳定，太小则训练速度过慢。
BATCH_SIZE = 4  # 一次处理的样本数量，并计算这4张图片的平均损失，然后更新一次权重。
NUM_EPOCHS = 10  # 一个完整训练的重复次数
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
DATA_DIR_TRAIN = os.path.join(PROJECT_ROOT, "data", "train")
DATA_DIR_VAL = os.path.join(PROJECT_ROOT, "data", "val")

now = datetime.now()
timestamp_str = now.strftime("%Y%m%d_%H%M%S")

WEIGHTS_SAVE_PATH = os.path.join(PROJECT_ROOT, "weights", f"best_model_{timestamp_str}.pth")

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

    # 记录本次训练超参数
    logger.info("--- 本次训练超参数 ---")
    logger.info(f"计算设备: {DEVICE}")
    logger.info(f"初始学习率: {LEARNING_RATE}")
    logger.info(f"一次处理的样本数: {BATCH_SIZE}")
    logger.info(f"模型训练次数: {NUM_EPOCHS}")
    logger.info("-----------------------------")
    # --- 1. 准备数据 ---
    train_dataset = RemoteSensingDataset(data_dir=DATA_DIR_TRAIN, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)

    val_dataset = RemoteSensingDataset(data_dir=DATA_DIR_VAL, transform=transform)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)

    # --- 2. 构建模型、损失函数和优化器 ---
    model = build_unet(device=DEVICE)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # --- 3. 训练循环 ---
    best_val_loss = float("inf")  # 初始化一个无穷大的最佳损失值

    for epoch in range(NUM_EPOCHS):
        logger.info(f"--- 开始第{epoch + 1}/{NUM_EPOCHS} epoch 训练！ ---")
        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, DEVICE)
        val_loss = evaluate(model, val_loader, loss_fn, DEVICE)
        logger.info(f"第{epoch + 1} epoch 训练结果: 训练集损失值: {train_loss:.4f}, 验证集损失值: {val_loss:.4f}")

        # --- 4. 保存最佳模型 ---
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), WEIGHTS_SAVE_PATH)
            logger.info(f"最佳验证集损失值: {best_val_loss:.4f}。 保存当前模型!")

    # 训练结束
    logger.info("训练结束! 模型保存位置：{WEIGHTS_SAVE_PATH}")


if __name__ == "__main__":
    main()

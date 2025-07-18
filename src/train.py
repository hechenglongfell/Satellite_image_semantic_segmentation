#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
train.py

模型训练

Author he.cl
Date 2025-06-25
"""

import torch
import os
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from datetime import datetime
import argparse

from config import load_config
from utils.logger import setup_logger

from utils.data_loader import MyDataLoader, DatasetStatistics, get_train_transforms, get_val_transforms
from networks.model import build_unet

logger = setup_logger(__name__, "train.log")


def train_one_epoch(model, dataloader, optimizer, loss_fn, device, logger):
    """
    在单个epoch上训练模型
    """
    model.train()  # 设置为训练模式
    epoch_loss = 0.0
    progress_bar = tqdm(dataloader, desc="Training")

    for images, masks in progress_bar:
        images = images.to(device)
        masks = masks.to(device)

        # 前向传播
        optimizer.zero_grad()  # 清空上一轮
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


def evaluate(model, dataloader, loss_fn, device, logger):
    """
    在验证集上评估模型
    """
    model.eval()  # 设置为评估模式
    total_loss = 0.0
    progress_bar = tqdm(dataloader, desc="Evaluating")

    with torch.no_grad():  # 关闭梯度计算
        for images, masks in progress_bar:
            images = images.to(device)
            masks = masks.to(device)
            outputs = model(images)
            loss = loss_fn(outputs, masks)
            total_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())

    avg_loss = total_loss / len(dataloader)
    logger.info(f"验证集平均损失值: {avg_loss:.4f}")
    return avg_loss


def main(config):
    """
    主训练函数，接收配置字典作为参数。
    """
    # --- 1. 设置计算设备 ---
    device = torch.device(config["vars"]["device"])
    architecture = config["vars"]["architecture"]
    encoder = config["vars"]["encoder"]
    learning_rate = config["vars"]["learning_rate"]
    batch_size = config["vars"]["batch_size"]
    epochs = config["vars"]["epochs"]
    mean = config["vars"]["mean"]
    std = config["vars"]["std"]
    class_weights = config["vars"]["class_weights"]
    num_classes = (config["vars"]["num_classes"],)

    # --- 2. 计算训练集统计信息 ---
    stats_calculator = DatasetStatistics(
        data_dir=config["paths"]["train_dir"],
        num_classes=num_classes,
        num_bands=config["vars"]["num_bands"],
    )
    # 更新到配置字典（内存）中。
    config["vars"].update(stats_calculator.get_stats())

    # 记录本次训练超参数配置
    logger.info("--- 训练超参数配置 ---")
    logger.info(f"计算设备: {device}")
    logger.info(f"模型架构: {architecture}")
    logger.info(f"骨干网络: {encoder}")
    logger.info(f"初始学习率: {learning_rate}")
    logger.info(f"批次大小: {batch_size}")
    logger.info(f"训练轮次: {epochs}")
    logger.info(f"任务类型(num_classes): {num_classes}")
    logger.info(f"均值(Mean): {mean}")
    logger.info(f"标准差(Std): {std}")
    logger.info(f"类别权重(class_weights): {num_classes}")
    logger.info("------------------------------------------------")

    # --- 3. 准备数据 ---
    train_dataset = MyDataLoader(
        data_dir=config["paths"]["train_dir"],
        num_bands=config["vars"]["num_bands"],
        num_classes=num_classes,
        transform=get_train_transforms(config),
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )

    val_dataset = MyDataLoader(
        data_dir=config["paths"]["val_dir"],
        num_bands=config["vars"]["num_bands"],
        num_classes=num_classes,
        transform=get_val_transforms(config),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    logger.info(f"训练集样本数: {len(train_dataset)}, 验证集样本数: {len(val_dataset)}")

    # --- 4. 构建模型、损失函数和优化器 ---
    model = build_unet(device=device, config=config)

    # todo :完善损失函数

    if num_classes == 1:
        # 二分类任务
        pos_weight = torch.tensor(class_weights, device=device).float() if class_weights else None
        loss_fn = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        logger.info(f"任务类型: 二分类")
        if pos_weight is not None:
            logger.info(f"Positive class weight: {pos_weight.item()}")
    else:
        # 多分类任务
        weight = torch.tensor(class_weights, device=device).float() if class_weights else None
        loss_fn = torch.nn.CrossEntropyLoss(weight=weight)
        logger.info(f"任务类型: 多分类,共 {num_classes} 类")
        if weight is not None:
            logger.info(f"Class weights: {weight.cpu().numpy()}")
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # --- 5. 训练循环 ---
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = f"{num_classes}_{num_classes}"
    save_dir = config["paths"]["output_dir"]
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"best_model_{model_name}_{timestamp_str}.pth")

    best_val_loss = float("inf")
    model.train()  # 确保模型在训练前处于训练模式

    for epoch in range(epochs):
        logger.info(f"--- 开始第 {epoch + 1}/{epochs} epoch 训练！ ---")

        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device, logger)
        val_loss = evaluate(model, val_loader, loss_fn, device, logger)

        logger.info(f"第 {epoch + 1} epoch 结束. 训练损失: {train_loss:.4f}, 验证损失: {val_loss:.4f}")

        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), save_path)
            logger.info(f"验证集损失值提升至: {best_val_loss:.4f}。保存当前最佳模型至: {save_path}")

    logger.info(f"训练结束! 最终模型保存在：{save_path}")


def get_args():
    """
    解析命令行参数
    """
    parser = argparse.ArgumentParser(description="训练语义分割模型训练...")
    parser.add_argument("--config", type=str, default="config.yaml", help="配置文件的路径")
    parser.add_argument("--lr", type=float, help="学习率")
    parser.add_argument("--batch_size", type=int, help="批次大小")
    parser.add_argument("--epochs", type=int, help="训练轮次")
    parser.add_argument("--device", type=str, help="计算设备 (例如 'cuda' 或 'cpu')")
    parser.add_argument("--num_classes", type=int, help="类别数")
    parser.add_argument("--num_bands", type=int, help="影像波段数")
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()

    # 1. 从 YAML 文件加载基础配置
    config = load_config(args.config)

    # 2. 使用命令行参数覆盖配置
    if args.lr:
        config["vars"]["learning_rate"] = args.lr
    if args.batch_size:
        config["vars"]["batch_size"] = args.batch_size
    if args.epochs:
        config["vars"]["epochs"] = args.epochs
    if args.device:
        config["vars"]["device"] = args.device
    if args.num_classes:
        config["vars"]["num_classes"] = args.num_classes
    if args.num_bands:
        config["vars"]["num_bands"] = args.num_bands

    # 3. 运行主训练函数
    main(config)

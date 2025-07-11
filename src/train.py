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

from utils.data_loader import RemoteSensingDataset, get_train_transforms, get_val_transforms
from networks.model import build_unet

logger = setup_logger(__name__, "train.log")


def train_one_epoch(model, dataloader, optimizer, loss_fn, device, logger):
    """在单个epoch上训练模型"""
    model.train()  # 设置为训练模式
    epoch_loss = 0.0
    progress_bar = tqdm(dataloader, desc="Training")

    for images, masks in progress_bar:
        images = images.to(device)
        masks = masks.float().to(device)  # loss 函数可能需要 float 类型

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
    """在验证集上评估模型"""
    model.eval()  # 设置为评估模式
    total_loss = 0.0
    progress_bar = tqdm(dataloader, desc="Evaluating")

    with torch.no_grad():  # 关闭梯度计算
        for images, masks in progress_bar:
            images = images.to(device)
            masks = masks.float().to(device)
            outputs = model(images)
            loss = loss_fn(outputs, masks)
            total_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())

    avg_loss = total_loss / len(dataloader)
    logger.info(f"验证集平均损失值: {avg_loss:.4f}")
    model.train()  # 恢复训练模式
    return avg_loss


def main(config):
    """
    主训练函数，接收配置字典作为参数。
    """
    # --- 1. 设置计算设备 ---
    device = torch.device(config["vars"]["device"])

    # --- 2. 超参数配置 ---
    logger.info("--- 训练超参数配置 ---")
    logger.info(f"计算设备: {device}")
    logger.info(f"模型架构: {config['vars']['architecture']}")
    logger.info(f"骨干网络: {config['vars']['encoder']}")
    logger.info(f"初始学习率: {config['vars']['learning_rate']}")
    logger.info(f"批次大小: {config['vars']['batch_size']}")
    logger.info(f"训练轮次: {config['vars']['epochs']}")
    logger.info("-----------------------------")

    # --- 3. 准备数据 ---
    train_dataset = RemoteSensingDataset(data_dir=config["paths"]["train_dir"], transform=get_train_transforms(config))
    train_loader = DataLoader(
        train_dataset, batch_size=config["vars"]["batch_size"], shuffle=True, num_workers=4, pin_memory=True
    )

    val_dataset = RemoteSensingDataset(data_dir=config["paths"]["val_dir"], transform=get_val_transforms(config))
    val_loader = DataLoader(
        val_dataset, batch_size=config["vars"]["batch_size"], shuffle=False, num_workers=4, pin_memory=True
    )

    logger.info(f"训练集样本数: {len(train_dataset)}, 验证集样本数: {len(val_dataset)}")

    # --- 4. 构建模型、损失函数和优化器 ---
    model = build_unet(device=device, config=config)
    loss_fn = torch.nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=config["vars"]["learning_rate"])

    # --- 5. 训练循环 ---
    now = datetime.now()
    timestamp_str = now.strftime("%Y%m%d_%H%M%S")
    model_name = f"{config['vars']['architecture']}_{config['vars']['encoder']}"
    save_dir = config["paths"]["output_dir"]
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"best_model_{model_name}_{timestamp_str}.pth")

    best_val_loss = float("inf")

    for epoch in range(config["vars"]["epochs"]):
        logger.info(f"--- 开始第{epoch + 1}/{config['vars']['epochs']} epoch 训练！ ---")

        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device, logger)
        val_loss = evaluate(model, val_loader, loss_fn, device, logger)

        logger.info(f"第{epoch + 1} epoch 结束. 训练损失: {train_loss:.4f}, 验证损失: {val_loss:.4f}")

        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), save_path)
            logger.info(f"验证集损失值提升至: {best_val_loss:.4f}。保存当前最佳模型至: {save_path}")

    logger.info(f"训练结束! 最终模型保存在：{save_path}")


def get_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description="训练语义分割模型...")
    parser.add_argument("--config", type=str, default="config.yaml", help="配置文件的路径")
    parser.add_argument("--lr", type=float, help="学习率")
    parser.add_argument("--batch_size", type=int, help="批次大小")
    parser.add_argument("--epochs", type=int, help="训练轮次")
    parser.add_argument("--device", type=str, help="计算设备 (例如 'cuda' 或 'cpu')")
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()

    # 1. 从 YAML 文件加载基础配置
    config = load_config(args.config)

    # 2. 使用命令行参数覆盖配置（如果提供了的话）
    if args.lr:
        config["vars"]["learning_rate"] = args.lr
    if args.batch_size:
        config["vars"]["batch_size"] = args.batch_size
    if args.epochs:
        config["vars"]["epochs"] = args.epochs
    if args.device:
        config["vars"]["device"] = args.device

    # 3. 运行主训练函数
    main(config)

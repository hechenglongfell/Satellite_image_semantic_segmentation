#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Description 日志
Author he.cl
Date 2025-07-08
"""

import logging
import sys
from pathlib import Path
from logging.handlers import TimedRotatingFileHandler

from config import global_config

# 日志格式
FORMATTER = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s", datefmt="%Y-%m-%d %H:%M:%S")

# 日志文件路径
LOG_DIR = Path(global_config["paths"]["log_dir"])

# 判断logs是否存在并创建
LOG_DIR.mkdir(exist_ok=True)


def get_console_handler():
    """
    获取一个输出到控制台的处理器
    """
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(FORMATTER)
    return console_handler


def get_file_handler(log_file_name: str):
    """
    获取一个输出到文件的处理器
    - TimedRotatingFileHandler: 可以按时间自动分割日志文件
    - when='D': 表示每天分割一次
    - backupCount=30: 表示最多保留30个旧日志文件
    """
    file_handler = TimedRotatingFileHandler(LOG_DIR / log_file_name, when="D", backupCount=30, encoding="utf-8")
    file_handler.setFormatter(FORMATTER)
    return file_handler


def setup_logger(name: str, log_file_name: str, level=logging.INFO):
    """
    配置并获取一个日志记录器

    Args:
        name (str): 日志记录器的名称，通常使用 __name__
        log_file_name (str): 日志文件名，例如 'train.log' 或 'predict.log'
        level (int, optional): 日志级别. Defaults to logging.INFO.

    Returns:
        logging.Logger: 配置好的日志记录器
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # 防止重复添加处理器
    if not logger.handlers:
        logger.addHandler(get_console_handler())
        logger.addHandler(get_file_handler(log_file_name))

    # 设置不向上传递日志信息
    logger.propagate = False

    return logger

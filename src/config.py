#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Description 加载YAML配置文件

Author he.cl
Date 2025-07-08
"""

import yaml
from pathlib import Path


def load_config(config_path: str = "config.yaml"):
    """
    加载并解析YAML配置文件。

    Args:
        config_path (str, optional): YAML配置文件的路径. 默认为 "config.yaml".

    Raises:
        FileNotFoundError: 如果找不到配置文件，则抛出此异常。

    Returns:
        dict: 包含所有配置参数的字典。
    """
    config_file = Path(config_path)
    if not config_file.is_file():
        raise FileNotFoundError(f"Config file not found at the project root: {config_path}")

    with open(config_file, "r", encoding="utf-8") as f:
        # 使用 safe_load 防止执行任意代码，更安全
        config = yaml.safe_load(f)
    return config


# --- 创建一个全局配置对象 ---
# 项目中的任何模块都可以直接 `from config import global_config`
# 来获取配置，避免了重复加载文件。
try:
    global_config = load_config()
except FileNotFoundError as e:
    print(f"Error: {e}. Please make sure 'config.yaml' exists in the project root directory.")
    global_config = {}

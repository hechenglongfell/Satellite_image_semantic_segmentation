"""
Description
Author he.cl
Date 2025-06-27
"""

import torch
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
WEIGHTS_PATH = os.path.join(PROJECT_ROOT, "..\weights", "best_model.pth")  # 模型路径

try:
    # 加载文件
    data = torch.load(WEIGHTS_PATH)

    # 判断否为字典
    if isinstance(data, dict):
        print(f"文件 '{WEIGHTS_PATH}' 是一个字典，包含以下内容：")
        print("=" * 50)

        # 遍历字典的键和值
        for key, value in data.items():
            print(f"\n--- 键 (Key): {key} ---")

            # 如果值是 PyTorch 张量 (Tensor)
            if isinstance(value, torch.Tensor):
                print(f"  值 (Value): 是一个 Tensor")
                print(f"    - 形状 (Shape): {value.shape}")
                print(f"    - 数据类型 (dtype): {value.dtype}")
                print(f"    - 设备 (Device): {value.device}")

            # 如果值是字典
            elif isinstance(value, dict):
                print(f"  值 (Value): 是一个字典，包含 {len(value)} 个项目。")
                # 打印字典的前几个键
                print(f"    - 内部的键 (Keys): {list(value.keys())[:5]} ...")

            # 如果是其他基本类型 (如 int, float, str)
            else:
                print(f"  值 (Value): {value}")

        print("\n" + "=" * 50)
        print("遍历完成。")

    else:
        print(f"文件 '{WEIGHTS_PATH}' 不是一个字典。")
        print(f"加载出的数据类型是: {type(data)}")

except FileNotFoundError:
    print(f"错误：找不到文件 '{WEIGHTS_PATH}'，请检查路径是否正确。")
except Exception as e:
    print(f"加载文件时发生错误: {e}")

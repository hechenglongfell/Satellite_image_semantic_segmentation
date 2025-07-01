"""
Description 模型格式转化
Author he.cl
Date 2025-06-30
"""

# pth_to_onnx.py

import os
import sys
import torch
import torch.onnx
import onnx
import onnxruntime

from datetime import datetime

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_ROOT = os.path.dirname(SCRIPT_DIR)
PROJECT_ROOT = os.path.dirname(SRC_ROOT)

sys.path.append(SRC_ROOT)

from networks.model import build_unet

pth_model_path = os.path.join(PROJECT_ROOT, "weights", "best_model.pth")

now = datetime.now()
timestamp_str = now.strftime("%Y%m%d_%H%M%S")
onnx_model_path = os.path.join(PROJECT_ROOT, "outputs", f"model_{timestamp_str}.onnx")

# 确认模型的输入参数，对于图像分割模型，通常是 (batch_size, channels, height, width)
BATCH_SIZE = 1  # 导出为ONNX时，batch_size通常设为1，并设为动态轴
INPUT_CHANNELS = 3  # RGB图像
INPUT_HEIGHT = 256
INPUT_WIDTH = 256


def main():
    """主转换函数"""
    # 检查.pth文件是否存在
    if not os.path.exists(pth_model_path):
        print(f"错误: Pytorch权重文件不存在于 '{pth_model_path}'")
        return

    # --- 检查是否有可用的GPU ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 实例化模型
    model = build_unet(device=device)

    # 加载.pth权重文件
    model.load_state_dict(torch.load(pth_model_path, map_location=device))

    # 设置为评估（推理）模式
    model.eval()

    # 创建虚拟输入
    dummy_input = torch.randn(BATCH_SIZE, INPUT_CHANNELS, INPUT_HEIGHT, INPUT_WIDTH, device=device)

    # 导出到ONNX
    print(f"正在将模型导出为ONNX格式，保存至 '{onnx_model_path}'...")
    try:
        torch.onnx.export(
            model,  # 要导出的模型
            dummy_input,  # 模型的虚拟输入
            onnx_model_path,  # ONNX文件保存路径
            export_params=True,  # 导出模型权重
            opset_version=12,  # ONNX算子集版本，11或12是常用稳定版
            do_constant_folding=True,  # 是否执行常量折叠优化
            input_names=["input"],  # 输入张量的名字
            output_names=["output"],  # 输出张量的名字
            dynamic_axes={
                "input": {0: "batch_size"},  # 设置动态轴 (让batch_size可变)
                "output": {0: "batch_size"},
            },
        )
        print(f"模型成功导出为ONNX格式: {onnx_model_path}")
    except Exception as e:
        print(f"导出ONNX时发生错误: {e}")
        return

    # 验证ONNX模型
    if onnx is not None and onnxruntime is not None:
        verify_onnx_model(dummy_input)


def verify_onnx_model(dummy_input):
    """验证生成的ONNX模型是否正确"""
    try:
        onnx_model = onnx.load(onnx_model_path)
        onnx.checker.check_model(onnx_model)

        # 使用ONNX Runtime进行一次推理测试
        ort_session = onnxruntime.InferenceSession(
            onnx_model_path, providers=["CPUExecutionProvider"]
        )  # 强制使用CPU验证

        ort_inputs = {ort_session.get_inputs()[0].name: dummy_input.cpu().numpy()}
        ort_outs = ort_session.run(None, ort_inputs)

        print("ONNX Runtime推理测试成功！")
        print(f"模型输出形状: {ort_outs[0].shape}")

    except Exception as e:
        print(f"验证ONNX模型时发生错误: {e}")


if __name__ == "__main__":
    main()

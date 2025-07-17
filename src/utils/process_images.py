"""
Description: 批量处理图像，将多通道（大于3）图像转换为3通道RGB图像。
【最终诊断版】
"""

import os
import sys
import numpy as np
from PIL import Image

# ------------------- 关键诊断区 -------------------
try:
    # 尝试导入真正的tqdm库
    from tqdm import tqdm

    # 检查导入的tqdm是否是我们需要的那个可调用的“类”
    if "function" not in str(type(tqdm)) and "class" not in str(type(tqdm)):
        print("\n\n[诊断错误]: 导入的 'tqdm' 无法被调用！")
        print("这几乎可以100%确定是由于文件名冲突造成的。")
        print("请检查您的项目文件夹（尤其是 'src/utils/'）下，是否存在一个您自己创建的、名为 'tqdm.py' 的文件。")
        print("如果存在，请将其重命名或删除。然后，务必删除 'src/utils/__pycache__' 文件夹后重试。")
        sys.exit()  # 立即退出程序
except ImportError:
    print("\n\n[诊断错误]: 未能找到 'tqdm' 库。")
    print("请先在命令行运行 'pip install tqdm' 进行安装。")
    sys.exit()
# ----------------- 诊断区结束 -----------------


# --- 配置 ---
INPUT_DIR = "D:/BD_Project/Satellite_image_semantic_segmentation/data/val/images"
OUTPUT_DIR = "D:/BD_Project/Satellite_image_semantic_segmentation/data/val/data_rgb"
SUPPORTED_EXTENSIONS = (".tif", ".tiff", ".png")
# --- 配置结束 ---


def process_images(input_dir, output_dir):
    print(f"开始扫描图像文件...")
    print(f"输入目录: {input_dir}")
    print(f"输出目录: {output_dir}")

    image_files_to_process = []
    for root, _, files in os.walk(input_dir):
        for filename in files:
            if filename.lower().endswith(SUPPORTED_EXTENSIONS):
                input_path = os.path.join(root, filename)
                relative_path = os.path.relpath(root, input_dir)
                current_output_dir = os.path.join(output_dir, relative_path)
                output_path = os.path.join(current_output_dir, filename)
                image_files_to_process.append((input_path, output_path, current_output_dir))

    if not image_files_to_process:
        print("\n警告：在输入目录中没有找到任何支持的图像文件。")
        return

    print(f"共找到 {len(image_files_to_process)} 个图像文件，开始处理...")

    processed_count = 0
    skipped_count = 0

    # 这里的tqdm现在可以保证是正确的了
    for input_path, output_path, current_output_dir in tqdm(image_files_to_process, desc="处理图像中"):
        try:
            os.makedirs(current_output_dir, exist_ok=True)
            with Image.open(input_path) as img:
                img_array = np.array(img)
                if img_array.ndim == 3 and img_array.shape[2] > 3:
                    rgb_array = img_array[:, :, :3]
                    new_img = Image.fromarray(rgb_array)
                    new_img.save(output_path)
                    processed_count += 1
                else:
                    img.save(output_path)
                    skipped_count += 1
        except Exception as e:
            print(f"\n处理文件 {os.path.basename(input_path)} 时发生错误: {e}")

    print("\n处理完成！")
    print(f"总共转换了 {processed_count} 个多波段图像。")
    print(f"总共复制了 {skipped_count} 个无需转换的图像。")


if __name__ == "__main__":
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    process_images(INPUT_DIR, OUTPUT_DIR)

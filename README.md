# Satellite Image Semantic Segmentation

[![Python-3.9](https://img.shields.io/badge/Python-3.9-blue.svg)](https://www.python.org/downloads/release/python-390/)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)](https://pytorch.org/)
[![Conda](https://img.shields.io/badge/conda-4.10%2B-green)](https://docs.conda.io/en/latest/)

这是一个基于 PyTorch 实现的卫星影像语义分割项目。项目旨在提供一个清晰、可扩展的框架，用于训练和部署不同的深度学习模型（如 U-Net, DeepLabV3 等）来解决遥感图像的像素级分类任务。

## 项目简介

语义分割是计算机视觉中的一项关键技术，其目标是为图像中的每个像素分配一个类别标签。在卫星遥感领域，这项技术可以用于地物分类、土地利用监测、城市规划、环境变化分析等多种场景。本项目通过深度学习方法，实现了对遥感影像的自动化、高精度地物信息提取。

## ✨ 功能特性

* **模型支持**:
    * ✅ **U-Net**: 已实现的经典图像分割网络。
    * 🔄 **DeepLabV3+**: 计划支持，适用于复杂场景的先进模型。
    * **可扩展性**: 框架设计清晰，方便集成更多其他分割网络。
* **数据处理**: 包含完整的数据加载、预处理和数据增强流程。
* **训练与评估**: 提供完整的模型训练、验证和评估脚本。
* **环境管理**: 使用 Conda 进行环境和依赖管理，确保环境的可复现性。

## 🚀 快速开始

请按照以下步骤在您的本地环境中设置并运行此项目。

### 1. 先决条件

* 确保您已经安装了 [Anaconda](https://www.anaconda.com/products/distribution) 或 [Miniconda](https://docs.conda.io/en/latest/miniconda.html)。
* 硬件：推荐使用支持 CUDA 的 NVIDIA GPU 以加速模型训练。

### 2. 克隆项目

```bash
git clone [https://github.com/hechenglongfell/Satellite_image_semantic_segmentation.git](https://github.com/hechenglongfell/Satellite_image_semantic_segmentation.git)
cd Satellite_image_semantic_segmentation
```

### 3. 创建 Conda 环境并安装依赖

项目依赖已在 `environment.yml` 文件中定义。请使用以下命令创建并激活 Conda 环境：

```bash
# 从 environment.yml 文件创建环境
conda env create -f environment.yml

# 激活新创建的环境
conda activate satseg
```

> **备选方案 (requirements.txt):**
> 如果您倾向于使用 `pip` 和 `requirements.txt`，可以先创建一个新的 Conda 环境，然后安装依赖：
> ```bash
> conda create --name satseg python=3.9 -y
> conda activate satseg
> pip install -r requirements.txt
> ```
> *注意：您需要手动创建 `requirements.txt` 文件。*

## 📁 数据集准备

1.  **下载数据集**:
    本项目使用一个公开的航拍/卫星影像数据集。请从 [此处提供数据集下载链接] 下载数据集，并将其解压。
    *(请在此处替换为您的数据集的实际下载链接和描述)*

2.  **组织目录结构**:
    请将您的数据集按照以下结构进行组织，以便代码能够正确读取：

    ```
    data/
    ├── train/
    │   ├── images/
    │   │   ├── 001.png
    │   │   ├── 002.png
    │   │   └── ...
    │   └── masks/
    │       ├── 001.png
    │       ├── 002.png
    │       └── ...
    └── val/
        ├── images/
        │   ├── 101.png
        │   └── ...
        └── masks/
            ├── 101.png
            └── ...
    ```

    * `images/` 目录下存放原始卫星影像。
    * `masks/` 目录下存放对应的语义分割标签图（mask）。
    * **重要**: 请确保训练集和验证集中，影像和其对应的标签图文件名完全一致。

## ⚙️ 使用方法

### 1. 模型训练

通过运行 `train.py` 脚本来启动模型的训练。您可以自定义训练参数。

**基本训练命令:**
```bash
python train.py --data_path ./data --epochs 50 --batch_size 4 --lr 1e-4
```

**可配置参数:**

* `--data_path`: 数据集根目录的路径。
* `--model`: 选择要使用的模型 (例如: `unet`)。默认为 `unet`。
* `--epochs`: 训练的总轮数。
* `--batch_size`: 训练的批量大小。
* `--lr`: 学习率 (learning rate)。
* `--device`: 指定训练设备 (`cuda` 或 `cpu`)。
* `--img_size`: 输入图像的尺寸。

查看所有可用参数：
```bash
python train.py --help
```

训练好的模型权重将默认保存在 `checkpoints/` 目录下。

### 2. 模型预测

使用 `predict.py` 脚本和一个已经训练好的模型权重，对新的单张或多张图像进行分割预测。

```bash
python predict.py --input ./data/test/images/ --output ./preds/ --model_path ./checkpoints/best_model.pth
```

* `--input`: 待预测图像所在的文件夹路径。
* `--output`: 保存预测结果的文件夹路径。
* `--model_path`: 训练好的模型权重文件 (`.pth` 文件) 的路径。

## 💡 未来工作

* [ ] 实现 DeepLabV3+ 网络结构。
* [ ] 引入更丰富的数据增强策略。
* [ ] 添加 TensorBoard 可视化支持，以监控训练过程。
* [ ] 优化 `predict.py` 脚本，支持对大尺寸遥感影像的滑窗预测。

## 🤝 贡献

欢迎任何形式的贡献！如果您有好的想法或发现了问题，请随时提交 Pull Request 或创建 Issue。

1.  Fork 本项目
2.  创建您的新分支 (`git checkout -b feature/AmazingFeature`)
3.  提交您的更改 (`git commit -m 'Add some AmazingFeature'`)
4.  将您的分支推送到远程仓库 (`git push origin feature/AmazingFeature`)
5.  创建一个 Pull Request

## 📄 许可证

本项目采用 [MIT License](LICENSE) 开源许可证。
*(请在您的项目中添加一个名为 `LICENSE` 的文件，并将 MIT 许可证文本粘贴进去)*
#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
智能小车目标板数据集划分脚本
============================

功能说明:
    将 png_smartcar 文件夹中的图片数据集按 8:2 比例划分为训练集和测试集。
    专门为智能小车目标识别任务设计。

数据集来源:
    png_smartcar/
    ├── 交通工具-直行/
    ├── 武器-左/
    └── 物资-右/

使用方法:
    # 直接运行（使用默认配置）
    python split_dataset.py

    # 运行前请确保数据集路径正确

输出结构:
    data/smartcar/
    ├── train/
    │   ├── 交通工具-直行/  (12张)
    │   ├── 武器-左/        (13张)
    │   └── 物资-右/        (20张)
    └── test/
        ├── 交通工具-直行/  (4张)
        ├── 武器-左/        (4张)
        └── 物资-右/        (6张)

配置说明:
    SRC_DIR    : 源数据集路径
    DATA_DIR   : 输出数据集路径
    TRAIN_RATIO: 训练集比例，默认为 0.8
"""

import os
import shutil
import random
from pathlib import Path


# ==================== 配置参数 ====================
# 源数据集路径（请根据实际情况修改）
SRC_DIR = Path("png_smartcar")

# 输出数据集路径
DATA_DIR = Path("data/smartcar")

# 训练集比例：0.8 表示 80% 训练，20% 测试
TRAIN_RATIO = 0.8

# 随机种子，确保结果可复现
RANDOM_SEED = 42
# =============================================


def split_smartcar_dataset():
    """
    划分智能小车目标板数据集

    该函数会:
    1. 扫描源目录中的所有类别文件夹
    2. 将每个类别的图片随机划分为训练集和测试集
    3. 将划分后的图片复制到对应的输出目录

    Returns:
        None
    """
    if not SRC_DIR.exists():
        print(f"错误: 源数据集目录不存在: {SRC_DIR}")
        return

    random.seed(RANDOM_SEED)

    print("=" * 50)
    print("智能小车目标板数据集划分")
    print("=" * 50)
    print(f"源目录: {SRC_DIR}")
    print(f"输出目录: {DATA_DIR}")
    print(f"训练集比例: {TRAIN_RATIO * 100:.0f}%")
    print("-" * 50)

    class_names = ["交通工具-直行", "武器-左", "物资-右"]

    for class_name in class_names:
        class_dir = SRC_DIR / class_name

        if not class_dir.exists():
            print(f"警告: 类别目录不存在: {class_dir}，跳过")
            continue

        images = list(class_dir.glob("*.png"))

        if not images:
            print(f"警告: {class_name} 目录下未找到图片，跳过")
            continue

        random.shuffle(images)

        split_idx = int(len(images) * TRAIN_RATIO)
        train_images = images[:split_idx]
        test_images = images[split_idx:]

        for img in train_images:
            dst = DATA_DIR / "train" / class_name / img.name
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(img, dst)

        for img in test_images:
            dst = DATA_DIR / "test" / class_name / img.name
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(img, dst)

        print(f"{class_name}: {len(train_images)} 训练, {len(test_images)} 测试")

    print("-" * 50)
    print("数据集划分完成!")
    print(f"训练集: {DATA_DIR}/train/")
    print(f"测试集: {DATA_DIR}/test/")
    print("=" * 50)


if __name__ == "__main__":
    split_smartcar_dataset()

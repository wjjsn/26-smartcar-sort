#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
通用数据集整理脚本
==================

功能说明:
    将任意格式的图片数据集自动整理为 PyTorch 标准的 train/test 划分结构。
    支持多类别分类任务，每个类别对应一个子文件夹。

使用方法:
    # 基本用法（默认 8:2 划分）
    python organize_dataset.py png_smartcar

    # 指定输出目录
    python organize_dataset.py png_smartcar --output mydata

    # 指定训练集比例
    python organize_dataset.py png_smartcar --ratio 0.9

    # 组合使用
    python organize_dataset.py png_smartcar -o mydata -r 0.8

输入要求:
    - 源文件夹中每个子文件夹视为一个类别
    - 支持的图片格式: .png, .jpg, .jpeg
    - 图片格式混合也可以正常处理

输出结构:
    output_dir/
    ├── train/
    │   ├── 类别A/
    │   │   ├── img1.png
    │   │   └── img2.png
    │   └── 类别B/
    │       └── img3.png
    └── test/
        ├── 类别A/
        └── 类别B/

参数说明:
    src_dir   : 源数据集路径（必填）
    --output  : 输出目录路径，默认为 "data"
    --ratio   : 训练集比例，默认为 0.8（80%训练，20%测试）
    -h        : 显示帮助信息
"""

import os
import shutil
import random
from pathlib import Path
from typing import Optional


def organize_dataset(
    src_dir: str,
    output_dir: str = "data",
    train_ratio: float = 0.8,
    seed: int = 42,
    pattern: Optional[list[str]] = None,
) -> None:
    """
    整理数据集为 train/test 结构

    Args:
        src_dir: 源数据集目录路径，该目录下应包含多个子目录（每个子目录为一个类别）
        output_dir: 输出目录路径，默认为 "data"
        train_ratio: 训练集比例，范围 0-1，默认为 0.8
        seed: 随机种子，确保结果可复现，默认为 42
        pattern: 文件名匹配模式列表，如 ["warped_*.png", "*.jpg"]。如果为 None，则匹配所有图片格式

    Returns:
        None

    Example:
        >>> organize_dataset("png_smartcar", "data", 0.8)
        物资-右: 20 训练, 6 测试
        交通工具-直行: 12 训练, 4 测试
        武器-左: 13 训练, 4 测试

        # 只处理 warped_*.png 和 *.jpg 文件
        >>> organize_dataset("out", "data/smartcar", pattern=["warped_*.png", "*.jpg"])
    """
    src_path = Path(src_dir)
    if not src_path.exists():
        print(f"错误: 目录 {src_dir} 不存在")
        return

    random.seed(seed)

    class_dirs = [d for d in src_path.iterdir() if d.is_dir()]

    if not class_dirs:
        print(f"错误: 在 {src_dir} 中未找到任何子目录（类别）")
        return

    print(f"找到 {len(class_dirs)} 个类别，开始整理数据集...\n")

    for class_dir in class_dirs:
        class_name = class_dir.name

        if pattern:
            images = []
            for p in pattern:
                images.extend(list(class_dir.glob(p)))
            images = list(set(images))
        else:
            images = (
                list(class_dir.glob("*.png"))
                + list(class_dir.glob("*.jpg"))
                + list(class_dir.glob("*.jpeg"))
            )

        if not images:
            print(f"警告: {class_name} 目录下未找到图片文件，跳过")
            continue

        random.shuffle(images)

        split_idx = int(len(images) * train_ratio)

        train_images = images[:split_idx]
        test_images = images[split_idx:]

        for img in train_images:
            dst = Path(output_dir) / "train" / class_name / img.name
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(img, dst)

        for img in test_images:
            dst = Path(output_dir) / "test" / class_name / img.name
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(img, dst)

        print(f"{class_name}: {len(train_images)} 训练, {len(test_images)} 测试")

    print(f"\n数据集整理完成!")
    print(f"训练集: {output_dir}/train/")
    print(f"测试集: {output_dir}/test/")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="通用数据集整理工具 - 将图片数据集自动划分为训练集和测试集",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:
  python organize_dataset.py png_smartcar
  python organize_dataset.py png_smartcar --output mydata
  python organize_dataset.py images -o dataset -r 0.9
        """,
    )

    parser.add_argument(
        "src_dir", help="源数据集目录路径（该目录下应包含多个类别子目录）"
    )

    parser.add_argument(
        "--output", "-o", default="data", help="输出目录路径（默认: data）"
    )

    parser.add_argument(
        "--ratio",
        "-r",
        type=float,
        default=0.8,
        help="训练集比例，范围 0-1（默认: 0.8）",
    )

    parser.add_argument(
        "--pattern",
        "-p",
        type=str,
        nargs="+",
        default=None,
        help="文件名匹配模式列表，如 'warped_*.png' '*.jpg'（默认: 匹配所有图片格式）",
    )

    args = parser.parse_args()

    organize_dataset(
        src_dir=args.src_dir,
        output_dir=args.output,
        train_ratio=args.ratio,
        pattern=args.pattern,
    )

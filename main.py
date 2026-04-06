#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
SmartCar 图像分类完整流程脚本
==============================

本脚本通过调用现有的模块函数完成完整流程：
    1. 预处理 - 调用 dataset.scripts.organize_dataset.organize_dataset
    2. 训练   - 调用 training.smartcar_train.train
    3. 推理   - 调用 inference.smartcar_predict.main

使用方法:
    python main.py                      # 运行完整流程
    python main.py --stage preprocess   # 仅预处理数据集
    python main.py --stage train        # 仅训练模型
    python main.py --stage inference    # 仅进行推理
    python main.py --epochs 30          # 自定义训练参数
"""

import sys
import argparse
from pathlib import Path

# 将项目根目录添加到模块搜索路径
sys.path.insert(0, str(Path(__file__).parent))


def main():
    """
    主函数入口

    参数说明:
        --stage: 运行阶段
            - all: 运行完整流程（预处理 + 训练 + 推理）
            - preprocess: 仅进行数据集预处理
            - train: 仅进行模型训练
            - inference: 仅进行模型推理
        --src_dir: 源数据集目录（默认: png_smartcar）
        --output_dir: 输出目录（默认: data/smartcar）
        --train_ratio: 训练集比例（默认: 0.8）
        --epochs: 训练轮数（默认: 20）
    """
    parser = argparse.ArgumentParser(
        description="SmartCar 图像分类 - 完整流程脚本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--stage",
        type=str,
        default="all",
        choices=["all", "preprocess", "train", "inference"],
        help="运行阶段: all(完整流程), preprocess(预处理), train(训练), inference(推理)",
    )
    parser.add_argument(
        "--src_dir",
        type=str,
        default="png_smartcar",
        help="源数据集目录（默认: png_smartcar）",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/smartcar",
        help="输出目录（默认: data/smartcar）",
    )
    parser.add_argument(
        "--train_ratio", type=float, default=0.8, help="训练集比例 0-1（默认: 0.8）"
    )
    parser.add_argument("--epochs", type=int, default=50, help="训练轮数（默认: 50）")
    parser.add_argument(
        "--framework",
        type=str,
        default="pytorch",
        choices=["pytorch", "tensorflow"],
        help="深度学习框架: pytorch 或 tensorflow（默认: pytorch）",
    )

    args = parser.parse_args()

    print("=" * 60)
    print("SmartCar 图像分类系统")
    print(f"运行阶段: {args.stage}")
    print(f"框架: {args.framework}")
    print("=" * 60)

    # ----------------------------------------
    # 阶段1: 数据集预处理
    # 步骤1.1: 调用 preprocessing.detect_red.main
    # 功能: 检测A4纸红色区域并进行透视校正，输出到 out/ 目录
    # 步骤1.2: 调用 dataset.scripts.organize_dataset.organize_dataset
    # 功能: 将校正后的图片整理为 train/test 目录结构
    # ----------------------------------------
    if args.stage in ["all", "preprocess"]:
        print("\n>>> 阶段1: 数据集预处理")
        print("-" * 40)

        # 步骤1.1: A4纸检测与透视校正
        print("\n[1.1] A4纸检测与透视校正")
        print("-" * 30)
        from preprocessing.detect_red import main as detect_red_main

        detect_red_main(
            input_dir=args.src_dir,
            output_dir="out",
            categories=["交通工具-直行", "武器-左", "物资-右"],
        )

        # 步骤1.2: 整理数据集为 train/test 结构
        print("\n[1.2] 整理数据集为 train/test 结构")
        print("-" * 30)
        from dataset.scripts.organize_dataset import organize_dataset

        organize_dataset(
            src_dir="out",
            output_dir=args.output_dir,
            train_ratio=args.train_ratio,
            seed=42,
            pattern="warped_*.png",
        )

    # ----------------------------------------
    # 阶段2: 模型训练
    # 调用 training.smartcar_train.train
    # 功能: 使用CNN模型进行训练并保存模型
    # ----------------------------------------
    if args.stage in ["all", "train"]:
        print("\n>>> 阶段2: 模型训练")
        print("-" * 40)

        if args.framework == "pytorch":
            from training.smartcar_train import train
        else:
            from training.smartcar_train_tf import train

        train(epochs=args.epochs)

    # ----------------------------------------
    # 阶段3: 模型推理
    # 调用 inference.smartcar_predict.main
    # 功能: 加载模型并对测试集进行预测
    # ----------------------------------------
    if args.stage in ["all", "inference"]:
        print("\n>>> 阶段3: 模型推理")
        print("-" * 40)

        if args.framework == "pytorch":
            from inference.smartcar_predict import main as inference_main
        else:
            from inference.smartcar_predict_tf import main as inference_main

        inference_main()

    print("\n" + "=" * 60)
    print("所有任务已完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()

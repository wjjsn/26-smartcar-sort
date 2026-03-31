import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
from PIL import Image
import logging
import sys

from models.cnn import MNISTCNN
from utils.device import get_device
from utils.transforms import MNIST_TRANSFORM

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def predict(image_path):
    logger.info(f"开始预测图像: {image_path}")

    device = get_device()
    logger.info(f"使用设备: {device}")

    logger.info("初始化CNN模型...")
    model = MNISTCNN().to(device)
    logger.debug(f"模型结构: {model}")

    logger.info("从'mnist_model.pth'加载模型权重...")
    model.load_state_dict(torch.load("mnist_model.pth", weights_only=True))
    logger.info("模型权重加载成功")

    model.eval()
    logger.debug("模型设置为评估模式")

    logger.info("定义图像转换管道...")
    transform = MNIST_TRANSFORM
    logger.debug(f"转换管道: {transform}")

    logger.info(f"加载并处理图像: {image_path}")
    image = Image.open(image_path)
    logger.debug(f"原始图像尺寸: {image.size}, 模式: {image.mode}")

    image = transform(image).unsqueeze(0).to(device)
    logger.debug(f"转换后图像形状: {image.shape}, 数据类型: {image.dtype}")

    logger.info("运行推理...")
    with torch.no_grad():
        output = model(image)
        logger.debug(f"模型输出(logits): {output}")

        probs = torch.softmax(output, dim=1)
        logger.debug(f"Softmax概率: {probs}")

        pred = output.argmax(dim=1).item()
        confidence = probs[0][pred].item()
        logger.debug(f"预测类别索引: {pred}, 置信度: {confidence:.4f}")

    logger.info(f"预测完成 - 预测数字: {pred} (置信度: {confidence:.4f})")
    print(f"Predicted digit: {pred}")
    return pred


if __name__ == "__main__":
    if len(sys.argv) > 1:
        predict(sys.argv[1])
    else:
        print("Usage: python predict.py <image_path>")

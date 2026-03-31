SMARTCAR_CLASSES = ["交通工具-直行", "武器-左", "物资-右"]

IDX_TO_CLASS = {i: c for i, c in enumerate(SMARTCAR_CLASSES)}
CLASS_TO_IDX = {c: i for i, c in enumerate(SMARTCAR_CLASSES)}

MNIST_CLASSES = [str(i) for i in range(10)]

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import shutil
import random

from config.classes import SMARTCAR_CLASSES


def prepare_dataset(test_ratio=0.2):
    src_dir = Path("out")
    dst_dir = Path("data/smartcar")

    categories = SMARTCAR_CLASSES

    for split in ["train", "test"]:
        for cat in categories:
            (dst_dir / split / cat).mkdir(parents=True, exist_ok=True)

    for cat in categories:
        cat_path = src_dir / cat
        if not cat_path.exists():
            print(f"Warning: {cat_path} does not exist")
            continue

        warped_files = list(cat_path.glob("warped_*.png"))
        print(f"{cat}: {len(warped_files)} warped images")

        random.shuffle(warped_files)
        split_idx = int(len(warped_files) * (1 - test_ratio))

        for i, f in enumerate(warped_files):
            split = "train" if i < split_idx else "test"
            dst = dst_dir / split / cat / f.name
            shutil.copy2(f, dst)

    print(f"\nDataset prepared at {dst_dir}")
    for split in ["train", "test"]:
        print(f"\n{split}:")
        for cat in categories:
            count = len(list((dst_dir / split / cat).glob("*.png")))
            print(f"  {cat}: {count} images")


if __name__ == "__main__":
    random.seed(42)
    prepare_dataset()

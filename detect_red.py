import cv2
import numpy as np
from pathlib import Path


def detect_a4_by_red(image_path):
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"Failed to load {image_path}")
        return None

    h, w = img.shape[:2]

    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    lower_red1 = np.array([0, 100, 100])
    upper_red1 = np.array([15, 255, 255])
    lower_red2 = np.array([150, 100, 100])
    upper_red2 = np.array([180, 255, 255])

    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    mask = cv2.bitwise_or(mask1, mask2)

    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    result = img.copy()

    if not contours:
        print(f"{image_path.name}: No red region found")
        return img

    center_red_contours = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        x, y, cw, ch = cv2.boundingRect(cnt)
        cnt_center_x = x + cw / 2

        if 0.25 * w < cnt_center_x < 0.75 * w and area > 1000 and cw > 40 and ch > 20:
            center_red_contours.append((cnt, area, x, y, cw, ch))

    if not center_red_contours:
        print(f"{image_path.name}: No center red region found")
        return img

    center_red_contours.sort(key=lambda r: r[1], reverse=True)
    best_cnt, best_area, rx, ry, rw, rh = center_red_contours[0]

    cv2.drawContours(result, [best_cnt], -1, (255, 0, 0), 1)

    return result


def main():
    input_dir = Path("png_smartcar")
    output_dir = Path("out")
    output_dir.mkdir(exist_ok=True)

    categories = ["交通工具-直行", "武器-左", "物资-右"]

    for category in categories:
        category_path = input_dir / category
        if not category_path.exists():
            continue

        category_out = output_dir / category
        category_out.mkdir(exist_ok=True)

        for img_path in category_path.glob("*.png"):
            result = detect_a4_by_red(img_path)
            if result is not None:
                output_path = category_out / img_path.name
                cv2.imwrite(str(output_path), result)


if __name__ == "__main__":
    main()

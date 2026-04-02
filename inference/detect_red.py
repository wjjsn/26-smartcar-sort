"""
A4纸红色区域检测与透视校正模块
=============================

通过检测图像中的红色区域来定位A4纸，并进行透视校正。

主要功能：
- 检测图像中的红色标记区域
- 根据红色区域计算A4纸的四个角点
- 进行透视变换，输出校正后的图像

使用方法：
    from inference.detect_red import detect_a4_by_red

    # 基本用法
    result, warped = detect_a4_by_red("image.png")

    # 自定义参数
    result, warped = detect_a4_by_red(
        image_path="image.png",
        red_hsv_ranges=(
            np.array([0, 100, 100]),    # 红色范围1下限
            np.array([15, 255, 255]),   # 红色范围1上限
            np.array([150, 100, 100]), # 红色范围2下限
            np.array([180, 255, 255]), # 红色范围2上限
        ),
        center_ratio=0.5,       # 中心区域占比
        min_area=1000,          # 最小红色区域面积
        phys_width=12.0,        # A4纸宽度(cm)
        phys_img_height=12.0,    # 图像区域高度(cm)
        phys_red_height=5.0,    # 红色标记高度(cm)
    )
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Optional, Tuple, List


def create_red_mask(
    hsv: np.ndarray,
    lower_red1: np.ndarray,
    upper_red1: np.ndarray,
    lower_red2: np.ndarray,
    upper_red2: np.ndarray,
) -> np.ndarray:
    """
    根据HSV颜色空间创建红色区域掩码。

    红色在HSV空间中分布在两端(0-15和150-180)，因此需要两个范围。

    参数:
        hsv: HSV颜色空间的图像
        lower_red1: 红色范围1的下限 (H, S, V)
        upper_red1: 红色范围1的上限 (H, S, V)
        lower_red2: 红色范围2的下限 (H, S, V)
        upper_red2: 红色范围2的上限 (H, S, V)

    返回:
        二值掩码图像，红色区域为白色(255)，其他区域为黑色(0)
    """
    mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
    return cv2.bitwise_or(mask1, mask2)


def apply_morphology(mask: np.ndarray, kernel_size: int = 3) -> np.ndarray:
    """
    对掩码图像进行形态学操作，去除噪声和填充空洞。

    操作流程:
        1. MORPH_CLOSE: 先膨胀后腐蚀，填充白色区域内部的小黑洞
        2. MORPH_OPEN: 先腐蚀后膨胀，去除白色区域边缘的细小噪点

    参数:
        mask: 输入的二值掩码图像
        kernel_size: 形态学操作的核大小，默认3x3

    返回:
        处理后的二值掩码图像
    """
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    return mask


def filter_center_contours(
    contours: List,
    img_width: int,
    center_ratio: float = 0.5,
    min_area: float = 1000,
    min_width: int = 40,
    min_height: int = 20,
) -> List:
    """
    筛选位于图像中心区域的红色轮廓。

    筛选条件:
        - 轮廓中心X坐标位于图像中央的center_ratio范围内
        - 轮廓面积大于min_area
        - 轮廓宽高大于指定的最小值

    参数:
        contours: opencv找到的所有轮廓列表
        img_width: 原始图像宽度，用于计算中心区域范围
        center_ratio: 中心区域占图像宽度的比例，默认0.5表示中心50%区域
        min_area: 轮廓最小面积阈值，默认1000
        min_width: 轮廓最小宽度(像素)，默认40
        min_height: 轮廓最小高度(像素)，默认20

    返回:
        按面积降序排列的筛选后轮廓列表，每个元素为(轮廓, 面积, x, y, 宽, 高)
    """
    center_range = (
        (0.5 - center_ratio / 2) * img_width,
        (0.5 + center_ratio / 2) * img_width,
    )
    filtered = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        x, y, cw, ch = cv2.boundingRect(cnt)
        cnt_center_x = x + cw / 2
        if (
            center_range[0] < cnt_center_x < center_range[1]
            and area > min_area
            and cw > min_width
            and ch > min_height
        ):
            filtered.append((cnt, area, x, y, cw, ch))
    return sorted(filtered, key=lambda r: r[1], reverse=True)


def find_corner_points(
    points: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    从轮廓点集中找出四个角点(左上、右上、右下、左下)。

    算法原理:
        - 左上角: x+y 的值最小
        - 右下角: x+y 的值最大
        - 右上角: y-x 的值最小 (即 x-y 最大)
        - 左下角: y-x 的值最大 (即 x-y 最小)

    参数:
        points: 轮廓上所有点的坐标数组，形状为(N, 2)，每行是[x, y]

    返回:
        tl, tr, br, bl: 左上、右上、右下、左下四个角点的坐标
    """
    add = points.sum(axis=1)
    diff = np.diff(points, axis=1)
    tl = points[np.argmin(add)]
    br = points[np.argmax(add)]
    tr = points[np.argmin(diff)]
    bl = points[np.argmax(diff)]
    return tl, tr, br, bl


def draw_contour(
    result: np.ndarray,
    contour: np.ndarray,
    color: Tuple[int, int, int] = (255, 0, 0),
    thickness: int = 1,
) -> None:
    """
    在图像上绘制轮廓线。

    参数:
        result: 绘制目标图像
        contour: 要绘制的轮廓
        color: 轮廓颜色，默认蓝色(255, 0, 0)
        thickness: 线条粗细，默认1
    """
    cv2.drawContours(result, [contour], -1, color, thickness)


def draw_corner_points(
    result: np.ndarray,
    corners: List[np.ndarray],
    color: Tuple[int, int, int] = (0, 255, 0),
    radius: int = 1,
    thickness: int = -1,
) -> None:
    """
    在图像上绘制角点(圆点)。

    参数:
        result: 绘制目标图像
        corners: 角点坐标列表
        color: 圆点颜色，默认绿色(0, 255, 0)
        radius: 圆点半径，默认1
        thickness: 线条粗细，-1表示实心圆，默认实心
    """
    for point in corners:
        px, py = point.ravel()
        cv2.circle(result, (px, py), radius, color, thickness)


def extend_line(
    point1: np.ndarray,
    point2: np.ndarray,
    extend_length: float,
) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    """
    计算线段的延长线两端点。

    以两点中点为起点，沿线段方向延长指定长度，返回延长线的两个端点。

    参数:
        point1: 线段第一个端点 [x, y]
        point2: 线段第二个端点 [x, y]
        extend_length: 延长长度(像素)，从线段中点向两侧各延伸的长度

    返回:
        延长线两个端点的坐标 (pt1, pt2)，每个是(int, int)元组
    """
    mid_x = (point1[0] + point2[0]) / 2
    mid_y = (point1[1] + point2[1]) / 2
    dir_x = point1[0] - point2[0]
    dir_y = point1[1] - point2[1]
    length = np.sqrt(dir_x**2 + dir_y**2)
    dir_x /= length
    dir_y /= length
    pt1 = (int(mid_x - dir_x * extend_length), int(mid_y - dir_y * extend_length))
    pt2 = (int(mid_x + dir_x * extend_length), int(mid_y + dir_y * extend_length))
    return pt1, pt2


def draw_extended_lines(
    result: np.ndarray,
    tl: np.ndarray,
    bl: np.ndarray,
    tr: np.ndarray,
    br: np.ndarray,
    extend_length: float = 500,
    color: Tuple[int, int, int] = (0, 255, 0),
    thickness: int = 1,
) -> None:
    """
    绘制左右两条延长线，用于辅助确定A4纸的边界。

    左侧线连接左上(TL)和左下(BL)角点，右侧线连接右上(TR)和右下(BR)角点。

    参数:
        result: 绘制目标图像
        tl: 左上角点坐标 [x, y]
        bl: 左下角点坐标 [x, y]
        tr: 右上角点坐标 [x, y]
        br: 右下角点坐标 [x, y]
        extend_length: 延长线从角点延伸的长度，默认500像素
        color: 线条颜色，默认绿色(0, 255, 0)
        thickness: 线条粗细，默认1
    """
    left_pt1, left_pt2 = extend_line(tl, bl, extend_length)
    right_pt1, right_pt2 = extend_line(tr, br, extend_length)
    cv2.line(result, left_pt1, left_pt2, color, thickness)
    cv2.line(result, right_pt1, right_pt2, color, thickness)


def compute_top_edge_points(
    src_pts: np.ndarray,
    phys_width: float,
    phys_img_height: float,
    phys_red_height: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    根据透视变换关系计算A4纸顶部边缘的两个角点。

    利用已知的物理尺寸比例，通过透视变换矩阵推算出纸张顶边角点位置。

    物理坐标系假设:
        - 原点位于红色标记区域的左上角
        - X轴向右，Y轴向下
        - 红色区域: (0, img_height) 到 (width, img_height)
        - 红色区域下方: (0, img_height) 到 (width, img_height + red_height)

    参数:
        src_pts: 图像中红色区域四个角点的坐标 [[tl], [tr], [br], [bl]]
        phys_width: A4纸物理宽度，默认12cm
        phys_img_height: 红色标记区域上边缘的物理高度，默认12cm
        phys_red_height: 红色标记区域的物理高度，默认5cm

    返回:
        pt_top_left, pt_top_right: 纸张顶部边缘的左角点和右角点坐标
    """
    w, h_img, h_red = phys_width, phys_img_height, phys_red_height
    dst_pts = np.array(
        [[0, h_img], [w, h_img], [w, h_img + h_red], [0, h_img + h_red]],
        dtype=np.float32,
    )
    M = cv2.getPerspectiveTransform(dst_pts, src_pts)
    top_points_physical = np.array([[[0, 0], [w, 0]]], dtype=np.float32)
    top_points_image = cv2.perspectiveTransform(top_points_physical, M)[0]
    return tuple(top_points_image[0].astype(int)), tuple(
        top_points_image[1].astype(int)
    )


def draw_top_edge(
    result: np.ndarray,
    tl: np.ndarray,
    tr: np.ndarray,
    pt_top_left: Tuple[int, int],
    pt_top_right: Tuple[int, int],
    line_color: Tuple[int, int, int] = (0, 0, 255),
    corner_color: Tuple[int, int, int] = (0, 0, 255),
    thickness: int = 1,
    corner_radius: int = 2,
) -> None:
    """
    绘制A4纸顶部边缘线及其角点。

    绘制内容:
        1. 顶部边缘线(红色)
        2. 顶部左右两个角点(红色圆点)
        3. 连接红色区域角点与顶部角点的斜线(绿色)

    参数:
        result: 绘制目标图像
        tl: 红色区域左上角点 [x, y]
        tr: 红色区域右上角点 [x, y]
        pt_top_left: 计算出的纸张顶部左上角点 (x, y)
        pt_top_right: 计算出的纸张顶部右上角点 (x, y)
        line_color: 顶部边缘线颜色，默认红色(0, 0, 255)
        corner_color: 角点颜色，默认红色(0, 0, 255)
        thickness: 线条粗细，默认1
        corner_radius: 角点圆点半径，默认2
    """
    cv2.line(result, pt_top_left, pt_top_right, line_color, thickness)
    cv2.circle(result, pt_top_left, corner_radius, corner_color, -1)
    cv2.circle(result, pt_top_right, corner_radius, corner_color, -1)
    cv2.line(result, tuple(tl.astype(int)), pt_top_left, (0, 255, 0), thickness)
    cv2.line(result, tuple(tr.astype(int)), pt_top_right, (0, 255, 0), thickness)


def perspective_crop(
    img: np.ndarray,
    tl: np.ndarray,
    tr: np.ndarray,
    pt_top_left: Tuple[int, int],
    pt_top_right: Tuple[int, int],
) -> np.ndarray:
    """
    对图像进行透视变换，裁剪出校正后的A4纸区域。

    参数:
        img: 原始输入图像
        tl: 左上角点 [x, y]
        tr: 右上角点 [x, y]
        pt_top_left: 顶部左上角点 (x, y)
        pt_top_right: 顶部右上角点 (x, y)

    返回:
        透视变换后的图像，尺寸根据角点位置自动计算
    """
    src_quad = np.array([tl, tr, pt_top_right, pt_top_left], dtype=np.float32)
    width = int(np.linalg.norm(tr - tl))
    height = int(np.linalg.norm(pt_top_left - tl))
    dst_quad = np.array(
        [[0, 0], [width, 0], [width, height], [0, height]], dtype=np.float32
    )
    M_warp = cv2.getPerspectiveTransform(src_quad, dst_quad)
    return cv2.warpPerspective(img, M_warp, (width, height))


def detect_a4_by_red(
    image_path: str,
    red_hsv_ranges: Optional[
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    ] = None,
    morphology_kernel_size: int = 3,
    center_ratio: float = 0.5,
    min_area: float = 1000,
    min_width: int = 40,
    min_height: int = 20,
    phys_width: float = 12.0,
    phys_img_height: float = 12.0,
    phys_red_height: float = 5.0,
    extend_length: float = 500.0,
    draw_result: bool = True,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """
    检测图像中的A4纸红色标记区域并进行透视校正。

    处理流程:
        1. 将图像转换为HSV颜色空间
        2. 创建红色区域掩码
        3. 形态学操作去除噪声
        4. 查找并筛选中心区域的红色轮廓
        5. 找出红色区域的四个角点
        6. 根据物理尺寸计算A4纸顶部边缘
        7. 进行透视变换，裁剪校正后的图像

    参数:
        image_path: 输入图像路径，支持字符串或Path对象

        red_hsv_ranges: 红色HSV颜色范围元组
            - lower_red1: 红色范围1的下限 [H, S, V]
            - upper_red1: 红色范围1的上限 [H, S, V]
            - lower_red2: 红色范围2的下限 [H, S, V]
            - upper_red2: 红色范围2的上限 [H, S, V]
            默认值None使用标准红色范围:
            ([0, 100, 100], [15, 255, 255], [150, 100, 100], [180, 255, 255])

        morphology_kernel_size: 形态学操作核大小，默认3x3
            - 值越大，噪声去除效果越好，但可能丢失细节

        center_ratio: 中心区域占图像宽度的比例，默认0.5
            - 0.5表示只接受位于图像中心50%区域的红色标记
            - 值越小，筛选越严格

        min_area: 红色轮廓最小面积(像素²)，默认1000
            - 用于过滤误检测的小轮廓

        min_width: 红色轮廓最小宽度(像素)，默认40

        min_height: 红色轮廓最小高度(像素)，默认20

        phys_width: A4纸物理宽度(cm)，默认12.0cm
            - 用于透视变换的物理坐标计算

        phys_img_height: 图像区域相对于红色标记的物理高度(cm)，默认12.0cm
            - 即红色标记上边缘距离A4纸顶边的距离

        phys_red_height: 红色标记条物理高度(cm)，默认5.0cm

        extend_length: 延长线延伸长度(像素)，默认500
            - 用于绘制辅助线，帮助可视化检测结果

        draw_result: 是否在结果图像上绘制检测信息，默认True
            - True: 返回标注了角点和延长线的原图 + 透视校正图
            - False: 返回(None, 透视校正图)

    返回:
        如果成功检测:
            (result, warped) - 标注图像和透视校正后的图像
        如果检测失败:
            (原图, None) 如果 draw_result=True
            None 如果 draw_result=False
    """
    img = cv2.imread(str(image_path))
    if img is None:
        print(f"Failed to load {image_path}")
        return None

    h, w = img.shape[:2]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    if red_hsv_ranges is None:
        lower_red1 = np.array([0, 100, 100])
        upper_red1 = np.array([15, 255, 255])
        lower_red2 = np.array([150, 100, 100])
        upper_red2 = np.array([180, 255, 255])
    else:
        lower_red1, upper_red1, lower_red2, upper_red2 = red_hsv_ranges

    mask = create_red_mask(hsv, lower_red1, upper_red1, lower_red2, upper_red2)
    mask = apply_morphology(mask, morphology_kernel_size)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        print(f"{Path(image_path).name}: No red region found")
        return img if draw_result else None

    center_contours = filter_center_contours(
        contours, w, center_ratio, min_area, min_width, min_height
    )

    if not center_contours:
        print(f"{Path(image_path).name}: No center red region found")
        return img if draw_result else None

    best_cnt = center_contours[0][0]
    points = best_cnt.reshape(-1, 2)
    tl, tr, br, bl = find_corner_points(points)

    if draw_result:
        result = img.copy()
        draw_contour(result, best_cnt)
        draw_corner_points(result, [tl, tr, br, bl])
        draw_extended_lines(result, tl, bl, tr, br, extend_length)

        src_pts = np.array([tl, tr, br, bl], dtype=np.float32)
        pt_top_left, pt_top_right = compute_top_edge_points(
            src_pts, phys_width, phys_img_height, phys_red_height
        )
        draw_top_edge(result, tl, tr, pt_top_left, pt_top_right)
    else:
        result = None

    warped = perspective_crop(img, tl, tr, pt_top_left, pt_top_right)

    return result, warped


def main(
    input_dir: str = "png_smartcar",
    output_dir: str = "out",
    categories: Optional[List[str]] = ["交通工具-直行", "武器-左", "物资-右"],
) -> None:
    """
    批量处理目录中的图像，检测并校正A4纸区域。

    参数:
        input_dir: 输入图像目录路径，默认"png_smartcar"
        output_dir: 输出目录路径，默认"out"
        categories: 类别子目录列表，默认["交通工具-直行", "武器-左", "物资-右"]
    """
    input_path = Path(input_dir)
    out_path = Path(output_dir)
    out_path.mkdir(exist_ok=True)

    for category in categories:
        category_path = input_path / category
        if not category_path.exists():
            continue

        category_out = out_path / category
        category_out.mkdir(exist_ok=True)

        for img_path in category_path.glob("*.png"):
            output_data = detect_a4_by_red(img_path)
            if output_data is not None:
                result, warped = output_data
                output_path = category_out / img_path.name
                cv2.imwrite(str(output_path), result)
                warped_path = category_out / f"warped_{img_path.name}"
                cv2.imwrite(str(warped_path), cv2.rotate(warped, cv2.ROTATE_180))


if __name__ == "__main__":
    main()

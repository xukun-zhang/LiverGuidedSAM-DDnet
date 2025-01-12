import os
import numpy as np
from PIL import Image
from scipy.spatial import cKDTree


def dice_coefficient(mask, label):
    smooth = 1e-5
    intersection = np.sum((mask == 1) & (label == 1))
    total = np.sum(mask == 1) + np.sum(label == 1)
    return 2.0 * intersection / (total + smooth)


def iou(mask, label):
    smooth = 1e-5
    intersection = np.sum((mask == 1) & (label == 1))
    union = np.sum((mask == 1) | (label == 1))
    return intersection / (union + smooth)


def chamfer_distance(mask, label):
    mask_points = np.argwhere(mask == 1)
    label_points = np.argwhere(label == 1)

    if len(mask_points) == 0 or len(label_points) == 0:
        return 500 #np.inf  # 如果其中一个没有点，返回宽的一半长度，设置为500

    # 使用 KD-Tree 查找最近邻点
    mask_tree = cKDTree(mask_points)
    label_tree = cKDTree(label_points)

    # 计算从 mask_points 到 label_points 的最近邻距离
    mask_to_label_distances, _ = label_tree.query(mask_points)
    label_to_mask_distances, _ = mask_tree.query(label_points)

    # 计算 Chamfer Distance
    chamfer_dist = np.mean(mask_to_label_distances) + np.mean(label_to_mask_distances)

    return chamfer_dist


def calculate_metrics(mask, label):
    dice_vals = []
    iou_vals = []
    chamfer_vals = []

    # 分别计算每个通道的指标
    for i in range(mask.shape[0]):
        dice = dice_coefficient(mask[i], label[i])
        iou_val = iou(mask[i], label[i])
        chamfer = chamfer_distance(mask[i], label[i])

        dice_vals.append(dice)
        iou_vals.append(iou_val)
        chamfer_vals.append(chamfer)

    # 将3个通道合并为1种类别计算整体指标
    combined_mask = np.any(mask == 1, axis=0).astype(np.uint8)
    combined_label = np.any(label == 1, axis=0).astype(np.uint8)
    combined_dice = dice_coefficient(combined_mask, combined_label)
    combined_iou = iou(combined_mask, combined_label)
    combined_chamfer = chamfer_distance(combined_mask, combined_label)

    return dice_vals, iou_vals, chamfer_vals, combined_dice, combined_iou, combined_chamfer


def process_image(image_path):
    # 读取RGB PNG图片
    image = Image.open(image_path).convert("RGB")
    image_array = np.array(image)

    # 将RGB图片的每个通道值大于1的像素设置为1，背景保持为0
    binary_mask = (image_array > 125).astype(np.uint8)
    return np.moveaxis(binary_mask, -1, 0)  # 将通道维度移到第一个维度 (3, H, W)


def main(mask_folder, label_folder):
    mask_files = [f for f in os.listdir(mask_folder) if f.endswith('.png')]

    all_dice_vals = []
    all_iou_vals = []
    all_chamfer_vals = []

    combined_dice_vals = []
    combined_iou_vals = []
    combined_chamfer_vals = []

    # 遍历每个 mask 文件并找到对应的 label 文件
    for mask_file in mask_files:
        mask_path = os.path.join(mask_folder, mask_file)
        label_path = os.path.join(label_folder, mask_file)

        if not os.path.exists(label_path):
            print(f"Label file not found for {mask_file}")
            continue

        # 处理图片，将其转换为二值掩码
        mask_array = process_image(mask_path)
        label_array = process_image(label_path)

        # 确保 mask 和 label 的形状匹配
        if mask_array.shape != label_array.shape:
            print(f"Shape mismatch for {mask_file}")
            continue

        # 计算各项指标
        dice_vals, iou_vals, chamfer_vals, combined_dice, combined_iou, combined_chamfer = calculate_metrics(mask_array,
                                                                                                             label_array)

        # 打印每个样本的结果
        print(f"Results for {mask_file}:")
        print(f"  Dice values for each channel: {dice_vals}")
        print(f"  IOU values for each channel: {iou_vals}")
        print(f"  Chamfer distance for each channel: {chamfer_vals}")
        print(f"  Combined Dice value: {combined_dice}")
        print(f"  Combined IOU value: {combined_iou}")
        print(f"  Combined Chamfer distance: {combined_chamfer}")

        # 记录结果以便计算平均值
        all_dice_vals.append(dice_vals)
        all_iou_vals.append(iou_vals)
        all_chamfer_vals.append(chamfer_vals)

        combined_dice_vals.append(combined_dice)
        combined_iou_vals.append(combined_iou)
        combined_chamfer_vals.append(combined_chamfer)

    # 计算平均值
    avg_dice_vals = np.mean(all_dice_vals, axis=0) if all_dice_vals else [0, 0, 0]
    avg_iou_vals = np.mean(all_iou_vals, axis=0) if all_iou_vals else [0, 0, 0]
    avg_chamfer_vals = np.mean(all_chamfer_vals, axis=0) if all_chamfer_vals else [np.inf, np.inf, np.inf]

    avg_combined_dice = np.mean(combined_dice_vals) if combined_dice_vals else 0
    avg_combined_iou = np.mean(combined_iou_vals) if combined_iou_vals else 0
    avg_combined_chamfer = np.mean(combined_chamfer_vals) if combined_chamfer_vals else np.inf

    print("\nAverage results across all samples:")
    print(f"  Average Dice values for each channel: {avg_dice_vals}")
    print(f"  Average IOU values for each channel: {avg_iou_vals}")
    print(f"  Average Chamfer distances for each channel: {avg_chamfer_vals}")
    print(f"  Average combined Dice value: {avg_combined_dice}")
    print(f"  Average combined IOU value: {avg_combined_iou}")
    print(f"  Average combined Chamfer distance: {avg_combined_chamfer}")

    return avg_dice_vals, avg_iou_vals, avg_chamfer_vals, avg_combined_dice, avg_combined_iou, avg_combined_chamfer



# 对比实验结果
mask_folder = "G:/output/mask"  # 替换为实际的mask文件夹路径
label_folder = "G:/Code/20241002-landmark2d/D2PGL_DATA/Test/labels"  # 替换为实际的label文件夹路径

main(mask_folder, label_folder)

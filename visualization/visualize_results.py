import os
from PIL import Image
import numpy as np


def overlay_images(jpg_dir, png_dir, output_dir, alpha=0.5):
    # 创建输出路径（如果不存在）
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 获取jpg图片列表
    jpg_files = [f for f in os.listdir(jpg_dir) if f.endswith('.jpg')]

    for jpg_file in jpg_files:
        jpg_path = os.path.join(jpg_dir, jpg_file)
        png_file = jpg_file.replace('.jpg', '.png')
        png_path = os.path.join(png_dir, png_file)

        # 检查png图片是否存在
        if not os.path.exists(png_path):
            print(f"PNG file {png_file} not found for JPG file {jpg_file}. Skipping...")
            continue

        # 打开jpg和png图片
        jpg_image = Image.open(jpg_path).convert('RGBA')
        png_image = Image.open(png_path).convert('RGBA')

        # 转换图片为numpy数组
        jpg_array = np.array(jpg_image)
        png_array = np.array(png_image)

        # 创建一个透明度蒙版：仅保留png中非黑色部分并应用alpha透明度
        non_black_mask = np.any(png_array[:, :, :3] > 0, axis=-1)  # 仅当RGB值大于0时，才生成蒙版
        alpha_mask = non_black_mask.astype(np.float32) * alpha  # 将非黑色部分应用alpha透明度

        # 叠加png图片到jpg图片，仅在非黑色区域进行叠加
        for c in range(3):  # 处理RGB三个通道
            jpg_array[:, :, c] = jpg_array[:, :, c] * (1 - alpha_mask) + png_array[:, :, c] * alpha_mask

        # 转换回PIL图片，移除Alpha通道
        overlaid_image = Image.fromarray(jpg_array.astype(np.uint8), 'RGBA')
        overlaid_image = overlaid_image.convert('RGB')  # 将图片转换为RGB以保存为JPEG

        output_path = os.path.join(output_dir, jpg_file)
        overlaid_image.save(output_path, format='JPEG')
        print(f"Saved overlaid image to {output_path}")



# 示例调用
jpg_dir = 'G:/Results/images'  # JPG图片路径

png_dir = 'G:/Results/labels'  # PNG图片路径
output_dir = 'G://Results/Combine_label'  # 保存叠加后的图片
alpha = 1  # 透明度设置

overlay_images(jpg_dir, png_dir, output_dir, alpha)

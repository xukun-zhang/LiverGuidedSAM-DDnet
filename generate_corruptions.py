#!/usr/bin/env python3
"""
batch_augmentation_suite.py
--------------------------------------------
对文件夹内所有 .jpg 执行：
  • motion blur (3 强度)
  • JPEG 质量压缩 (60, 40)
  • 高斯噪声、泊松噪声
  • 雾化 / haze
结果以 <out_dir>/<tag>/<原文件名>.jpg 保存
--------------------------------------------
$ python batch_augmentation_suite.py --in_dir ./images \
                                     --out_dir ./aug
"""

import cv2
import numpy as np
import argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

# -------------------------------------------------------------
# 单幅图像增强函数
# -------------------------------------------------------------
def motion_blur(img, ksize=15, angle=0):
    kernel = np.zeros((ksize, ksize), np.float32)
    kernel[ksize // 2, :] = 1.0 / ksize
    M = cv2.getRotationMatrix2D((ksize / 2, ksize / 2), angle, 1)
    kernel = cv2.warpAffine(kernel, M, (ksize, ksize))
    return cv2.filter2D(img, -1, kernel)

def jpeg_compress(img, quality):
    _, enc = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return cv2.imdecode(enc, cv2.IMREAD_COLOR)

def add_gaussian(img, sigma=10):
    noise = np.random.normal(0, sigma, img.shape).astype(np.float32)
    return np.clip(img.astype(np.float32) + noise, 0, 255).astype(np.uint8)

def add_poisson(img):
    vals = 2 ** np.ceil(np.log2(len(np.unique(img))))
    noisy = np.random.poisson(img * vals) / vals
    return np.clip(noisy, 0, 255).astype(np.uint8)

def add_haze(img, fog_factor=0.65, kernel_size=15):
    haze = cv2.addWeighted(img, 1 - fog_factor,
                           np.full(img.shape, 255, np.uint8),
                           fog_factor, 0)
    return cv2.GaussianBlur(haze, (kernel_size, kernel_size), 0)

# -------------------------------------------------------------
# 处理单张并保存
# -------------------------------------------------------------
def process_one(jpg_path: Path, out_root: Path):
    img = cv2.imread(str(jpg_path))
    if img is None:
        print(" [!] skip unreadable:", jpg_path.name)
        return

    fname = jpg_path.name
    variants = {
        "motion7":  motion_blur(img, 7, 0),
        "motion15": motion_blur(img, 15, 0),
        "motion31": motion_blur(img, 31, 0),
        "jpeg60":   jpeg_compress(img, 60),
        "jpeg40":   jpeg_compress(img, 40),
        "gauss":    add_gaussian(img, sigma=15),
        "poisson":  add_poisson(img),
        "haze":     add_haze(img, fog_factor=0.6, kernel_size=21),
    }

    for tag, im in variants.items():
        out_dir = out_root / tag
        out_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_dir / fname), im)
    print(" [+] done", fname)

# -------------------------------------------------------------
# main
# -------------------------------------------------------------
def main(in_dir: str, out_dir: str, workers: int = 4):
    in_root = Path(in_dir)
    out_root = Path(out_dir)
    jpg_files = sorted(in_root.rglob("*.jpg"))
    if not jpg_files:
        raise FileNotFoundError(f"No .jpg found in {in_dir}")

    print(f"Processing {len(jpg_files)} images …")
    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(lambda p: process_one(p, out_root), jpg_files))
    print("All done!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_dir",  default="./images",
                        help="输入 jpg 文件夹 (默认 ./images)")
    parser.add_argument("--out_dir", default="./aug_p2ilf",
                        help="输出文件夹 (默认 ./aug)")
    parser.add_argument("--workers", type=int, default=4,
                        help="并行线程数")
    args = parser.parse_args()
    main(args.in_dir, args.out_dir, args.workers)

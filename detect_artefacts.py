import os
import cv2
import numpy as np
import scipy.fftpack as fftpack
import shutil

def calculate_dct(image, block_size = 8):
    h, w = image.shape
    dct_blocks = []
    for i in range(0, h, block_size):
        for j in range(0, w, block_size):
            block = image[i:i+block_size, j:j+block_size]
            if block.shape == (block_size, block_size):
                dct_block = fftpack.dct(fftpack.dct(block.T, norm='ortho').T, norm='ortho')
                dct_blocks.append(dct_block)
    return dct_blocks

def count_small_coefficients(dct_blocks, threshold = 0.1):
    count = 0
    total = 0
    for block in dct_blocks:
        total += block.size
        count += np.sum(np.abs(block) < threshold)
    return count / total

def is_low_quality(image_path):
    image = cv2.imread(image_path, 0)
    dct_blocks = calculate_dct(image)
    small_coefficients_ratio = count_small_coefficients(dct_blocks)
    return small_coefficients_ratio > 0.4

def move_low_quality_images(directory_path):
    if not os.path.isdir(directory_path):
        print(f"The provided directory path does not exist: {directory_path}")
        return

    low_quality_dir = os.path.join(directory_path, "low_quality")
    os.makedirs(low_quality_dir, exist_ok=True)

    for filename in os.listdir(directory_path):
        if filename.endswith(".jpg") or filename.endswith(".png") or filename.endswith(".webp"):
            image_path = os.path.join(directory_path, filename)
            if is_low_quality(image_path):
                shutil.move(image_path, low_quality_dir)
                print(f"Moved low quality image: {filename}")

if __name__ == "__main__":
    directory_path = input("Enter the directory path: ")
    move_low_quality_images(directory_path)

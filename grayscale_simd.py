import cv2
import numpy as np
from numba import vectorize, float32
import time

@vectorize([float32(float32, float32, float32)], target='parallel')
def rgb_to_grayscale(r, g, b):
    return 0.299 * r + 0.587 * g + 0.114 * b

def convert_image_to_grayscale(img_path, save_path):
    img = cv2.imread(img_path)
    img = img.astype(np.float32)


    b, g, r = cv2.split(img)

    start = time.time()
    gray = rgb_to_grayscale(r, g, b)
    duration = time.time() - start

    gray_img = gray.astype(np.uint8)
    cv2.imwrite(save_path, gray_img)
    print(f"Grayscale SIMD selesai dalam {duration:.4f} detik")

if __name__ == "__main__":
    convert_image_to_grayscale("input/archive/Garbage_Collective_Data/biological/biological1.jpg", "output/sample_gray.jpg")

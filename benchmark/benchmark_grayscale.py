import cv2
import numpy as np
import time
from numba import vectorize, float32

@vectorize([float32(float32, float32, float32)], target='parallel')
def grayscale_simd(r, g, b):
    return 0.299 * r + 0.587 * g + 0.114 * b

# NON-SIMD 
def grayscale_nonsimd(img):
    r = img[:, :, 2]
    g = img[:, :, 1]
    b = img[:, :, 0]
    return (0.299 * r + 0.587 * g + 0.114 * b).astype(np.uint8)

def run_benchmark(img_path):
    img = cv2.imread(img_path).astype(np.float32)

    # SIMD
    r, g, b = cv2.split(img)
    start = time.time()
    gray_simd = grayscale_simd(r, g, b)
    simd_time = time.time() - start

    # NON-SIMD
    start = time.time()
    gray_nonsimd = grayscale_nonsimd(img)
    nonsimd_time = time.time() - start

    print("Grayscale Benchmark:")
    print(f"  SIMD     : {simd_time:.6f} detik")
    print(f"  Non-SIMD : {nonsimd_time:.6f} detik")

if __name__ == "__main__":
    run_benchmark("input/archive/Garbage_Collective_Data/biological/biological1.jpg")

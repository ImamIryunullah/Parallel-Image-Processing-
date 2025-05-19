import cv2
import numpy as np
import time
from numba import njit, prange

@njit(parallel=True)
def sobel_simd(gray):
    h, w = gray.shape
    out = np.zeros_like(gray)
    Kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    Ky = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    for i in prange(1, h-1):
        for j in range(1, w-1):
            gx = 0.0
            gy = 0.0
            for m in range(3):
                for n in range(3):
                    val = gray[i+m-1, j+n-1]
                    gx += Kx[m,n] * val
                    gy += Ky[m,n] * val
            out[i,j] = min(255, int(np.sqrt(gx**2 + gy**2)))
    return out

# NON-SIMD
def sobel_nonsimd(gray):
    Kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    Ky = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    gx = cv2.filter2D(gray, -1, Kx)
    gy = cv2.filter2D(gray, -1, Ky)
    return cv2.magnitude(gx.astype(np.float32), gy.astype(np.float32)).astype(np.uint8)

def run_benchmark(img_path):
    gray = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)

    # SIMD
    start = time.time()
    result_simd = sobel_simd(gray)
    simd_time = time.time() - start

    # NON-SIMD
    start = time.time()
    result_nonsimd = sobel_nonsimd(gray)
    nonsimd_time = time.time() - start

    print("Edge Detection Benchmark:")
    print(f"  SIMD     : {simd_time:.6f} detik")
    print(f"  Non-SIMD : {nonsimd_time:.6f} detik")

if __name__ == "__main__":
    run_benchmark("input/archive/Garbage_Collective_Data/biological/biological1.jpg")

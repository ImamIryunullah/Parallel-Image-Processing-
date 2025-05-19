import cv2
import numpy as np
from numba import njit, prange
import time
import os

@njit(parallel=True)
def sobel_edge_detection(gray_img):
    h, w = gray_img.shape
    output = np.zeros((h, w), dtype=np.uint8)

    # Kernel Sobel X dan Y
    Kx = np.array([[-1, 0, 1],
                   [-2, 0, 2],
                   [-1, 0, 1]])

    Ky = np.array([[-1, -2, -1],
                   [ 0,  0,  0],
                   [ 1,  2,  1]])

    for i in prange(1, h - 1):
        for j in range(1, w - 1):
            gx = 0.0
            gy = 0.0
            for m in range(3):
                for n in range(3):
                    px = gray_img[i + m - 1, j + n - 1]
                    gx += Kx[m, n] * px
                    gy += Ky[m, n] * px
            magnitude = np.sqrt(gx**2 + gy**2)
            output[i, j] = min(255, int(magnitude))

    return output

def process_edge_detection(input_path, output_path):
    img = cv2.imread(input_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"[ERROR] Gagal membuka gambar: {input_path}")
        return

    start = time.time()
    edge_img = sobel_edge_detection(img)
    duration = time.time() - start

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cv2.imwrite(output_path, edge_img)
    print(f"Deteksi tepi selesai dalam {duration:.4f} detik")

if __name__ == "__main__":
    process_edge_detection("input/archive/Garbage_Collective_Data/biological/biological1.jpg", "output/sample_edge.jpg")

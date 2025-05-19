import os
import time
import cv2
import numpy as np
from tkinter import Tk, filedialog, Button, Label, Text, END, Canvas, NW, Frame, BOTH
from PIL import Image, ImageTk
from matplotlib import pyplot as plt
from numba import njit, prange, vectorize, float32

# === GRAYSCALE ===
@vectorize([float32(float32, float32, float32)], target='parallel')
def grayscale_simd(r, g, b):
    return 0.299 * r + 0.587 * g + 0.114 * b

def grayscale_nonsimd(img):
    r = img[:, :, 2]
    g = img[:, :, 1]
    b = img[:, :, 0]
    return (0.299 * r + 0.587 * g + 0.114 * b).astype(np.uint8)

# === EDGE DETECTION ===
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

def sobel_nonsimd(gray):
    Kx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    Ky = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    gx = cv2.filter2D(gray, -1, Kx)
    gy = cv2.filter2D(gray, -1, Ky)
    return cv2.magnitude(gx.astype(np.float32), gy.astype(np.float32)).astype(np.uint8)

def process_images(filepaths, log_output, image_canvas, root):
    grayscale_times = []
    edge_times = []
    last_gray_img = None
    last_edge_img = None

    for img_path in filepaths:
        filename = os.path.basename(img_path)
        base = os.path.splitext(filename)[0]

        img = cv2.imread(img_path)
        if img is None:
            log_output.insert(END, f"[❌] Gagal membaca: {img_path}\n")
            continue

        r, g, b = cv2.split(img.astype(np.float32))
        start = time.time()
        gray_simd = grayscale_simd(r, g, b).astype(np.uint8)
        t_simd = time.time() - start
        os.makedirs("output/grayscale_simd", exist_ok=True)
        cv2.imwrite(f"output/grayscale_simd/{base}.jpg", gray_simd)

        start = time.time()
        gray_nonsimd = grayscale_nonsimd(img)
        t_nonsimd = time.time() - start
        os.makedirs("output/grayscale_nonsimd", exist_ok=True)
        cv2.imwrite(f"output/grayscale_nonsimd/{base}.jpg", gray_nonsimd)

        grayscale_times.append((filename, t_simd, t_nonsimd))

        start = time.time()
        edge_simd = sobel_simd(gray_simd)
        t_simd_edge = time.time() - start
        os.makedirs("output/edge_simd", exist_ok=True)
        cv2.imwrite(f"output/edge_simd/{base}.jpg", edge_simd)

        start = time.time()
        edge_nonsimd = sobel_nonsimd(gray_nonsimd)
        t_nonsimd_edge = time.time() - start
        os.makedirs("output/edge_nonsimd", exist_ok=True)
        cv2.imwrite(f"output/edge_nonsimd/{base}.jpg", edge_nonsimd)

        edge_times.append((filename, t_simd_edge, t_nonsimd_edge))

        log_output.insert(END, f"\n📂 {filename}\n")
        log_output.insert(END, f"  ✅ Grayscale SIMD: {t_simd:.5f}s | Non-SIMD: {t_nonsimd:.5f}s\n")
        log_output.insert(END, f"  ✅ Edge SIMD:     {t_simd_edge:.5f}s | Non-SIMD: {t_nonsimd_edge:.5f}s\n")
        log_output.see(END)

        last_gray_img = f"output/grayscale_simd/{base}.jpg"
        last_edge_img = f"output/edge_simd/{base}.jpg"

    if last_gray_img and last_edge_img:
        show_image_preview(image_canvas, last_gray_img, last_edge_img)

    show_time_chart(grayscale_times, edge_times, root)

def show_image_preview(canvas, gray_path, edge_path):
    canvas.delete("all")
    gray_img = Image.open(gray_path).resize((256, 256))
    edge_img = Image.open(edge_path).resize((256, 256))

    gray_tk = ImageTk.PhotoImage(gray_img)
    edge_tk = ImageTk.PhotoImage(edge_img)

    canvas.image1 = gray_tk
    canvas.image2 = edge_tk

    canvas.create_image(0, 0, anchor=NW, image=gray_tk)
    canvas.create_image(260, 0, anchor=NW, image=edge_tk)

def show_time_chart(grayscale_times, edge_times, root):
    fig, ax = plt.subplots(2, 1, figsize=(10, 6))

    files = [name for name, _, _ in grayscale_times]

    gray_simd = [t for _, t, _ in grayscale_times]
    gray_nonsimd = [t for _, _, t in grayscale_times]
    ax[0].bar(files, gray_simd, label="SIMD", alpha=0.7)
    ax[0].bar(files, gray_nonsimd, label="Non-SIMD", alpha=0.7, bottom=gray_simd)
    ax[0].set_title("Grayscale Time Comparison")
    ax[0].set_ylabel("Waktu (s)")
    ax[0].legend()

    edge_simd = [t for _, t, _ in edge_times]
    edge_nonsimd = [t for _, _, t in edge_times]
    ax[1].bar(files, edge_simd, label="SIMD", alpha=0.7)
    ax[1].bar(files, edge_nonsimd, label="Non-SIMD", alpha=0.7, bottom=edge_simd)
    ax[1].set_title("Edge Detection Time Comparison")
    ax[1].set_ylabel("Waktu (s)")
    ax[1].legend()

    plt.tight_layout()
    plt.show()

def launch_gui():
    root = Tk()
    root.title("Parallel Image Processing (SIMD)")

    Label(root, text="Parallel Image Processor", font=("Helvetica", 16, "bold")).pack(pady=10)

    frame = Frame(root)
    frame.pack()

    log_output = Text(frame, height=20, width=60)
    log_output.pack(side="left", padx=5, pady=5)

    image_canvas = Canvas(frame, width=520, height=260)
    image_canvas.pack(side="right", padx=5, pady=5)

    def select_and_process():
        filepaths = filedialog.askopenfilenames(
            title="Pilih gambar",
            filetypes=[("Image Files", "*.jpg *.jpeg *.png")]
        )
        if filepaths:
            log_output.insert(END, f"\n=== Memproses {len(filepaths)} file ===\n")
            process_images(filepaths, log_output, image_canvas, root)

    Button(root, text="Upload & Proses Gambar", command=select_and_process,
           bg="#4CAF50", fg="white", padx=10, pady=5).pack(pady=10)

    root.mainloop()

if __name__ == "__main__":
    launch_gui()

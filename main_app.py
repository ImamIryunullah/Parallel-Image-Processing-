import os
import cv2
import time
import numpy as np
import argparse
from numba import njit, prange, vectorize, float32
from tabulate import tabulate
import tkinter as tk
from tkinter import filedialog
from matplotlib import pyplot as plt
from PIL import Image, ImageTk
from tqdm import tqdm
import tkinter as tk
from tkinter import ttk, messagebox
import threading
import csv


# === SIMD / NON-SIMD FUNCTIONS ===

@vectorize([float32(float32, float32, float32)], target='parallel')
def grayscale_simd(r, g, b):
    return 0.299 * r + 0.587 * g + 0.114 * b

def grayscale_nonsimd(img):
    r = img[:, :, 2]
    g = img[:, :, 1]
    b = img[:, :, 0]
    return (0.299 * r + 0.587 * g + 0.114 * b).astype(np.uint8)

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

def process_batch(input_folder="input"):
    grayscale_results = []
    edge_results = []

    # Kumpulkan semua file yang valid
    all_files = []
    for root, dirs, files in os.walk(input_folder):
        for filename in files:
            if filename.lower().endswith(('.jpg', '.png', '.jpeg')):
                all_files.append(os.path.join(root, filename))

    print(f"Memproses {len(all_files)} file gambar di '{input_folder}'...")

    for img_path in tqdm(all_files, desc="Proses gambar"):
        filename = os.path.basename(img_path)
        img = cv2.imread(img_path)
        if img is None:
            print(f"Warning: gagal membaca {img_path}, dilewati.")
            continue

        base = os.path.splitext(filename)[0]
        r, g, b = cv2.split(img.astype(np.float32))

        # Grayscale SIMD
        start = time.time()
        gray_simd = grayscale_simd(r, g, b).astype(np.uint8)
        t_simd = time.time() - start
        os.makedirs("output/grayscale_simd", exist_ok=True)
        cv2.imwrite(f"output/grayscale_simd/{base}.jpg", gray_simd)

        # Grayscale Non-SIMD
        start = time.time()
        gray_nonsimd = grayscale_nonsimd(img)
        t_nonsimd = time.time() - start
        os.makedirs("output/grayscale_nonsimd", exist_ok=True)
        cv2.imwrite(f"output/grayscale_nonsimd/{base}.jpg", gray_nonsimd)

        grayscale_results.append([filename, f"{t_simd:.5f}", f"{t_nonsimd:.5f}"])

        # Sobel SIMD
        start = time.time()
        edge_simd = sobel_simd(gray_simd)
        t_simd_edge = time.time() - start
        os.makedirs("output/edge_simd", exist_ok=True)
        cv2.imwrite(f"output/edge_simd/{base}.jpg", edge_simd)

        # Sobel Non-SIMD
        start = time.time()
        edge_nonsimd = sobel_nonsimd(gray_nonsimd)
        t_nonsimd_edge = time.time() - start
        os.makedirs("output/edge_nonsimd", exist_ok=True)
        cv2.imwrite(f"output/edge_nonsimd/{base}.jpg", edge_nonsimd)

        edge_results.append([filename, f"{t_simd_edge:.5f}", f"{t_nonsimd_edge:.5f}"])

        # Simpan juga versi output umum (optional)
        os.makedirs("output", exist_ok=True)
        cv2.imwrite(f"output/{base}_gray_simd.jpg", gray_simd)
        cv2.imwrite(f"output/{base}_gray_nonsimd.jpg", gray_nonsimd)
        cv2.imwrite(f"output/{base}_edge_simd.jpg", edge_simd)
        cv2.imwrite(f"output/{base}_edge_nonsimd.jpg", edge_nonsimd)

    # Tampilkan hasil di CLI
    print("\n Grayscale Benchmark (detik):")
    print(tabulate(grayscale_results, headers=["File", "SIMD", "Non-SIMD"], tablefmt="github"))

    print("\n Edge Detection Benchmark (detik):")
    print(tabulate(edge_results, headers=["File", "SIMD", "Non-SIMD"], tablefmt="github"))

    return grayscale_results, edge_results

def show_cli_result_gui(grayscale_results, edge_results):
    result_window = tk.Toplevel()
    result_window.title("Hasil Benchmark CLI")
    result_window.geometry("800x600")

    tab_control = ttk.Notebook(result_window)

    # Tab Grayscale
    tab_gray = ttk.Frame(tab_control)
    tab_control.add(tab_gray, text="Grayscale")

    tree_gray = ttk.Treeview(tab_gray, columns=("file", "simd", "nonsimd"), show="headings")
    tree_gray.heading("file", text="File")
    tree_gray.heading("simd", text="SIMD (s)")
    tree_gray.heading("nonsimd", text="Non-SIMD (s)")

    for row in grayscale_results:
        tree_gray.insert("", "end", values=row)

    tree_gray.pack(fill="both", expand=True)

    # Tab Edge Detection
    tab_edge = ttk.Frame(tab_control)
    tab_control.add(tab_edge, text="Edge Detection")

    tree_edge = ttk.Treeview(tab_edge, columns=("file", "simd", "nonsimd"), show="headings")
    tree_edge.heading("file", text="File")
    tree_edge.heading("simd", text="SIMD (s)")
    tree_edge.heading("nonsimd", text="Non-SIMD (s)")

    for row in edge_results:
        tree_edge.insert("", "end", values=row)

    tree_edge.pack(fill="both", expand=True)

    tab_control.pack(expand=True, fill="both")
    
    def export_csv():
        file_path = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV Files", "*.csv")])
        if not file_path:
            return

        with open(file_path, "w", newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["Grayscale Benchmark"])
            writer.writerow(["File", "SIMD", "Non-SIMD"])
            writer.writerows(grayscale_results)

            writer.writerow([])
            writer.writerow(["Edge Detection Benchmark"])
            writer.writerow(["File", "SIMD", "Non-SIMD"])
            writer.writerows(edge_results)

        messagebox.showinfo("Sukses", f"Hasil diekspor ke {file_path}")

    export_button = tk.Button(result_window, text="Ekspor ke CSV", command=export_csv)
    export_button.pack(pady=10)

    
def run_cli_batch():
    popup = tk.Toplevel()
    popup.title("Proses Berjalan")
    popup.geometry("300x100")
    label = tk.Label(popup, text="Sedang memproses gambar...\nMohon tunggu...")
    label.pack(expand=True)

    def run_batch():
        try:
            grayscale_results, edge_results = process_batch("input")
            popup.destroy()
            show_cli_result_gui(grayscale_results, edge_results)
        except Exception as e:
            popup.destroy()
            messagebox.showerror("Error", f"Gagal menjalankan batch CLI:\n{str(e)}")

    threading.Thread(target=run_batch).start()

def run_gui():
    def process_image(path):
        img = cv2.imread(path)
        r, g, b = cv2.split(img.astype(np.float32))

        t0 = time.time()
        gray_simd = grayscale_simd(r, g, b).astype(np.uint8)
        t1 = time.time()
        gray_nonsimd = grayscale_nonsimd(img)
        t2 = time.time()

        edge_simd = sobel_simd(gray_simd)
        t3 = time.time()
        edge_nonsimd = sobel_nonsimd(gray_nonsimd)
        t4 = time.time()

        return {
            "gray_simd": gray_simd,
            "gray_nonsimd": gray_nonsimd,
            "edge_simd": edge_simd,
            "edge_nonsimd": edge_nonsimd,
            "times": {
                "gray_simd": t1 - t0,
                "gray_nonsimd": t2 - t1,
                "edge_simd": t3 - t2,
                "edge_nonsimd": t4 - t3
            }
        }

    def show_result(result):
        def show_image(title, img_array):
            img_rgb = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
            img_pil = Image.fromarray(img_rgb)
            img_tk = ImageTk.PhotoImage(img_pil.resize((256, 256)))
            label = tk.Label(img_frame, text=title, image=img_tk, compound=tk.TOP)
            label.image = img_tk
            label.pack(side=tk.LEFT, padx=10)

        # Clear old
        for widget in img_frame.winfo_children():
            widget.destroy()

        show_image("Grayscale SIMD", result["gray_simd"])
        show_image("Grayscale Non-SIMD", result["gray_nonsimd"])
        show_image("Edge SIMD", result["edge_simd"])
        show_image("Edge Non-SIMD", result["edge_nonsimd"])

        # Show chart
        times = result["times"]
        keys = list(times.keys())
        values = [times[k] for k in keys]

        plt.figure(figsize=(6, 4))
        plt.bar(keys, values, color=['green', 'blue', 'red', 'orange'])
        plt.title("Execution Time Comparison")
        plt.ylabel("Seconds")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

    def upload_image():
        path = filedialog.askopenfilename(filetypes=[("Images", "*.jpg *.png *.jpeg")])
        if path:
            result = process_image(path)
            show_result(result)

    # === GUI Init ===
    window = tk.Tk()
    window.title("GraviPix (Grayscale & Vision Pixel)- SIMD vs Non-SIMD")
    window.geometry("1100x400")

    btn = tk.Button(window, text="Upload Image", command=upload_image, font=("Arial", 14), bg="skyblue")
    btn.pack(pady=10)

    global img_frame
    img_frame = tk.Frame(window)
    img_frame.pack()

    window.mainloop()

# === ENTRY POINT ===

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["cli", "gui"], required=True)
    parser.add_argument("--folder", default="input")
    args = parser.parse_args()

    if args.mode == "cli":
        process_batch(args.folder)
    else:
        # jalankan GUI (tidak perlu argumen folder)
        run_gui()
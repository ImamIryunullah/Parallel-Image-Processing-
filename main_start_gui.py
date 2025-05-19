import tkinter as tk
from tkinter import messagebox, filedialog, ttk
import subprocess
import sys
import threading

def run_cli_mode(folder, status_label, btn_cli, btn_gui, progress):
    def task():
        btn_cli.config(state="disabled")
        btn_gui.config(state="disabled")
        status_label.config(text="⏳ Menjalankan proses CLI...")
        progress.start(10)

        cmd = [sys.executable, "main_app.py", "--mode", "cli", "--folder", folder]
        try:
            subprocess.run(cmd, check=True)
            messagebox.showinfo("Selesai", "✅ Proses CLI selesai.")
        except subprocess.CalledProcessError as e:
            messagebox.showerror("Error", f"Proses CLI gagal:\n{e}")

        progress.stop()
        status_label.config(text="")
        btn_cli.config(state="normal")
        btn_gui.config(state="normal")

    threading.Thread(target=task, daemon=True).start()

def run_gui_mode(status_label, btn_cli, btn_gui, progress):
    def task():
        btn_cli.config(state="disabled")
        btn_gui.config(state="disabled")
        status_label.config(text="🔄 Menjalankan GUI...")
        progress.start(10)

        cmd = [sys.executable, "main_app.py", "--mode", "gui"]
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            messagebox.showerror("Error", f"Proses GUI gagal:\n{e}")

        progress.stop()
        status_label.config(text="")
        btn_cli.config(state="normal")
        btn_gui.config(state="normal")

    threading.Thread(target=task, daemon=True).start()

def select_cli_folder(status_label, btn_cli, btn_gui, progress):
    folder_selected = filedialog.askdirectory(title="Pilih Folder Input Gambar")
    if folder_selected:
        run_cli_mode(folder_selected, status_label, btn_cli, btn_gui, progress)

def main():
    root = tk.Tk()
    root.title("GraviPix (Grayscale & Vision Pixel)")

    w, h = 420, 260
    ws = root.winfo_screenwidth()
    hs = root.winfo_screenheight()
    x = (ws // 2) - (w // 2)
    y = (hs // 2) - (h // 2)
    root.geometry(f"{w}x{h}+{x}+{y}")
    root.resizable(False, False)

    frame = tk.Frame(root, padx=20, pady=20)
    frame.pack(expand=True, fill="both")

    label = tk.Label(frame, text="Pilih Mode Aplikasi:", font=("Arial", 16))
    label.pack(pady=(0, 20))

    btn_cli = tk.Button(frame, text="CLI Mode (Batch Processing)", width=30)
    btn_cli.pack(pady=8)

    btn_gui = tk.Button(frame, text="GUI Mode (Interactive)", width=30)
    btn_gui.pack(pady=8)

    status_label = tk.Label(frame, text="", fg="green", font=("Arial", 10))
    status_label.pack(pady=(10, 5))

    progress = ttk.Progressbar(frame, mode="indeterminate", length=300)
    progress.pack()

    btn_cli.config(command=lambda: select_cli_folder(status_label, btn_cli, btn_gui, progress))
    btn_gui.config(command=lambda: run_gui_mode(status_label, btn_cli, btn_gui, progress))

    def on_enter(e): e.widget.config(bg="#cce6ff")
    def on_leave(e): e.widget.config(bg="SystemButtonFace")
    for btn in (btn_cli, btn_gui):
        btn.bind("<Enter>", on_enter)
        btn.bind("<Leave>", on_leave)

    root.mainloop()

if __name__ == "__main__":
    main()

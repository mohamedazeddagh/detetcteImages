import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk, ImageDraw
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image as keras_image
import cv2
import threading
import time
import os
import shutil

# ── Load model ───────────────────────────────────────────────────────────────
model = tf.keras.models.load_model('CNN_ImageProcessing_Manab.h5')

CLASSES = ['Aeroplane', 'Automobile', 'Bird', 'Cat', 'Deer',
           'Dog', 'Frog', 'Horse', 'Ship', 'Truck']

CLASS_ICONS = {
    'Aeroplane': '✈', 'Automobile': '🚗', 'Bird': '🐦', 'Cat': '🐱',
    'Deer': '🦌', 'Dog': '🐶', 'Frog': '🐸', 'Horse': '🐴',
    'Ship': '🚢', 'Truck': '🚛'
}

# ── Color Palette ─────────────────────────────────────────────────────────────
BG_DEEP       = "#080b14"
BG_PANEL      = "#0d1120"
BG_CARD       = "#111827"
BORDER        = "#1e2d40"
ACCENT_CYAN   = "#00e5ff"
ACCENT_PINK   = "#ff2d78"
ACCENT_GREEN  = "#00ff9f"
ACCENT_YELLOW = "#ffd700"
ACCENT_PURPLE = "#b060ff"
TEXT_PRIMARY  = "#e8f0fe"
TEXT_DIM      = "#4a5568"
TEXT_MID      = "#8892a4"

# ── Fonts ─────────────────────────────────────────────────────────────────────
FONT_TITLE   = ("Courier New", 26, "bold")
FONT_MONO_SM = ("Courier New", 9)
FONT_RESULT  = ("Courier New", 20, "bold")
FONT_BTN     = ("Courier New", 10, "bold")
FONT_SMALL   = ("Courier New", 9)

# ── Category storage folder ───────────────────────────────────────────────────
SAVE_DIR = "category_data"


def ensure_dirs():
    for cls in CLASSES:
        os.makedirs(os.path.join(SAVE_DIR, cls), exist_ok=True)


ensure_dirs()


# ─────────────────────────────────────────────────────────────────────────────
class ScanlineCanvas(tk.Canvas):
    def __init__(self, master, **kw):
        super().__init__(master, **kw)
        self._offset = 0
        self._animate()

    def _animate(self):
        w, h = self.winfo_width() or 400, self.winfo_height() or 300
        self.delete("scanline")
        y = self._offset % h
        self.create_line(0, y, w, y, fill="#0a2a30", width=2, tags="scanline")
        self._offset += 2
        self.after(40, self._animate)


class ConfidenceBar(tk.Canvas):
    def __init__(self, master, **kw):
        super().__init__(master, bg=BG_CARD, highlightthickness=0, height=6, **kw)
        self._value = 0

    def set_value(self, pct):
        self._value = pct
        self.after(10, self._draw)

    def _draw(self):
        self.update_idletasks()
        w = self.winfo_width()
        self.delete("all")
        self.create_rectangle(0, 0, w, 6, fill=BORDER, outline="")
        fill_w = int(w * self._value / 100)
        if fill_w > 0:
            color = (ACCENT_GREEN if self._value > 75
                     else ACCENT_CYAN if self._value > 45
                     else ACCENT_PINK)
            self.create_rectangle(0, 0, fill_w, 6, fill=color, outline="")
        for t in [25, 50, 75]:
            tx = int(w * t / 100)
            self.create_line(tx, 0, tx, 6, fill="#2a3040", width=1)


# ─────────────────────────────────────────────────────────────────────────────
class GalleryWindow:
    """Pop-up gallery showing all saved category images."""

    THUMB = 90

    def __init__(self, parent):
        self.win = tk.Toplevel(parent)
        self.win.title("CATEGORY GALLERY")
        self.win.configure(bg=BG_DEEP)
        self.win.geometry("960x680")
        self.win.resizable(True, True)
        self._thumb_refs = []
        self._build()

    def _build(self):
        win = self.win

        # Header
        hdr = tk.Frame(win, bg=BG_DEEP)
        hdr.pack(fill="x", padx=20, pady=(16, 0))
        tk.Label(hdr, text="◈ CATEGORY GALLERY",
                 font=("Courier New", 18, "bold"),
                 bg=BG_DEEP, fg=ACCENT_CYAN).pack(side="left")

        total = sum(
            len([f for f in os.listdir(os.path.join(SAVE_DIR, c))
                 if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))])
            for c in CLASSES
            if os.path.isdir(os.path.join(SAVE_DIR, c))
        )
        tk.Label(hdr, text=f"  {total} image(s) saved",
                 font=FONT_MONO_SM, bg=BG_DEEP, fg=TEXT_MID).pack(side="left", padx=12)

        tk.Button(hdr, text="✕  CLOSE", command=self.win.destroy,
                  font=FONT_BTN, bg=BG_CARD, fg=ACCENT_PINK,
                  activebackground=BORDER, activeforeground=ACCENT_PINK,
                  relief="flat", cursor="hand2", bd=0,
                  highlightthickness=1, highlightbackground=ACCENT_PINK
                  ).pack(side="right")

        sep = tk.Canvas(win, bg=BG_DEEP, highlightthickness=0, height=1)
        sep.pack(fill="x", padx=20, pady=8)
        sep.create_line(0, 0, 960, 0, fill=BORDER)

        # Scrollable canvas
        container = tk.Frame(win, bg=BG_DEEP)
        container.pack(fill="both", expand=True, padx=10, pady=4)

        canvas    = tk.Canvas(container, bg=BG_DEEP, highlightthickness=0)
        scrollbar = tk.Scrollbar(container, orient="vertical",
                                 command=canvas.yview, bg=BG_PANEL)
        canvas.configure(yscrollcommand=scrollbar.set)
        scrollbar.pack(side="right", fill="y")
        canvas.pack(side="left", fill="both", expand=True)

        inner    = tk.Frame(canvas, bg=BG_DEEP)
        inner_id = canvas.create_window((0, 0), window=inner, anchor="nw")

        def _on_configure(e):
            canvas.configure(scrollregion=canvas.bbox("all"))
        inner.bind("<Configure>", _on_configure)
        canvas.bind("<Configure>",
                    lambda e: canvas.itemconfig(inner_id, width=canvas.winfo_width()))

        def _on_wheel(e):
            canvas.yview_scroll(int(-1 * (e.delta / 120)), "units")
        canvas.bind_all("<MouseWheel>", _on_wheel)

        # Populate categories
        for cls in CLASSES:
            folder = os.path.join(SAVE_DIR, cls)
            files  = sorted([
                f for f in os.listdir(folder)
                if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))
            ]) if os.path.isdir(folder) else []

            cat_row = tk.Frame(inner, bg=BG_DEEP)
            cat_row.pack(fill="x", padx=8, pady=(14, 4))
            icon        = CLASS_ICONS.get(cls, "?")
            count_color = ACCENT_CYAN if files else TEXT_DIM
            tk.Label(cat_row,
                     text=f"{icon}  {cls.upper()}",
                     font=("Courier New", 13, "bold"),
                     bg=BG_DEEP, fg=count_color).pack(side="left")
            tk.Label(cat_row,
                     text=f"  [{len(files)} image(s)]",
                     font=FONT_MONO_SM,
                     bg=BG_DEEP, fg=TEXT_DIM).pack(side="left")

            div = tk.Canvas(inner, bg=BG_DEEP, highlightthickness=0, height=1)
            div.pack(fill="x", padx=8, pady=(0, 6))
            div.bind("<Configure>",
                     lambda e, d=div: d.create_line(0, 0, e.width, 0, fill=BORDER))

            if not files:
                tk.Label(inner, text="  — no images saved yet —",
                         font=FONT_MONO_SM, bg=BG_DEEP, fg=TEXT_DIM
                         ).pack(anchor="w", padx=20, pady=4)
                continue

            # Thumbnail grid — wrap every 9
            grid_frame = tk.Frame(inner, bg=BG_DEEP)
            grid_frame.pack(fill="x", padx=16, pady=(0, 4))
            for idx, fname in enumerate(files):
                fpath = os.path.join(folder, fname)
                self._add_thumb(grid_frame, fpath, fname, idx // 9, idx % 9)

    def _add_thumb(self, parent, fpath, fname, row, col):
        T    = self.THUMB
        cell = tk.Frame(parent, bg=BG_CARD,
                        highlightthickness=1, highlightbackground=BORDER)
        cell.grid(row=row, column=col, padx=4, pady=4, sticky="n")

        try:
            img = Image.open(fpath).convert("RGB")
            img.thumbnail((T, T), Image.LANCZOS)
            square = Image.new("RGB", (T, T), (17, 24, 39))
            ox = (T - img.width) // 2
            oy = (T - img.height) // 2
            square.paste(img, (ox, oy))
            photo = ImageTk.PhotoImage(square)
            self._thumb_refs.append(photo)

            lbl = tk.Label(cell, image=photo, bg=BG_CARD, cursor="hand2")
            lbl.pack()
            lbl.bind("<Button-1>", lambda e, p=fpath: self._enlarge(p))
        except Exception:
            tk.Label(cell, text="ERR", width=8, height=4,
                     bg=BG_CARD, fg=ACCENT_PINK, font=FONT_MONO_SM).pack()

        short = (fname[:10] + "…") if len(fname) > 12 else fname
        tk.Label(cell, text=short, font=("Courier New", 7),
                 bg=BG_CARD, fg=TEXT_DIM).pack(pady=(0, 3))

    def _enlarge(self, fpath):
        top = tk.Toplevel(self.win)
        top.title(os.path.basename(fpath))
        top.configure(bg=BG_DEEP)
        top.resizable(False, False)

        img   = Image.open(fpath).convert("RGB")
        img.thumbnail((520, 520), Image.LANCZOS)
        photo = ImageTk.PhotoImage(img)

        lbl       = tk.Label(top, image=photo, bg=BG_DEEP)
        lbl.image = photo
        lbl.pack(padx=16, pady=16)

        tk.Label(top, text=os.path.basename(fpath),
                 font=FONT_MONO_SM, bg=BG_DEEP, fg=TEXT_MID).pack(pady=(0, 8))
        tk.Button(top, text="CLOSE", command=top.destroy,
                  font=FONT_BTN, bg=BG_CARD, fg=ACCENT_PINK,
                  relief="flat", cursor="hand2", bd=0,
                  highlightthickness=1, highlightbackground=ACCENT_PINK
                  ).pack(pady=(0, 12))


# ─────────────────────────────────────────────────────────────────────────────
class App:
    def __init__(self, root):
        self.root = root
        self.root.title("NEURAL · VISION")
        self.root.geometry("820x780")
        self.root.resizable(False, False)
        self.root.configure(bg=BG_DEEP)

        self.file_path    = None
        self._last_label  = None
        self._cam_running = False
        self._cam_thread  = None
        self._cap         = None
        self._autosave    = False

        self._build_ui()
        self._draw_grid()

    # ── Build UI ──────────────────────────────────────────────────────────────
    def _build_ui(self):
        root = self.root

        self.bg_canvas = tk.Canvas(root, bg=BG_DEEP, highlightthickness=0)
        self.bg_canvas.place(x=0, y=0, relwidth=1, relheight=1)

        # Header
        header = tk.Frame(root, bg=BG_DEEP)
        header.place(x=0, y=0, width=820, height=70)
        tk.Label(header, text="◈ NEURAL·VISION",
                 font=FONT_TITLE, bg=BG_DEEP, fg=ACCENT_CYAN).place(x=28, y=18)
        tk.Label(header, text="IMAGE CLASSIFICATION v2.6",
                 font=FONT_MONO_SM, bg=BG_DEEP, fg=TEXT_DIM).place(x=34, y=50)
        self.status_dot = tk.Label(header, text="●", font=("Courier New", 14),
                                   bg=BG_DEEP, fg=ACCENT_GREEN)
        self.status_dot.place(x=770, y=25)
        self.status_lbl = tk.Label(header, text="READY", font=FONT_MONO_SM,
                                   bg=BG_DEEP, fg=TEXT_MID)
        self.status_lbl.place(x=736, y=48)

        sep = tk.Canvas(root, bg=BG_DEEP, highlightthickness=0, height=1)
        sep.place(x=0, y=68, width=820)
        sep.create_line(28, 0, 792, 0, fill=BORDER, width=1)

        # Left panel — image viewer
        left = tk.Frame(root, bg=BG_CARD, bd=0)
        left.place(x=28, y=88, width=480, height=380)
        self._corner_accents(root, 28, 88, 480, 380)

        self.view_canvas = ScanlineCanvas(left, bg=BG_CARD, highlightthickness=0,
                                          width=480, height=380)
        self.view_canvas.pack(fill="both", expand=True)
        self.view_canvas.create_text(240, 190, text="[ NO INPUT ]",
                                     font=("Courier New", 16, "bold"),
                                     fill=BORDER, tags="placeholder")
        self.view_canvas.create_text(240, 215, text="UPLOAD IMAGE OR START CAMERA",
                                     font=FONT_MONO_SM, fill=TEXT_DIM, tags="placeholder")

        # Right panel
        right = tk.Frame(root, bg=BG_DEEP)
        right.place(x=524, y=88, width=268, height=380)

        info_card = tk.Frame(right, bg=BG_PANEL, bd=0)
        info_card.place(x=0, y=0, width=268, height=130)
        self._border_frame(root, 524, 88, 268, 130)
        tk.Label(info_card, text="MODEL INFO", font=FONT_SMALL,
                 bg=BG_PANEL, fg=ACCENT_CYAN).place(x=12, y=10)
        for i, (k, v) in enumerate([("ARCH", "CNN · CIFAR-10"),
                                     ("CLASSES", "10 categories"),
                                     ("INPUT", "32 × 32 px")]):
            y = 34 + i * 28
            tk.Label(info_card, text=f"{k:<9}", font=FONT_MONO_SM,
                     bg=BG_PANEL, fg=TEXT_DIM).place(x=12, y=y)
            tk.Label(info_card, text=v, font=FONT_MONO_SM,
                     bg=BG_PANEL, fg=TEXT_PRIMARY).place(x=90, y=y)

        res_card = tk.Frame(right, bg=BG_PANEL, bd=0)
        res_card.place(x=0, y=145, width=268, height=235)
        self._border_frame(root, 524, 233, 268, 235)
        tk.Label(res_card, text="PREDICTION", font=FONT_SMALL,
                 bg=BG_PANEL, fg=ACCENT_PINK).place(x=12, y=10)

        self.icon_lbl = tk.Label(res_card, text="?",
                                 font=("Courier New", 42, "bold"),
                                 bg=BG_PANEL, fg=BORDER)
        self.icon_lbl.place(x=12, y=34)

        self.pred_lbl = tk.Label(res_card, text="———",
                                 font=FONT_RESULT, bg=BG_PANEL, fg=TEXT_DIM)
        self.pred_lbl.place(x=12, y=100)

        tk.Label(res_card, text="CONFIDENCE", font=FONT_SMALL,
                 bg=BG_PANEL, fg=TEXT_DIM).place(x=12, y=145)
        self.conf_pct = tk.Label(res_card, text="—%",
                                 font=("Courier New", 13, "bold"),
                                 bg=BG_PANEL, fg=ACCENT_CYAN)
        self.conf_pct.place(x=186, y=143)

        self.conf_bar = ConfidenceBar(res_card, width=244)
        self.conf_bar.place(x=12, y=166)

        tk.Label(res_card, text="TOP RESULTS", font=FONT_SMALL,
                 bg=BG_PANEL, fg=TEXT_DIM).place(x=12, y=183)
        self.top3_labels = []
        for i in range(3):
            lbl = tk.Label(res_card, text="", font=FONT_MONO_SM,
                           bg=BG_PANEL, fg=TEXT_MID, anchor="w")
            lbl.place(x=12, y=202 + i * 18, width=244)
            self.top3_labels.append(lbl)

        # ── ROW 1: Upload / Predict / Camera ──
        row1 = [
            ("⬆  UPLOAD IMAGE", self.load_image,   ACCENT_CYAN),
            ("◉  PREDICT",       self.predict_image, ACCENT_PINK),
            ("▶  LIVE CAMERA",   self.toggle_camera, ACCENT_GREEN),
        ]
        for i, (txt, cmd, fg) in enumerate(row1):
            btn = tk.Button(root, text=txt, command=cmd,
                            font=FONT_BTN, bg=BG_CARD, fg=fg,
                            activebackground=BORDER, activeforeground=fg,
                            relief="flat", cursor="hand2", bd=0,
                            highlightthickness=1, highlightbackground=fg)
            btn.place(x=28 + i * 264, y=484, width=250, height=44)
            if "CAMERA" in txt:
                self.cam_btn = btn

        # ── ROW 2: Add to Category / Visualiser / Auto-Save toggle ──
        self.add_btn = tk.Button(root,
                                 text="＋  ADD TO CATEGORY",
                                 command=self.add_to_category,
                                 font=FONT_BTN, bg=BG_CARD, fg=ACCENT_YELLOW,
                                 activebackground=BORDER, activeforeground=ACCENT_YELLOW,
                                 relief="flat", cursor="hand2", bd=0,
                                 highlightthickness=1, highlightbackground=ACCENT_YELLOW)
        self.add_btn.place(x=28, y=536, width=246, height=44)

        tk.Button(root, text="◧  VISUALISER",
                  command=self.open_gallery,
                  font=FONT_BTN, bg=BG_CARD, fg=ACCENT_PURPLE,
                  activebackground=BORDER, activeforeground=ACCENT_PURPLE,
                  relief="flat", cursor="hand2", bd=0,
                  highlightthickness=1, highlightbackground=ACCENT_PURPLE
                  ).place(x=282, y=536, width=246, height=44)

        self.autosave_btn = tk.Button(root,
                                      text="⏺  AUTO-SAVE: OFF",
                                      command=self.toggle_autosave,
                                      font=FONT_BTN, bg=BG_CARD, fg=TEXT_DIM,
                                      activebackground=BORDER, activeforeground=ACCENT_GREEN,
                                      relief="flat", cursor="hand2", bd=0,
                                      highlightthickness=1, highlightbackground=BORDER)
        self.autosave_btn.place(x=536, y=536, width=256, height=44)

        # ── Log panel ──
        log_frame = tk.Frame(root, bg=BG_PANEL)
        log_frame.place(x=28, y=594, width=764, height=158)
        self._border_frame(root, 28, 594, 764, 158)
        tk.Label(log_frame, text="SYSTEM LOG", font=FONT_SMALL,
                 bg=BG_PANEL, fg=TEXT_DIM).place(x=10, y=8)
        self.log_text = tk.Text(log_frame, bg=BG_PANEL, fg=ACCENT_GREEN,
                                font=FONT_MONO_SM, relief="flat",
                                state="disabled", insertbackground=ACCENT_CYAN,
                                selectbackground=BORDER, wrap="word",
                                highlightthickness=0)
        self.log_text.place(x=10, y=28, width=744, height=120)

        self._log("System initialized.")
        self._log(f"Model loaded · {len(CLASSES)} classes detected.")
        self._log("Awaiting input…")

    # ── Helpers ───────────────────────────────────────────────────────────────
    def _corner_accents(self, parent, x, y, w, h, color=ACCENT_CYAN, size=18):
        c = self.bg_canvas
        for cx, cy in [(x, y), (x+w, y), (x, y+h), (x+w, y+h)]:
            dx = 1 if cx == x else -1
            dy = 1 if cy == y else -1
            c.create_line(cx, cy, cx + dx*size, cy, fill=color, width=1)
            c.create_line(cx, cy, cx, cy + dy*size, fill=color, width=1)

    def _border_frame(self, parent, x, y, w, h, color=BORDER):
        self.bg_canvas.create_rectangle(x, y, x+w, y+h,
                                        outline=color, width=1, fill="")

    def _draw_grid(self):
        self.root.update_idletasks()
        for gx in range(0, 820, 40):
            for gy in range(0, 780, 40):
                self.bg_canvas.create_oval(gx-1, gy-1, gx+1, gy+1,
                                           fill="#1a2030", outline="")

    def _log(self, msg):
        ts = time.strftime("%H:%M:%S")
        self.log_text.config(state="normal")
        self.log_text.insert("end", f"[{ts}] {msg}\n")
        self.log_text.see("end")
        self.log_text.config(state="disabled")

    def _set_status(self, text, color=ACCENT_GREEN):
        self.status_lbl.config(text=text, fg=color)
        self.status_dot.config(fg=color)

    # ── Load image ────────────────────────────────────────────────────────────
    def load_image(self):
        self._stop_camera()
        path = filedialog.askopenfilename(
            filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp *.gif *.webp")])
        if not path:
            return
        self.file_path   = path
        self._last_label = None
        img = Image.open(path).convert("RGB")
        self._display_pil(img)
        self._log(f"Image loaded: {os.path.basename(path)}")
        self._set_status("IMAGE LOADED", ACCENT_CYAN)

    def _display_pil(self, pil_img):
        pil_img = pil_img.resize((480, 380), Image.LANCZOS)
        self.view_canvas.delete("placeholder")
        photo = ImageTk.PhotoImage(pil_img)
        self.view_canvas.delete("img")
        self.view_canvas.create_image(0, 0, anchor="nw", image=photo, tags="img")
        self.view_canvas.image = photo

    # ── Predict ───────────────────────────────────────────────────────────────
    def predict_image(self):
        if not self.file_path:
            messagebox.showerror("Error", "Upload an image first.")
            return
        img = keras_image.load_img(self.file_path, target_size=(32, 32))
        self._run_prediction(img)

    def _run_prediction(self, pil_img_32):
        arr    = keras_image.img_to_array(pil_img_32)
        arr    = np.expand_dims(arr, axis=0)
        result = model.predict(arr, verbose=0)[0]

        idx   = int(np.argmax(result))
        conf  = int(np.max(result) * 100)
        label = CLASSES[idx]

        self._last_label = label

        self.pred_lbl.config(text=label.upper(), fg=ACCENT_CYAN)
        self.icon_lbl.config(text=CLASS_ICONS.get(label, "?"), fg=ACCENT_CYAN)
        self.conf_pct.config(text=f"{conf}%")
        self.conf_bar.set_value(conf)

        top3_idx = np.argsort(result)[::-1][:3]
        for i, ti in enumerate(top3_idx):
            bar = "█" * int(result[ti] * 16)
            self.top3_labels[i].config(
                text=f"{CLASSES[ti]:<12} {int(result[ti]*100):>3}%  {bar}",
                fg=ACCENT_CYAN if i == 0 else TEXT_MID)

        self._log(f"Prediction: {label} ({conf}%)")
        self._set_status("PREDICTED", ACCENT_GREEN)
        return label, conf

    # ── Add to Category ───────────────────────────────────────────────────────
    def add_to_category(self):
        if not self.file_path:
            messagebox.showerror("Error", "No image loaded.")
            return
        if not self._last_label:
            messagebox.showerror("Error", "Run PREDICT first so the category is known.")
            return
        self._save_to_category(self._last_label)

    def _save_to_category(self, category):
        dest_dir = os.path.join(SAVE_DIR, category)
        os.makedirs(dest_dir, exist_ok=True)

        ext  = os.path.splitext(self.file_path)[1] or ".jpg"
        ts   = time.strftime("%Y%m%d_%H%M%S")
        dest = os.path.join(dest_dir, f"{category}_{ts}{ext}")
        shutil.copy2(self.file_path, dest)

        self._log(f"Saved → [{category}]  {os.path.basename(dest)}")
        self._set_status(f"SAVED → {category.upper()}", ACCENT_YELLOW)
        messagebox.showinfo("Saved ✔",
                            f"Image added to category:\n\n"
                            f"  {CLASS_ICONS.get(category,'')}  {category}\n\n"
                            f"{os.path.basename(dest)}")

    # ── Gallery / Visualiser ──────────────────────────────────────────────────
    def open_gallery(self):
        GalleryWindow(self.root)
        self._log("Gallery opened.")

    # ── Camera ────────────────────────────────────────────────────────────────
    def toggle_camera(self):
        if self._cam_running:
            self._stop_camera()
        else:
            self._start_camera()

    def toggle_autosave(self):
        self._autosave = not self._autosave
        if self._autosave:
            self.autosave_btn.config(text="⏺  AUTO-SAVE: ON ",
                                     fg=ACCENT_GREEN,
                                     highlightbackground=ACCENT_GREEN)
            self._log("Auto-save ENABLED — saves every detection ≥60% confidence (max 1 per 3s).")
        else:
            self.autosave_btn.config(text="⏺  AUTO-SAVE: OFF",
                                     fg=TEXT_DIM,
                                     highlightbackground=BORDER)
            self._log("Auto-save DISABLED.")

    def _start_camera(self):
        self._cap = cv2.VideoCapture(0)
        if not self._cap.isOpened():
            messagebox.showerror("Camera Error", "Cannot open camera.")
            self._log("ERROR: Camera not found.")
            self._set_status("CAM ERROR", ACCENT_PINK)
            return
        self._cam_running = True
        self.cam_btn.config(text="■  STOP CAMERA", fg=ACCENT_PINK)
        self._set_status("LIVE", ACCENT_GREEN)
        self._log("Live camera started.")
        self.file_path   = None
        self._last_label = None
        self._cam_thread = threading.Thread(target=self._camera_loop, daemon=True)
        self._cam_thread.start()

    def _stop_camera(self):
        if not self._cam_running:
            return
        self._cam_running = False
        time.sleep(0.15)
        if self._cap:
            self._cap.release()
            self._cap = None
        self.cam_btn.config(text="▶  LIVE CAMERA", fg=ACCENT_GREEN)
        self._set_status("READY", ACCENT_GREEN)
        self._log("Camera stopped.")

    def _camera_loop(self):
        last_pred_time = 0
        last_save_time = 0
        PRED_INTERVAL  = 0.5   # predict every 0.5s
        SAVE_INTERVAL  = 3.0   # auto-save at most once every 3s
        label, conf    = None, None

        while self._cam_running:
            ret, frame = self._cap.read()
            if not ret:
                break

            rgb       = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_frame = Image.fromarray(rgb)

            now = time.time()
            if now - last_pred_time >= PRED_INTERVAL:
                small  = pil_frame.resize((32, 32), Image.LANCZOS)
                arr    = keras_image.img_to_array(small)
                arr    = np.expand_dims(arr, axis=0)
                result = model.predict(arr, verbose=0)[0]
                idx    = int(np.argmax(result))
                conf   = int(np.max(result) * 100)
                label  = CLASSES[idx]
                last_pred_time = now

                # ── Auto-save if enabled, confidence ≥ 60%, cooldown elapsed ──
                if self._autosave and conf >= 60 and (now - last_save_time) >= SAVE_INTERVAL:
                    save_frame = pil_frame.copy()
                    save_label = label
                    threading.Thread(
                        target=self._autosave_frame,
                        args=(save_frame, save_label),
                        daemon=True
                    ).start()
                    last_save_time = now

                self.root.after(0, lambda l=label, c=conf, r=result.copy():
                                self._update_live_result(l, c, r))

            if label:
                draw = ImageDraw.Draw(pil_frame)
                draw.rectangle([0, 0, pil_frame.width, 40], fill=(0, 0, 0))
                draw.text((10, 8),
                          f"{CLASS_ICONS.get(label, '?')} {label}  {conf}%",
                          fill=(0, 229, 255))
                # Green dot = auto-save active & confident, grey = not
                if self._autosave:
                    dot_color = (0, 255, 159) if conf >= 60 else (60, 60, 60)
                    draw.ellipse([pil_frame.width - 22, 12,
                                  pil_frame.width - 10, 24], fill=dot_color)

            self.root.after(0, lambda f=pil_frame: self._display_pil(f))
            time.sleep(0.03)

    def _autosave_frame(self, pil_img, category):
        """Save a raw camera frame to its category folder (background thread)."""
        dest_dir = os.path.join(SAVE_DIR, category)
        os.makedirs(dest_dir, exist_ok=True)
        ts   = time.strftime("%Y%m%d_%H%M%S")
        dest = os.path.join(dest_dir, f"{category}_{ts}_cam.jpg")
        pil_img.save(dest, "JPEG", quality=90)
        self.root.after(0, lambda: (
            self._log(f"[CAM] Auto-saved → [{category}]  {os.path.basename(dest)}"),
            self._set_status(f"AUTO-SAVED → {category.upper()}", ACCENT_GREEN)
        ))

    def _update_live_result(self, label, conf, result):
        self._last_label = label
        self.pred_lbl.config(text=label.upper(), fg=ACCENT_CYAN)
        self.icon_lbl.config(text=CLASS_ICONS.get(label, "?"), fg=ACCENT_CYAN)
        self.conf_pct.config(text=f"{conf}%")
        self.conf_bar.set_value(conf)
        top3 = np.argsort(result)[::-1][:3]
        for i, ti in enumerate(top3):
            bar = "█" * int(result[ti] * 16)
            self.top3_labels[i].config(
                text=f"{CLASSES[ti]:<12} {int(result[ti]*100):>3}%  {bar}",
                fg=ACCENT_CYAN if i == 0 else TEXT_MID)


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    root = tk.Tk()
    app = App(root)
    root.mainloop()
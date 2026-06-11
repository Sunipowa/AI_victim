"""
3D Printed Object Defect Detection
Deep Learning Project — YOLO + SAHI Pipeline
"""

import tkinter as tk
from tkinter import ttk, font
import threading
import time
import random
import math
from datetime import datetime
from collections import deque

# ─── Try importing optional heavy deps ───────────────────────────────────────
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False

try:
    from sahi import AutoDetectionModel
    from sahi.predict import get_sliced_prediction
    SAHI_AVAILABLE = True
except ImportError:
    SAHI_AVAILABLE = False

try:
    from PIL import Image, ImageTk, ImageDraw, ImageFont
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

# ─── Color Palette ────────────────────────────────────────────────────────────
BG_DARK      = "#0D0F14"
BG_PANEL     = "#141720"
BG_CARD      = "#1C2030"
BG_CARD2     = "#222840"
ACCENT_CYAN  = "#00D4FF"
ACCENT_GREEN = "#00FF9C"
ACCENT_RED   = "#FF4757"
ACCENT_AMBER = "#FFB300"
TEXT_PRIMARY = "#E8EAF0"
TEXT_MUTED   = "#6B7280"
TEXT_DIM     = "#3D4455"
BORDER       = "#252A3A"

# ─── Defect Classes (example for 3D print) ───────────────────────────────────
DEFECT_CLASSES = [
    "Layer Separation",
    "Under-Extrusion",
    "Over-Extrusion",
    "Stringing",
    "Warping",
    "Blob / Zit",
    "Elephant Foot",
]
DEFECT_COLORS = {
    "Layer Separation": ACCENT_RED,
    "Under-Extrusion":  ACCENT_AMBER,
    "Over-Extrusion":   "#FF6B35",
    "Stringing":        "#A855F7",
    "Warping":          ACCENT_RED,
    "Blob / Zit":       ACCENT_AMBER,
    "Elephant Foot":    "#06B6D4",
}


# ══════════════════════════════════════════════════════════════════════════════
class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("3D Print Defect Detection")
        self.configure(bg=BG_DARK)
        self.geometry("1280x760")
        self.minsize(1100, 680)

        self._running    = False
        self._cap        = None
        self._model      = None
        self._use_sahi   = tk.BooleanVar(value=True)
        self._conf_thresh = tk.DoubleVar(value=0.45)
        self._history    = deque(maxlen=200)
        self._frame_count = 0
        self._fps_time   = time.time()
        self._fps        = 0.0
        self._total_det  = 0
        self._demo_mode  = not (CV2_AVAILABLE and PIL_AVAILABLE)

        self._setup_fonts()
        self._build_ui()
        self._animate_idle()

    # ── Fonts ─────────────────────────────────────────────────────────────────
    def _setup_fonts(self):
        self.fn_title  = ("Segoe UI", 11, "bold")
        self.fn_label  = ("Segoe UI", 9)
        self.fn_small  = ("Segoe UI", 8)
        self.fn_mono   = ("Consolas", 9)
        self.fn_big    = ("Segoe UI", 22, "bold")
        self.fn_medium = ("Segoe UI", 13, "bold")

    # ══════════════════════════════════════════════════════════════════════════
    # UI BUILD
    # ══════════════════════════════════════════════════════════════════════════
    def _build_ui(self):
        # ── Top bar ──────────────────────────────────────────────────────────
        topbar = tk.Frame(self, bg=BG_PANEL, height=54)
        topbar.pack(fill="x", side="top")
        topbar.pack_propagate(False)

        dot_frame = tk.Frame(topbar, bg=BG_PANEL)
        dot_frame.pack(side="left", padx=16, pady=0)
        for col in [ACCENT_RED, ACCENT_AMBER, ACCENT_GREEN]:
            tk.Label(dot_frame, bg=col, width=2, height=1,
                     relief="flat").pack(side="left", padx=3, pady=18)

        tk.Label(topbar, text="3D Print Defect Detection",
                 bg=BG_PANEL, fg=TEXT_PRIMARY,
                 font=("Segoe UI", 13, "bold")).pack(side="left", padx=4)
        tk.Label(topbar, text="  ·  YOLO + SAHI Pipeline",
                 bg=BG_PANEL, fg=TEXT_MUTED,
                 font=("Segoe UI", 10)).pack(side="left")

        # Status pill
        self._status_lbl = tk.Label(topbar, text="● IDLE",
                                    bg=BG_PANEL, fg=TEXT_MUTED,
                                    font=("Segoe UI", 9, "bold"))
        self._status_lbl.pack(side="right", padx=20)

        # FPS badge
        self._fps_lbl = tk.Label(topbar, text="0.0 fps",
                                 bg=BG_CARD, fg=ACCENT_CYAN,
                                 font=self.fn_mono, padx=10, pady=4,
                                 relief="flat")
        self._fps_lbl.pack(side="right", padx=4)

        # ── Separator ────────────────────────────────────────────────────────
        tk.Frame(self, bg=BORDER, height=1).pack(fill="x")

        # ── Main body ────────────────────────────────────────────────────────
        body = tk.Frame(self, bg=BG_DARK)
        body.pack(fill="both", expand=True, padx=0, pady=0)

        # Left: camera + controls
        left = tk.Frame(body, bg=BG_DARK)
        left.pack(side="left", fill="both", expand=True, padx=16, pady=14)

        # Right: sidebar
        right = tk.Frame(body, bg=BG_DARK, width=320)
        right.pack(side="right", fill="y", padx=(0, 14), pady=14)
        right.pack_propagate(False)

        self._build_camera_panel(left)
        self._build_controls(left)
        self._build_sidebar(right)

    # ── Camera panel ──────────────────────────────────────────────────────────
    def _build_camera_panel(self, parent):
        wrapper = tk.Frame(parent, bg=BG_CARD, highlightthickness=1,
                           highlightbackground=BORDER)
        wrapper.pack(fill="both", expand=True)

        # Header row
        hdr = tk.Frame(wrapper, bg=BG_CARD2, height=34)
        hdr.pack(fill="x")
        hdr.pack_propagate(False)

        tk.Label(hdr, text="⬡  LIVE FEED", bg=BG_CARD2, fg=ACCENT_CYAN,
                 font=("Segoe UI", 9, "bold")).pack(side="left", padx=12,
                                                    pady=8)
        self._sahi_badge = tk.Label(hdr, text="SAHI ✓", bg=BG_CARD2,
                                    fg=ACCENT_GREEN,
                                    font=("Segoe UI", 8, "bold"))
        self._sahi_badge.pack(side="right", padx=12)

        # Canvas
        self._canvas = tk.Canvas(wrapper, bg="#060810", cursor="crosshair",
                                 highlightthickness=0)
        self._canvas.pack(fill="both", expand=True)
        self._canvas.bind("<Configure>", self._on_canvas_resize)

    # ── Controls bar ──────────────────────────────────────────────────────────
    def _build_controls(self, parent):
        bar = tk.Frame(parent, bg=BG_DARK)
        bar.pack(fill="x", pady=(10, 0))

        # Start / Stop
        self._start_btn = self._btn(bar, "▶  START", ACCENT_GREEN,
                                    self._toggle_detection)
        self._start_btn.pack(side="left")

        # Snapshot
        self._btn(bar, "⊙  SNAPSHOT", ACCENT_CYAN,
                  self._snapshot).pack(side="left", padx=(8, 0))

        # Clear history
        self._btn(bar, "✕  CLEAR", ACCENT_RED,
                  self._clear_history).pack(side="left", padx=(8, 0))

        # SAHI toggle
        sahi_chk = tk.Checkbutton(bar, text="SAHI Slicing",
                                  variable=self._use_sahi,
                                  bg=BG_DARK, fg=TEXT_PRIMARY,
                                  selectcolor=BG_CARD,
                                  activebackground=BG_DARK,
                                  activeforeground=TEXT_PRIMARY,
                                  font=self.fn_label,
                                  command=self._on_sahi_toggle)
        sahi_chk.pack(side="left", padx=(16, 0))

        # Conf threshold
        tk.Label(bar, text="Conf:", bg=BG_DARK, fg=TEXT_MUTED,
                 font=self.fn_small).pack(side="left", padx=(16, 4))

        conf_slider = tk.Scale(bar, variable=self._conf_thresh,
                               from_=0.1, to=0.95, resolution=0.05,
                               orient="horizontal", length=120,
                               bg=BG_DARK, fg=TEXT_PRIMARY,
                               troughcolor=BG_CARD, highlightthickness=0,
                               activebackground=ACCENT_CYAN,
                               showvalue=True, font=self.fn_small)
        conf_slider.pack(side="left")

        # Model path label
        self._model_lbl = tk.Label(bar, text="Model: not loaded",
                                   bg=BG_DARK, fg=TEXT_MUTED,
                                   font=self.fn_small)
        self._model_lbl.pack(side="right", padx=4)

    def _btn(self, parent, text, color, cmd):
        btn = tk.Label(parent, text=text, bg=BG_CARD, fg=color,
                       font=("Segoe UI", 9, "bold"),
                       padx=14, pady=7, cursor="hand2",
                       relief="flat")
        btn.bind("<Button-1>", lambda e: cmd())
        btn.bind("<Enter>",  lambda e, b=btn, c=color: b.config(bg=BG_CARD2))
        btn.bind("<Leave>",  lambda e, b=btn: b.config(bg=BG_CARD))
        return btn

    # ── Sidebar ───────────────────────────────────────────────────────────────
    def _build_sidebar(self, parent):
        # ── Stats row ────────────────────────────────────────────────────────
        stats_frame = tk.Frame(parent, bg=BG_DARK)
        stats_frame.pack(fill="x", pady=(0, 10))

        self._det_count_lbl  = self._stat_card(stats_frame, "Total Det.", "0",   ACCENT_CYAN)
        self._session_lbl    = self._stat_card(stats_frame, "Session",    "0:00", TEXT_MUTED)

        # ── Confidence gauge ─────────────────────────────────────────────────
        conf_card = tk.Frame(parent, bg=BG_CARD, highlightthickness=1,
                             highlightbackground=BORDER)
        conf_card.pack(fill="x", pady=(0, 10))

        tk.Label(conf_card, text="CONFIDENCE METER",
                 bg=BG_CARD, fg=TEXT_MUTED,
                 font=("Segoe UI", 8, "bold")).pack(anchor="w",
                                                    padx=12, pady=(10, 4))

        self._conf_canvas = tk.Canvas(conf_card, height=56,
                                      bg=BG_CARD, highlightthickness=0)
        self._conf_canvas.pack(fill="x", padx=12, pady=(0, 10))
        self._conf_val = tk.Label(conf_card, text="—",
                                  bg=BG_CARD, fg=ACCENT_CYAN,
                                  font=("Segoe UI", 18, "bold"))
        self._conf_val.pack(pady=(0, 8))

        # ── Detection history ─────────────────────────────────────────────────
        hist_card = tk.Frame(parent, bg=BG_CARD, highlightthickness=1,
                             highlightbackground=BORDER)
        hist_card.pack(fill="both", expand=True)

        hdr = tk.Frame(hist_card, bg=BG_CARD2, height=34)
        hdr.pack(fill="x")
        hdr.pack_propagate(False)
        tk.Label(hdr, text="⏱  DETECTION LOG",
                 bg=BG_CARD2, fg=ACCENT_CYAN,
                 font=("Segoe UI", 9, "bold")).pack(side="left",
                                                    padx=12, pady=8)
        self._log_count = tk.Label(hdr, text="0 events",
                                   bg=BG_CARD2, fg=TEXT_MUTED,
                                   font=self.fn_small)
        self._log_count.pack(side="right", padx=10)

        # Scrollable list
        list_frame = tk.Frame(hist_card, bg=BG_CARD)
        list_frame.pack(fill="both", expand=True)

        scrollbar = tk.Scrollbar(list_frame, orient="vertical",
                                 bg=BG_CARD, troughcolor=BG_CARD)
        scrollbar.pack(side="right", fill="y")

        self._history_list = tk.Listbox(
            list_frame,
            bg=BG_CARD, fg=TEXT_PRIMARY,
            font=self.fn_mono,
            selectbackground=BG_CARD2,
            selectforeground=ACCENT_CYAN,
            activestyle="none",
            highlightthickness=0,
            borderwidth=0,
            yscrollcommand=scrollbar.set,
        )
        self._history_list.pack(fill="both", expand=True)
        scrollbar.config(command=self._history_list.yview)

    def _stat_card(self, parent, label, value, color):
        card = tk.Frame(parent, bg=BG_CARD, highlightthickness=1,
                        highlightbackground=BORDER)
        card.pack(side="left", fill="x", expand=True, padx=(0, 6))
        tk.Label(card, text=label, bg=BG_CARD, fg=TEXT_MUTED,
                 font=("Segoe UI", 8)).pack(pady=(8, 0))
        val_lbl = tk.Label(card, text=value, bg=BG_CARD, fg=color,
                           font=("Segoe UI", 18, "bold"))
        val_lbl.pack(pady=(0, 8))
        return val_lbl

    # ══════════════════════════════════════════════════════════════════════════
    # LOGIC
    # ══════════════════════════════════════════════════════════════════════════
    def _toggle_detection(self):
        if self._running:
            self._stop_detection()
        else:
            self._start_detection()

    def _start_detection(self):
        self._running    = True
        self._session_start = time.time()
        self._start_btn.config(text="⏹  STOP", fg=ACCENT_RED)
        self._status_lbl.config(text="● RUNNING", fg=ACCENT_GREEN)

        if not self._demo_mode:
            # Real camera
            self._cap = cv2.VideoCapture(0)
            if not self._cap.isOpened():
                self._demo_mode = True  # fallback

            # Load YOLO if available
            if YOLO_AVAILABLE and self._model is None:
                try:
                    self._model = YOLO("yolov8n.pt")
                    self._model_lbl.config(text="Model: YOLOv8n",
                                           fg=ACCENT_GREEN)
                except Exception:
                    self._model = None

        thread = threading.Thread(target=self._detection_loop, daemon=True)
        thread.start()

    def _stop_detection(self):
        self._running = False
        self._start_btn.config(text="▶  START", fg=ACCENT_GREEN)
        self._status_lbl.config(text="● IDLE", fg=TEXT_MUTED)
        if self._cap:
            self._cap.release()
            self._cap = None

    def _detection_loop(self):
        while self._running:
            t0 = time.time()

            if self._demo_mode:
                frame, detections = self._demo_frame()
            else:
                frame, detections = self._real_frame()

            self._frame_count += 1
            dt = time.time() - t0
            self._fps = 0.8 * self._fps + 0.2 * (1.0 / max(dt, 0.001))

            self.after(0, self._update_ui, frame, detections)
            sleep_t = max(0, (1 / 30) - dt)
            time.sleep(sleep_t)

    # ── Demo / Simulation ─────────────────────────────────────────────────────
    def _demo_frame(self):
        """Generate a synthetic frame with random detections."""
        w, h = 640, 480

        if PIL_AVAILABLE:
            img = Image.new("RGB", (w, h), color=(6, 8, 16))
            draw = ImageDraw.Draw(img)

            # Grid lines (scanner aesthetic)
            for x in range(0, w, 40):
                draw.line([(x, 0), (x, h)], fill=(20, 28, 48), width=1)
            for y in range(0, h, 40):
                draw.line([(0, y), (w, y)], fill=(20, 28, 48), width=1)

            # Fake 3D print object silhouette
            cx, cy = w // 2, h // 2
            for ring in range(5, 0, -1):
                r = ring * 28
                col = (15 + ring * 6, 20 + ring * 8, 35 + ring * 10)
                draw.ellipse([cx - r, cy - r // 2, cx + r, cy + r // 2],
                             fill=col)

            detections = []
            if random.random() > 0.35:
                n = random.randint(1, 3)
                for _ in range(n):
                    cls  = random.choice(DEFECT_CLASSES)
                    conf = random.uniform(0.50, 0.97)
                    if conf < self._conf_thresh.get():
                        continue
                    x1 = random.randint(60, 420)
                    y1 = random.randint(60, 320)
                    x2 = x1 + random.randint(60, 160)
                    y2 = y1 + random.randint(40, 120)
                    color_hex = DEFECT_COLORS.get(cls, ACCENT_CYAN)
                    r, g, b = (int(color_hex[i:i+2], 16)
                               for i in (1, 3, 5))

                    # Bounding box
                    for th in range(2, 0, -1):
                        draw.rectangle([x1 - th, y1 - th, x2 + th, y2 + th],
                                       outline=(r, g, b, 180))

                    # Label bg
                    label = f"{cls}  {conf:.0%}"
                    lw = len(label) * 7
                    draw.rectangle([x1, y1 - 18, x1 + lw, y1],
                                   fill=(r // 3, g // 3, b // 3))
                    draw.text((x1 + 3, y1 - 16), label,
                              fill=(r, g, b))

                    detections.append({"class": cls, "conf": conf})

            # Scanline overlay
            for y in range(0, h, 4):
                draw.line([(0, y), (w, y)], fill=(0, 0, 0, 30), width=1)

            return img, detections
        else:
            return None, []

    # ── Real inference ────────────────────────────────────────────────────────
    def _real_frame(self):
        detections = []
        if not self._cap:
            return None, detections

        ret, frame = self._cap.read()
        if not ret:
            return None, detections

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        if self._model:
            conf = self._conf_thresh.get()
            if self._use_sahi.get() and SAHI_AVAILABLE:
                try:
                    result = get_sliced_prediction(
                        frame_rgb, self._model,
                        slice_height=256, slice_width=256,
                        overlap_height_ratio=0.2,
                        overlap_width_ratio=0.2,
                    )
                    for obj in result.object_prediction_list:
                        if obj.score.value >= conf:
                            bb = obj.bbox
                            cv2.rectangle(frame_rgb,
                                          (int(bb.minx), int(bb.miny)),
                                          (int(bb.maxx), int(bb.maxy)),
                                          (0, 212, 255), 2)
                            detections.append({
                                "class": obj.category.name,
                                "conf":  obj.score.value,
                            })
                except Exception:
                    pass
            else:
                results = self._model(frame_rgb, conf=conf, verbose=False)
                for r in results:
                    for box in r.boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        c    = float(box.conf[0])
                        cls  = self._model.names[int(box.cls[0])]
                        cv2.rectangle(frame_rgb, (x1, y1), (x2, y2),
                                      (0, 212, 255), 2)
                        detections.append({"class": cls, "conf": c})

        if PIL_AVAILABLE:
            return Image.fromarray(frame_rgb), detections
        return None, detections

    # ── UI update (main thread) ───────────────────────────────────────────────
    def _update_ui(self, frame, detections):
        # Canvas
        cw = self._canvas.winfo_width()
        ch = self._canvas.winfo_height()
        if cw < 2 or ch < 2:
            return

        if frame is not None and PIL_AVAILABLE:
            frame_resized = frame.resize((cw, ch), Image.LANCZOS)
            self._tk_img = ImageTk.PhotoImage(frame_resized)
            self._canvas.delete("all")
            self._canvas.create_image(0, 0, anchor="nw",
                                      image=self._tk_img)
        else:
            self._draw_placeholder(cw, ch)

        # FPS
        self._fps_lbl.config(text=f"{self._fps:.1f} fps")

        # Session time
        elapsed = int(time.time() - self._session_start)
        m, s    = divmod(elapsed, 60)
        self._session_lbl.config(text=f"{m}:{s:02d}")

        # Detections
        if detections:
            for det in detections:
                self._add_history_entry(det)
            self._total_det += len(detections)
            self._det_count_lbl.config(text=str(self._total_det))

            best = max(detections, key=lambda d: d["conf"])
            self._update_conf_gauge(best["conf"])
            self._conf_val.config(
                text=f"{best['conf']:.1%}",
                fg=self._conf_color(best["conf"])
            )

    def _add_history_entry(self, det):
        ts    = datetime.now().strftime("%H:%M:%S")
        cls   = det["class"]
        conf  = det["conf"]
        entry = f"  {ts}  {cls:<22} {conf:.0%}"
        self._history.appendleft(entry)
        self._history_list.insert(0, entry)
        color = DEFECT_COLORS.get(cls, ACCENT_CYAN)

        # Color by conf
        self._history_list.itemconfig(0, fg=color)
        self._log_count.config(text=f"{len(self._history)} events")

    def _update_conf_gauge(self, value):
        c  = self._conf_canvas
        cw = c.winfo_width()
        if cw < 2:
            return
        c.delete("all")
        # Track
        c.create_rectangle(0, 20, cw, 38, fill=BG_CARD2, outline="")
        # Fill
        fill_w = int(cw * value)
        fc = self._conf_color(value)
        c.create_rectangle(0, 20, fill_w, 38, fill=fc, outline="")
        # Tick marks
        for p in [0.25, 0.5, 0.75]:
            x = int(cw * p)
            c.create_line(x, 14, x, 44, fill=TEXT_DIM, width=1)
            c.create_text(x, 10, text=f"{p:.0%}",
                          fill=TEXT_DIM, font=self.fn_small)

    def _conf_color(self, v):
        if v >= 0.80:
            return ACCENT_GREEN
        elif v >= 0.55:
            return ACCENT_AMBER
        else:
            return ACCENT_RED

    def _on_sahi_toggle(self):
        state = "✓" if self._use_sahi.get() else "✗"
        col   = ACCENT_GREEN if self._use_sahi.get() else TEXT_MUTED
        self._sahi_badge.config(text=f"SAHI {state}", fg=col)

    def _on_canvas_resize(self, event):
        if not self._running:
            self._draw_placeholder(event.width, event.height)

    def _draw_placeholder(self, w, h):
        self._canvas.delete("all")
        # Background grid
        for x in range(0, w, 40):
            self._canvas.create_line(x, 0, x, h, fill="#0F1420")
        for y in range(0, h, 40):
            self._canvas.create_line(0, y, w, y, fill="#0F1420")

        # Center icon
        cx, cy = w // 2, h // 2
        r = 44
        self._canvas.create_oval(cx - r, cy - r, cx + r, cy + r,
                                 outline=TEXT_DIM, width=2)
        self._canvas.create_oval(cx - 14, cy - 14, cx + 14, cy + 14,
                                 outline=TEXT_DIM, width=2)
        self._canvas.create_line(cx - r - 12, cy, cx + r + 12, cy,
                                 fill=TEXT_DIM, width=1)
        self._canvas.create_line(cx, cy - r - 12, cx, cy + r + 12,
                                 fill=TEXT_DIM, width=1)

        msg = ("Camera feed will appear here"
               if CV2_AVAILABLE else
               "Install opencv-python to use camera")
        self._canvas.create_text(cx, cy + r + 28, text=msg,
                                 fill=TEXT_MUTED,
                                 font=("Segoe UI", 10))
        if self._demo_mode:
            self._canvas.create_text(cx, cy + r + 50,
                                     text="Running in DEMO MODE",
                                     fill=ACCENT_AMBER,
                                     font=("Segoe UI", 9, "bold"))

    # ── Idle animation ────────────────────────────────────────────────────────
    def _animate_idle(self):
        if not self._running:
            w = self._canvas.winfo_width()
            h = self._canvas.winfo_height()
            if w > 2 and h > 2:
                self._draw_placeholder(w, h)
                # Pulsing ring
                t  = time.time()
                cx, cy = w // 2, h // 2
                pulse = 44 + 8 * math.sin(t * 2)
                self._canvas.create_oval(
                    cx - pulse, cy - pulse, cx + pulse, cy + pulse,
                    outline=ACCENT_CYAN,
                    width=1,
                    dash=(4, 6),
                )
        self.after(80, self._animate_idle)

    # ── Actions ───────────────────────────────────────────────────────────────
    def _snapshot(self):
        if not PIL_AVAILABLE:
            return
        ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = f"snapshot_{ts}.png"
        # Try to grab current canvas as screenshot
        try:
            x = self.winfo_rootx() + self._canvas.winfo_x()
            y = self.winfo_rooty() + self._canvas.winfo_y()
            w = self._canvas.winfo_width()
            h = self._canvas.winfo_height()
            import pyautogui
            img = pyautogui.screenshot(region=(x, y, w, h))
            img.save(path)
            self._status_lbl.config(text=f"Saved {path}", fg=ACCENT_GREEN)
        except Exception:
            self._status_lbl.config(text="Snapshot: install pyautogui",
                                    fg=ACCENT_AMBER)

    def _clear_history(self):
        self._history.clear()
        self._history_list.delete(0, "end")
        self._total_det = 0
        self._det_count_lbl.config(text="0")
        self._conf_val.config(text="—")
        self._log_count.config(text="0 events")
        self._conf_canvas.delete("all")

    def on_close(self):
        self._running = False
        if self._cap:
            self._cap.release()
        self.destroy()


# ══════════════════════════════════════════════════════════════════════════════
def main():
    app = App()
    app.protocol("WM_DELETE_WINDOW", app.on_close)
    app.mainloop()


if __name__ == "__main__":
    main()
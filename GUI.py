import cv2
import time
import queue
import threading
import tkinter as tk
from tkinter import ttk, messagebox
from datetime import datetime

from PIL import Image, ImageTk
from ultralytics import YOLO


# =========================
# CONFIG
# =========================
MODEL_PATH = r"C:/Users/Dell/Downloads/runs/detect/nhan_dien_in_3d/weights/best.pt"
CAMERA_INDEX = 0

WINDOW_WIDTH = 1600
WINDOW_HEIGHT = 900


class YOLODashboard:
    def __init__(self, root):
        self.root = root
        self.root.title("AI Detection of Defective 3D Printed Parts")
        self.root.geometry(f"{WINDOW_WIDTH}x{WINDOW_HEIGHT}")
        self.root.configure(bg="#0B1120")
        self.root.minsize(1200, 720)

        self.running = True
        self.detection_active = True

        self.frame_queue = queue.Queue(maxsize=1)
        self.prev_ui_time = time.time()

        self.conf_threshold = tk.DoubleVar(value=0.50)
        self.metric_status = tk.StringVar(value="Running")

        # Load model
        try:
            self.model = YOLO(MODEL_PATH)
        except Exception as e:
            messagebox.showerror("Model Error", f"Cannot load YOLO model:\n{e}")
            self.root.destroy()
            return

        # Open camera
        self.cap = cv2.VideoCapture(CAMERA_INDEX)
        if not self.cap.isOpened():
            messagebox.showerror("Camera Error", f"Cannot open camera index {CAMERA_INDEX}")
            self.root.destroy()
            return

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        self._build_styles()
        self._build_ui()

        self.worker_thread = threading.Thread(target=self.process_camera, daemon=True)
        self.worker_thread.start()

        self.update_ui()

        self.root.bind("<space>", lambda e: self.toggle_detection())
        self.root.bind("<Escape>", lambda e: self.on_close())
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    # -------------------------
    # Styles
    # -------------------------
    def _build_styles(self):
        style = ttk.Style(self.root)
        style.theme_use("clam")

        style.configure("TButton", font=("Segoe UI", 10))
        style.configure("TScale", background="#111827")

    # -------------------------
    # UI
    # -------------------------
    def _build_ui(self):
        # Header
        header = tk.Frame(self.root, bg="#0B1120", height=72)
        header.pack(fill="x", padx=18, pady=(12, 6))
        header.pack_propagate(False)

        title = tk.Label(
            header,
            text="AI Detection of Defective 3D Printed Parts",
            font=("Segoe UI", 24, "bold"),
            fg="#F8FAFC",
            bg="#0B1120"
        )
        title.pack(side="left")

        self.top_info = tk.Label(
            header,
            textvariable=self.metric_status,
            font=("Segoe UI", 11, "bold"),
            fg="#38BDF8",
            bg="#0B1120"
        )
        self.top_info.pack(side="right", padx=(0, 12), pady=8)

        # Main layout
        main = tk.Frame(self.root, bg="#0B1120")
        main.pack(fill="both", expand=True, padx=18, pady=(0, 18))

        # Left: Camera
        left = tk.Frame(main, bg="#0B1120")
        left.pack(side="left", fill="both", expand=True)

        cam_card = tk.Frame(
            left,
            bg="#111827",
            bd=0,
            highlightthickness=1,
            highlightbackground="#1F2937"
        )
        cam_card.pack(fill="both", expand=True, padx=(0, 12), pady=6)

        cam_header = tk.Frame(cam_card, bg="#111827", height=56)
        cam_header.pack(fill="x", padx=12, pady=(12, 6))
        cam_header.pack_propagate(False)

        tk.Label(
            cam_header,
            text="Live Camera Feed",
            font=("Segoe UI", 16, "bold"),
            fg="#F8FAFC",
            bg="#111827"
        ).pack(side="left")

        self.cam_badge = tk.Label(
            cam_header,
            text="ACTIVE",
            font=("Segoe UI", 10, "bold"),
            fg="#0B1120",
            bg="#38BDF8",
            padx=8,
            pady=4
        )
        self.cam_badge.pack(side="right")

        self.video_canvas = tk.Canvas(
            cam_card,
            bg="#0F172A",
            highlightthickness=0
        )
        self.video_canvas.pack(fill="both", expand=True, padx=12, pady=(0, 12))

        # Right: Controls only
        right = tk.Frame(
            main,
            bg="#111827",
            width=360,
            bd=0,
            highlightthickness=1,
            highlightbackground="#1F2937"
        )
        right.pack(side="right", fill="y")
        right.pack_propagate(False)

        tk.Label(
            right,
            text="System Controls",
            font=("Segoe UI", 16, "bold"),
            fg="#F8FAFC",
            bg="#111827"
        ).pack(anchor="w", padx=18, pady=(18, 8))

        ctrl_frame = tk.Frame(right, bg="#111827")
        ctrl_frame.pack(fill="x", padx=18)

        tk.Label(
            ctrl_frame,
            text="Confidence Threshold",
            font=("Segoe UI", 11, "bold"),
            fg="#94A3B8",
            bg="#111827"
        ).pack(anchor="w")

        self.threshold_label = tk.Label(
            ctrl_frame,
            text=f"{self.conf_threshold.get():.2f}",
            font=("Segoe UI", 11),
            fg="#38BDF8",
            bg="#111827"
        )
        self.threshold_label.pack(anchor="e")

        self.threshold_slider = ttk.Scale(
            ctrl_frame,
            from_=0.10,
            to=1.00,
            orient="horizontal",
            variable=self.conf_threshold,
            command=self._on_threshold_change
        )
        self.threshold_slider.pack(fill="x", pady=(4, 18))

        self.start_btn = ttk.Button(
            ctrl_frame,
            text="Start Detection (Space)",
            command=self.start_detection
        )
        self.start_btn.pack(fill="x", pady=6)

        self.stop_btn = ttk.Button(
            ctrl_frame,
            text="Stop Detection",
            command=self.stop_detection
        )
        self.stop_btn.pack(fill="x", pady=6)

        self.exit_btn = ttk.Button(
            ctrl_frame,
            text="Exit (Esc)",
            command=self.on_close
        )
        self.exit_btn.pack(fill="x", pady=6)

    # -------------------------
    # Events
    # -------------------------
    def _on_threshold_change(self, _=None):
        self.threshold_label.config(text=f"{self.conf_threshold.get():.2f}")

    def start_detection(self):
        self.detection_active = True
        self.metric_status.set("Running")
        self.cam_badge.config(text="ACTIVE", bg="#38BDF8", fg="#0B1120")

    def stop_detection(self):
        self.detection_active = False
        self.metric_status.set("Stopped")
        self.cam_badge.config(text="STOPPED", bg="#475569", fg="#F8FAFC")

    def toggle_detection(self):
        if self.detection_active:
            self.stop_detection()
        else:
            self.start_detection()

    # -------------------------
    # Worker thread
    # -------------------------
    def process_camera(self):
        while self.running:
            ret, frame = self.cap.read()

            if not ret:
                self.metric_status.set("Camera Error")
                time.sleep(0.1)
                continue

            if self.detection_active:
                try:
                    threshold = float(self.conf_threshold.get())
                    results = self.model.predict(
                        source=frame,
                        conf=threshold,
                        imgsz=640,
                        verbose=False
                    )[0]

                    annotated = results.plot()
                    packet = {"frame": annotated}
                except Exception:
                    self.metric_status.set("Inference Error")
                    time.sleep(0.05)
                    continue
            else:
                packet = {"frame": frame}

            if self.frame_queue.full():
                try:
                    self.frame_queue.get_nowait()
                except queue.Empty:
                    pass

            self.frame_queue.put(packet)
            time.sleep(0.005)

    # -------------------------
    # UI update
    # -------------------------
    def update_ui(self):
        if not self.running:
            return

        try:
            packet = self.frame_queue.get_nowait()
        except queue.Empty:
            self.root.after(30, self.update_ui)
            return

        frame = packet.get("frame")
        if frame is None:
            self.root.after(30, self.update_ui)
            return

        try:
            frame_bgr = frame.copy()

            # only keep datetime overlay
            current_time_text = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            overlay = frame_bgr.copy()
            cv2.rectangle(overlay, (0, 0), (320, 45), (11, 17, 32), -1)
            frame_bgr = cv2.addWeighted(overlay, 0.45, frame_bgr, 0.55, 0)

            cv2.putText(
                frame_bgr,
                current_time_text,
                (15, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (230, 230, 230),
                2
            )

            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            pil = Image.fromarray(frame_rgb)

            canvas_w = max(200, self.video_canvas.winfo_width())
            canvas_h = max(120, self.video_canvas.winfo_height())
            pil = pil.resize((canvas_w, canvas_h), Image.LANCZOS)

            tk_img = ImageTk.PhotoImage(pil)
            self.video_canvas.delete("all")
            self.video_canvas.create_image(0, 0, anchor="nw", image=tk_img)
            self.video_canvas.image = tk_img

        except Exception as e:
            self.video_canvas.delete("all")
            self.video_canvas.create_text(
                10, 10,
                anchor="nw",
                text=f"Frame error: {e}",
                fill="#F8FAFC",
                font=("Segoe UI", 12)
            )

        self.root.after(30, self.update_ui)

    # -------------------------
    # Close
    # -------------------------
    def on_close(self):
        self.running = False
        try:
            if self.cap is not None:
                self.cap.release()
        except Exception:
            pass
        self.root.destroy()


if __name__ == "__main__":
    root = tk.Tk()
    app = YOLODashboard(root)
    root.mainloop()

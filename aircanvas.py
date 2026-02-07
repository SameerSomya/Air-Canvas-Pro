import cv2
import mediapipe as mp
import numpy as np
import customtkinter as ctk
from PIL import Image


ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("dark-blue")


class AirCanvas:

    def __init__(self):

        self.app = ctk.CTk()
        self.app.title("Air Canvas Pro")
        self.app.geometry("1200x720")
        self.app.minsize(950, 600)

        # ---------- BACKGROUND ----------
        bg = Image.open("bg.jpeg")

        self.bg_img = ctk.CTkImage(
            light_image=bg,
            dark_image=bg,
            size=(1920,1080)
        )

        bg_label = ctk.CTkLabel(self.app, image=self.bg_img, text="")
        bg_label.place(relwidth=1, relheight=1)

        # ---------- FLOATING FRAME ----------
        self.main_frame = ctk.CTkFrame(
            self.app,
            fg_color="#111111",
            corner_radius=30
        )
        self.main_frame.place(relx=0.5, rely=0.5, anchor="center")

        # ---------- FIXED VIDEO SIZE (prevents crash) ----------
        self.video_width = 960
        self.video_height = 540

        # ---------- CAMERA ----------
        self.cap = None
        self.running = False
        self.canvas_img = None
        self.prev_x, self.prev_y = None, None

        self.draw_color = (255, 0, 255)
        self.brush_size = 7
        self.eraser_size = 60

        # ---------- MEDIAPIPE ----------
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(
            min_detection_confidence=0.8,
            min_tracking_confidence=0.8
        )
        self.mpDraw = mp.solutions.drawing_utils

        # ---------- VIDEO LABEL ----------
        self.video_label = ctk.CTkLabel(self.main_frame, text="")
        self.video_label.pack(padx=25, pady=25)

        # ---------- BUTTONS ----------
        controls = ctk.CTkFrame(self.main_frame, fg_color="transparent")
        controls.pack(pady=10)

        ctk.CTkButton(
            controls,
            text="Start Camera",
            width=140,
            command=self.start_camera
        ).grid(row=0, column=0, padx=10)

        ctk.CTkButton(
            controls,
            text="Clear",
            width=140,
            command=self.clear_canvas
        ).grid(row=0, column=1, padx=10)

        ctk.CTkButton(
            controls,
            text="Stop",
            width=140,
            command=self.stop_camera
        ).grid(row=0, column=2, padx=10)

        # ---------- COLOR PALETTE ----------
        palette = ctk.CTkFrame(self.main_frame, fg_color="transparent")
        palette.pack(pady=10)

        self.colors = [
            (255,0,255),
            (255,255,0),
            (0,255,0),
            (0,140,255),
            (0,0,255)
        ]

        for i, color in enumerate(self.colors):

            btn = ctk.CTkButton(
                palette,
                text="",
                width=40,
                height=40,
                corner_radius=20,
                fg_color=self.rgb_to_hex(color),
                hover_color=self.rgb_to_hex(color),
                command=lambda c=color: self.set_color(c)
            )
            btn.grid(row=0, column=i, padx=8)

        # ---------- BRUSH SLIDER ----------
        slider_frame = ctk.CTkFrame(self.main_frame)
        slider_frame.pack(pady=15)

        ctk.CTkLabel(slider_frame, text="Brush Size").pack()

        self.slider = ctk.CTkSlider(
            slider_frame,
            from_=1,
            to=35,
            command=self.change_brush
        )
        self.slider.set(7)
        self.slider.pack(padx=20, pady=5)

    # ---------- UTIL ----------

    def rgb_to_hex(self, color):
        return '#%02x%02x%02x' % (color[2], color[1], color[0])

    def set_color(self, color):
        self.draw_color = color

    def change_brush(self, value):
        self.brush_size = int(value)

    # ---------- CAMERA ----------

    def start_camera(self):
        if not self.running:

            self.cap = cv2.VideoCapture(0)

            # limit resolution → better performance
            self.cap.set(3, 1280)
            self.cap.set(4, 720)

            self.running = True

            # delay prevents GUI race condition
            self.app.after(120, self.update_frame)

    def stop_camera(self):
        self.running = False
        if self.cap:
            self.cap.release()

    def clear_canvas(self):
        self.canvas_img = None

    # ---------- GESTURE ----------

    def fingers_closed(self, lmList):
        tips = [8, 12, 16, 20]
        return sum(lmList[tip][2] > lmList[tip-2][2] for tip in tips) >= 3

    # ---------- FRAME LOOP ----------

    def update_frame(self):

        if not self.running:
            return

        success, img = self.cap.read()
        img = cv2.flip(img, 1)

        if self.canvas_img is None:
            self.canvas_img = np.zeros_like(img)

        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.hands.process(imgRGB)

        if results.multi_hand_landmarks:

            hand = results.multi_hand_landmarks[0]
            lmList = []

            for id, lm in enumerate(hand.landmark):
                h, w, _ = img.shape
                lmList.append((id, int(lm.x*w), int(lm.y*h)))

            self.mpDraw.draw_landmarks(
                img, hand, self.mpHands.HAND_CONNECTIONS)

            x1, y1 = lmList[4][1], lmList[4][2]
            x2, y2 = lmList[8][1], lmList[8][2]

            distance = np.hypot(x2-x1, y2-y1)

            if distance < 35:

                if self.prev_x is None:
                    self.prev_x, self.prev_y = x2, y2

                cv2.line(
                    self.canvas_img,
                    (self.prev_x, self.prev_y),
                    (x2, y2),
                    self.draw_color,
                    self.brush_size
                )

                self.prev_x, self.prev_y = x2, y2

            elif self.fingers_closed(lmList):

                cv2.circle(
                    self.canvas_img,
                    (x2, y2),
                    self.eraser_size,
                    (0,0,0),
                    -1
                )

                self.prev_x, self.prev_y = x2, y2

            else:
                self.prev_x, self.prev_y = None, None

        # ---------- MERGE ----------
        gray = cv2.cvtColor(self.canvas_img, cv2.COLOR_BGR2GRAY)
        _, inv = cv2.threshold(gray, 20, 255, cv2.THRESH_BINARY_INV)
        inv = cv2.cvtColor(inv, cv2.COLOR_GRAY2BGR)

        img = cv2.bitwise_and(img, inv)
        img = cv2.bitwise_or(img, self.canvas_img)

        # ---------- FIXED SIZE RENDER ----------
        img = cv2.resize(img, (self.video_width, self.video_height))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)

        ctk_img = ctk.CTkImage(
            light_image=img,
            dark_image=img,
            size=(self.video_width, self.video_height)
        )

        self.video_label.configure(image=ctk_img)
        self.video_label.image = ctk_img

        self.app.after(10, self.update_frame)

    def run(self):
        self.app.mainloop()


AirCanvas().run()

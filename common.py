"""
common.py - tudo que e compartilhado entre o menu e os dois jogos:
camera (picamera2), deteccao ArUco + calibracao por homografia, a thread
de deteccao de mao (MediaPipe), e um punhado de utilitarios de desenho
usados nas telas de espera/fim de cada jogo.

Nao roda nada sozinho -- e importado por main.py, menu.py, fruit_ninja/game.py
e breakout/game.py.
"""

import os
import time
import threading

import cv2
import numpy as np
import mediapipe as mp
from picamera2 import Picamera2

# ----------------------------- CONFIG GERAL ------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))

GAME_W, GAME_H = 1280, 720
CAM_W, CAM_H = 640, 480
SWAP_RB = False
CALIB_FILE = os.path.join(_HERE, "homography.npy")
WINDOW = "game"

TARGET_FPS = 60
GESTURE_HOLD_S = 2
BOUNDS_MARGIN = 25

HOME_CENTER = (70, 70)     # canto superior esquerdo, no espaco do jogo
HOME_RADIUS = 55
HOME_HOLD_S = 1.2          # mais longo: e uma acao "destrutiva" (sai do jogo)

mp_hands = mp.solutions.hands

# ------------------------- CAMERA (picamera2) ------------------------
def make_camera():
    picam2 = Picamera2()
    picam2.configure(picam2.create_preview_configuration(
        main={"format": "RGB888", "size": (CAM_W, CAM_H)}))
    picam2.start()
    time.sleep(0.5)
    return picam2

def grab(picam2):
    frame = picam2.capture_array()
    if SWAP_RB:
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    return frame

# ------------------------- ARUCO (compat) -----------------------------
def get_aruco():
    aruco = cv2.aruco
    dictionary = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
    try:
        detector = aruco.ArucoDetector(dictionary, aruco.DetectorParameters())
        def detect(gray):
            corners, ids, _ = detector.detectMarkers(gray)
            return corners, ids
        def make_marker(i, size):
            return aruco.generateImageMarker(dictionary, i, size)
    except AttributeError:
        params = aruco.DetectorParameters_create()
        def detect(gray):
            corners, ids, _ = aruco.detectMarkers(gray, dictionary, parameters=params)
            return corners, ids
        def make_marker(i, size):
            return aruco.drawMarker(dictionary, i, size)
    return detect, make_marker

# ------------------------- CALIBRACAO ----------------------------------
def build_calib_canvas(make_marker, marker_px=160, quiet=40, margin=140):
    canvas = np.zeros((GAME_H, GAME_W, 3), dtype=np.uint8)
    pad = marker_px + 2 * quiet
    targets = [
        (margin, margin),
        (GAME_W - margin, margin),
        (GAME_W - margin, GAME_H - margin),
        (margin, GAME_H - margin),
    ]
    for i, (cx, cy) in enumerate(targets):
        m = cv2.cvtColor(make_marker(i, marker_px), cv2.COLOR_GRAY2BGR)
        px0, py0 = cx - pad // 2, cy - pad // 2
        canvas[py0:py0 + pad, px0:px0 + pad] = 255
        mx0, my0 = cx - marker_px // 2, cy - marker_px // 2
        canvas[my0:my0 + marker_px, mx0:mx0 + marker_px] = m
    return canvas, np.array(targets, dtype=np.float32)

def calibrate(picam2, detect, make_marker):
    base, game_targets = build_calib_canvas(make_marker)
    print("Calibrando... deixe a mesa livre. ESC cancela.")
    H = None
    last_report = 0.0
    while True:
        frame = grab(picam2)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids = detect(gray)

        found = sorted(int(i) for i in ids.flatten()) if ids is not None else []
        if time.time() - last_report > 1.0:
            print(f"marcadores detectados: {len(found)}/4  ids={found}")
            last_report = time.time()

        canvas = base.copy()
        _centered_text(canvas, f"Calibrando: {len(found)}/4",
                       GAME_H // 2, 1.4, (0, 200, 255), 3)
        cv2.imshow(WINDOW, canvas)
        key = cv2.waitKey(30) & 0xFF
        if key == 27:
            break

        if ids is not None:
            centers = {int(i): c.reshape(4, 2).mean(axis=0)
                       for c, i in zip(corners, ids.flatten())}
            if all(i in centers for i in range(4)):
                cam_pts = np.array([centers[i] for i in range(4)], dtype=np.float32)
                H = cv2.getPerspectiveTransform(cam_pts, game_targets)
                np.save(CALIB_FILE, H)
                print("Calibracao OK -> salva em", CALIB_FILE)
                break
    return H

def cam_to_game(pts, H):
    arr = np.array([pts], dtype=np.float32)
    return cv2.perspectiveTransform(arr, H)[0]

def extended_fingers(lm):
    wrist = np.array([lm[0].x, lm[0].y])
    joints = {"index": (8, 6), "middle": (12, 10), "ring": (16, 14), "pinky": (20, 18)}
    out = {}
    for name, (tip, pip) in joints.items():
        d_tip = np.linalg.norm(np.array([lm[tip].x, lm[tip].y]) - wrist)
        d_pip = np.linalg.norm(np.array([lm[pip].x, lm[pip].y]) - wrist)
        out[name] = d_tip > d_pip
    return out

def is_fist(lm):
    f = extended_fingers(lm)
    return not f["index"] and not f["middle"] and not f["ring"] and not f["pinky"]

def is_open_hand(lm):
    f = extended_fingers(lm)
    return sum(f.values()) >= 3

def in_bounds(pt, margin=BOUNDS_MARGIN):
    x, y = pt
    return -margin <= x <= GAME_W + margin and -margin <= y <= GAME_H + margin

# ------------------- THREAD DE DETECCAO DA MAO (compartilhada) --------
class HandTracker(threading.Thread):
    """Uma unica thread serve o menu e os dois jogos: publica tudo que
    qualquer consumidor possa precisar; cada um le so o que usa.
      - get_point()  -> ponta do indicador (menu e hotspot "voltar ao menu")
      - get_paddle() -> (indicador, polegar), usado pelo Breakout
      - get_blade()  -> (ponto_anterior, ponto_atual), usado pelo Fruit Ninja
      - is_fist_gesture() / is_open_hand_gesture() -> gestos de inicio
    """
    def __init__(self, picam2, H):
        super().__init__(daemon=True)
        self.picam2 = picam2
        self.H = H
        self._point = None
        self._paddle = None
        self._blade_prev = None
        self._blade_cur = None
        self._blade_ts = 0.0
        self._is_fist = False
        self._is_open = False
        self._lock = threading.Lock()
        self.running = True
        self.fps = 0.0

    def run(self):
        with mp_hands.Hands(model_complexity=0,
                            max_num_hands=1,
                            min_detection_confidence=0.5,
                            min_tracking_confidence=0.5) as hands:
            t_prev = time.time()
            while self.running:
                frame = grab(self.picam2)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                rgb.flags.writeable = False
                results = hands.process(rgb)

                point = paddle = None
                fist = openh = False
                if results.multi_hand_landmarks:
                    lm = results.multi_hand_landmarks[0].landmark
                    h, w = rgb.shape[:2]
                    ix = lm[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * w
                    iy = lm[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * h
                    tx = lm[mp_hands.HandLandmark.THUMB_TIP].x * w
                    ty = lm[mp_hands.HandLandmark.THUMB_TIP].y * h
                    pts = cam_to_game([(ix, iy), (tx, ty)], self.H)
                    p_i = (float(pts[0][0]), float(pts[0][1]))
                    p_t = (float(pts[1][0]), float(pts[1][1]))
                    if in_bounds(p_i):
                        point = p_i
                        fist = is_fist(lm)
                        openh = is_open_hand(lm)
                        if in_bounds(p_t):
                            paddle = (p_i, p_t)

                with self._lock:
                    self._paddle = paddle
                    self._is_fist = fist
                    self._is_open = openh
                    if point is not None:
                        self._blade_prev = self._blade_cur
                        self._blade_cur = point
                        self._blade_ts = time.time()
                    else:
                        self._blade_prev = None
                        self._blade_cur = None
                    self._point = point

                now = time.time()
                self.fps = 0.9 * self.fps + 0.1 * (1.0 / max(now - t_prev, 1e-3))
                t_prev = now

    def get_point(self):
        with self._lock:
            return self._point

    def get_paddle(self):
        with self._lock:
            return self._paddle

    def get_blade(self):
        with self._lock:
            if self._blade_prev is None or self._blade_cur is None:
                return None
            if time.time() - self._blade_ts > 0.25:
                return None
            return self._blade_prev, self._blade_cur

    def is_fist_gesture(self):
        with self._lock:
            return self._is_fist

    def is_open_hand_gesture(self):
        with self._lock:
            return self._is_open

    def stop(self):
        self.running = False

# ----------------------- TELAS (utilitarios comuns) --------------------
def _centered_text(canvas, text, y, scale, color, thick):
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
    x = (GAME_W - tw) // 2
    cv2.putText(canvas, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)
    return th

def _hold_bar(canvas, progress, y0=None):
    if progress <= 0:
        return
    bw, bh = 400, 22
    x0 = (GAME_W - bw) // 2
    if y0 is None:
        y0 = GAME_H // 2 + 110
    cv2.rectangle(canvas, (x0, y0), (x0 + bw, y0 + bh), (120, 120, 120), 2)
    fill = int(bw * min(progress, 1.0))
    cv2.rectangle(canvas, (x0, y0), (x0 + fill, y0 + bh), (0, 255, 0), -1)

def format_time(seconds):
    m = int(seconds) // 60
    s = seconds - m * 60
    return f"{m:02d}:{s:05.2f}"

def draw_home_hotspot(canvas, progress):
    """Circulo discreto no canto p/ voltar ao menu. Enche conforme segura."""
    cx, cy = HOME_CENTER
    cv2.circle(canvas, (cx, cy), HOME_RADIUS, (90, 90, 90), 2)
    cv2.putText(canvas, "M", (cx - 10, cy + 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (110, 110, 110), 2, cv2.LINE_AA)
    if progress > 0:
        angle = int(360 * min(progress, 1.0))
        cv2.ellipse(canvas, (cx, cy), (HOME_RADIUS, HOME_RADIUS), -90, 0, angle,
                    (0, 255, 0), 4)

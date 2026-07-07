#!/usr/bin/env python3
"""
AR Fruit Ninja - versao projetor top-down (Raspberry Pi 5 + picamera2)
=======================================================================

Reaproveita a MESMA arquitetura do seu AR Breakout:
  - Projetor mira de cima pra baixo numa mesa (superficie plana).
  - PiCamera fixa junto do projetor, olhando pra MESMA area da mesa.
  - Camera e so SENSOR (nunca aparece na projecao); jogo desenhado
    num canvas proprio e a homografia converte mao (espaco camera)
    -> espaco do jogo, exatamente como no Breakout.

O que muda em relacao ao Breakout:
  - Em vez de uma "raquete" fixa (linha indicador-polegar), a lamina
    e o SEGMENTO entre a posicao do indicador no frame anterior e no
    frame atual -- ou seja, o proprio movimento de "cortar" no ar.
  - Em vez de uma bola quicando, ha uma lista de frutas que sobem e
    caem com gravidade (fisica de canvas 2D, igual a bola do Breakout).
  - Bombas encerram o jogo na hora se cortadas.

Assets opcionais (mesmos nomes do ai-fruit-ninja original):
  ./assets/sandia.png   (melancia)
  ./assets/naranja.png  (laranja)
  ./assets/banana.png
  ./assets/bomb.png
  Precisam ter canal alpha (PNG transparente). Se nao existirem,
  o jogo cai automaticamente pra circulos coloridos -- funciona sem
  nenhum asset, so fica menos bonito.

Dependencias (Pi OS Bookworm 64-bit):
  sudo apt install -y python3-picamera2
  pip install mediapipe opencv-contrib-python numpy

Uso:
  python3 ar_fruit_ninja_projector.py
  - 1a vez: calibracao com 4 marcadores ArUco (igual ao Breakout).
  - Para recalibrar: apague homography.npy
  - Mao aberta (parada) = comecar/reiniciar. ESC sai. 'r' reinicio manual.
"""

import os
import time
import random
import threading
from collections import deque

import cv2
import numpy as np
import mediapipe as mp
from picamera2 import Picamera2

# ----------------------------- CONFIG -----------------------------
GAME_W, GAME_H = 1280, 720
CAM_W, CAM_H = 640, 480
SWAP_RB = False
CALIB_FILE = "homography.npy"
WINDOW = "game"
ASSET_DIR = "assets"
BEST_SCORE_FILE = "bestscore.txt"

TARGET_FPS = 30
GESTURE_HOLD_S = 0.6
BOUNDS_MARGIN = 25

FRUIT_RADIUS = 45
GRAVITY = 0.35              # px/frame^2 (mesma escala do Breakout)
SPAWN_MIN_S, SPAWN_MAX_S = 0.7, 1.3
BOMB_CHANCE = 0.15
LIVES_START = 3
SLICE_TOLERANCE = 10        # folga (px) alem do raio da fruta pra contar corte
MIN_SLICE_SPEED = 4.0       # px/frame minimo entre frames p/ contar como "corte"
                             # (evita fatiar so encostando parado)

mp_hands = mp.solutions.hands

FRUIT_KINDS = {
    "sandia":  {"color": (60, 60, 220),  "points": 1},   # melancia (BGR)
    "naranja": {"color": (0, 140, 255),  "points": 1},   # laranja
    "banana":  {"color": (0, 220, 255),  "points": 2},   # banana (menor, mais rara)
}
BOMB_COLOR = (40, 40, 40)

# ------------------------- CAMERA (picamera2) ---------------------
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

# ------------------------- ARUCO (compat) -------------------------
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

# ------------------------- CALIBRACAO -----------------------------
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

def is_open_hand(lm):
    """Gesto de comecar/reiniciar: mao aberta (3+ dedos estendidos, fora o polegar)."""
    f = extended_fingers(lm)
    return sum(f.values()) >= 3

def in_bounds(pt, margin=BOUNDS_MARGIN):
    x, y = pt
    return -margin <= x <= GAME_W + margin and -margin <= y <= GAME_H + margin

# ------------------- THREAD DE DETECCAO DA MAO --------------------
class HandTracker(threading.Thread):
    """Roda camera + MediaPipe em paralelo; publica a LAMINA (segmento de
    movimento do indicador entre duas leituras) no espaco do jogo."""
    def __init__(self, picam2, H):
        super().__init__(daemon=True)
        self.picam2 = picam2
        self.H = H
        self._prev_point = None
        self._cur_point = None
        self._blade_ts = 0.0
        self._open_hand = False
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

                open_hand = False
                new_point = None
                if results.multi_hand_landmarks:
                    lm = results.multi_hand_landmarks[0].landmark
                    h, w = rgb.shape[:2]
                    ix = lm[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * w
                    iy = lm[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * h
                    p = cam_to_game([(ix, iy)], self.H)
                    p_i = (float(p[0][0]), float(p[0][1]))
                    if in_bounds(p_i):
                        new_point = p_i
                        open_hand = is_open_hand(lm)

                with self._lock:
                    if new_point is not None:
                        self._prev_point = self._cur_point
                        self._cur_point = new_point
                        self._blade_ts = time.time()
                    else:
                        # perdeu a mao: zera a trilha p/ nao "congelar" uma lamina antiga
                        self._prev_point = None
                        self._cur_point = None
                    self._open_hand = open_hand

                now = time.time()
                self.fps = 0.9 * self.fps + 0.1 * (1.0 / max(now - t_prev, 1e-3))
                t_prev = now

    def get_blade(self):
        """Retorna (ponto_anterior, ponto_atual) ou None se nao ha lamina valida."""
        with self._lock:
            if self._prev_point is None or self._cur_point is None:
                return None
            if time.time() - self._blade_ts > 0.25:   # leitura muito velha: ignora
                return None
            return self._prev_point, self._cur_point

    def is_open_hand_gesture(self):
        with self._lock:
            return self._open_hand

    def stop(self):
        self.running = False

# ------------------------- ASSETS (opcional) -----------------------
def load_assets():
    """Carrega PNGs com alpha se existirem; senao retorna dict vazio
    (o jogo cai pra circulos coloridos automaticamente).
    IMPORTANTE: ja redimensiona aqui (uma vez so, no load) pro tamanho final
    de tela (FRUIT_RADIUS*2). Antes o resize rodava a cada frame por fruta
    dentro de overlay_transparent, o que gerava lag desnecessario no Pi."""
    files = {"sandia": "sandia.png", "naranja": "naranja.png", "banana": "banana.png",
             "bomb": "bomb.png"}
    box_size = FRUIT_RADIUS * 2
    imgs = {}
    for kind, fname in files.items():
        path = os.path.join(ASSET_DIR, fname)
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is not None and img.shape[2] == 4:
            imgs[kind] = cv2.resize(img, (box_size, box_size))
        elif img is not None:
            print(f"aviso: {path} sem canal alpha, ignorando (use PNG transparente)")
    return imgs

def overlay_transparent(background, overlay, cx, cy):
    """Desenha overlay (BGRA) ja no tamanho final, centralizado em (cx, cy).
    Faz clipping nas bordas do canvas. Sem resize aqui -- isso agora e feito
    uma unica vez em load_assets()."""
    if overlay is None:
        return
    ov = overlay
    h, w = ov.shape[:2]
    x, y = int(cx - w / 2), int(cy - h / 2)
    x0, y0 = max(x, 0), max(y, 0)
    x1, y1 = min(x + w, background.shape[1]), min(y + h, background.shape[0])
    if x1 <= x0 or y1 <= y0:
        return
    ox0, oy0 = x0 - x, y0 - y
    ox1, oy1 = ox0 + (x1 - x0), oy0 + (y1 - y0)
    crop = ov[oy0:oy1, ox0:ox1]
    region = background[y0:y1, x0:x1]
    alpha = crop[:, :, 3:4].astype(np.float32) / 255.0
    rgb = crop[:, :, :3].astype(np.float32)
    blended = (rgb * alpha + region.astype(np.float32) * (1 - alpha)).astype(np.uint8)
    background[y0:y1, x0:x1] = blended

# ----------------------- LOGICA DO JOGO ----------------------------
class Fruit:
    def __init__(self, x, y, vx, vy, kind):
        self.x, self.y = x, y
        self.vx, self.vy = vx, vy
        self.kind = kind
        self.radius = FRUIT_RADIUS

    def update(self):
        self.vy += GRAVITY
        self.x += self.vx
        self.y += self.vy

    @property
    def pos(self):
        return (self.x, self.y)

def spawn_fruit():
    is_bomb = random.random() < BOMB_CHANCE
    kind = "bomb" if is_bomb else random.choice(list(FRUIT_KINDS.keys()))
    x = random.uniform(GAME_W * 0.15, GAME_W * 0.85)
    y = GAME_H + FRUIT_RADIUS
    apex_height = random.uniform(GAME_H * 0.35, GAME_H * 0.70)
    vy0 = -np.sqrt(2 * GRAVITY * apex_height)     # velocidade inicial p/ atingir o apice
    vx = random.uniform(-2.5, 2.5)
    return Fruit(x, y, vx, vy0, kind)

def point_segment_distance(p, a, b):
    """Distancia do ponto p ao segmento a-b (todos tuplas (x,y))."""
    p, a, b = np.array(p, dtype=np.float64), np.array(a, dtype=np.float64), np.array(b, dtype=np.float64)
    ab = b - a
    len_sq = float(np.dot(ab, ab))
    if len_sq < 1e-6:
        return float(np.linalg.norm(p - a))
    t = np.clip(np.dot(p - a, ab) / len_sq, 0, 1)
    closest = a + t * ab
    return float(np.linalg.norm(p - closest))

def draw_fruit(canvas, fruit, assets):
    img = assets.get(fruit.kind)
    if img is not None:
        overlay_transparent(canvas, img, fruit.x, fruit.y)
    elif fruit.kind == "bomb":
        cv2.circle(canvas, (int(fruit.x), int(fruit.y)), fruit.radius, BOMB_COLOR, -1)
        cv2.putText(canvas, "X", (int(fruit.x) - 12, int(fruit.y) + 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3, cv2.LINE_AA)
    else:
        color = FRUIT_KINDS[fruit.kind]["color"]
        cv2.circle(canvas, (int(fruit.x), int(fruit.y)), fruit.radius, color, -1)

# ----------------------- TELAS ------------------------------------
def _centered_text(canvas, text, y, scale, color, thick):
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
    x = (GAME_W - tw) // 2
    cv2.putText(canvas, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)
    return th

def _hold_bar(canvas, progress):
    if progress <= 0:
        return
    bw, bh = 400, 22
    x0 = (GAME_W - bw) // 2
    y0 = GAME_H // 2 + 110
    cv2.rectangle(canvas, (x0, y0), (x0 + bw, y0 + bh), (120, 120, 120), 2)
    fill = int(bw * min(progress, 1.0))
    cv2.rectangle(canvas, (x0, y0), (x0 + fill, y0 + bh), (0, 255, 0), -1)

def start_screen(canvas, progress):
    y = GAME_H // 2
    _centered_text(canvas, "AR Fruit Ninja", y - 60, 2, (255, 255, 255), 3)
    _centered_text(canvas, "Mao aberta para comecar", y + 20, 1.1, (0, 255, 0), 2)
    _hold_bar(canvas, progress)

def game_over_screen(canvas, progress, score, is_new_record):
    y = GAME_H // 2
    _centered_text(canvas, "Game Over", y - 60, 2, (0, 0, 255), 3)
    _centered_text(canvas, f"Pontos: {score}", y - 10, 1.2, (255, 255, 255), 2)
    if is_new_record:
        _centered_text(canvas, "NOVO RECORDE!", y + 30, 1.0, (0, 215, 255), 2)
    _centered_text(canvas, "Mao aberta para jogar de novo", y + 75, 1, (255, 255, 255), 2)
    _hold_bar(canvas, progress)

def load_best_score():
    try:
        with open(BEST_SCORE_FILE) as f:
            return int(f.read().strip())
    except (FileNotFoundError, ValueError):
        return None

def save_best_score(value):
    try:
        with open(BEST_SCORE_FILE, "w") as f:
            f.write(str(value))
    except OSError as e:
        print("Nao foi possivel salvar o recorde:", e)

def draw_hud(canvas, score, lives, best_score, tracker_fps):
    cv2.putText(canvas, f"Pontos: {score}", (16, GAME_H - 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    text = f"Recorde: {best_score}" if best_score is not None else "Recorde: --"
    (tw, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
    cv2.putText(canvas, text, (GAME_W - tw - 16, GAME_H - 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 215, 255), 2, cv2.LINE_AA)
    cv2.putText(canvas, f"Vidas: {'*' * lives}", (16, 34),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 80, 80), 2, cv2.LINE_AA)
    cv2.putText(canvas, f"hand {tracker_fps:4.1f} fps", (GAME_W - 130, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (90, 90, 90), 1, cv2.LINE_AA)

# ----------------------------- MAIN -------------------------------
def main():
    picam2 = make_camera()
    detect, make_marker = get_aruco()
    assets = load_assets()
    if assets:
        print(f"Assets carregados: {list(assets.keys())}")
    else:
        print("Nenhum asset em ./assets -- usando circulos coloridos.")

    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(WINDOW, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    if os.path.exists(CALIB_FILE):
        H = np.load(CALIB_FILE)
        print("Homografia carregada de", CALIB_FILE, "(apague o arquivo p/ recalibrar)")
    else:
        H = calibrate(picam2, detect, make_marker)

    if H is None:
        print("Sem calibracao. Encerrando.")
        picam2.stop()
        cv2.destroyAllWindows()
        return

    tracker = HandTracker(picam2, H)
    tracker.start()

    state = "waiting"          # "waiting" | "playing" | "over"
    score = 0
    lives = LIVES_START
    is_new_record = False
    fruits = []
    next_spawn_at = 0.0
    gesture_since = None
    best_score = load_best_score()
    target_dt = 1.0 / TARGET_FPS
    trail = deque(maxlen=8)    # so pra desenhar o rastro da lamina

    def gesture_progress():
        if gesture_since is None:
            return 0.0
        return (time.time() - gesture_since) / GESTURE_HOLD_S

    def reset_game():
        nonlocal score, lives, fruits, next_spawn_at
        score = 0
        lives = LIVES_START
        fruits = []
        next_spawn_at = time.time() + random.uniform(SPAWN_MIN_S, SPAWN_MAX_S)

    while True:
        t0 = time.time()
        canvas = np.zeros((GAME_H, GAME_W, 3), dtype=np.uint8)

        # ---- Gesto de iniciar/reiniciar (so nas telas de espera/fim) ----
        if state in ("waiting", "over"):
            if tracker.is_open_hand_gesture():
                if gesture_since is None:
                    gesture_since = time.time()
                elif time.time() - gesture_since >= GESTURE_HOLD_S:
                    reset_game()
                    state = "playing"
                    is_new_record = False
                    gesture_since = None
            else:
                gesture_since = None

        # ---- Pega a lamina atual (segmento prev->cur no espaco do jogo) ----
        blade = tracker.get_blade()
        if blade is not None:
            trail.append(blade[1])

        # ---- Render por estado ----
        if state == "waiting":
            start_screen(canvas, gesture_progress())

        elif state == "over":
            game_over_screen(canvas, gesture_progress(), score, is_new_record)

        else:  # playing
            now = time.time()
            if now >= next_spawn_at:
                fruits.append(spawn_fruit())
                next_spawn_at = now + random.uniform(SPAWN_MIN_S, SPAWN_MAX_S)

            for fruit in fruits[:]:
                fruit.update()

                sliced = False
                if blade is not None:
                    dist = point_segment_distance(fruit.pos, blade[0], blade[1])
                    if dist <= fruit.radius + SLICE_TOLERANCE:
                        sliced = True

                if sliced:
                    fruits.remove(fruit)
                    if fruit.kind == "bomb":
                        state = "over"
                        if best_score is None or score > best_score:
                            best_score = score
                            save_best_score(best_score)
                            is_new_record = True
                        else:
                            is_new_record = False
                    else:
                        score += FRUIT_KINDS[fruit.kind]["points"]
                    continue

                # fruta caiu sem ser cortada
                if fruit.y - fruit.radius > GAME_H:
                    fruits.remove(fruit)
                    if fruit.kind != "bomb":
                        lives -= 1
                        if lives <= 0:
                            state = "over"
                            if best_score is None or score > best_score:
                                best_score = score
                                save_best_score(best_score)
                                is_new_record = True
                            else:
                                is_new_record = False

            for fruit in fruits:
                draw_fruit(canvas, fruit, assets)

            # rastro da lamina (visual, ultimos N pontos)
            pts = list(trail)
            for i in range(1, len(pts)):
                cv2.line(canvas, (int(pts[i-1][0]), int(pts[i-1][1])),
                          (int(pts[i][0]), int(pts[i][1])), (255, 255, 255), 4)

        draw_hud(canvas, score, lives, best_score, tracker.fps)

        cv2.imshow(WINDOW, canvas)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break
        elif key == ord('r'):
            reset_game()
            state, gesture_since = "playing", None
            is_new_record = False

        dt = time.time() - t0
        if dt < target_dt:
            time.sleep(target_dt - dt)

    tracker.stop()
    time.sleep(0.2)
    picam2.stop()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
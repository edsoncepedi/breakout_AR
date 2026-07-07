#!/usr/bin/env python3
"""
AR Breakout - versao projetor top-down (Raspberry Pi 5 + picamera2)
==================================================================

Setup fisico assumido:
  - Projetor mira de cima pra baixo numa mesa (superficie plana).
  - PiCamera fixa junto do projetor, olhando pra MESMA area da mesa.
  - A camera NAO pode se mexer depois de calibrar, senao a homografia quebra.

Como funciona:
  - A camera e apenas SENSOR: detecta a mao. Ela nunca aparece na projecao.
  - O jogo e desenhado num canvas proprio 1280x720 (fundo preto) e projetado.
  - Uma homografia converte a posicao da mao (espaco da camera) -> espaco do jogo.
  - A "raquete" e a linha entre a ponta do indicador e a ponta do polegar.

Dependencias (Pi OS Bookworm 64-bit):
  sudo apt install -y python3-picamera2
  pip install mediapipe opencv-contrib-python numpy   # de preferencia num venv

Uso:
  python3 ar_breakout_projector.py
  - Na 1a vez roda a calibracao: 4 marcadores ArUco aparecem nos cantos.
    Deixe a mesa livre; assim que os 4 forem detectados, salva homography.npy.
  - Para recalibrar depois: apague o arquivo homography.npy.
  - Em jogo: 'r' reinicia, ESC sai.

Config do projetor: configure-o como tela ESTENDIDA (nao espelhada) e deixe
esta janela na tela do projetor antes de rodar (ou ajuste seu WM).
"""

import os
import time
import threading

import cv2
import numpy as np
import mediapipe as mp
from picamera2 import Picamera2

# ----------------------------- CONFIG -----------------------------
GAME_W, GAME_H = 1280, 720      # resolucao nativa do projetor = canvas do jogo
CAM_W, CAM_H = 640, 480         # resolucao de captura (baixa = deteccao mais rapida)
SWAP_RB = False                 # ligue (True) se as cores vierem trocadas do picamera2
CALIB_FILE = "homography.npy"
WINDOW = "game"

# Propriedades do jogo (iguais ao original)
ball_radius = 20
ball_color = (0, 255, 0)        # verde (BGR)
box_size = (80, 40)             # largura, altura
gap = 5

TARGET_FPS = 60                 # trava o loop principal p/ velocidade consistente da bola
BALL_SPEED = 7.0                # velocidade da bola em px/frame (menor = mais facil)
GESTURE_HOLD_S = 0.6            # tempo segurando a mao aberta p/ iniciar/reiniciar

mp_hands = mp.solutions.hands

# ------------------------- CAMERA (picamera2) ---------------------
def make_camera():
    picam2 = Picamera2()
    picam2.configure(picam2.create_preview_configuration(
        main={"format": "RGB888", "size": (CAM_W, CAM_H)}))
    picam2.start()
    time.sleep(0.5)  # deixa o auto-exposure estabilizar
    return picam2

def grab(picam2):
    """Retorna um frame em convencao BGR (padrao do OpenCV)."""
    frame = picam2.capture_array()
    if SWAP_RB:
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    return frame

# ------------------------- ARUCO (compat) -------------------------
def get_aruco():
    aruco = cv2.aruco
    dictionary = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
    try:  # OpenCV >= 4.7
        detector = aruco.ArucoDetector(dictionary, aruco.DetectorParameters())
        def detect(gray):
            corners, ids, _ = detector.detectMarkers(gray)
            return corners, ids
        def make_marker(i, size):
            return aruco.generateImageMarker(dictionary, i, size)
    except AttributeError:  # OpenCV < 4.7
        params = aruco.DetectorParameters_create()
        def detect(gray):
            corners, ids, _ = aruco.detectMarkers(gray, dictionary, parameters=params)
            return corners, ids
        def make_marker(i, size):
            return aruco.drawMarker(dictionary, i, size)
    return detect, make_marker

# ------------------------- CALIBRACAO -----------------------------
def build_calib_canvas(make_marker, marker_px=160, quiet=40, margin=140):
    """Canvas com 4 marcadores (ids 0..3) em cantos conhecidos do espaco do jogo.
    IMPORTANTE: cada marcador precisa de uma borda BRANCA em volta (quiet zone),
    senao o detector nao consegue separar o marcador (borda preta) do fundo preto."""
    canvas = np.zeros((GAME_H, GAME_W, 3), dtype=np.uint8)
    pad = marker_px + 2 * quiet
    targets = [
        (margin, margin),                    # id 0: sup-esq
        (GAME_W - margin, margin),           # id 1: sup-dir
        (GAME_W - margin, GAME_H - margin),  # id 2: inf-dir
        (margin, GAME_H - margin),           # id 3: inf-esq
    ]
    for i, (cx, cy) in enumerate(targets):
        m = cv2.cvtColor(make_marker(i, marker_px), cv2.COLOR_GRAY2BGR)
        # quadrado branco (quiet zone) atras do marcador
        px0, py0 = cx - pad // 2, cy - pad // 2
        canvas[py0:py0 + pad, px0:px0 + pad] = 255
        # marcador centralizado sobre o branco
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
        if time.time() - last_report > 1.0:      # feedback no terminal (1x/s)
            print(f"marcadores detectados: {len(found)}/4  ids={found}")
            last_report = time.time()

        # feedback na propria projecao
        canvas = base.copy()
        _centered_text(canvas, f"Calibrando: {len(found)}/4",
                       GAME_H // 2, 1.4, (0, 200, 255), 3)
        cv2.imshow(WINDOW, canvas)
        key = cv2.waitKey(30) & 0xFF
        if key == 27:  # ESC
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
    """Converte lista de pontos (espaco camera) -> espaco do jogo."""
    arr = np.array([pts], dtype=np.float32)      # shape (1, N, 2)
    return cv2.perspectiveTransform(arr, H)[0]   # N pontos (x, y)

def count_extended_fingers(lm):
    """Conta dedos estendidos (ignora o polegar). Robusto a rotacao no plano:
    o dedo esta estendido se a ponta esta mais longe do pulso que a junta PIP."""
    wrist = np.array([lm[0].x, lm[0].y])
    count = 0
    for tip, pip in [(8, 6), (12, 10), (16, 14), (20, 18)]:
        d_tip = np.linalg.norm(np.array([lm[tip].x, lm[tip].y]) - wrist)
        d_pip = np.linalg.norm(np.array([lm[pip].x, lm[pip].y]) - wrist)
        if d_tip > d_pip:
            count += 1
    return count

# ------------------- THREAD DE DETECCAO DA MAO --------------------
class HandTracker(threading.Thread):
    """Roda camera + MediaPipe em paralelo; publica a raquete no espaco do jogo."""
    def __init__(self, picam2, H):
        super().__init__(daemon=True)
        self.picam2 = picam2
        self.H = H
        self._paddle = None          # (ponto_indicador, ponto_polegar) ou None
        self._open_hand = False      # mao aberta (gesto de iniciar/reiniciar)
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
                frame = grab(self.picam2)                    # BGR
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                rgb.flags.writeable = False
                results = hands.process(rgb)

                paddle = None
                open_hand = False
                if results.multi_hand_landmarks:
                    lm = results.multi_hand_landmarks[0].landmark
                    h, w = rgb.shape[:2]
                    ix = lm[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * w
                    iy = lm[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * h
                    tx = lm[mp_hands.HandLandmark.THUMB_TIP].x * w
                    ty = lm[mp_hands.HandLandmark.THUMB_TIP].y * h
                    p = cam_to_game([(ix, iy), (tx, ty)], self.H)
                    paddle = ((float(p[0][0]), float(p[0][1])),
                              (float(p[1][0]), float(p[1][1])))
                    open_hand = count_extended_fingers(lm) >= 4

                with self._lock:
                    self._paddle = paddle
                    self._open_hand = open_hand

                now = time.time()
                self.fps = 0.9 * self.fps + 0.1 * (1.0 / max(now - t_prev, 1e-3))
                t_prev = now

    def get_paddle(self):
        with self._lock:
            return self._paddle

    def is_open_hand(self):
        with self._lock:
            return self._open_hand

    def stop(self):
        self.running = False

# ----------------------- LOGICA DO JOGO ---------------------------
def initialize_boxes(image_width, box_size, gap, max_rows=3):
    boxes = []
    box_width, box_height = box_size
    num_per_row = image_width // (box_width + gap)
    num_boxes = int(num_per_row * max_rows)
    for i in range(num_boxes):
        x = (i % num_per_row) * (box_width + gap)
        y = gap + (i // num_per_row) * (box_height + gap)
        boxes.append([x, y, box_width, box_height])
    return boxes

def wall_collision(w, h, ball_position, ball_velocity):
    if ball_position[0] - ball_radius < 0:
        ball_position[0] = ball_radius
        ball_velocity[0] = -ball_velocity[0]
    if ball_position[0] + ball_radius > w:
        ball_position[0] = w - ball_radius
        ball_velocity[0] = -ball_velocity[0]
    if ball_position[1] - ball_radius < 0:
        ball_position[1] = ball_radius
        ball_velocity[1] = -ball_velocity[1]
    if ball_position[1] + ball_radius > h:
        return ball_position, [0.0, 0.0], True   # game over
    return ball_position, ball_velocity, False

def line_collision(line_start, line_end, ball_position, ball_velocity):
    line_vec = np.array([line_end[0] - line_start[0], line_end[1] - line_start[1]])
    line_length_sq = float(np.dot(line_vec, line_vec))
    if line_length_sq < 1e-6:      # indicador e polegar coincidem: ignora
        return ball_velocity

    ball_vec = np.array([ball_position[0] - line_start[0],
                         ball_position[1] - line_start[1]])
    t = np.clip(np.dot(ball_vec, line_vec) / line_length_sq, 0, 1)
    closest = np.array(line_start) + t * line_vec
    dist = np.linalg.norm(np.array(ball_position) - closest)

    if dist <= ball_radius + 5:
        normal = np.array([-line_vec[1], line_vec[0]]) / np.sqrt(line_length_sq)
        v = np.array(ball_velocity, dtype=np.float64)
        v -= 2 * np.dot(v, normal) * normal
        return v
    return ball_velocity

# ----------------------- TELAS ------------------------------------
def _centered_text(canvas, text, y, scale, color, thick):
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
    x = (GAME_W - tw) // 2
    cv2.putText(canvas, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)
    return th

def _hold_bar(canvas, progress):
    """Barra que enche conforme a mao aberta e mantida (0.0 a 1.0)."""
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
    _centered_text(canvas, "AR Breakout", y - 60, 2, (255, 255, 255), 3)
    _centered_text(canvas, "Mao aberta para comecar", y + 20, 1.1, (0, 255, 0), 2)
    _hold_bar(canvas, progress)

def game_over_screen(canvas, progress):
    y = GAME_H // 2
    _centered_text(canvas, "Game Over", y - 40, 2, (0, 0, 255), 3)
    _centered_text(canvas, "Mao aberta para jogar de novo", y + 30, 1, (255, 255, 255), 2)
    _hold_bar(canvas, progress)

def win_screen(canvas, progress):
    y = GAME_H // 2
    _centered_text(canvas, "You Win!", y - 40, 2, (0, 255, 0), 3)
    _centered_text(canvas, "Mao aberta para jogar de novo", y + 30, 1, (255, 255, 255), 2)
    _hold_bar(canvas, progress)

# ----------------------------- MAIN -------------------------------
def new_game():
    bx = np.random.randint(ball_radius + 50, GAME_W - ball_radius - 50)
    ball_position = [float(bx), float(GAME_H - 50)]
    # velocidade com magnitude fixa (BALL_SPEED) e angulo aleatorio p/ cima
    angle = np.random.uniform(-0.6, 0.6)  # radianos em torno da vertical
    ball_velocity = [float(BALL_SPEED * np.sin(angle)),
                     float(-BALL_SPEED * np.cos(angle))]
    boxes = initialize_boxes(GAME_W, box_size, gap, max_rows=1)
    return ball_position, ball_velocity, boxes

def main():
    picam2 = make_camera()
    detect, make_marker = get_aruco()

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

    # Estados: "waiting" (antes de comecar), "playing", "over" (game over/vitoria)
    state = "waiting"
    is_winner = False
    ball_position, ball_velocity, boxes = new_game()
    gesture_since = None          # quando a mao aberta comecou a ser mantida
    target_dt = 1.0 / TARGET_FPS

    def gesture_progress():
        if gesture_since is None:
            return 0.0
        return (time.time() - gesture_since) / GESTURE_HOLD_S

    while True:
        t0 = time.time()
        canvas = np.zeros((GAME_H, GAME_W, 3), dtype=np.uint8)

        # ---- Gesto de iniciar/reiniciar (so nas telas de espera/fim) ----
        if state in ("waiting", "over"):
            if tracker.is_open_hand():
                if gesture_since is None:
                    gesture_since = time.time()
                elif time.time() - gesture_since >= GESTURE_HOLD_S:
                    ball_position, ball_velocity, boxes = new_game()
                    state = "playing"
                    is_winner = False
                    gesture_since = None
            else:
                gesture_since = None  # soltou a mao: zera o hold

        # ---- Render por estado ----
        if state == "waiting":
            start_screen(canvas, gesture_progress())

        elif state == "over":
            if is_winner:
                win_screen(canvas, gesture_progress())
            else:
                game_over_screen(canvas, gesture_progress())

        else:  # playing
            # Vitoria: acabou as caixas
            if not boxes:
                state, is_winner = "over", True

            if state == "playing":
                # Fisica da bola
                ball_position[0] += ball_velocity[0]
                ball_position[1] += ball_velocity[1]
                ball_position, ball_velocity, dead = wall_collision(
                    GAME_W, GAME_H, ball_position, ball_velocity)
                if dead:
                    state, is_winner = "over", False

                # Raquete (mao) - vem da thread, ja no espaco do jogo
                paddle = tracker.get_paddle()
                if paddle is not None and state == "playing":
                    p_i, p_t = paddle
                    cv2.line(canvas, (int(p_i[0]), int(p_i[1])),
                             (int(p_t[0]), int(p_t[1])), (0, 255, 0), 6)
                    ball_velocity = line_collision(p_i, p_t, ball_position, ball_velocity)

                # Bola
                cv2.circle(canvas, (int(ball_position[0]), int(ball_position[1])),
                           ball_radius, ball_color, -1)

                # Caixas + colisao (centro dentro da caixa -> inverte vy)
                for box in boxes[:]:
                    bx, by, bw, bh = box
                    cv2.rectangle(canvas, (bx, by), (bx + bw, by + bh), (255, 0, 0), -1)
                    if bx < ball_position[0] < bx + bw and by < ball_position[1] < by + bh:
                        boxes.remove(box)
                        ball_velocity[1] = -ball_velocity[1]

        # HUD discreto: fps da deteccao da mao
        cv2.putText(canvas, f"hand {tracker.fps:4.1f} fps", (12, GAME_H - 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (90, 90, 90), 1, cv2.LINE_AA)

        cv2.imshow(WINDOW, canvas)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:            # ESC sai
            break
        elif key == ord('r'):    # 'r' ainda funciona como reinicio manual
            ball_position, ball_velocity, boxes = new_game()
            state, is_winner, gesture_since = "playing", False, None

        # Trava de framerate: mantem a velocidade da bola consistente
        dt = time.time() - t0
        if dt < target_dt:
            time.sleep(target_dt - dt)

    tracker.stop()
    time.sleep(0.2)
    picam2.stop()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
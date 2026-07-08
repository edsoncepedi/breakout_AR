"""
fruit_ninja/game.py - AR Fruit Ninja (modo tempo).

Gesto de comecar/reiniciar: mao aberta. Cada rodada dura TIME_LIMIT_S
segundos; o objetivo e fazer o maior numero de pontos possivel nesse
intervalo. O recorde e a maior pontuacao alcancada dentro do mesmo tempo
fixo (por isso as tentativas sao comparaveis entre si).

A rodada tambem pode terminar antes do tempo acabar se cortar uma bomba
ou perder todas as vidas (fruta caindo sem cortar).

Assets opcionais (PNGs com alpha) ficam em fruit_ninja/assets/.
Recorde salvo em fruit_ninja/bestscore.txt.
"""

import os
import random
import time
from collections import deque

import cv2
import numpy as np

from common import GAME_W, GAME_H, GESTURE_HOLD_S, _centered_text, _hold_bar

_HERE = os.path.dirname(os.path.abspath(__file__))
ASSET_DIR = os.path.join(_HERE, "assets")
BEST_SCORE_FILE = os.path.join(_HERE, "bestscore.txt")

FRUIT_RADIUS = 45
GRAVITY = 0.35
SPAWN_MIN_S, SPAWN_MAX_S = 0.7, 1.3
BOMB_CHANCE = 0.15
LIVES_START = 3
SLICE_TOLERANCE = 10
TIME_LIMIT_S = 45           # duracao de cada rodada, em segundos

FRUIT_KINDS = {
    "sandia":  {"color": (60, 60, 220),  "points": 1},
    "naranja": {"color": (0, 140, 255),  "points": 1},
    "banana":  {"color": (0, 220, 255),  "points": 2},
}
BOMB_COLOR = (40, 40, 40)


def load_assets():
    """Carrega PNGs com alpha se existirem; senao retorna dict vazio
    (o jogo cai pra circulos coloridos automaticamente). Ja redimensiona
    aqui (uma vez so, no load) pro tamanho final de tela."""
    files = {"sandia": "sandia.png", "naranja": "naranja.png", "banana": "banana.png",
             "bomb": "bomb.png"}
    box_size_px = FRUIT_RADIUS * 2
    imgs = {}
    for kind, fname in files.items():
        path = os.path.join(ASSET_DIR, fname)
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is not None and img.shape[2] == 4:
            imgs[kind] = cv2.resize(img, (box_size_px, box_size_px))
        elif img is not None:
            print(f"aviso: {path} sem canal alpha, ignorando (use PNG transparente)")
    return imgs

def overlay_transparent(background, overlay, cx, cy):
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
    vy0 = -np.sqrt(2 * GRAVITY * apex_height)
    vx = random.uniform(-2.5, 2.5)
    return Fruit(x, y, vx, vy0, kind)

def point_segment_distance(p, a, b):
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

def format_time(seconds):
    seconds = max(0, seconds)
    m = int(seconds) // 60
    s = seconds - m * 60
    return f"{m:02d}:{s:05.2f}"

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


class FruitNinjaGame:
    def __init__(self):
        os.makedirs(ASSET_DIR, exist_ok=True)
        self.assets = load_assets()
        if self.assets:
            print(f"Assets carregados: {list(self.assets.keys())}")
        else:
            print(f"Nenhum asset em {ASSET_DIR} -- usando circulos coloridos.")
        self.best_score = load_best_score()
        self.trail = deque(maxlen=8)
        self.reset()

    def reset(self):
        self.state = "waiting"
        self.score = 0
        self.lives = LIVES_START
        self.is_new_record = False
        self.end_reason = None     # "time" | "bomb" | "lives"
        self.fruits = []
        self.next_spawn_at = time.time() + random.uniform(SPAWN_MIN_S, SPAWN_MAX_S)
        self.gesture_since = None
        self.round_start = None
        self.trail.clear()

    def _gesture_progress(self):
        if self.gesture_since is None:
            return 0.0
        return (time.time() - self.gesture_since) / GESTURE_HOLD_S

    def _end_round(self, reason):
        self.state = "over"
        self.end_reason = reason
        if self.best_score is None or self.score > self.best_score:
            self.best_score = self.score
            save_best_score(self.best_score)
            self.is_new_record = True
        else:
            self.is_new_record = False

    def update_and_draw(self, canvas, tracker):
        if self.state in ("waiting", "over"):
            if tracker.is_open_hand_gesture():
                if self.gesture_since is None:
                    self.gesture_since = time.time()
                elif time.time() - self.gesture_since >= GESTURE_HOLD_S:
                    self.reset()
                    self.state = "playing"
                    self.round_start = time.time()
                    self.gesture_since = None
            else:
                self.gesture_since = None

        blade = tracker.get_blade()
        if blade is not None:
            self.trail.append(blade[1])

        if self.state == "waiting":
            y = GAME_H // 2
            _centered_text(canvas, "AR Fruit Ninja", y - 60, 2, (255, 255, 255), 3)
            _centered_text(canvas, f"Voce tem {TIME_LIMIT_S}s -- faca o maior placar!",
                           y - 15, 0.8, (180, 180, 180), 2)
            _centered_text(canvas, "Mao aberta para comecar", y + 20, 1.1, (0, 255, 0), 2)
            _hold_bar(canvas, self._gesture_progress())

        elif self.state == "over":
            y = GAME_H // 2
            title = "Tempo esgotado!" if self.end_reason == "time" else "Game Over"
            color = (0, 215, 255) if self.end_reason == "time" else (0, 0, 255)
            _centered_text(canvas, title, y - 60, 2, color, 3)
            _centered_text(canvas, f"Pontos: {self.score}", y - 10, 1.2, (255, 255, 255), 2)
            if self.is_new_record:
                _centered_text(canvas, "NOVO RECORDE!", y + 30, 1.0, (0, 215, 255), 2)
            _centered_text(canvas, "Mao aberta para jogar de novo", y + 75, 1, (255, 255, 255), 2)
            _hold_bar(canvas, self._gesture_progress())

        else:  # playing
            now = time.time()
            elapsed = now - self.round_start
            remaining = TIME_LIMIT_S - elapsed
            if remaining <= 0:
                self._end_round("time")
            else:
                if now >= self.next_spawn_at:
                    self.fruits.append(spawn_fruit())
                    self.next_spawn_at = now + random.uniform(SPAWN_MIN_S, SPAWN_MAX_S)

                for fruit in self.fruits[:]:
                    fruit.update()

                    sliced = False
                    if blade is not None:
                        dist = point_segment_distance(fruit.pos, blade[0], blade[1])
                        if dist <= fruit.radius + SLICE_TOLERANCE:
                            sliced = True

                    if sliced:
                        self.fruits.remove(fruit)
                        if fruit.kind == "bomb":
                            self._end_round("bomb")
                        else:
                            self.score += FRUIT_KINDS[fruit.kind]["points"]
                        continue

                    if fruit.y - fruit.radius > GAME_H:
                        self.fruits.remove(fruit)
                        if fruit.kind != "bomb":
                            self.lives -= 1
                            if self.lives <= 0:
                                self._end_round("lives")

                for fruit in self.fruits:
                    draw_fruit(canvas, fruit, self.assets)

                pts = list(self.trail)
                for i in range(1, len(pts)):
                    cv2.line(canvas, (int(pts[i-1][0]), int(pts[i-1][1])),
                              (int(pts[i][0]), int(pts[i][1])), (255, 255, 255), 4)

                # cronometro regressivo, bem visivel no topo central
                _centered_text(canvas, format_time(remaining), 50, 1.4,
                               (0, 215, 255) if remaining > 10 else (0, 0, 255), 3)

        cv2.putText(canvas, f"Pontos: {self.score}", (16, GAME_H - 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        text = f"Recorde: {self.best_score}" if self.best_score is not None else "Recorde: --"
        (tw, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
        cv2.putText(canvas, text, (GAME_W - tw - 16, GAME_H - 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 215, 255), 2, cv2.LINE_AA)
        cv2.putText(canvas, f"Vidas: {'*' * self.lives}", (16, 34),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 80, 80), 2, cv2.LINE_AA)

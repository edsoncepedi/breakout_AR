"""
breakout/game.py - AR Breakout.

Gesto de comecar/reiniciar: feche a mao (punho). Recorde = menor tempo
para zerar as caixas, salvo em breakout/besttime.txt.
"""

import os

import cv2
import numpy as np

from common import GAME_W, GAME_H, GESTURE_HOLD_S, _centered_text, _hold_bar, format_time
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
BEST_TIME_FILE = os.path.join(_HERE, "besttime.txt")

ball_radius = 20
ball_color = (255, 255, 255)
box_size = (80, 40)
gap = 5
BALL_SPEED = 7.0
PADDLE_THICKNESS = 6
PADDLE_COOLDOWN = 5


def initialize_boxes(image_width, box_size, gap, max_rows=1):
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
        return ball_position, [0.0, 0.0], True
    return ball_position, ball_velocity, False

def line_collision(line_start, line_end, ball_position, ball_velocity, cooldown):
    if cooldown > 0:
        return ball_velocity, ball_position, cooldown - 1

    ls = np.array(line_start, dtype=np.float64)
    le = np.array(line_end, dtype=np.float64)
    bp = np.array(ball_position, dtype=np.float64)

    line_vec = le - ls
    line_length_sq = float(line_vec @ line_vec)
    if line_length_sq < 1e-6:
        return ball_velocity, ball_position, 0

    t = np.clip(((bp - ls) @ line_vec) / line_length_sq, 0, 1)
    closest = ls + t * line_vec
    to_ball = bp - closest
    dist = float(np.linalg.norm(to_ball))
    hit_dist = ball_radius + PADDLE_THICKNESS

    if dist <= hit_dist:
        if dist > 1e-6:
            normal = to_ball / dist
        else:
            normal = np.array([-line_vec[1], line_vec[0]]) / np.sqrt(line_length_sq)

        v = np.array(ball_velocity, dtype=np.float64)
        if float(v @ normal) < 0:
            v = v - 2 * float(v @ normal) * normal
            bp = closest + normal * (hit_dist + 1)
            return v.tolist(), bp.tolist(), PADDLE_COOLDOWN

    return ball_velocity, ball_position, 0

def load_best_time():
    try:
        with open(BEST_TIME_FILE) as f:
            return float(f.read().strip())
    except (FileNotFoundError, ValueError):
        return None

def save_best_time(value):
    try:
        with open(BEST_TIME_FILE, "w") as f:
            f.write(str(value))
    except OSError as e:
        print("Nao foi possivel salvar o recorde:", e)


class BreakoutGame:
    def __init__(self):
        self.best_time = load_best_time()
        self.reset()

    def _new_round(self):
        bx = np.random.randint(ball_radius + 50, GAME_W - ball_radius - 50)
        ball_position = [float(bx), float(GAME_H - 50)]
        angle = np.random.uniform(-0.6, 0.6)
        ball_velocity = [float(BALL_SPEED * np.sin(angle)),
                         float(-BALL_SPEED * np.cos(angle))]
        boxes = initialize_boxes(GAME_W, box_size, gap, max_rows=1)
        return ball_position, ball_velocity, boxes

    def reset(self):
        self.state = "waiting"
        self.ball_position, self.ball_velocity, self.boxes = self._new_round()
        self.paddle_cooldown = 0
        self.is_winner = False
        self.is_new_record = False
        self.gesture_since = None
        self.run_start = None
        self.last_elapsed = 0.0

    def _gesture_progress(self):
        if self.gesture_since is None:
            return 0.0
        return (time.time() - self.gesture_since) / GESTURE_HOLD_S

    def update_and_draw(self, canvas, tracker):
        if self.state in ("waiting", "over"):
            if tracker.is_fist_gesture():
                if self.gesture_since is None:
                    self.gesture_since = time.time()
                elif time.time() - self.gesture_since >= GESTURE_HOLD_S:
                    self.ball_position, self.ball_velocity, self.boxes = self._new_round()
                    self.state = "playing"
                    self.is_winner = False
                    self.is_new_record = False
                    self.paddle_cooldown = 0
                    self.run_start = time.time()
                    self.gesture_since = None
            else:
                self.gesture_since = None

        elapsed = (time.time() - self.run_start) if (self.state == "playing" and self.run_start) else self.last_elapsed

        if self.state == "waiting":
            y = GAME_H // 2
            _centered_text(canvas, "AR Breakout", y - 60, 2, (255, 255, 255), 3)
            _centered_text(canvas, "Feche a mao para comecar", y + 20, 1.1, (0, 255, 0), 2)
            _hold_bar(canvas, self._gesture_progress())

        elif self.state == "over":
            y = GAME_H // 2
            if self.is_winner:
                _centered_text(canvas, "You Win!", y - 40, 2, (0, 255, 0), 3)
                _centered_text(canvas, f"Tempo: {format_time(self.last_elapsed)}", y + 10, 1, (255, 255, 255), 2)
                if self.is_new_record:
                    _centered_text(canvas, "Novo recorde!", y + 50, 1, (0, 215, 255), 2)
                    _centered_text(canvas, "Feche a mao para jogar de novo", y + 90, 1, (255, 255, 255), 2)
                else:
                    _centered_text(canvas, "Feche a mao para jogar de novo", y + 55, 1, (255, 255, 255), 2)
            else:
                _centered_text(canvas, "Game Over", y - 40, 2, (0, 0, 255), 3)
                _centered_text(canvas, f"Tempo: {format_time(self.last_elapsed)}", y + 10, 1, (255, 255, 255), 2)
                _centered_text(canvas, "Feche a mao para jogar de novo", y + 55, 1, (255, 255, 255), 2)
            _hold_bar(canvas, self._gesture_progress())

        else:  # playing
            if not self.boxes:
                self.last_elapsed = elapsed
                self.state, self.is_winner = "over", True
                if self.best_time is None or self.last_elapsed < self.best_time:
                    self.best_time = self.last_elapsed
                    save_best_time(self.best_time)
                    self.is_new_record = True
                else:
                    self.is_new_record = False

            if self.state == "playing":
                self.ball_position[0] += self.ball_velocity[0]
                self.ball_position[1] += self.ball_velocity[1]
                self.ball_position, self.ball_velocity, dead = wall_collision(
                    GAME_W, GAME_H, self.ball_position, self.ball_velocity)
                if dead:
                    self.last_elapsed = elapsed
                    self.state, self.is_winner = "over", False

                paddle = tracker.get_paddle()
                if paddle is not None and self.state == "playing":
                    p_i, p_t = paddle
                    cv2.line(canvas, (int(p_i[0]), int(p_i[1])),
                             (int(p_t[0]), int(p_t[1])), (255, 255, 255), 6)
                    self.ball_velocity, self.ball_position, self.paddle_cooldown = line_collision(
                        p_i, p_t, self.ball_position, self.ball_velocity, self.paddle_cooldown)
                elif self.paddle_cooldown > 0:
                    self.paddle_cooldown -= 1

                cv2.circle(canvas, (int(self.ball_position[0]), int(self.ball_position[1])),
                           ball_radius, ball_color, -1)

                for box in self.boxes[:]:
                    bx, by, bw, bh = box
                    cv2.rectangle(canvas, (bx, by), (bx + bw, by + bh), (255, 0, 0), -1)
                    if bx < self.ball_position[0] < bx + bw and by < self.ball_position[1] < by + bh:
                        self.boxes.remove(box)
                        self.ball_velocity[1] = -self.ball_velocity[1]

        cv2.putText(canvas, f"Tempo: {format_time(elapsed)}", (16, GAME_H - 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        text = f"Recorde: {format_time(self.best_time)}" if self.best_time is not None else "Recorde: --:--.--"
        (tw, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
        cv2.putText(canvas, text, (GAME_W - tw - 16, GAME_H - 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 215, 255), 2, cv2.LINE_AA)

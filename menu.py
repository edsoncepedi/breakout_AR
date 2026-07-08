"""
menu.py - tela de selecao entre os jogos.

Aponte o indicador pro lado esquerdo (Breakout) ou direito (Fruit Ninja)
e segure por GESTURE_HOLD_S segundos pra escolher.
"""

import time

import cv2

from common import GAME_W, GAME_H, GESTURE_HOLD_S, _centered_text, _hold_bar


class Menu:
    LABELS = {"breakout": "AR Breakout", "fruitninja": "AR Fruit Ninja"}

    def __init__(self):
        self.hover = None
        self.hover_since = None

    def _zone_for(self, pt):
        if pt is None:
            return None
        return "breakout" if pt[0] < GAME_W / 2 else "fruitninja"

    def update_and_draw(self, canvas, tracker):
        """Desenha o menu e retorna 'breakout' / 'fruitninja' quando uma
        escolha e confirmada, ou None enquanto ainda nao foi."""
        pt = tracker.get_point()
        zone = self._zone_for(pt)

        if zone != self.hover:
            self.hover = zone
            self.hover_since = time.time() if zone else None

        progress = 0.0
        selected = None
        if self.hover is not None:
            progress = (time.time() - self.hover_since) / GESTURE_HOLD_S
            if progress >= 1.0:
                selected = self.hover
                self.hover = None
                self.hover_since = None

        _centered_text(canvas, "AR Arcade", 90, 1.6, (255, 255, 255), 3)
        _centered_text(canvas, "Aponte para um lado e segure para escolher",
                       140, 0.8, (180, 180, 180), 2)

        mid = GAME_W // 2
        cv2.line(canvas, (mid, 170), (mid, GAME_H - 60), (60, 60, 60), 2)

        zones = {"breakout": (0, mid), "fruitninja": (mid, GAME_W)}
        for kind, (x0, x1) in zones.items():
            cx = (x0 + x1) // 2
            active = (self.hover == kind)
            color = (0, 255, 0) if active else (150, 150, 150)
            label = self.LABELS[kind]
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 1.3, 3)
            cv2.putText(canvas, label, (cx - tw // 2, GAME_H // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.3, color, 3, cv2.LINE_AA)
            if active:
                _hold_bar(canvas, progress, y0=GAME_H // 2 + 60)

        return selected

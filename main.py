#!/usr/bin/env python3
"""
main.py - AR Arcade: ponto de entrada.

So orquestra: camera, calibracao (ArUco + homografia), a thread de mao
compartilhada, o menu e os dois jogos. A logica de cada jogo mora em
breakout/game.py e fruit_ninja/game.py.

Navegacao:
  - Menu: aponte pro lado esquerdo (Breakout) ou direito (Fruit Ninja) e
    segure ~0.6s.
  - Dentro de um jogo: segure o dedo no circulo "M" (canto sup. esquerdo)
    por ~1.2s pra voltar ao menu a qualquer momento.
  - Gestos de comecar/reiniciar de cada jogo: Breakout = punho fechado,
    Fruit Ninja = mao aberta (cada um manteve o que ja usava).
  - ESC sai do programa. 'm' forca volta ao menu (atalho de operador).
    'r' reinicia o jogo ativo na hora (atalho de operador).
"""

import os
import time

import cv2
import numpy as np

from common import (GAME_W, GAME_H, WINDOW, CALIB_FILE, TARGET_FPS, HOME_CENTER,
                     HOME_RADIUS, HOME_HOLD_S, make_camera, get_aruco, calibrate,
                     HandTracker, draw_home_hotspot)
from menu import Menu
from breakout.game import BreakoutGame
from fruit_ninja.game import FruitNinjaGame


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

    menu = Menu()
    breakout = BreakoutGame()
    fruitninja = FruitNinjaGame()

    active = "menu"        # "menu" | "breakout" | "fruitninja"
    home_since = None
    target_dt = 1.0 / TARGET_FPS

    while True:
        t0 = time.time()
        canvas = np.zeros((GAME_H, GAME_W, 3), dtype=np.uint8)

        # ---- Hotspot "voltar ao menu", ativo em qualquer jogo ----
        home_progress = 0.0
        if active != "menu":
            pt = tracker.get_point()
            hovering_home = (pt is not None and
                              np.hypot(pt[0] - HOME_CENTER[0], pt[1] - HOME_CENTER[1]) <= HOME_RADIUS)
            if hovering_home:
                if home_since is None:
                    home_since = time.time()
                home_progress = (time.time() - home_since) / HOME_HOLD_S
                if home_progress >= 1.0:
                    active = "menu"
                    home_since = None
                    home_progress = 0.0
            else:
                home_since = None

        # ---- Roda o estado ativo ----
        if active == "menu":
            selected = menu.update_and_draw(canvas, tracker)
            if selected == "breakout":
                breakout.reset()
                active = "breakout"
            elif selected == "fruitninja":
                fruitninja.reset()
                active = "fruitninja"
        elif active == "breakout":
            breakout.update_and_draw(canvas, tracker)
            draw_home_hotspot(canvas, home_progress)
        elif active == "fruitninja":
            fruitninja.update_and_draw(canvas, tracker)
            draw_home_hotspot(canvas, home_progress)

        cv2.putText(canvas, f"hand {tracker.fps:4.1f} fps", (GAME_W - 130, 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (90, 90, 90), 1, cv2.LINE_AA)

        cv2.imshow(WINDOW, canvas)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break
        elif key == ord('m'):
            active = "menu"
            home_since = None
        elif key == ord('r'):
            if active == "breakout":
                breakout.reset()
                breakout.state = "playing"
                breakout.run_start = time.time()
            elif active == "fruitninja":
                fruitninja.reset()
                fruitninja.state = "playing"

        dt = time.time() - t0
        if dt < target_dt:
            time.sleep(target_dt - dt)

    tracker.stop()
    time.sleep(0.2)
    picam2.stop()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()

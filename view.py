#!/usr/bin/env python3
"""
Visualizador de camera - diagnostico de enquadramento/foco (Pi 5 + picamera2)
============================================================================

Abre a camera numa janela no monitor e desenha por cima os marcadores ArUco
detectados (com os IDs). Serve para:
  - posicionar a camera ate ela enxergar a area projetada inteira
  - focar a lente (a IMX500 tem foco por anel manual) ate a imagem ficar nitida
  - conferir se os 4 marcadores (ids 0..3) sao detectados

Teclas:
  c  -> alterna a ordem de cor (use se as cores aparecerem trocadas)
  s  -> salva um print em camera_snapshot.png
  ESC -> sai

Rode DIRETO no Pi (nao por SSH sem display), com o projetor mostrando a tela
de calibracao do jogo OU qualquer coisa com os marcadores.
"""

import time
import cv2
import numpy as np
from picamera2 import Picamera2

# Use uma resolucao boa de visualizar (pode ser maior que a da deteccao do jogo)
VIEW_W, VIEW_H = 1280, 720
SWAP_RB = False   # comeca assim; aperte 'c' pra alternar ao vivo

def get_aruco_detect():
    aruco = cv2.aruco
    dictionary = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
    try:  # OpenCV >= 4.7
        detector = aruco.ArucoDetector(dictionary, aruco.DetectorParameters())
        def detect(gray):
            return detector.detectMarkers(gray)
    except AttributeError:  # OpenCV < 4.7
        params = aruco.DetectorParameters_create()
        def detect(gray):
            return aruco.detectMarkers(gray, dictionary, parameters=params)
    return detect

def main():
    global SWAP_RB

    picam2 = Picamera2()
    picam2.configure(picam2.create_preview_configuration(
        main={"format": "RGB888", "size": (VIEW_W, VIEW_H)}))
    picam2.start()
    time.sleep(0.5)

    detect = get_aruco_detect()

    cv2.namedWindow("camera", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("camera", VIEW_W, VIEW_H)

    print("Janela 'camera' aberta. c=cor  s=salvar  ESC=sair")
    fps = 0.0
    t_prev = time.time()

    while True:
        frame = picam2.capture_array()
        if SWAP_RB:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = detect(gray)

        found = []
        if ids is not None:
            cv2.aruco.drawDetectedMarkers(frame, corners, ids)
            found = sorted(int(i) for i in ids.flatten())
            # marca o centro de cada marcador
            for c, i in zip(corners, ids.flatten()):
                cx, cy = c.reshape(4, 2).mean(axis=0).astype(int)
                cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)

        # HUD
        now = time.time()
        fps = 0.9 * fps + 0.1 * (1.0 / max(now - t_prev, 1e-3))
        t_prev = now
        status = f"ArUco: {len(found)}/4  ids={found}   {fps:4.1f} fps   SWAP_RB={SWAP_RB}"
        color = (0, 255, 0) if len(found) == 4 else (0, 200, 255)
        cv2.rectangle(frame, (0, 0), (VIEW_W, 34), (0, 0, 0), -1)
        cv2.putText(frame, status, (10, 24), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, color, 2, cv2.LINE_AA)

        # cruz central (ajuda a centralizar o enquadramento)
        cv2.drawMarker(frame, (VIEW_W // 2, VIEW_H // 2), (120, 120, 120),
                       cv2.MARKER_CROSS, 40, 1)

        cv2.imshow("camera", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:            # ESC
            break
        elif key == ord('c'):
            SWAP_RB = not SWAP_RB
            print("SWAP_RB =", SWAP_RB)
        elif key == ord('s'):
            cv2.imwrite("camera_snapshot.png", frame)
            print("Salvo em camera_snapshot.png")

    picam2.stop()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
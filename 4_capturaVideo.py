import cv2
import numpy as np

cam = cv2.VideoCapture(0)

if not cam.isOpened():
    print("No se pudo abrir la cámara")
    exit()

while True:
    ret, frame = cam.read()
    if not ret:
        print("No se pudo leer la cámara")
        break

    # BGR original
    bgr = frame

    # Escala de grises
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    # HSV
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # CIE Lab
    cielab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)

    # Redimensionar para mostrar en 4 ventanas dentro de una sola imagen
    size = (320, 240)
    bgr = cv2.resize(bgr, size)
    gray_bgr = cv2.resize(gray_bgr, size)
    hsv = cv2.resize(hsv, size)
    cielab = cv2.resize(cielab, size)

    # Agregar títulos
    cv2.putText(bgr, "BGR", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(gray_bgr, "Grises", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(hsv, "HSV", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(cielab, "CIE Lab", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Unir las 4 vistas
    fila1 = np.hstack((bgr, gray_bgr))
    fila2 = np.hstack((hsv, cielab))
    mosaico = np.vstack((fila1, fila2))

    cv2.imshow("Camara - BGR | Grises | HSV | CIE Lab", mosaico)

    # Presiona ESC para salir
    if cv2.waitKey(1) & 0xFF == 27:
        break

cam.release()
cv2.destroyAllWindows()
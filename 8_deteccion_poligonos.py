import cv2
import numpy as np
import os
import time

# ==========================
# CONFIGURACION
# ==========================

CLASE = "carta_1"      # cambiar a carta_2, carta_3, carta_4, carta_5
MAX_IMAGENES = 100
INTERVALO = 2          # segundos

# ==========================
# CARPETA
# ==========================

ruta_dataset = os.path.join("Imagenes", CLASE)
os.makedirs(ruta_dataset, exist_ok=True)

contador = len(os.listdir(ruta_dataset))
ultimo_guardado = time.time()

# ==========================
# CAMARA
# ==========================

cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)

if not cap.isOpened():
    print("No se pudo abrir la cámara")
    exit()

# ==========================
# INTERFAZ
# ==========================

def nada(x):
    pass

cv2.namedWindow("Detector de Cartas")

cv2.createTrackbar(
    "Canny Bajo",
    "Detector de Cartas",
    50,
    255,
    nada
)

cv2.createTrackbar(
    "Canny Alto",
    "Detector de Cartas",
    150,
    255,
    nada
)

cv2.createTrackbar(
    "Area Minima",
    "Detector de Cartas",
    1000,
    20000,
    nada
)

print(f"Capturando {CLASE}")
print("Presiona Q para salir")

while True:

    ret, frame = cap.read()

    if not ret:
        break

    salida = frame.copy()

    bajo = cv2.getTrackbarPos(
        "Canny Bajo",
        "Detector de Cartas"
    )

    alto = cv2.getTrackbarPos(
        "Canny Alto",
        "Detector de Cartas"
    )

    area_minima = cv2.getTrackbarPos(
        "Area Minima",
        "Detector de Cartas"
    )

    gris = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2GRAY
    )

    blur = cv2.GaussianBlur(
        gris,
        (5, 5),
        0
    )

    bordes = cv2.Canny(
        blur,
        bajo,
        alto
    )

    contornos, _ = cv2.findContours(
        bordes,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    mejor_contorno = None
    mejor_area = 0

    for contorno in contornos:

        area = cv2.contourArea(contorno)

        if area < area_minima:
            continue

        perimetro = cv2.arcLength(
            contorno,
            True
        )

        aproximacion = cv2.approxPolyDP(
            contorno,
            0.02 * perimetro,
            True
        )

        if len(aproximacion) == 4:

            if area > mejor_area:
                mejor_area = area
                mejor_contorno = aproximacion

    if mejor_contorno is not None:

        x, y, w, h = cv2.boundingRect(
            mejor_contorno
        )

        # Contorno verde
        cv2.drawContours(
            salida,
            [mejor_contorno],
            -1,
            (0, 255, 0),
            3
        )

        # Rectángulo azul
        cv2.rectangle(
            salida,
            (x, y),
            (x + w, y + h),
            (255, 0, 0),
            2
        )

        cv2.putText(
            salida,
            "Carta Detectada",
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

        tiempo_actual = time.time()

        if tiempo_actual - ultimo_guardado >= INTERVALO:

            roi = frame[y:y+h, x:x+w]

            if roi.size > 0:

                nombre = os.path.join(
                    ruta_dataset,
                    f"{contador:04d}.png"
                )

                cv2.imwrite(
                    nombre,
                    roi
                )

                print(
                    f"Guardada: {nombre}"
                )

                contador += 1
                ultimo_guardado = tiempo_actual

    cv2.putText(
        salida,
        f"Imagenes: {contador}/{MAX_IMAGENES}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    cv2.imshow(
        "Detector de Cartas",
        salida
    )

    cv2.imshow(
        "Bordes Canny",
        bordes
    )

    if contador >= MAX_IMAGENES:
        print("100 imagenes capturadas")
        break

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
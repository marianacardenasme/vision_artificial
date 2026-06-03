import cv2
import os
import time

# ==========================
# CONFIGURACION
# ==========================

os.makedirs("carta_3", exist_ok=True)

contador = 0
ultimo_guardado = time.time()

# ==========================
# CAMARA
# ==========================

cap = cv2.VideoCapture(0, cv2.CAP_AVFOUNDATION)

if not cap.isOpened():
    print("No se pudo abrir la camara")
    exit()

# ==========================
# CONTROLES
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
    30000,
    nada
)

print("Mostrando camara...")
print("Las cartas se guardaran cada 2 segundos")
print("Presiona Q para salir")

# ==========================
# LOOP
# ==========================

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

    gris = cv2.GaussianBlur(
        gris,
        (5, 5),
        0
    )

    bordes = cv2.Canny(
        gris,
        bajo,
        alto
    )

    contornos, _ = cv2.findContours(
        bordes,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

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

            x, y, w, h = cv2.boundingRect(
                aproximacion
            )

            cv2.drawContours(
                salida,
                [aproximacion],
                -1,
                (0, 255, 0),
                3
            )

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

            if tiempo_actual - ultimo_guardado >= 2:

                carta = frame[
                    y:y+h,
                    x:x+w
                ]

                if carta.size > 0:

                    nombre = (
                        f"dataset_cartas/"
                        f"carta_{contador:04d}.jpg"
                    )

                    cv2.imwrite(
                        nombre,
                        carta
                    )

                    print(
                        f"Guardada: {nombre}"
                    )

                    contador += 1
                    ultimo_guardado = tiempo_actual

    cv2.putText(
        salida,
        f"Imagenes: {contador}",
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

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
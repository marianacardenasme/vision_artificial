import cv2
import numpy as np

def nothing(x):
    pass

modo = "cielab"   # opciones: "bgr", "hsv", "cielab"

cam = cv2.VideoCapture(0)

if not cam.isOpened():
    print("No se pudo abrir la cámara")
    exit()

ventana_controles = f"Controles {modo.upper()}"
cv2.namedWindow(ventana_controles)

if modo == "bgr":
    canales = ["B", "G", "R"]
    maximos = [255, 255, 255]
    codigo_color = None

elif modo == "hsv":
    canales = ["H", "S", "V"]
    maximos = [179, 255, 255]
    codigo_color = cv2.COLOR_BGR2HSV

elif modo == "cielab":
    canales = ["L", "A", "B"]
    maximos = [255, 255, 255]
    codigo_color = cv2.COLOR_BGR2LAB

else:
    print("Modo no válido")
    exit()

for canal, max_val in zip(canales, maximos):
    cv2.createTrackbar(f"{canal} min", ventana_controles, 0, max_val, nothing)
    cv2.createTrackbar(f"{canal} max", ventana_controles, max_val, max_val, nothing)

while True:
    ret, frame = cam.read()

    if not ret:
        break

    if codigo_color is not None:
        imagen_color = cv2.cvtColor(frame, codigo_color)
    else:
        imagen_color = frame

    valores_min = []
    valores_max = []

    for canal in canales:
        valores_min.append(cv2.getTrackbarPos(f"{canal} min", ventana_controles))
        valores_max.append(cv2.getTrackbarPos(f"{canal} max", ventana_controles))

    lower = np.array(valores_min)
    upper = np.array(valores_max)

    mask = cv2.inRange(imagen_color, lower, upper)
    result = cv2.bitwise_and(frame, frame, mask=mask)

    mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

    size = (320, 240)

    original_view = cv2.resize(frame, size)
    color_view = cv2.resize(imagen_color, size)
    mask_view = cv2.resize(mask_bgr, size)
    result_view = cv2.resize(result, size)

    cv2.putText(original_view, "Original BGR", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.putText(color_view, modo.upper(), (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.putText(mask_view, "Mascara", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.putText(result_view, "Resultado", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    fila1 = np.hstack((original_view, color_view))
    fila2 = np.hstack((mask_view, result_view))

    mosaico = np.vstack((fila1, fila2))

    cv2.imshow(f"Editor de Mascaras - {modo.upper()}", mosaico)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cam.release()
cv2.destroyAllWindows()

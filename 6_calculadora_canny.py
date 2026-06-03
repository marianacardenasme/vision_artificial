qimport cv2
import numpy as np

def nada(x):
    pass

cap = cv2.VideoCapture(0)

cv2.namedWindow("Calculadora Canny")
cv2.createTrackbar("Umbral bajo", "Calculadora Canny", 100, 255, nada)
cv2.createTrackbar("Umbral alto", "Calculadora Canny", 200, 255, nada)
cv2.createTrackbar("Kernel Gaussiano", "Calculadora Canny", 5, 31, nada)

while True:
    ret, frame = cap.read()

    if not ret:
        print("No se pudo acceder a la cámara")
        break

    bajo = cv2.getTrackbarPos("Umbral bajo", "Calculadora Canny")
    alto = cv2.getTrackbarPos("Umbral alto", "Calculadora Canny")
    kernel = cv2.getTrackbarPos("Kernel Gaussiano", "Calculadora Canny")

    if kernel < 1:
        kernel = 1

    if kernel % 2 == 0:
        kernel += 1

    gris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gaussian = cv2.GaussianBlur(gris, (kernel, kernel), 0)
    canny = cv2.Canny(gaussian, bajo, alto)

    gris_bgr = cv2.cvtColor(gris, cv2.COLOR_GRAY2BGR)
    gaussian_bgr = cv2.cvtColor(gaussian, cv2.COLOR_GRAY2BGR)
    canny_bgr = cv2.cvtColor(canny, cv2.COLOR_GRAY2BGR)

    fila1 = np.hstack((frame, gris_bgr))
    fila2 = np.hstack((gaussian_bgr, canny_bgr))
    salida = np.vstack((fila1, fila2))

    cv2.putText(salida, "Original", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    cv2.putText(salida, "Grises", (frame.shape[1] + 20, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    cv2.putText(salida, "Gaussiano", (20, frame.shape[0] + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
    cv2.putText(salida, "Canny", (frame.shape[1] + 20, frame.shape[0] + 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

    cv2.imshow("Calculadora Canny", salida)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
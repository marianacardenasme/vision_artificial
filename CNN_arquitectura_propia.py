import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt
import os

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Dense, Flatten, Reshape
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report

# =========================
# CARGAR DATOS
# =========================
def cargarDatos(rutaOrigen, categorias, ancho=28, alto=28):
    imagenesCargadas = []
    valorEsperado = []

    for idxCategoria, categoria in enumerate(categorias):
        rutaCategoria = os.path.join(rutaOrigen, categoria)

        archivos = sorted([
            f for f in os.listdir(rutaCategoria)
            if f.lower().endswith(".jpg")
        ])

        for archivo in archivos:
            ruta = os.path.join(rutaCategoria, archivo)
            imagen = cv2.imread(ruta)
            if imagen is None:
                print(f"[SKIP] No se pudo cargar: {ruta}")
                continue

            if len(imagen.shape) == 2:
                pass
            elif imagen.shape[2] == 4:
                imagen = cv2.cvtColor(imagen, cv2.COLOR_BGRA2GRAY)
            else:
                imagen = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)

            imagen = cv2.resize(imagen, (ancho, alto))
            imagen = imagen.astype("float32") / 255.0
            imagen = imagen[..., np.newaxis]
            imagenesCargadas.append(imagen)

            probabilidades = np.zeros(len(categorias))
            probabilidades[idxCategoria] = 1
            valorEsperado.append(probabilidades)

    imagenes = np.array(imagenesCargadas, dtype="float32")
    valoresEsperados = np.array(valorEsperado, dtype="float32")
    return imagenes, valoresEsperados


# =========================
# CONFIGURACIÓN
# =========================
ancho = 28
alto = 28

numeroCanales = 1
numeroCategorias = 5

porcentajeValidacion = 0.2



# =========================
# CARGAR TRAIN
# =========================
cat = ['carta_1', 'carta_2', 'carta_3', 'carta_4', 'carta_5']
imagenes, probabilidades = cargarDatos(
    "images/test/",
    cat
)


# =========================
# SEPARAR TRAIN Y VALIDACIÓN
# =========================
x_train, x_val, y_train, y_val = train_test_split(
    imagenes,
    probabilidades,
    test_size=porcentajeValidacion,
    random_state=42,
    shuffle=True,
    stratify=np.argmax(probabilidades, axis=1)
)

print("Datos entrenamiento:", x_train.shape)
print("Datos validación   :", x_val.shape)


# =========================
# CREAR MODELO
# =========================
model = Sequential([
    Input(shape=(ancho, alto, numeroCanales)),

    Conv2D(
        filters=16,
        kernel_size=5,
        strides=2,
        padding="same",
        activation="relu",
        name="capa_1"
    ),
    MaxPooling2D(pool_size=2, strides=2),

    Conv2D(
        filters=36,
        kernel_size=3,
        strides=1,
        padding="same",
        activation="relu",
        name="capa_2"
    ),
    MaxPooling2D(pool_size=2, strides=2),

    Flatten(),
    Dense(128, activation="relu"),
    Dense(numeroCategorias, activation="softmax")
])


# =========================
# COMPILAR
# =========================
model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)


# =========================
# EARLY STOPPING
# =========================
early_stop = EarlyStopping(
    monitor="val_loss",
    patience=3,
    min_delta=0.0001,
    restore_best_weights=True,
    verbose=1
)


# =========================
# ENTRENAR
# =========================
historial = model.fit(
    x=x_train,
    y=y_train,
    validation_data=(x_val, y_val),
    epochs=100,
    batch_size=60,
    callbacks=[early_stop],
    shuffle=True
)


# =========================
# CURVA DE PÉRDIDA
# =========================
plt.figure()
plt.plot(historial.history["loss"], label="Pérdida entrenamiento")
plt.plot(historial.history["val_loss"], label="Pérdida validación")
plt.xlabel("Época")
plt.ylabel("Pérdida")
plt.title("Curva de pérdida")
plt.legend()
plt.grid(True)
plt.show()


# =========================
# CARGAR DATOS DE PRUEBA
# =========================

cat = ['carta_1', 'carta_2', 'carta_3', 'carta_4', 'carta_5']
imagenesPrueba, probabilidadesPrueba = cargarDatos(
    "images/test/",
    cat
)


# =========================
# EVALUAR MODELO
# =========================
resultados = model.evaluate(
    x=imagenesPrueba,
    y=probabilidadesPrueba
)

print("Loss test =", resultados[0])
print("Accuracy test =", resultados[1])


# =========================
# MÉTRICAS SKLEARN Y MATRIZ DE CONFUSIÓN
# =========================
predicciones = model.predict(imagenesPrueba)

y_pred = np.argmax(predicciones, axis=1)
y_true = np.argmax(probabilidadesPrueba, axis=1)

print("\nReporte de clasificación:")
print(classification_report(y_true, y_pred, digits=4))

matriz = confusion_matrix(y_true, y_pred)

disp = ConfusionMatrixDisplay(
    confusion_matrix=matriz,
    display_labels=list(range(numeroCategorias))
)

disp.plot(cmap="Blues")
plt.title("Matriz de confusión")
plt.show()


# =========================
# GUARDAR MODELO
# =========================
ruta = "models/modeloA.keras"
model.save(ruta)

model.summary()
import os
import shutil
import random

# ─────────────────────────────────────────────
# CONFIGURACIÓN — ajusta esta ruta
# ─────────────────────────────────────────────
BASE_DIR   = "."          # Ruta raíz del repositorio (cambia si es necesario)
TRAIN_DIR  = os.path.join(BASE_DIR, "images/train")
TEST_DIR   = os.path.join(BASE_DIR, "images/test")
CARPETAS   = ["carta_1", "carta_2", "carta_3", "carta_4", "carta_5"]
TRAIN_RATIO = 0.80
SEED        = 42          # Semilla para reproducibilidad (pon None para aleatorio puro)

EXTENSIONES_VALIDAS = {".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp", ".tiff"}
# ─────────────────────────────────────────────

def es_imagen(nombre: str) -> bool:
    return os.path.splitext(nombre)[1].lower() in EXTENSIONES_VALIDAS


def dividir_carpeta(nombre_carpeta: str):
    origen = os.path.join(BASE_DIR, nombre_carpeta)

    if not os.path.isdir(origen):
        print(f"  [AVISO] No se encontró la carpeta: {origen}")
        return

    imagenes = [f for f in os.listdir(origen) if es_imagen(f)]

    if not imagenes:
        print(f"  [AVISO] No hay imágenes en: {origen}")
        return

    random.shuffle(imagenes)

    corte       = int(len(imagenes) * TRAIN_RATIO)
    train_imgs  = imagenes[:corte]
    test_imgs   = imagenes[corte:]

    dest_train = os.path.join(TRAIN_DIR, nombre_carpeta)
    dest_test  = os.path.join(TEST_DIR,  nombre_carpeta)
    os.makedirs(dest_train, exist_ok=True)
    os.makedirs(dest_test,  exist_ok=True)

    for img in train_imgs:
        shutil.copy2(os.path.join(origen, img), os.path.join(dest_train, img))

    for img in test_imgs:
        shutil.copy2(os.path.join(origen, img), os.path.join(dest_test, img))

    print(f"  {nombre_carpeta}: {len(train_imgs)} → train  |  {len(test_imgs)} → test")


def main():
    if SEED is not None:
        random.seed(SEED)

    print("=" * 50)
    print("  División de dataset  80 % train / 20 % test")
    print("=" * 50)

    for carpeta in CARPETAS:
        dividir_carpeta(carpeta)

    print("=" * 50)
    print("¡Listo! Imágenes copiadas correctamente.")
    print(f"  train/ → {TRAIN_DIR}")
    print(f"  test/  → {TEST_DIR}")


if __name__ == "__main__":
    main()
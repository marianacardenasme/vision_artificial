# Vision Artificial - Primera Sesion

Proyecto de practica con OpenCV para captura de video, mascaras por color, bordes Canny, operaciones morfologicas y deteccion de poligonos en tiempo real.

## Requisitos

- Python 3.10 o superior
- Camara web disponible
- Sistema operativo con soporte para ventanas de OpenCV

## Instalacion del proyecto

### 1. Clonar el repositorio

```bash
git clone <url-del-repositorio>
cd primera_sesion
```

### 2. Crear entorno virtual

En Windows (PowerShell):

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

En Linux/macOS:

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Instalar dependencias

```bash
pip install -r requirements.txt
```

## Ejecucion de scripts

Ejecuta cualquiera de los siguientes archivos:

```bash
python 4_capturaVideo.py
python 5_calculadora_mascaras_rgb.py
python 6_calculadora_canny.py
python 7_calculadora_canny_erosion.py
python 8_deteccion_poligonos.py
```

## Contenido del proyecto

- `4_capturaVideo.py`: Visualizacion de la camara en mosaico (BGR, grises, HSV y CIE Lab).
- `5_calculadora_mascaras_rgb.py`: Calculadora interactiva de mascaras en BGR, HSV o CIE Lab.
- `6_calculadora_canny.py`: Ajuste de umbrales y suavizado para deteccion de bordes Canny.
- `7_calculadora_canny_erosion.py`: Canny con operaciones morfologicas y selector de modo.
- `8_deteccion_poligonos.py`: Deteccion y clasificacion de poligonos sobre la imagen capturada.

## Controles generales

- En la mayoria de scripts se usa `q` para salir.
- En algunos scripts se usa `ESC` para salir.
- Ajusta los trackbars en cada ventana para modificar parametros en tiempo real.

## Solucion de problemas

- Si aparece error de camara, verifica que no este en uso por otra aplicacion.
- Si OpenCV no abre ventanas, revisa permisos de acceso a camara y entorno grafico.
- Si `pip` no reconoce comandos, activa primero el entorno virtual.

# Sistema de estimación de la densidad aparente de madera de balsa

Sistema mecatrónico que mide, sin contacto, las dimensiones de una probeta de madera de balsa con dos celulares como cámaras, la pesa con una celda de carga y calcula su densidad aparente (masa dividida entre volumen). Proyecto Integrador, Ingeniería Mecatrónica, Universidad Santo Tomás Bucaramanga, 2026.

## Cómo funciona

1. La probeta se coloca sobre la plataforma de la celda de carga.
2. Dos celulares (iPhone 15 Pro Max y iPhone 11) con la app DroidCam envían video por WiFi al computador.
3. Un modelo U-Net separa la probeta del fondo en cada imagen y se miden sus dimensiones en centímetros con la calibración de cada cámara.
4. Las medidas de las dos cámaras se combinan en el volumen.
5. El Arduino UNO Q pesa la probeta y controla la iluminación LED. El computador le pide la masa y las luces por WiFi.
6. La densidad se calcula como masa dividida entre volumen.

## Estructura del repositorio

| Carpeta | Contenido |
|---|---|
| linux/captura | Conexión con las cámaras IP |
| linux/calibracion | Calibración de cámaras (método de Zhang) y ajuste de escala |
| linux/segmentacion | Prueba en vivo del modelo U-Net |
| linux/medicion_dimensional | Medición del ancho y el alto de la cara de la probeta |
| linux/calculo_densidad | Medición con las dos cámaras, volumen y densidad |
| linux/registro, linux/interfaz_web | En desarrollo |
| entrenamiento | Cuaderno de Colab que entrena la U-Net y programas para crear el dataset |
| bridge | App del UNO Q en Python: servidor web, balanza y luces por el Bridge |
| stm32 | Sketch del microcontrolador: balanza HX711 e iluminación PCA9685 |
| tests | Programa de prueba de la estación |
| docs | Requisitos, matrices QFD y Pugh, diagramas |

## Requisitos

- Computador con Windows y Python 3.
- Librerías: opencv-python, numpy y onnxruntime.
- Dos celulares con DroidCam, en la misma red WiFi que el computador.
- Arduino UNO Q con App Lab, celda de carga de 1 kg con HX711 y driver PCA9685 con módulo MOSFET y 4 módulos LED.

## Instalación

pip install opencv-python numpy onnxruntime

## Ejecución del lado del computador

Los programas se importan entre sí y leen sus archivos desde su propia carpeta, así que se ejecutan juntos en una carpeta de trabajo. Desde la raíz del repositorio, en CMD:

mkdir trabajo
for /r linux %f in (*.py) do copy "%f" trabajo

Copie también a la carpeta trabajo el modelo unet_balsa.onnx. Luego, dentro de trabajo:

1. python calibrar_stream.py : calibra cada cámara y guarda un archivo .npz. La primera vez se crea config.json, donde se escribe la IP de cada celular.
2. python ajustar_escala_modelo.py : ajusta la escala de cada cámara con una probeta de medidas conocidas.
3. python medir_dos_camaras_modelo.py : mide con las dos cámaras, calcula el volumen y la densidad y guarda un CSV.

## Ejecución del lado del UNO Q

1. En App Lab, crear la app estacion_balsa y copiar en ella stm32/celda_carga/sketch.ino, bridge/main.py y bridge/app.yaml.
2. Pulsar Run. La app abre un servidor en el puerto 8000.
3. En el computador, escribir la IP del UNO Q en tests/probar_estacion.py y ejecutarlo para probar las luces y la balanza.

## Estado actual

Hecho: calibración de las dos cámaras, segmentación con U-Net, medición dimensional con dos cámaras, balanza calibrada con la probeta de referencia, control de la iluminación por cámara y comunicación de la masa por WiFi.

Pendiente: unir cámaras, luces y masa en un solo flujo automático, registro de las mediciones, interfaz web y validación con varias probetas.

## Autores

Catalina Alarcón Zabala, Kizzye Vanessa Hidalgo Ariza y Marlon Yesid Rodríguez Mateus.

"""
Genera un patron de tablero de ajedrez para calibracion de camara
(metodo de Zhang), ajustado para caber en una hoja tamano oficio
(21.6 cm x 33.0 cm), dejando margen. El tamano de cuadro se calcula
en pixeles a una resolucion conocida (300 DPI) para que, al imprimir
SIN escalar ("tamano real" / "100%", no "ajustar a pagina"), el
tablero completo quede impreso con las medidas reales indicadas.

Patron: 9x6 esquinas internas (10x7 cuadros).
"""

import cv2
import numpy as np

ESQUINAS_ANCHO = 9   # esquinas internas horizontales
ESQUINAS_ALTO = 6    # esquinas internas verticales
DPI = 300

PAPEL_ANCHO_CM = 21.6   # oficio
PAPEL_ALTO_CM = 33.0    # oficio
MARGEN_CM = 1.0         # margen a cada lado

cuadros_ancho = ESQUINAS_ANCHO + 1
cuadros_alto = ESQUINAS_ALTO + 1

ancho_disponible_cm = PAPEL_ANCHO_CM - 2 * MARGEN_CM
alto_disponible_cm = PAPEL_ALTO_CM - 2 * MARGEN_CM

# El tamano de cuadro se limita por el lado del papel que resulte mas
# estrecho para el patron (normalmente el ancho, en hoja vertical).
tamano_cuadro_cm = min(ancho_disponible_cm / cuadros_ancho,
                        alto_disponible_cm / cuadros_alto)
# Redondeo hacia abajo a 1 decimal, para que el numero sea facil de
# verificar con una regla comun.
tamano_cuadro_cm = int(tamano_cuadro_cm * 10) / 10

pixeles_por_cm = DPI / 2.54
tamano_cuadro_px = int(tamano_cuadro_cm * pixeles_por_cm)

ancho_px = cuadros_ancho * tamano_cuadro_px
alto_px = cuadros_alto * tamano_cuadro_px

patron = np.zeros((alto_px, ancho_px), dtype=np.uint8)
for fila in range(cuadros_alto):
    for columna in range(cuadros_ancho):
        if (fila + columna) % 2 == 0:
            y0, y1 = fila * tamano_cuadro_px, (fila + 1) * tamano_cuadro_px
            x0, x1 = columna * tamano_cuadro_px, (columna + 1) * tamano_cuadro_px
            patron[y0:y1, x0:x1] = 255

cv2.imwrite("patron_calibracion.png", patron)

print(f"Patron guardado: patron_calibracion.png")
print(f"Tamano de cuadro calculado: {tamano_cuadro_cm} cm por lado")
print(f"Tamano total del tablero: {ancho_px/pixeles_por_cm:.1f} cm x {alto_px/pixeles_por_cm:.1f} cm")
print(f"(cabe en hoja oficio {PAPEL_ANCHO_CM} cm x {PAPEL_ALTO_CM} cm con {MARGEN_CM} cm de margen)")
print(f"IMPORTANTE: imprimir a {DPI} DPI, escala 100% / 'tamano real' (NO 'ajustar a pagina').")
print(f"Verifica con una regla que cada cuadro mida exactamente {tamano_cuadro_cm} cm por lado tras imprimir.")

"""
medir_con_modelo.py
Paso 17: medicion automatica de la cara frontal de la probeta usando el modelo U-Net.

Flujo: pones la probeta en la posicion; cuando el modelo la ve quieta durante
unos cuadros, el programa mide solo. Para medir otra vez, retira la probeta
(se ve SIN PROBETA) y vuelve a ponerla. No usa fondo ni teclas para medir.

Escala: cm/px = distancia / fx * factor (misma calibracion de medir_probeta.py).
Si la cara esta girada, el largo y el ancho se calculan con el promedio de los
lados opuestos del cuadrilatero.

Teclas (ventana del video activa):
  ENTER : terminar la configuracion de camara (solo al inicio)
  Q     : terminar y ver el resumen
"""
import csv
import json
import os
import time
from collections import deque
from datetime import datetime

import cv2
import numpy as np

from calibrar_stream import DISTANCIAS_CM, LectorStream
import probar_modelo_vivo as pm

BASE = os.path.dirname(os.path.abspath(__file__))
CARPETA_SALIDA = os.path.join(BASE, "pruebas_medicion")
RUTA_CSV = os.path.join(CARPETA_SALIDA, "mediciones_modelo.csv")

CUADROS_ESTABLES = 10       # cuadros seguidos quietos para medir
MOVIMIENTO_MAXIMO = 1.5     # px maximo que se mueve una esquina entre cuadros
CUADROS_AUSENTE = 3         # cuadros sin probeta para rearmar
CAMBIO_MINIMO = 8.0         # px que debe moverse una esquina para contar como otra probeta
CUADROS_CAMBIO = 3          # cuadros seguidos con ese cambio para rearmar
FUENTE = cv2.FONT_HERSHEY_SIMPLEX


# ---------------------------------------------------------------- escala
def cargar_escala(nombre_corto, nombre_camara):
    ruta = os.path.join(BASE, f"calibracion_stream_{nombre_corto}.npz")
    if not os.path.exists(ruta):
        raise FileNotFoundError(f"No encuentro {ruta}.")
    datos = np.load(ruta, allow_pickle=True)
    fx = float(datos["matriz_camara"][0, 0])
    if "distancia_cm" in datos.files:
        distancia = float(datos["distancia_cm"])
    else:
        distancia = DISTANCIAS_CM.get(nombre_camara, 36.0)
    factor = 1.0
    ruta_f = os.path.join(BASE, f"ajuste_escala_{nombre_corto}.json")
    if os.path.exists(ruta_f):
        with open(ruta_f, "r", encoding="utf-8") as f:
            factor = float(json.load(f).get("factor", 1.0))
    return distancia / fx * factor, fx, distancia, factor


# -------------------------------------------------------------- medicion
def area_poligono(q):
    x, y = q[:, 0], q[:, 1]
    return 0.5 * abs(float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))))


def medidas_de(lados_px, area_px, escala):
    """Largo y ancho en cm con el promedio de lados opuestos."""
    a = 0.5 * (lados_px[0] + lados_px[2])
    b = 0.5 * (lados_px[1] + lados_px[3])
    return {
        "largo_cm": max(a, b) * escala,
        "ancho_cm": min(a, b) * escala,
        "area_cm2": area_px * escala ** 2,
    }


class Disparador:
    """Decide cuando medir: probeta quieta N cuadros.
    Se rearma para medir otra vez si:
      - la probeta desaparece unos cuadros (la retiraste), o
      - la cara cambia de posicion o tamano (pusiste otra o la moviste)."""

    def __init__(self):
        self.ventana = deque(maxlen=CUADROS_ESTABLES)
        self.ausentes = 0
        self.cambios = 0
        self.armado = True
        self.quad_medido = None

    def _rearmar(self):
        self.armado = True
        self.cambios = 0
        self.ventana.clear()

    def actualizar(self, q):
        """q: esquinas (4x2) o None. Devuelve (estado, lista_de_quads_para_medir)."""
        if q is None:
            self.ausentes += 1
            self.ventana.clear()
            if self.ausentes >= CUADROS_AUSENTE and not self.armado:
                self._rearmar()
            return "ESPERANDO", None
        self.ausentes = 0

        if not self.armado:
            if np.abs(q - self.quad_medido).max() > CAMBIO_MINIMO:
                self.cambios += 1
            else:
                self.cambios = 0
            if self.cambios >= CUADROS_CAMBIO:
                self._rearmar()
            else:
                return "MEDIDO - cambia la probeta para otra medicion", None

        if self.ventana and np.abs(q - self.ventana[-1]).max() > MOVIMIENTO_MAXIMO:
            self.ventana.clear()
        self.ventana.append(q)
        if len(self.ventana) < CUADROS_ESTABLES:
            return "ESTABILIZANDO %d/%d" % (len(self.ventana), CUADROS_ESTABLES), None
        quads = list(self.ventana)
        self.ventana.clear()
        self.armado = False
        self.cambios = 0
        self.quad_medido = np.median(np.array(quads), axis=0)
        return "MEDIR", quads


def mediana_quads(quads):
    return np.median(np.array(quads), axis=0).astype(np.float32)


# ---------------------------------------------------------------- dibujo
def dibujar(frame, mask, q, metodo, estado, ultima, esperado, ms, n):
    vista = frame.copy()
    if q is not None:
        capa = vista.copy()
        capa[mask > 0] = (0, 255, 0)
        vista = cv2.addWeighted(capa, 0.25, vista, 0.75, 0)
        cv2.polylines(vista, [q.astype(np.int32)], True, (0, 255, 0), 2)
        for pt in q:
            cv2.circle(vista, tuple(int(v) for v in pt), 5, (0, 0, 255), -1)
    color = (0, 255, 255) if estado.startswith("ESTAB") else (
        (0, 255, 0) if estado.startswith("MEDIDO") else (255, 255, 255))
    cv2.putText(vista, estado, (20, 40), FUENTE, 0.9, color, 2)
    if ultima is not None:
        l1 = "Ultima: largo %.2f cm  ancho %.2f cm  area %.2f cm2" % (
            ultima["largo_cm"], ultima["ancho_cm"], ultima["area_cm2"])
        cv2.putText(vista, l1, (20, 80), FUENTE, 0.8, (0, 255, 255), 2)
        if esperado is not None:
            el = 100 * (ultima["largo_cm"] - esperado[0]) / esperado[0]
            ea = 100 * (ultima["ancho_cm"] - esperado[1]) / esperado[1]
            cv2.putText(vista, "Error: largo %+.1f %%  ancho %+.1f %%" % (el, ea),
                        (20, 115), FUENTE, 0.8, (0, 255, 255), 2)
    cv2.putText(vista, "Mediciones: %d   %.0f ms   Q = terminar" % (n, ms),
                (20, vista.shape[0] - 20), FUENTE, 0.7, (255, 255, 255), 2)
    return vista


def resumen(filas, esperado):
    print("\n==== Resumen ====")
    if not filas:
        print("No hubo mediciones.")
        return
    for k, nombre in enumerate(("largo_cm", "ancho_cm", "area_cm2")):
        v = np.array([f[nombre] for f in filas])
        print("%-9s media %.3f   desviacion %.3f   (n=%d)" % (nombre, v.mean(), v.std(), len(v)))
    if esperado is not None:
        lg = np.mean([f["largo_cm"] for f in filas])
        an = np.mean([f["ancho_cm"] for f in filas])
        print("Error medio: largo %+.2f %%   ancho %+.2f %%"
              % (100 * (lg - esperado[0]) / esperado[0],
                 100 * (an - esperado[1]) / esperado[1]))


def guardar_csv(fila, camara, esperado, lados_px):
    os.makedirs(CARPETA_SALIDA, exist_ok=True)
    nuevo = not os.path.exists(RUTA_CSV)
    with open(RUTA_CSV, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if nuevo:
            w.writerow(["fecha", "camara", "largo_cm", "ancho_cm", "area_cm2",
                        "lado1_px", "lado2_px", "lado3_px", "lado4_px",
                        "largo_esperado", "ancho_esperado"])
        w.writerow([datetime.now().isoformat(timespec="seconds"), camara,
                    "%.3f" % fila["largo_cm"], "%.3f" % fila["ancho_cm"],
                    "%.3f" % fila["area_cm2"]]
                   + ["%.1f" % v for v in lados_px]
                   + ["" if esperado is None else esperado[0],
                      "" if esperado is None else esperado[1]])


def main():
    from camara import CamaraIP, cargar_configuracion

    if not os.path.exists(pm.RUTA_MODELO):
        print("No encuentro", pm.RUTA_MODELO)
        return
    sesion, entrada = pm.cargar_modelo()

    config = cargar_configuracion()
    claves = list(config.keys())
    print("Camaras disponibles:")
    for i, clave in enumerate(claves, start=1):
        print(f"  {i}. {config[clave]['nombre']} ({config[clave]['ip']})")
    datos = config[claves[int(input("Numero de camara: ").strip()) - 1]]
    nombre = datos["nombre"]
    nombre_corto = nombre.lower().replace(" pro max", "").replace(" ", "")
    nueva_ip = input(f"IP de {nombre} (ENTER para usar {datos['ip']}): ").strip()
    if nueva_ip:
        datos["ip"] = nueva_ip

    escala, fx, distancia, factor = cargar_escala(nombre_corto, nombre)
    print(f"Calibracion: fx = {fx:.1f} px, distancia = {distancia} cm, "
          f"factor = {factor:.4f}  ->  escala {escala:.4f} cm/px")
    esperado = None   # no se piden medidas: el programa mide lo que vea

    cam = CamaraIP(datos)
    print(f"\nConectando con {nombre} ({cam.base_url}) ...")
    try:
        cam.iniciar_sesion()
    except ConnectionError as e:
        print(e)
        return
    print("Conectado. Fija EV, WB y AF y pulsa ENTER en la ventana del video.")

    ventana = f"Medicion automatica {nombre}"
    lector = LectorStream(cam._captura)
    disp = Disparador()
    filas = []
    ultima = None
    try:
        if not pm.configurar_camara(lector, ventana):
            return
        while True:
            frame, _ = lector.ultimo()
            if frame is None:
                time.sleep(0.01)
                continue
            if not lector.vivo:
                print("Se perdio la conexion con el stream.")
                break

            t0 = time.time()
            mask = pm.predecir(sesion, entrada, frame)
            contorno, mask = pm.mayor_region(mask)
            ms = (time.time() - t0) * 1000.0

            q, metodo = None, ""
            area = 0.0 if contorno is None else cv2.contourArea(contorno)
            if contorno is not None and area >= pm.AREA_MINIMA * frame.shape[0] * frame.shape[1]:
                q, metodo = pm.cuadrilatero(contorno)
            else:
                mask[:] = 0

            estado, quads = disp.actualizar(q)
            if quads is not None:
                qm = mediana_quads(quads)
                lp = pm.lados(qm)
                ultima = medidas_de(lp, area_poligono(qm), escala)
                filas.append(ultima)
                guardar_csv(ultima, nombre, esperado, lp)
                print("Medicion %d: largo %.2f cm  ancho %.2f cm  area %.2f cm2"
                      % (len(filas), ultima["largo_cm"], ultima["ancho_cm"], ultima["area_cm2"]))
                estado = "MEDIDO - cambia la probeta para otra medicion"

            cv2.imshow(ventana, dibujar(frame, mask, q, metodo, estado, ultima,
                                        esperado, ms, len(filas)))
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        lector.detener()
        cv2.destroyAllWindows()
        cam.cerrar_sesion()
        resumen(filas, esperado)
        if filas:
            print("Mediciones guardadas en:", RUTA_CSV)


if __name__ == "__main__":
    main()

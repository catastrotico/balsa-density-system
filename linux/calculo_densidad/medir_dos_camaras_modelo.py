"""
medir_dos_camaras_modelo.py
Paso 19: medicion automatica con las DOS camaras y el modelo U-Net.

Cada camara ve una cara distinta de la probeta y la mide con su propia calibracion.
Las dos caras comparten una arista. El volumen se calcula como

    volumen = medida_compartida (promedio) x otra_medida_cam1 x otra_medida_cam2
    densidad = masa / volumen

La medicion es automatica: pones la probeta, cada camara espera a verla quieta,
mide, y cuando las dos midieron se registra el volumen. Para otra probeta, retirala
o cambiala: se rearma sola. No hay fondo ni teclas para medir.

Teclas (ventana del video activa):
  ENTER : terminar la configuracion de camaras (solo al inicio)
  Q     : terminar y ver el resumen
"""
import csv
import os
import time
from datetime import datetime

import cv2
import numpy as np

from calibrar_stream import LectorStream
import probar_modelo_vivo as pm
import medir_con_modelo as mc

CARPETA_SALIDA = mc.CARPETA_SALIDA
RUTA_CSV = os.path.join(CARPETA_SALIDA, "mediciones_dual_modelo.csv")
ANCHO_PANEL, ALTO_PANEL = 640, 360
TIEMPO_MAX_PAR = 20.0      # s maximos entre la medicion de una camara y la de la otra
DIF_COMPARTIDA_MAX = 0.08  # si las dos medidas "compartidas" difieren mas, se avisa
FUENTE = cv2.FONT_HERSHEY_SIMPLEX


# ------------------------------------------------------------ calculo
def volumen_y_densidad(m1, m2, masa_g):
    """Combina las dos caras. Devuelve un dict con el volumen y la densidad."""
    d1 = [m1["largo_cm"], m1["ancho_cm"]]
    d2 = [m2["largo_cm"], m2["ancho_cm"]]
    mejor = None
    for i in range(2):
        for j in range(2):
            dif = abs(d1[i] - d2[j]) / (0.5 * (d1[i] + d2[j]))
            if mejor is None or dif < mejor[0]:
                mejor = (dif, i, j)
    dif, i, j = mejor
    compartida = 0.5 * (d1[i] + d2[j])
    x = d1[1 - i]
    y = d2[1 - j]
    volumen = compartida * x * y
    return {
        "volumen_cm3": volumen,
        "compartida_cm": compartida,
        "otra_cam1_cm": x,
        "otra_cam2_cm": y,
        "dif_compartida": dif,
        "densidad": (masa_g / volumen) if masa_g else None,
    }


def emparejar(e1, e2, ahora):
    """Devuelve (m1, m2) si las dos camaras tienen medicion nueva y reciente."""
    if not (e1.nueva and e2.nueva):
        return None
    if abs(e1.t_ultima - e2.t_ultima) <= TIEMPO_MAX_PAR:
        e1.nueva = e2.nueva = False
        return e1.ultima, e2.ultima
    # no son de la misma colocacion: se descarta la mas vieja
    if e1.t_ultima < e2.t_ultima:
        e1.nueva = False
    else:
        e2.nueva = False
    return None


# ------------------------------------------------------------ estacion
class Estacion:
    """Una camara con su calibracion, su video y su disparador de medicion."""

    def __init__(self, cam, datos, sesion, entrada):
        self.cam = cam
        self.nombre = datos["nombre"]
        self.nombre_corto = self.nombre.lower().replace(" pro max", "").replace(" ", "")
        self.escala, self.fx, self.distancia, self.factor = mc.cargar_escala(
            self.nombre_corto, self.nombre)
        self.sesion = sesion
        self.entrada = entrada
        self.disp = mc.Disparador()
        self.lector = None
        self.ultima = None
        self.t_ultima = 0.0
        self.nueva = False
        self.n = 0
        self.estado = "ESPERANDO"
        self.q = None
        self.mask = None
        self.ms = 0.0

    def abrir(self):
        self.lector = LectorStream(self.cam._captura)

    def cerrar(self):
        if self.lector is not None:
            self.lector.detener()
        try:
            self.cam.cerrar_sesion()
        except Exception:
            pass

    def procesar(self):
        """Lee un cuadro, corre el modelo y actualiza el estado. Devuelve el cuadro."""
        frame, _ = self.lector.ultimo()
        if frame is None:
            return None
        if not self.lector.vivo:
            raise ConnectionError("Se perdio la conexion con " + self.nombre)
        t0 = time.time()
        mask = pm.predecir(self.sesion, self.entrada, frame)
        contorno, mask = pm.mayor_region(mask)
        self.ms = (time.time() - t0) * 1000.0

        q = None
        area = 0.0 if contorno is None else cv2.contourArea(contorno)
        if contorno is not None and area >= pm.AREA_MINIMA * frame.shape[0] * frame.shape[1]:
            q, _ = pm.cuadrilatero(contorno)
        else:
            mask[:] = 0
        self.q, self.mask = q, mask

        estado, quads = self.disp.actualizar(q)
        if quads is not None:
            qm = mc.mediana_quads(quads)
            self.ultima = mc.medidas_de(pm.lados(qm), mc.area_poligono(qm), self.escala)
            self.ultima["lados_px"] = pm.lados(qm)
            self.t_ultima = time.time()
            self.nueva = True
            self.n += 1
            print("  %s: largo %.2f cm  ancho %.2f cm"
                  % (self.nombre, self.ultima["largo_cm"], self.ultima["ancho_cm"]))
            estado = "MEDIDO - cambia la probeta para otra medicion"
        self.estado = estado
        return frame

    def panel(self, frame):
        vista = frame.copy()
        if self.q is not None:
            capa = vista.copy()
            capa[self.mask > 0] = (0, 255, 0)
            vista = cv2.addWeighted(capa, 0.25, vista, 0.75, 0)
            cv2.polylines(vista, [self.q.astype(np.int32)], True, (0, 255, 0), 3)
            for pt in self.q:
                cv2.circle(vista, tuple(int(v) for v in pt), 7, (0, 0, 255), -1)
        vista = cv2.resize(vista, (ANCHO_PANEL, ALTO_PANEL))
        cv2.putText(vista, self.nombre, (10, 24), FUENTE, 0.6, (255, 255, 255), 2)
        color = (0, 255, 255) if self.estado.startswith("ESTAB") else (
            (0, 255, 0) if self.estado.startswith("MEDIDO") else (255, 255, 255))
        cv2.putText(vista, self.estado, (10, 52), FUENTE, 0.55, color, 2)
        if self.ultima is not None:
            cv2.putText(vista, "Ultima: %.2f x %.2f cm" % (
                self.ultima["largo_cm"], self.ultima["ancho_cm"]),
                (10, ALTO_PANEL - 12), FUENTE, 0.6, (0, 255, 255), 2)
        return vista


# ------------------------------------------------------------ interfaz
def barra(texto1, texto2=""):
    b = np.zeros((70, 2 * ANCHO_PANEL, 3), np.uint8)
    cv2.putText(b, texto1, (10, 28), FUENTE, 0.7, (0, 255, 255), 2)
    if texto2:
        cv2.putText(b, texto2, (10, 58), FUENTE, 0.7, (255, 255, 255), 2)
    return b


def configurar(estaciones, ventana):
    """Video de las dos camaras para fijar EV, WB y AF. ENTER continua, Q sale."""
    while True:
        paneles = []
        for e in estaciones:
            frame, _ = e.lector.ultimo()
            if frame is None:
                time.sleep(0.01)
                break
            if not e.lector.vivo:
                print("Se perdio la conexion con", e.nombre)
                return False
            paneles.append(cv2.resize(frame, (ANCHO_PANEL, ALTO_PANEL)))
        if len(paneles) < len(estaciones):
            continue
        img = np.vstack([np.hstack(paneles),
                         barra("CONFIGURACION: ajusta EV, WB y AF en las dos camaras",
                               "Al terminar pulsa ENTER en esta ventana")])
        cv2.imshow(ventana, img)
        tecla = cv2.waitKey(30) & 0xFF
        if tecla in (13, 32):
            return True
        if tecla == ord("q"):
            return False


def conectar(config, clave, sesion, entrada):
    datos = config[clave]
    nombre = datos["nombre"]
    nueva_ip = input(f"IP de {nombre} (ENTER para usar {datos['ip']}): ").strip()
    if nueva_ip:
        datos["ip"] = nueva_ip
    from camara import CamaraIP
    cam = CamaraIP(datos)
    print(f"Conectando con {nombre} ({cam.base_url}) ...")
    cam.iniciar_sesion()
    e = Estacion(cam, datos, sesion, entrada)
    print(f"  fx = {e.fx:.1f} px, distancia = {e.distancia} cm, factor = {e.factor:.4f}, "
          f"escala = {e.escala:.4f} cm/px")
    return e


def guardar_csv(e1, e2, r, masa):
    os.makedirs(CARPETA_SALIDA, exist_ok=True)
    nuevo = not os.path.exists(RUTA_CSV)
    with open(RUTA_CSV, "a", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        if nuevo:
            w.writerow(["fecha", "cam1", "cam1_largo_cm", "cam1_ancho_cm",
                        "cam2", "cam2_largo_cm", "cam2_ancho_cm",
                        "compartida_cm", "otra_cam1_cm", "otra_cam2_cm",
                        "dif_compartida", "volumen_cm3", "masa_g", "densidad_g_cm3"])
        w.writerow([datetime.now().isoformat(timespec="seconds"),
                    e1.nombre, "%.3f" % e1.ultima["largo_cm"], "%.3f" % e1.ultima["ancho_cm"],
                    e2.nombre, "%.3f" % e2.ultima["largo_cm"], "%.3f" % e2.ultima["ancho_cm"],
                    "%.3f" % r["compartida_cm"], "%.3f" % r["otra_cam1_cm"],
                    "%.3f" % r["otra_cam2_cm"], "%.4f" % r["dif_compartida"],
                    "%.2f" % r["volumen_cm3"],
                    "" if not masa else masa,
                    "" if r["densidad"] is None else "%.5f" % r["densidad"]])


def resumen(registros, masa):
    print("\n==== Resumen ====")
    if not registros:
        print("No hubo mediciones completas de las dos camaras.")
        return
    v = np.array([r["volumen_cm3"] for r in registros])
    print("Volumen: media %.2f cm3   desviacion %.2f   (n=%d)" % (v.mean(), v.std(), len(v)))
    if masa:
        d = np.array([r["densidad"] for r in registros])
        print("Densidad: media %.5f g/cm3   desviacion %.5f" % (d.mean(), d.std()))
    print("Mediciones guardadas en:", RUTA_CSV)


def main():
    from camara import cargar_configuracion

    if not os.path.exists(pm.RUTA_MODELO):
        print("No encuentro", pm.RUTA_MODELO)
        return
    sesion, entrada = pm.cargar_modelo()

    config = cargar_configuracion()
    claves = list(config.keys())
    print("Camara 1 =", config[claves[0]]["nombre"], "| Camara 2 =", config[claves[1]]["nombre"])

    texto = input("Masa de la probeta en gramos (ENTER si todavia no la tienes): ").strip().replace(",", ".")
    masa = float(texto) if texto else None

    estaciones = []
    registros = []
    try:
        for clave in claves[:2]:
            estaciones.append(conectar(config, clave, sesion, entrada))
        for e in estaciones:
            e.abrir()
        e1, e2 = estaciones
        ventana = "Medicion con dos camaras"
        print("Conectadas. Fija EV, WB y AF y pulsa ENTER en la ventana del video.")
        if not configurar(estaciones, ventana):
            return

        ultimo = None
        while True:
            f1 = e1.procesar()
            f2 = e2.procesar()
            if f1 is None or f2 is None:
                time.sleep(0.01)
                continue

            par = emparejar(e1, e2, time.time())
            if par is not None:
                r = volumen_y_densidad(par[0], par[1], masa)
                registros.append(r)
                guardar_csv(e1, e2, r, masa)
                ultimo = r
                linea = "Volumen %.2f cm3" % r["volumen_cm3"]
                if r["densidad"] is not None:
                    linea += "  densidad %.4f g/cm3" % r["densidad"]
                print("Medicion %d: %s (arista compartida %.2f cm)"
                      % (len(registros), linea, r["compartida_cm"]))
                if r["dif_compartida"] > DIF_COMPARTIDA_MAX:
                    print("  AVISO: las dos caras no comparten una medida parecida "
                          "(%.1f %% de diferencia). Revisa la probeta." % (100 * r["dif_compartida"]))

            if ultimo is None:
                t1 = "Esperando la medicion de las dos camaras"
            else:
                t1 = "Volumen %.2f cm3" % ultimo["volumen_cm3"]
                if ultimo["densidad"] is not None:
                    t1 += "   Densidad %.4f g/cm3" % ultimo["densidad"]
            t2 = "Mediciones completas: %d   %.0f + %.0f ms   Q = terminar" % (
                len(registros), e1.ms, e2.ms)
            img = np.vstack([np.hstack([e1.panel(f1), e2.panel(f2)]), barra(t1, t2)])
            cv2.imshow(ventana, img)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    except ConnectionError as e:
        print(e)
    finally:
        for e in estaciones:
            e.cerrar()
        cv2.destroyAllWindows()
        resumen(registros, masa)


if __name__ == "__main__":
    main()

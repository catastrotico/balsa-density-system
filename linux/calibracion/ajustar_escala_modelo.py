"""
ajustar_escala_modelo.py
Calcula el factor de ajuste de escala de una camara usando el modelo U-Net.

Pones una cara de medidas conocidas (la escribes una vez al inicio), el programa
mide 10 veces solo (cambia o gira la probeta entre una y otra) y calcula el factor
que corrige la escala. Al final te pregunta si lo guarda en
ajuste_escala_<camara>.json (el mismo archivo que usan los demas programas).

Teclas (ventana del video activa):
  ENTER : terminar la configuracion de camara (solo al inicio)
  Q     : terminar antes de las 10 mediciones
"""
import json
import os
import time
from datetime import datetime

import cv2
import numpy as np

from calibrar_stream import LectorStream
import probar_modelo_vivo as pm
import medir_con_modelo as mc

OBJETIVO = 10


def pedir_medidas():
    while True:
        entrada = input("Medidas reales de la cara que ve esta camara, en cm, "
                        "largo y ancho (ej. 7.65 6.98): ").strip().replace(",", ".")
        try:
            a, b = [float(x) for x in entrada.split()[:2]]
            return max(a, b), min(a, b)
        except ValueError:
            print("Escribe dos numeros separados por un espacio.")


def calcular_factor(filas, esperado, factor_actual):
    """Devuelve (factor_nuevo, correccion, detalle) a partir de las mediciones."""
    largo = np.array([f["largo_cm"] for f in filas])
    ancho = np.array([f["ancho_cm"] for f in filas])
    corr_l = esperado[0] / largo.mean()
    corr_a = esperado[1] / ancho.mean()
    correccion = 0.5 * (corr_l + corr_a)
    return factor_actual * correccion, correccion, (largo, ancho, corr_l, corr_a)


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

    escala, fx, distancia, factor = mc.cargar_escala(nombre_corto, nombre)
    print(f"Calibracion: fx = {fx:.1f} px, distancia = {distancia} cm, "
          f"factor actual = {factor:.4f}  ->  escala {escala:.4f} cm/px")
    esperado = pedir_medidas()

    cam = CamaraIP(datos)
    print(f"\nConectando con {nombre} ({cam.base_url}) ...")
    try:
        cam.iniciar_sesion()
    except ConnectionError as e:
        print(e)
        return
    print("Conectado. Fija EV, WB y AF y pulsa ENTER en la ventana del video.")
    print(f"Se haran {OBJETIVO} mediciones. Entre una y otra retira la probeta "
          "o muevela/giratela un poco.")

    ventana = f"Ajuste de escala {nombre}"
    lector = LectorStream(cam._captura)
    disp = mc.Disparador()
    filas = []
    ultima = None
    try:
        if not pm.configurar_camara(lector, ventana):
            return
        while len(filas) < OBJETIVO:
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
                qm = mc.mediana_quads(quads)
                lp = pm.lados(qm)
                ultima = mc.medidas_de(lp, mc.area_poligono(qm), escala)
                filas.append(ultima)
                print("Medicion %d/%d: largo %.2f cm  ancho %.2f cm"
                      % (len(filas), OBJETIVO, ultima["largo_cm"], ultima["ancho_cm"]))
                estado = "MEDIDO - cambia la probeta para otra medicion"

            cv2.imshow(ventana, mc.dibujar(frame, mask, q, metodo, estado, ultima,
                                           esperado, ms, len(filas)))
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        lector.detener()
        cv2.destroyAllWindows()
        cam.cerrar_sesion()

    if len(filas) < 3:
        print("Muy pocas mediciones para calcular un factor confiable.")
        return

    nuevo, correccion, (largo, ancho, corr_l, corr_a) = calcular_factor(filas, esperado, factor)
    print("\n==== Resultado ====")
    print("Largo: media %.3f cm, desviacion %.3f (real %.2f)" % (largo.mean(), largo.std(), esperado[0]))
    print("Ancho: media %.3f cm, desviacion %.3f (real %.2f)" % (ancho.mean(), ancho.std(), esperado[1]))
    print("Correccion por largo %.4f, por ancho %.4f" % (corr_l, corr_a))
    print("Factor actual %.4f  ->  factor nuevo %.4f" % (factor, nuevo))
    print("Con el factor nuevo el largo daria %.3f cm y el ancho %.3f cm"
          % (largo.mean() * correccion, ancho.mean() * correccion))
    if abs(corr_l - corr_a) > 0.03:
        print("AVISO: las correcciones de largo y ancho difieren mas de 3 %. "
              "Revisa que las medidas reales escritas sean de la cara que ve la camara.")

    if input("\nGuardar este factor para esta camara? (s/n): ").strip().lower() == "s":
        ruta = os.path.join(mc.BASE, f"ajuste_escala_{nombre_corto}.json")
        with open(ruta, "w", encoding="utf-8") as f:
            json.dump({"factor": nuevo, "fecha": datetime.now().isoformat(timespec="seconds")}, f)
        print("Guardado en", ruta)


if __name__ == "__main__":
    main()

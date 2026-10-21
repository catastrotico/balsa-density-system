"""
probar_modelo_vivo.py
Paso 16: prueba en vivo del modelo U-Net (unet_balsa.onnx) con la camara.
No usa fondo ni fotos de referencia: el modelo decide solo.

Teclas (con la ventana del video activa):
  ENTER  : terminar la configuracion de camara (solo al inicio)
  S      : guardar el cuadro actual y la prediccion en pruebas_vivo/
           (sirve para guardar los casos donde el modelo falle)
  Q      : salir
"""
import math
import os
import time
from datetime import datetime

import cv2
import numpy as np
import onnxruntime as ort

from calibrar_stream import LectorStream

RUTA_MODELO = "unet_balsa.onnx"
ALTO, ANCHO = 256, 448
UMBRAL = 0.5
AREA_MINIMA = 0.005      # menos de 0.5 % del cuadro = sin probeta
CARPETA_SALIDA = "pruebas_vivo"
FUENTE = cv2.FONT_HERSHEY_SIMPLEX


def cargar_modelo():
    sesion = ort.InferenceSession(RUTA_MODELO, providers=["CPUExecutionProvider"])
    return sesion, sesion.get_inputs()[0].name


def predecir(sesion, nombre_entrada, frame):
    """Devuelve la mascara binaria (uint8 0/255) al tamano del cuadro.
    Se amplian las probabilidades (suave) y luego se aplica el umbral,
    asi el borde no queda en escalera."""
    h, w = frame.shape[:2]
    rgb = cv2.cvtColor(cv2.resize(frame, (ANCHO, ALTO), interpolation=cv2.INTER_AREA),
                       cv2.COLOR_BGR2RGB)
    x = rgb.astype(np.float32)[None]
    p = sesion.run(None, {nombre_entrada: x})[0][0, :, :, 0]
    grande = cv2.resize(p, (w, h), interpolation=cv2.INTER_CUBIC)
    return (grande > UMBRAL).astype(np.uint8) * 255


def mayor_region(mask):
    """Deja solo la region mas grande de la mascara."""
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts:
        return None, mask
    c = max(cnts, key=cv2.contourArea)
    limpia = np.zeros_like(mask)
    cv2.drawContours(limpia, [c], -1, 255, -1)
    return c, limpia


def ordenar(p):
    """Ordena 4 puntos: arriba-izq, arriba-der, abajo-der, abajo-izq."""
    p = np.array(p, np.float32).reshape(4, 2)
    s = p.sum(axis=1)
    d = p[:, 0] - p[:, 1]
    return np.array([p[np.argmin(s)], p[np.argmax(d)],
                     p[np.argmax(s)], p[np.argmin(d)]], np.float32)


def esquinas_iniciales(contorno):
    """Primera estimacion de las 4 esquinas (aproximacion poligonal)."""
    casco = cv2.convexHull(contorno)
    per = cv2.arcLength(casco, True)
    for eps in np.linspace(0.01, 0.08, 15):
        aprox = cv2.approxPolyDP(casco, eps * per, True)
        if len(aprox) == 4:
            return ordenar(aprox), True
    caja = cv2.boxPoints(cv2.minAreaRect(casco))
    return ordenar(caja), False


def interseccion(r1, r2):
    """Cruce de dos rectas dadas como (vx, vy, x0, y0). None si son paralelas."""
    v1 = np.array(r1[:2], np.float64)
    p1 = np.array(r1[2:], np.float64)
    v2 = np.array(r2[:2], np.float64)
    p2 = np.array(r2[2:], np.float64)
    A = np.array([v1, -v2]).T
    if abs(np.linalg.det(A)) < 1e-6:
        return None
    t = np.linalg.solve(A, p2 - p1)
    return (p1 + t[0] * v1).astype(np.float32)


def cuadrilatero(contorno):
    """Cuatro esquinas como cruce de 4 rectas ajustadas a los lados.
    Cada recta se ajusta con los puntos del contorno del tramo central
    de su lado (se evitan las esquinas, donde el borde se redondea)."""
    q0, es_cuad = esquinas_iniciales(contorno)
    pts = contorno.reshape(-1, 2).astype(np.float32)
    diag = float(np.linalg.norm(q0[2] - q0[0]))
    rectas = []
    for i in range(4):
        a, b = q0[i], q0[(i + 1) % 4]
        d = b - a
        largo = float(np.linalg.norm(d))
        if largo < 1:
            return q0, "rectangulo"
        u = d / largo
        n = np.array([-u[1], u[0]], np.float32)
        rel = pts - a
        t = rel @ u / largo
        dist = np.abs(rel @ n)
        sel = pts[(t > 0.15) & (t < 0.85) & (dist < 0.05 * diag + 4)]
        if len(sel) < 10:
            return q0, "rectangulo"
        vx, vy, x0, y0 = cv2.fitLine(sel, cv2.DIST_HUBER, 0, 0.01, 0.01).flatten()
        rectas.append((vx, vy, x0, y0))
    q = []
    for i in range(4):
        pt = interseccion(rectas[i - 1], rectas[i])
        if pt is None or np.linalg.norm(pt - q0[i]) > 0.08 * diag:
            return q0, "rectangulo"
        q.append(pt)
    q = ordenar(q)
    if not cv2.isContourConvex(q.reshape(-1, 1, 2).astype(np.float32)):
        return q0, "rectangulo"
    return q, "rectas"


def lados(q):
    return [float(np.linalg.norm(q[(i + 1) % 4] - q[i])) for i in range(4)]


def dibujar(frame, mask, q, metodo, ms):
    vista = frame.copy()
    if q is not None:
        capa = vista.copy()
        capa[mask > 0] = (0, 255, 0)
        vista = cv2.addWeighted(capa, 0.30, vista, 0.70, 0)
        color = (0, 255, 0) if metodo == "rectas" else (0, 165, 255)
        cv2.polylines(vista, [q.astype(np.int32)], True, color, 2)
        for k, pt in enumerate(q):
            cv2.circle(vista, tuple(int(v) for v in pt), 5, (0, 0, 255), -1)
        l = lados(q)
        cv2.putText(vista, "Lados px: %.0f %.0f %.0f %.0f" % tuple(l),
                    (20, 40), FUENTE, 0.8, (0, 255, 255), 2)
        cv2.putText(vista, "PROBETA DETECTADA (%s)" % metodo, (20, 78),
                    FUENTE, 0.8, color, 2)
    else:
        cv2.putText(vista, "SIN PROBETA", (20, 40), FUENTE, 1.0, (0, 0, 255), 2)
    cv2.putText(vista, "%.0f ms   S = guardar   Q = salir" % ms,
                (20, vista.shape[0] - 20), FUENTE, 0.7, (255, 255, 255), 2)
    return vista


def esperar_cuadro(lector):
    while True:
        frame, _ = lector.ultimo()
        if frame is not None:
            return frame
        time.sleep(0.01)


def configurar_camara(lector, ventana):
    while True:
        frame, _ = lector.ultimo()
        if frame is None:
            time.sleep(0.01)
            continue
        if not lector.vivo:
            print("Se perdio la conexion con el stream.")
            return False
        vista = frame.copy()
        cv2.putText(vista, "CONFIGURACION: ajusta EV, WB y AF en DroidCam", (20, 40),
                    FUENTE, 0.9, (0, 255, 255), 2)
        cv2.putText(vista, "Al terminar pulsa ENTER", (20, 78),
                    FUENTE, 0.8, (255, 255, 255), 2)
        cv2.imshow(ventana, vista)
        tecla = cv2.waitKey(30) & 0xFF
        if tecla in (13, 32):
            return True
        if tecla == ord("q"):
            return False


def main():
    from camara import CamaraIP, cargar_configuracion

    if not os.path.exists(RUTA_MODELO):
        print("No encuentro", RUTA_MODELO, "en esta carpeta.")
        return
    sesion, entrada = cargar_modelo()
    print("Modelo cargado.")

    config = cargar_configuracion()
    claves = list(config.keys())
    print("Camaras disponibles:")
    for i, clave in enumerate(claves, start=1):
        print(f"  {i}. {config[clave]['nombre']} ({config[clave]['ip']})")
    datos = config[claves[int(input("Numero de camara: ").strip()) - 1]]
    nombre = datos["nombre"]
    nueva_ip = input(f"IP de {nombre} (ENTER para usar {datos['ip']}): ").strip()
    if nueva_ip:
        datos["ip"] = nueva_ip

    cam = CamaraIP(datos)
    print(f"\nConectando con {nombre} ({cam.base_url}) ...")
    try:
        cam.iniciar_sesion()
    except ConnectionError as e:
        print(e)
        return
    print("Conectado. Configura EV, WB y AF y pulsa ENTER en la ventana del video.")

    ventana = f"Prueba en vivo {nombre}"
    lector = LectorStream(cam._captura)
    guardadas = 0
    try:
        if not configurar_camara(lector, ventana):
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
            mask = predecir(sesion, entrada, frame)
            contorno, mask = mayor_region(mask)
            ms = (time.time() - t0) * 1000.0

            q, metodo = None, ""
            area = 0.0 if contorno is None else cv2.contourArea(contorno)
            if contorno is not None and area >= AREA_MINIMA * frame.shape[0] * frame.shape[1]:
                q, metodo = cuadrilatero(contorno)
            else:
                mask[:] = 0

            vista = dibujar(frame, mask, q, metodo, ms)
            cv2.imshow(ventana, vista)
            tecla = cv2.waitKey(1) & 0xFF
            if tecla == ord("q"):
                break
            if tecla == ord("s"):
                os.makedirs(CARPETA_SALIDA, exist_ok=True)
                marca = datetime.now().strftime("%Y%m%d_%H%M%S")
                cv2.imwrite(os.path.join(CARPETA_SALIDA, marca + "_cuadro.jpg"), frame)
                cv2.imwrite(os.path.join(CARPETA_SALIDA, marca + "_mascara.png"), mask)
                cv2.imwrite(os.path.join(CARPETA_SALIDA, marca + "_vista.jpg"), vista)
                guardadas += 1
                print("Guardado", marca, "| lados px:",
                      "-" if q is None else " ".join("%.0f" % v for v in lados(q)))
    finally:
        lector.detener()
        cv2.destroyAllWindows()
        cam.cerrar_sesion()
        print(f"\nGuardadas {guardadas} capturas en {CARPETA_SALIDA}")


if __name__ == "__main__":
    main()

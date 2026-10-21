"""
Prueba de medicion de la probeta de balsa con el stream de DroidCam.

Segmenta la probeta por resta de fondo: se toma una foto de referencia de la escena
VACIA (tecla B) y despues se detecta lo que cambia al poner la probeta. Asi no importa
que el fondo se parezca a la madera. Sin referencia usa saturacion Otsu (solo sirve
con fondo liso). Calcula largo, ancho y area de la cara que ve la camara y los pasa
de pixeles a cm con la calibracion:

    escala (cm/px) = distancia lente-cara / focal fx

Si escribes las medidas reales de la cara, compara y te dice el error. Con varias
mediciones calcula el promedio y la repetibilidad. Al final puede guardar un
factor de ajuste de escala (la probeta conocida sirve de referencia).

Controles en la ventana de video:
    B       -> guarda la referencia del fondo (escena SIN la probeta ni la mano)
    ESPACIO -> registra la medicion (usa la mediana de los ultimos cuadros)
    R       -> dibuja un recuadro para limitar la zona donde se busca la probeta
    C       -> quita el recuadro
    M       -> muestra u oculta la mascara de segmentacion
    Q       -> termina y muestra el resumen

Uso: python medir_probeta.py
"""

import csv
import glob
import json
import os
import time
from collections import deque
from datetime import datetime

import cv2
import numpy as np

from calibrar_stream import DISTANCIAS_CM, LectorStream

BASE = os.path.dirname(os.path.abspath(__file__))
CARPETA_SALIDA = os.path.join(BASE, "pruebas_medicion")
RUTA_CSV = os.path.join(CARPETA_SALIDA, "mediciones_probeta.csv")
BRILLO_MINIMO = 40            # descarta lo muy oscuro (fondo negro)
AREA_MINIMA_FRACCION = 0.01   # ignora manchas menores al 1 % de la imagen
CUADROS_MEDIANA = 10          # cuadros que se promedian al registrar
UMBRAL_DIFERENCIA_MIN = 15    # diferencia minima (0-255) contra el fondo para contar como probeta
FRACCION_MAXIMA = 0.6         # si lo detectado ocupa mas de esto, la luz cambio: se descarta
FUENTE = cv2.FONT_HERSHEY_SIMPLEX


# --------------------------------------------------------------------------
# Escala
# --------------------------------------------------------------------------

def ruta_ajuste(nombre_corto):
    return os.path.join(BASE, f"ajuste_escala_{nombre_corto}.json")


def cargar_escala(nombre_corto, nombre_camara):
    """Devuelve (escala_cm_por_px, fx, distancia_cm, factor) segun la calibracion guardada."""
    ruta = os.path.join(BASE, f"calibracion_stream_{nombre_corto}.npz")
    if not os.path.exists(ruta):
        raise FileNotFoundError(f"No encuentro {ruta}. Corre antes calibrar_stream.py para esa camara.")
    datos = np.load(ruta, allow_pickle=True)
    fx = float(datos["matriz_camara"][0, 0])
    if "distancia_cm" in datos.files:
        distancia = float(datos["distancia_cm"])
    else:
        distancia = DISTANCIAS_CM.get(nombre_camara, 36.0)
    factor = 1.0
    if os.path.exists(ruta_ajuste(nombre_corto)):
        with open(ruta_ajuste(nombre_corto), "r", encoding="utf-8") as f:
            factor = float(json.load(f).get("factor", 1.0))
    return distancia / fx * factor, fx, distancia, factor


# --------------------------------------------------------------------------
# Segmentacion y medicion
# --------------------------------------------------------------------------

def segmentar(imagen, roi=None, fondo=None):
    """
    Devuelve (contorno, mascara). El contorno esta en coordenadas de la imagen
    completa. Si no hay probeta devuelve (None, mascara).
    Con fondo (foto de la escena vacia, mismo tamano) detecta lo que cambia; sin el,
    usa saturacion Otsu.
    """
    x0, y0 = 0, 0
    zona = imagen
    ref = fondo
    if roi is not None:
        x0, y0, w, h = roi
        zona = imagen[y0:y0 + h, x0:x0 + w]
        if ref is not None:
            ref = ref[y0:y0 + h, x0:x0 + w]
    if ref is not None and ref.shape != zona.shape:
        ref = None

    if ref is not None:
        dif = cv2.absdiff(cv2.GaussianBlur(zona, (7, 7), 0), cv2.GaussianBlur(ref, (7, 7), 0))
        dif = np.ascontiguousarray(dif.max(axis=2))
        otsu, _ = cv2.threshold(dif, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        _, mascara = cv2.threshold(dif, max(otsu, UMBRAL_DIFERENCIA_MIN), 255, cv2.THRESH_BINARY)
    else:
        hsv = cv2.cvtColor(zona, cv2.COLOR_BGR2HSV)
        saturacion = cv2.GaussianBlur(hsv[:, :, 1], (7, 7), 0)
        _, m_sat = cv2.threshold(saturacion, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        _, m_brillo = cv2.threshold(hsv[:, :, 2], BRILLO_MINIMO, 255, cv2.THRESH_BINARY)
        mascara = cv2.bitwise_and(m_sat, m_brillo)

    kernel = np.ones((7, 7), np.uint8)
    mascara = cv2.morphologyEx(mascara, cv2.MORPH_CLOSE, kernel)
    mascara = cv2.morphologyEx(mascara, cv2.MORPH_OPEN, kernel)

    contornos, _ = cv2.findContours(mascara, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    total = zona.shape[0] * zona.shape[1]
    minimo = AREA_MINIMA_FRACCION * total
    maximo = FRACCION_MAXIMA * total if ref is not None else total
    contornos = [c for c in contornos if minimo <= cv2.contourArea(c) <= maximo]
    if not contornos:
        return None, mascara
    mayor = max(contornos, key=cv2.contourArea)
    return mayor + np.array([[[x0, y0]]], dtype=mayor.dtype), mascara


def medir(contorno, escala):
    """Largo, ancho (cm), area (cm2) y rectangularidad del contorno."""
    area_px = cv2.contourArea(contorno)
    (cx, cy), (w, h), angulo = cv2.minAreaRect(contorno)
    largo_px, ancho_px = max(w, h), min(w, h)
    rectangularidad = area_px / (w * h) if w * h > 0 else 0.0
    return {
        "largo_cm": largo_px * escala, "ancho_cm": ancho_px * escala,
        "area_cm2": area_px * escala ** 2, "area_px": area_px,
        "rectangularidad": rectangularidad,
    }


def dibujar(imagen, contorno, m, esperado, roi, hay_fondo=True):
    salida = imagen.copy()
    if roi is not None:
        x, y, w, h = roi
        cv2.rectangle(salida, (x, y), (x + w, y + h), (255, 160, 0), 2)
    if contorno is None:
        cv2.putText(salida, "Probeta no detectada", (20, 40), FUENTE, 1.0, (0, 0, 255), 2)
        if not hay_fondo:
            cv2.putText(salida, "Sin fondo: quita la probeta y presiona B", (20, 78), FUENTE, 0.8, (0, 165, 255), 2)
        return salida
    caja = cv2.boxPoints(cv2.minAreaRect(contorno)).astype(int)
    cv2.drawContours(salida, [contorno], -1, (0, 255, 0), 2)
    cv2.drawContours(salida, [caja], -1, (0, 200, 255), 2)
    lineas = [f"Largo {m['largo_cm']:.2f} cm   Ancho {m['ancho_cm']:.2f} cm",
              f"Area {m['area_cm2']:.1f} cm2   Rect. {m['rectangularidad']:.2f}"]
    if esperado is not None:
        e_l, e_a = esperado
        err_l = 100 * (m["largo_cm"] - e_l) / e_l
        err_a = 100 * (m["ancho_cm"] - e_a) / e_a
        lineas.append(f"Error largo {err_l:+.1f} %   ancho {err_a:+.1f} %")
    for i, texto in enumerate(lineas):
        cv2.putText(salida, texto, (20, 40 + 38 * i), FUENTE, 0.9, (0, 255, 0), 2)
    return salida


# --------------------------------------------------------------------------
# Registro y resumen
# --------------------------------------------------------------------------

def registrar(camara, archivo, m, esperado):
    os.makedirs(CARPETA_SALIDA, exist_ok=True)
    nuevo = not os.path.exists(RUTA_CSV)
    fila = {"fecha": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "camara": camara, "archivo": archivo,
            "largo_cm": round(m["largo_cm"], 3), "ancho_cm": round(m["ancho_cm"], 3),
            "area_cm2": round(m["area_cm2"], 2), "rectangularidad": round(m["rectangularidad"], 3),
            "esperado_largo": "", "esperado_ancho": "", "err_largo_pct": "", "err_ancho_pct": "", "err_area_pct": ""}
    if esperado is not None:
        e_l, e_a = esperado
        fila.update({"esperado_largo": e_l, "esperado_ancho": e_a,
                     "err_largo_pct": round(100 * (m["largo_cm"] - e_l) / e_l, 2),
                     "err_ancho_pct": round(100 * (m["ancho_cm"] - e_a) / e_a, 2),
                     "err_area_pct": round(100 * (m["area_cm2"] - e_l * e_a) / (e_l * e_a), 2)})
    with open(RUTA_CSV, "a", newline="", encoding="utf-8") as f:
        escritor = csv.DictWriter(f, fieldnames=list(fila.keys()))
        if nuevo:
            escritor.writeheader()
        escritor.writerow(fila)


def vaciar_teclado():
    """Descarta teclas pendientes en la consola (por ejemplo ENTER presionados mientras corria el video)."""
    try:
        import msvcrt
        while msvcrt.kbhit():
            msvcrt.getch()
    except ImportError:
        pass


def preguntar_si_no(pregunta):
    """Pregunta s/n y repite hasta que se escriba una de las dos."""
    vaciar_teclado()
    while True:
        resp = input(f"{pregunta} (s/n): ").strip().lower()
        if resp in ("s", "n"):
            return resp == "s"
        print("Escribe s o n.")


def resumen(mediciones, esperado, nombre_corto, factor_actual=1.0):
    if not mediciones:
        print("\nNo se registro ninguna medicion.")
        return
    largos = np.array([m["largo_cm"] for m in mediciones])
    anchos = np.array([m["ancho_cm"] for m in mediciones])
    areas = np.array([m["area_cm2"] for m in mediciones])
    print(f"\nResumen de {len(mediciones)} mediciones")
    print(f"  Largo : {largos.mean():.3f} cm  (desv. {largos.std():.3f})")
    print(f"  Ancho : {anchos.mean():.3f} cm  (desv. {anchos.std():.3f})")
    print(f"  Area  : {areas.mean():.2f} cm2  (desv. {areas.std():.2f}, {100 * areas.std() / areas.mean():.2f} %)")
    if esperado is None:
        return
    e_l, e_a = esperado
    err_l = 100 * (largos.mean() - e_l) / e_l
    err_a = 100 * (anchos.mean() - e_a) / e_a
    err_area = 100 * (areas.mean() - e_l * e_a) / (e_l * e_a)
    print(f"  Real  : {e_l} x {e_a} cm = {e_l * e_a:.2f} cm2")
    print(f"  Error : largo {err_l:+.2f} %, ancho {err_a:+.2f} %, area {err_area:+.2f} %")
    # las medidas ya incluyen el factor actual, asi que el nuevo factor se acumula sobre el
    correccion = 0.5 * (e_l / largos.mean() + e_a / anchos.mean())
    factor = factor_actual * correccion
    print(f"  Factor actual {factor_actual:.4f} -> factor nuevo sugerido: {factor:.4f}")
    if abs(correccion - 1) < 0.005:
        print("  La escala ya coincide con la medida real (menos de 0.5 %). No hace falta ajustar.")
        return
    if preguntar_si_no("Guardar este factor para las proximas mediciones de esta camara?"):
        with open(ruta_ajuste(nombre_corto), "w", encoding="utf-8") as f:
            json.dump({"factor": factor, "fecha": datetime.now().isoformat(timespec="seconds")}, f)
        print(f"Factor guardado en {ruta_ajuste(nombre_corto)}")


def pedir_esperado():
    entrada = input("Medidas reales de la cara que ve la camara en cm, largo y ancho "
                    "(ej. 19.4 7.65). ENTER si no las sabes: ").strip().replace(",", ".")
    if not entrada:
        return None
    a, b = [float(x) for x in entrada.split()[:2]]
    return max(a, b), min(a, b)


# --------------------------------------------------------------------------
# Modos
# --------------------------------------------------------------------------

def modo_archivos(ruta, camara, escala, esperado):
    if os.path.isdir(ruta):
        archivos = sorted(glob.glob(os.path.join(ruta, "*.jpg")) + glob.glob(os.path.join(ruta, "*.png")))
    else:
        archivos = [ruta]
    mediciones = []
    os.makedirs(CARPETA_SALIDA, exist_ok=True)
    for archivo in archivos:
        imagen = cv2.imread(archivo)
        if imagen is None:
            continue
        contorno, _ = segmentar(imagen)
        if contorno is None:
            print(f"  {os.path.basename(archivo)}: no se detecto la probeta")
            continue
        m = medir(contorno, escala)
        mediciones.append(m)
        registrar(camara, os.path.basename(archivo), m, esperado)
        cv2.imwrite(os.path.join(CARPETA_SALIDA, "medida_" + os.path.basename(archivo)),
                    dibujar(imagen, contorno, m, esperado, None))
        print(f"  {os.path.basename(archivo)}: {m['largo_cm']:.2f} x {m['ancho_cm']:.2f} cm, "
              f"area {m['area_cm2']:.1f} cm2")
    return mediciones


def modo_vivo(cam, camara, escala, esperado):
    os.makedirs(CARPETA_SALIDA, exist_ok=True)
    lector = LectorStream(cam._captura)
    roi = None
    fondo = None
    ver_mascara = False
    recientes = deque(maxlen=CUADROS_MEDIANA)
    mediciones = []
    mensaje, t_mensaje = "", 0.0
    print("\nB = guardar fondo (escena vacia), ESPACIO = registrar, R = recuadro, C = quitar recuadro, "
          "M = mascara, Q = terminar.\n")
    try:
        while True:
            frame, _ = lector.ultimo()
            if frame is None:
                time.sleep(0.01)
                continue
            if not lector.vivo:
                print("Se perdio la conexion con el stream.")
                break

            contorno, mascara = segmentar(frame, roi, fondo)
            m = medir(contorno, escala) if contorno is not None else None
            if m is not None:
                recientes.append(m)
            else:
                recientes.clear()

            vista = dibujar(frame, contorno, m, esperado, roi, fondo is not None)
            cv2.putText(vista, f"Registradas: {len(mediciones)}", (20, vista.shape[0] - 20),
                        FUENTE, 0.8, (255, 255, 255), 2)
            if mensaje and time.time() - t_mensaje < 1.5:
                cv2.putText(vista, mensaje, (20, vista.shape[0] - 55), FUENTE, 0.8, (255, 255, 0), 2)
            cv2.imshow("Medicion de la probeta", vista)
            if ver_mascara:
                cv2.imshow("Mascara", mascara)

            tecla = cv2.waitKey(10) & 0xFF
            if tecla == ord("q"):
                break
            if tecla == ord("m"):
                ver_mascara = not ver_mascara
                if not ver_mascara:
                    cv2.destroyWindow("Mascara")
            if tecla == ord("c"):
                roi = None
            if tecla == ord("b"):
                fondo = frame.copy()
                recientes.clear()
                mensaje, t_mensaje = "Fondo guardado", time.time()
            if tecla == ord("r"):
                caja = cv2.selectROI("Medicion de la probeta", frame, showCrosshair=False)
                roi = tuple(int(v) for v in caja) if caja[2] > 20 and caja[3] > 20 else None
            if tecla == ord(" "):
                if len(recientes) < 3:
                    mensaje, t_mensaje = "No hay probeta detectada", time.time()
                    continue
                mediana = {k: float(np.median([r[k] for r in recientes])) for k in recientes[0]}
                mediciones.append(mediana)
                nombre = f"medida_{len(mediciones):03d}.jpg"
                cv2.imwrite(os.path.join(CARPETA_SALIDA, nombre), vista)
                registrar(camara, nombre, mediana, esperado)
                print(f"  [{len(mediciones)}] {mediana['largo_cm']:.2f} x {mediana['ancho_cm']:.2f} cm, "
                      f"area {mediana['area_cm2']:.1f} cm2")
                mensaje, t_mensaje = "Medicion registrada", time.time()
    finally:
        lector.detener()
        cv2.destroyAllWindows()
    return mediciones


def main():
    from camara import CamaraIP, cargar_configuracion

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
    print(f"\nCalibracion: fx = {fx:.1f} px, distancia = {distancia} cm, factor de ajuste = {factor:.4f}")
    print(f"Escala: {escala:.4f} cm/px")

    esperado = pedir_esperado()
    modo = input("1 = en vivo con la camara, 2 = desde una foto o carpeta: ").strip()

    if modo == "2":
        ruta = input("Ruta de la foto o carpeta: ").strip().strip('"')
        mediciones = modo_archivos(ruta, nombre, escala, esperado)
    else:
        cam = CamaraIP(datos)
        print(f"\nConectando con {nombre} ({cam.base_url}) ...")
        try:
            cam.iniciar_sesion()
        except ConnectionError as e:
            print(e)
            return
        input("Conectado. Fija EV, AWB y enfoque en DroidCam como en la calibracion y presiona ENTER...")
        mediciones = modo_vivo(cam, nombre, escala, esperado)
        cam.cerrar_sesion()

    resumen(mediciones, esperado, nombre_corto, factor)
    print(f"\nMediciones guardadas en: {RUTA_CSV}")


if __name__ == "__main__":
    main()

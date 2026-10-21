"""
Captura del dataset para entrenar la U-Net que reconoce el palo de balsa.

Por cada foto guarda la imagen y su mascara (la cara frontal del palo en blanco, el resto en negro),
todo de forma automatica: al inicio aprende el entorno vacio y despues, cada vez que pones la probeta,
detecta la cara frontal. Para ajustar la cara busca sus cuatro bordes (la zona clara de la cara frontal
contra las caras laterales y el fondo, que salen mas oscuras), traza una recta por cada borde y las
esquinas salen donde se cruzan las rectas. Asi, si la probeta esta girada y la cara se ve como un
trapecio por la perspectiva, la mascara tambien es un trapecio. Si no logra ajustar las cuatro
rectas, usa un rectangulo y lo anota en la columna metodo del csv.
La mascara es solo la etiqueta de entrenamiento; el modelo entrenado despues no necesitara fondo.

Flujo:
  1. Se abre el video en vivo. Con ese video a la vista configura en DroidCam EV, WB, AF y lo que
     uses siempre (puedes poner la probeta para enfocar). Cuando todo este fijo pulsa ENTER en la ventana.
  2. Cuenta atras de 20 s para retirar la probeta y el programa aprende el entorno (30 cuadros).
  3. Pon la probeta y muevela un poco entre foto y foto (posicion, giro, luz, otras probetas).
  4. Guarda con ESPACIO, o activa el modo automatico con A y desliza la probeta: guarda sola
     cada vez que queda quieta en una pose nueva.

Controles en la ventana de video:
    ENTER   -> (solo en la configuracion inicial) termina de configurar y sigue
    ESPACIO -> guarda la foto actual (solo si la probeta esta quieta y con forma de rectangulo)
    A       -> activa o desactiva el guardado automatico en poses nuevas
    N       -> guarda una foto negativa (SIN probeta, mascara vacia; sirve con mano u objetos)
    R       -> vuelve a aprender el entorno (quita la probeta antes)
    Q       -> termina

Salida: dataset_balsa/<camara>/imagenes, mascaras, previews, fondos y etiquetas.csv

Uso: python capturar_dataset.py
"""

import csv
import math
import os
import time
from collections import deque
from datetime import datetime

import cv2
import numpy as np

import medir_probeta as mp
from calibrar_stream import LectorStream

BASE = os.path.dirname(os.path.abspath(__file__))
CARPETA_DATASET = os.path.join(BASE, "dataset_balsa")
OBJETIVO = 200             # fotos con probeta por camara (conviene repartirlas en varias sesiones)
RECT_MINIMA = 0.85         # rectangularidad minima para aceptar la deteccion (descarta manos)
CUADROS_ESTABLES = 8       # cuadros seguidos para considerar la probeta quieta
VARIACION_AREA = 0.01      # variacion relativa maxima del area en esos cuadros
MOVIMIENTO_MAXIMO = 1.5    # px maximos de movimiento del centro en esos cuadros
DISTANCIA_POSE_NUEVA = 6.0 # px minimos de diferencia de posicion para contar como pose nueva
ANGULO_POSE_NUEVO = 1.5    # grados minimos de diferencia para contar como pose nueva
INTERVALO_AUTO = 0.7       # segundos minimos entre guardados automaticos
FUENTE = cv2.FONT_HERSHEY_SIMPLEX
SEGUNDOS_ESPERA = 20       # tiempo para retirar la probeta antes de aprender el entorno
CUADROS_ENTORNO = 30       # cuadros que se promedian como fondo
RUIDO_MAXIMO = 8.0         # si el entorno se mueve mas que esto, se repite el aprendizaje
VENTANA_BORDE = 10         # px a cada lado que se promedian para medir un escalon de brillo en un borde
SUAVIZADO = 25             # px a lo largo del borde que se promedian para ignorar la veta de la madera
FRACCION_PICO = 0.65       # un borde cuenta como candidato si llega a esta fraccion del escalon maximo
PASO = 3                   # px entre rectas candidatas al buscar cada borde
MIN_ESCALON = 6.0          # diferencia minima de brillo (L) para aceptar un borde
MARGEN_TRAMO = 0.12        # fraccion de cada lado que se descarta en las puntas (ahi estan las esquinas)


def esperar_cuadro(lector):
    while True:
        frame, _ = lector.ultimo()
        if frame is not None:
            return frame
        time.sleep(0.01)


def configurar_camara(lector, ventana):
    """Video en vivo para configurar EV, WB, AF en DroidCam. ENTER o ESPACIO continua, Q sale."""
    while True:
        frame, _ = lector.ultimo()
        if frame is None:
            time.sleep(0.01)
            continue
        if not lector.vivo:
            print("Se perdio la conexion con el stream.")
            return False
        vista = frame.copy()
        cv2.putText(vista, "CONFIGURACION: ajusta EV, WB y AF en DroidCam", (20, 40), FUENTE, 0.9, (0, 255, 255), 2)
        cv2.putText(vista, "Puedes poner la probeta para enfocar. Al terminar pulsa ENTER", (20, 78), FUENTE, 0.8, (255, 255, 255), 2)
        cv2.putText(vista, "Q = salir", (20, vista.shape[0] - 20), FUENTE, 0.8, (255, 255, 255), 2)
        cv2.imshow(ventana, vista)
        tecla = cv2.waitKey(30) & 0xFF
        if tecla in (13, 32):
            return True
        if tecla == ord("q"):
            return False


def aprender_entorno(lector, ventana):
    """Cuenta atras, luego promedia cuadros de la escena vacia. Devuelve el fondo o None si se movio."""
    t0 = time.time()
    while time.time() - t0 < SEGUNDOS_ESPERA:
        vista = esperar_cuadro(lector).copy()
        faltan = int(SEGUNDOS_ESPERA - (time.time() - t0)) + 1
        cv2.putText(vista, f"Retira la probeta... {faltan}", (20, 50), FUENTE, 1.2, (0, 165, 255), 3)
        cv2.imshow(ventana, vista)
        if cv2.waitKey(30) & 0xFF == ord("q"):
            return None
    cuadros = []
    for i in range(CUADROS_ENTORNO):
        frame = esperar_cuadro(lector)
        cuadros.append(frame.astype(np.float32))
        vista = frame.copy()
        cv2.putText(vista, f"Aprendiendo el entorno {i + 1}/{CUADROS_ENTORNO}", (20, 50),
                    FUENTE, 1.0, (0, 255, 255), 3)
        cv2.imshow(ventana, vista)
        cv2.waitKey(60)
    pila = np.stack(cuadros)
    ruido = float(pila.std(axis=0).mean())
    if ruido > RUIDO_MAXIMO:
        print(f"El entorno se movio durante la captura (ruido {ruido:.1f}). Revisa que no haya nada moviendose.")
        return None
    return pila.mean(axis=0).astype(np.uint8)


def a_rectangulo(contorno):
    """La cara es un rectangulo: ajusta el rectangulo minimo al contorno y lo devuelve como cuatro esquinas.
    Tambien devuelve que tanto se parece el contorno a ese rectangulo (1.0 = igual), que sirve para
    descartar manos u otros objetos."""
    casco = cv2.convexHull(contorno)
    (cx, cy), (w, h), ang = cv2.minAreaRect(casco)
    caja = cv2.boxPoints(((cx, cy), (w, h), ang)).astype(np.int32).reshape(-1, 1, 2)
    rect = cv2.contourArea(casco) / (w * h) if w * h > 0 else 0.0
    return caja, rect


def mapa_escalones(brillo):
    """Escalon de brillo en cada pixel: media de los VENTANA_BORDE pixeles de un lado menos la del otro.
    gx es positivo donde a la derecha es mas claro, gy donde abajo es mas claro."""
    k = VENTANA_BORDE
    nucleo = (np.concatenate([-np.ones(k), np.ones(k)]).astype(np.float32) / k).reshape(1, -1)
    gx = cv2.blur(cv2.filter2D(brillo, -1, nucleo), (1, SUAVIZADO))
    gy = cv2.blur(cv2.filter2D(brillo, -1, nucleo.reshape(-1, 1)), (SUAVIZADO, 1))
    return gx, gy


def buscar_borde(g, signo, vertical, rango, tramo, interno):
    """Busca la recta casi vertical (o casi horizontal) con el escalon de brillo mas fuerte a lo largo de
    su tramo. De las rectas que llegan a FRACCION_PICO del maximo se queda con la mas interna: asi se
    queda con el borde de la cara frontal y no con el contorno de la probeta (cara lateral o tapa)."""
    posiciones = np.arange(tramo[0], tramo[1], 4)
    inicio = np.arange(rango[0], rango[1], PASO)
    fin = np.arange(rango[0], rango[1], PASO)
    f = (posiciones - tramo[0]) / max(tramo[1] - 1 - tramo[0], 1)
    cruce_ = inicio[:, None, None] + (fin[None, :, None] - inicio[:, None, None]) * f[None, None, :]
    limite = g.shape[1] if vertical else g.shape[0]
    cruce_ = np.clip(np.round(cruce_).astype(int), 0, limite - 1)
    if vertical:
        valores = g[posiciones[None, None, :], cruce_]
    else:
        valores = g[cruce_, posiciones[None, None, :]]
    puntaje = (signo * valores.mean(axis=2)).astype(np.float32)
    mejor = float(puntaje.max())
    if mejor < MIN_ESCALON:
        return None
    picos = (puntaje == cv2.dilate(puntaje, np.ones((5, 5), np.uint8))) & (puntaje >= FRACCION_PICO * mejor)
    ia, ib = np.nonzero(picos)
    posicion_media = inicio[ia] + fin[ib]
    i = int(np.argmax(posicion_media)) if interno > 0 else int(np.argmin(posicion_media))
    return inicio[ia[i]], fin[ib[i]], posiciones, f


def puntos_del_borde(g, signo, vertical, ini, fin, posiciones, f, ancho=6):
    """Con la recta encontrada como guia, toma en cada fila (o columna) el punto de mayor escalon cerca de ella."""
    puntos = []
    linea = ini + (fin - ini) * f
    for p, c in zip(posiciones, linea):
        c = int(round(c))
        lo = max(c - ancho, 0)
        s_ = signo * (g[p, lo:c + ancho + 1] if vertical else g[lo:c + ancho + 1, p])
        if len(s_) == 0:
            continue
        i = int(np.argmax(s_))
        if s_[i] >= MIN_ESCALON:
            puntos.append((lo + i, p) if vertical else (p, lo + i))
    return puntos


def recta_robusta(puntos):
    """Ajusta una recta quitando de a poco los puntos que se alejan (tornillos, veta, ruido)."""
    p = np.array(puntos, np.float32)
    conservar = np.ones(len(p), bool)
    for _ in range(4):
        vx, vy, x0, y0 = cv2.fitLine(p[conservar], cv2.DIST_L2, 0, 0.01, 0.01).flatten()
        residuo = np.abs((p[:, 0] - x0) * vy - (p[:, 1] - y0) * vx)
        limite = max(1.5, 2.5 * float(np.median(residuo[conservar])))
        nuevo = residuo <= limite
        if nuevo.sum() < 5:
            break
        conservar = nuevo
    return vx, vy, x0, y0


def interseccion(r1, r2):
    vx1, vy1, x1, y1 = r1
    vx2, vy2, x2, y2 = r2
    A = np.array([[vx1, -vx2], [vy1, -vy2]], np.float64)
    if abs(np.linalg.det(A)) < 1e-6:
        return None
    t = np.linalg.solve(A, np.array([x2 - x1, y2 - y1], np.float64))[0]
    return x1 + t * vx1, y1 + t * vy1


def cuadrilatero_cara(frame, contorno):
    """Ajusta las cuatro rectas de la cara frontal dentro de la region que ocupa la probeta.
    La cara frontal es la zona mas clara: sus bordes son escalones de claro (cara) a oscuro (lateral,
    tapa o fondo). Devuelve las cuatro esquinas (izq-sup, sup-der, der-inf, inf-izq) o None."""
    x, y, w, h = cv2.boundingRect(contorno)
    alto, ancho = frame.shape[:2]
    pad = 20
    x0, y0 = max(x - pad, 0), max(y - pad, 0)
    x1, y1 = min(x + w + pad, ancho), min(y + h + pad, alto)
    recorte = frame[y0:y1, x0:x1]
    brillo = cv2.GaussianBlur(cv2.cvtColor(recorte, cv2.COLOR_BGR2LAB)[:, :, 0].astype(np.float32), (0, 0), 1.5)
    gx, gy = mapa_escalones(brillo)
    bx, by = x - x0, y - y0
    tramo_v = (int(by + MARGEN_TRAMO * h), int(by + (1 - MARGEN_TRAMO) * h))
    tramo_h = (int(bx + MARGEN_TRAMO * w), int(bx + (1 - MARGEN_TRAMO) * w))
    rango_izq = (max(bx - 6, 0), int(bx + 0.4 * w))
    rango_der = (int(bx + 0.6 * w), min(bx + w + 6, gx.shape[1] - 1))
    rango_sup = (max(by - 6, 0), int(by + 0.4 * h))
    rango_inf = (int(by + 0.6 * h), min(by + h + 6, gx.shape[0] - 1))
    rectas = {}
    for nombre, g, signo, vertical, rango, tramo, interno in (
            ("izq", gx, +1, True, rango_izq, tramo_v, +1), ("der", gx, -1, True, rango_der, tramo_v, -1),
            ("sup", gy, +1, False, rango_sup, tramo_h, +1), ("inf", gy, -1, False, rango_inf, tramo_h, -1)):
        encontrado = buscar_borde(g, signo, vertical, rango, tramo, interno)
        if encontrado is None:
            return None
        ini, fin, posiciones, f = encontrado
        puntos = puntos_del_borde(g, signo, vertical, ini, fin, posiciones, f)
        if len(puntos) < 10:
            return None
        rectas[nombre] = recta_robusta(puntos)
    esquinas = [interseccion(rectas["izq"], rectas["sup"]), interseccion(rectas["sup"], rectas["der"]),
                interseccion(rectas["der"], rectas["inf"]), interseccion(rectas["inf"], rectas["izq"])]
    if any(e is None for e in esquinas):
        return None
    esquinas = np.array(esquinas, np.float32) + np.array([x0, y0], np.float32)
    poligono = np.round(esquinas).astype(np.int32).reshape(-1, 1, 2)
    area_poligono = cv2.contourArea(poligono)
    area_region = cv2.contourArea(cv2.convexHull(contorno))
    dentro = (esquinas[:, 0].min() > x - 0.1 * w and esquinas[:, 0].max() < x + 1.1 * w
              and esquinas[:, 1].min() > y - 0.1 * h and esquinas[:, 1].max() < y + 1.1 * h)
    if not (cv2.isContourConvex(poligono) and dentro and 0.5 * area_region < area_poligono < 1.15 * area_region):
        return None
    return poligono


def cara_cuadrilatero(frame, contorno):
    """Cuadrilatero de la cara frontal. Devuelve (cuatro esquinas, parecido de la region detectada a un
    rectangulo, metodo). El parecido sirve para descartar manos u objetos raros. Si no se logran ajustar
    las cuatro rectas, usa el rectangulo minimo y el metodo queda como rectangulo."""
    _, rect = a_rectangulo(contorno)
    poligono = cuadrilatero_cara(frame, contorno)
    if poligono is not None:
        return poligono, rect, "rectas"
    caja, _ = a_rectangulo(contorno)
    return caja, rect, "rectangulo"


def pose_de(contorno):
    (cx, cy), (w, h), ang = cv2.minAreaRect(contorno)
    ang = ((ang + 45.0) % 90.0) - 45.0   # el angulo de minAreaRect es ambiguo cada 90 grados
    area = cv2.contourArea(contorno)
    rect = area / (w * h) if w * h > 0 else 0.0
    return {"cx": cx, "cy": cy, "w": max(w, h), "h": min(w, h), "ang": ang, "area": area, "rect": rect}


def es_pose_nueva(pose, guardadas):
    for g in guardadas:
        cerca = math.hypot(pose["cx"] - g["cx"], pose["cy"] - g["cy"]) <= DISTANCIA_POSE_NUEVA
        igual_giro = abs(pose["ang"] - g["ang"]) <= ANGULO_POSE_NUEVO
        if cerca and igual_giro:
            return False
    return True


def lados_px(p):
    """Longitud en pixeles de los cuatro lados del cuadrilatero, en orden."""
    return [float(np.linalg.norm(p[i] - p[(i + 1) % 4])) for i in range(4)]


def esta_quieta(recientes):
    if len(recientes) < CUADROS_ESTABLES:
        return False
    areas = np.array([r["area"] for r in recientes])
    cx = np.array([r["cx"] for r in recientes])
    cy = np.array([r["cy"] for r in recientes])
    return (areas.std() / areas.mean() < VARIACION_AREA
            and cx.std() < MOVIMIENTO_MAXIMO and cy.std() < MOVIMIENTO_MAXIMO)


class Dataset:
    def __init__(self, nombre_corto):
        self.raiz = os.path.join(CARPETA_DATASET, nombre_corto)
        for sub in ("imagenes", "mascaras", "previews", "fondos"):
            os.makedirs(os.path.join(self.raiz, sub), exist_ok=True)
        self.ruta_csv = os.path.join(self.raiz, "etiquetas.csv")
        self.sesion = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.positivas = 0
        self.negativas = 0
        self.poses = []
        existentes = len(os.listdir(os.path.join(self.raiz, "imagenes")))
        self.contador = existentes

    def guardar_fondo(self, fondo):
        cv2.imwrite(os.path.join(self.raiz, "fondos", f"{self.sesion}.jpg"), fondo)

    def guardar(self, frame, poligono=None, metodo=""):
        """Guarda imagen, mascara y vista previa. poligono=None guarda una foto negativa."""
        self.contador += 1
        nombre = f"{self.sesion}_{self.contador:04d}"
        mascara = np.zeros(frame.shape[:2], np.uint8)
        columnas = ["archivo", "sesion", "fecha", "tipo", "metodo", "x1", "y1", "x2", "y2", "x3", "y3", "x4", "y4",
                    "lado1_px", "lado2_px", "lado3_px", "lado4_px", "area_px"]
        fila = {c: "" for c in columnas}
        fila.update({"archivo": f"{nombre}.jpg", "sesion": self.sesion,
                     "fecha": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "tipo": "negativa"})
        preview = frame.copy()
        if poligono is not None:
            p = np.array(poligono, np.float32).reshape(4, 2)
            cv2.fillPoly(mascara, [np.round(p).astype(np.int32).reshape(-1, 1, 2)], 255)
            cv2.polylines(preview, [np.round(p).astype(np.int32).reshape(-1, 1, 2)], True, (0, 255, 0), 1)
            self.poses.append(pose_de(np.round(p).astype(np.int32).reshape(-1, 1, 2)))
            self.positivas += 1
            fila["tipo"] = "positiva"
            fila["metodo"] = metodo
            lados = lados_px(p)
            for i in range(4):
                fila[f"x{i + 1}"] = round(float(p[i][0]), 1)
                fila[f"y{i + 1}"] = round(float(p[i][1]), 1)
                fila[f"lado{i + 1}_px"] = round(lados[i], 1)
            fila["area_px"] = int((mascara > 0).sum())
        else:
            self.negativas += 1
        cv2.imwrite(os.path.join(self.raiz, "imagenes", f"{nombre}.jpg"), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
        cv2.imwrite(os.path.join(self.raiz, "mascaras", f"{nombre}.png"), mascara)
        cv2.imwrite(os.path.join(self.raiz, "previews", f"{nombre}.jpg"), preview)
        nuevo = not os.path.exists(self.ruta_csv)
        with open(self.ruta_csv, "a", newline="", encoding="utf-8") as f:
            escritor = csv.DictWriter(f, fieldnames=columnas)
            if nuevo:
                escritor.writeheader()
            escritor.writerow(fila)


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

    cam = CamaraIP(datos)
    print(f"\nConectando con {nombre} ({cam.base_url}) ...")
    try:
        cam.iniciar_sesion()
    except ConnectionError as e:
        print(e)
        return
    print("Conectado. Se abre el video para que configures EV, WB y AF en DroidCam.")
    print("Cuando todo este fijo, pulsa ENTER en la ventana del video (no en la consola).")

    dataset = Dataset(nombre_corto)
    ventana = f"Dataset {nombre}"
    lector = LectorStream(cam._captura)
    recientes = deque(maxlen=CUADROS_ESTABLES)
    automatico = False
    t_ultimo = 0.0
    mensaje, t_mensaje = "", 0.0
    try:
        if not configurar_camara(lector, ventana):
            return
        fondo = None
        while fondo is None:
            fondo = aprender_entorno(lector, ventana)
            if fondo is None and cv2.getWindowProperty(ventana, cv2.WND_PROP_VISIBLE) < 1:
                return
        dataset.guardar_fondo(fondo)
        print(f"\nEntorno aprendido. Sesion {dataset.sesion}. Pon la probeta y muevela entre foto y foto.")
        print("ESPACIO = guardar, A = automatico, N = negativa (sin probeta), R = reaprender entorno, Q = terminar.\n")

        while True:
            frame, _ = lector.ultimo()
            if frame is None:
                time.sleep(0.01)
                continue
            if not lector.vivo:
                print("Se perdio la conexion con el stream.")
                break

            contorno, _ = mp.segmentar(frame, None, fondo)
            pose = None
            metodo = ""
            if contorno is not None:
                contorno, rect_forma, metodo = cara_cuadrilatero(frame, contorno)
                pose = pose_de(contorno)
                pose["rect"] = rect_forma
            if pose is not None:
                recientes.append(pose)
            else:
                recientes.clear()

            quieta = esta_quieta(recientes)
            forma_ok = pose is not None and pose["rect"] >= RECT_MINIMA
            lista = quieta and forma_ok

            vista = frame.copy()
            if contorno is not None:
                color = (0, 0, 255) if not forma_ok else ((0, 255, 0) if metodo == "rectas" else (0, 165, 255))
                cv2.polylines(vista, [contorno], True, color, 1)
            if pose is None:
                estado, color_estado = "Esperando la probeta", (255, 255, 255)
            elif not forma_ok:
                estado, color_estado = f"Forma rara ({pose['rect']:.2f}): hay una mano u otro objeto?", (0, 0, 255)
            elif not quieta:
                estado, color_estado = "Quieta un momento...", (0, 255, 255)
            else:
                estado, color_estado = "LISTA para guardar", (0, 255, 0)

            cv2.putText(vista, estado, (20, 40), FUENTE, 0.9, color_estado, 2)
            cv2.putText(vista, f"Con probeta: {dataset.positivas}/{OBJETIVO}   Negativas: {dataset.negativas}   "
                               f"Auto: {'SI' if automatico else 'NO'}", (20, 78), FUENTE, 0.8, (255, 255, 255), 2)
            if mensaje and time.time() - t_mensaje < 1.5:
                cv2.putText(vista, mensaje, (20, vista.shape[0] - 20), FUENTE, 0.9, (255, 255, 0), 2)
            cv2.imshow(ventana, vista)

            if automatico and lista and time.time() - t_ultimo >= INTERVALO_AUTO and es_pose_nueva(pose, dataset.poses) :
                dataset.guardar(frame, contorno, metodo)
                t_ultimo = time.time()
                mensaje, t_mensaje = f"Guardada {dataset.positivas}", time.time()
                print(f"  [{dataset.positivas}] pose nueva guardada")

            tecla = cv2.waitKey(10) & 0xFF
            if tecla == ord("q"):
                break
            if tecla == ord("a"):
                automatico = not automatico
            if tecla == ord(" "):
                if not lista:
                    mensaje, t_mensaje = "No se guarda: " + estado, time.time()
                else:
                    dataset.guardar(frame, contorno, metodo)
                    t_ultimo = time.time()
                    mensaje, t_mensaje = f"Guardada {dataset.positivas}", time.time()
                    print(f"  [{dataset.positivas}] guardada")
            if tecla == ord("n"):
                dataset.guardar(frame, None)
                mensaje, t_mensaje = f"Negativa guardada ({dataset.negativas})", time.time()
                print(f"  negativa guardada ({dataset.negativas})")
            if tecla == ord("r"):
                nuevo = aprender_entorno(lector, ventana)
                if nuevo is not None:
                    fondo = nuevo
                    dataset.guardar_fondo(fondo)
                recientes.clear()
    finally:
        lector.detener()
        cv2.destroyAllWindows()
        cam.cerrar_sesion()
        print(f"\nSesion {dataset.sesion}: {dataset.positivas} con probeta y {dataset.negativas} negativas.")
        print(f"Dataset en: {dataset.raiz}")


if __name__ == "__main__":
    main()

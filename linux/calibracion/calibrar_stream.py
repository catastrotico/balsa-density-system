"""
Calibracion de camara (metodo de Zhang) con frames del stream de DroidCam.

Cambios de esta version respecto a la anterior:
  * La captura automatica guarda SOLO fotos con una pose nueva (posicion,
    distancia, inclinacion y giro distintos de las ya guardadas) y nitidas.
    Asi las 150 fotos aportan informacion distinta y no 150 copias casi iguales.
  * La ventana indica que poses faltan ("aleja el borde derecho", etc.).
  * La calibracion ya NO quita fotos hasta quedar con 12. Solo descarta
    outliers claros. Luego compara varios modelos de distorsion con validacion
    cruzada (se calibra con una parte de las fotos y se mide el error en las
    que no se usaron) y elige el que mejor generaliza, no el que mas se
    ajusta. Con eso la distorsion ya no sale absurda (k3 = 6, etc.).
  * Al final informa la cobertura de poses y la incertidumbre de la focal.

Controles en la ventana de video:
    ESPACIO -> guarda el cuadro actual (siempre)
    A       -> activa o desactiva el modo automatico
    N       -> activa o desactiva el filtro de poses nuevas
    + / -   -> cambia el intervalo minimo del modo automatico
    Q       -> termina la captura y calibra

Uso: python calibrar_stream.py
"""

import glob
import os
import shutil
import threading
import time
from collections import deque
from concurrent.futures import ProcessPoolExecutor

import cv2
import numpy as np

# Tablero recortado: 7 columnas x 8 filas de cuadros = 6 x 7 esquinas internas.
# Como una cantidad es par y la otra impar, OpenCV numera las esquinas siempre igual
# respecto al tablero fisico, sin importar como lo gires.
ESQUINAS_ANCHO = 6
ESQUINAS_ALTO = 7
TAMANO_CUADRO_CM = 2.9
FRAMES_OBJETIVO = 150          # meta de fotos guardadas
MAX_FRAMES_CALIBRACION = 200   # fotos que entran a la calibracion (repartidas)
FRAMES_MINIMOS = 25            # por debajo de esto no se descarta ni una foto mas
INTERVALO_AUTO = 0.3           # segundos minimos entre fotos en modo automatico
ESCALA_DETECCION = 0.75        # la vista en vivo detecta con la imagen reducida
DISTANCIA_NOVEDAD = 0.35       # cuanto debe diferir una pose para guardarse
NITIDEZ_MINIMA = 0.6           # fraccion de la nitidez mediana para aceptar una foto
FOCAL_APROX = 0.71             # f aproximada = 0.71 * ancho (solo para guiar la captura)
PLIEGUES_CV = 5                # validacion cruzada: numero de grupos
FUENTE = cv2.FONT_HERSHEY_SIMPLEX
BASE = os.path.dirname(os.path.abspath(__file__))
PATRON = (ESQUINAS_ANCHO, ESQUINAS_ALTO)
DISTANCIAS_CM = {"iPhone 15 Pro Max": 30.5, "iPhone 11": 36.0}


def puntos_del_patron():
    puntos = np.zeros((ESQUINAS_ALTO * ESQUINAS_ANCHO, 3), np.float32)
    puntos[:, :2] = np.mgrid[0:ESQUINAS_ANCHO, 0:ESQUINAS_ALTO].T.reshape(-1, 2)
    return puntos * TAMANO_CUADRO_CM


PUNTOS_3D = puntos_del_patron()


def pedir_ip(config, clave, guardar_configuracion):
    """Muestra la IP guardada y permite cambiarla. ENTER mantiene la actual."""
    datos = config[clave]
    print(f"\nIP guardada para {datos['nombre']}: {datos['ip']}  (puerto {datos['puerto']})")
    nueva_ip = input("Nueva IP (ENTER para mantener): ").strip()
    if nueva_ip:
        datos["ip"] = nueva_ip
        guardar_configuracion(config)
        print(f"IP actualizada a {nueva_ip} y guardada en config.json")
    return datos


def formatear(esquinas):
    """Deja las esquinas siempre como (N, 1, 2) float32."""
    return np.asarray(esquinas, dtype=np.float32).reshape(-1, 1, 2)


def normalizar_orden(esquinas):
    """Solo da formato. No reordena: OpenCV ya numera igual respecto al tablero fisico."""
    return formatear(esquinas)


def detectar(gris, exhaustivo=False):
    """
    Prueba varios metodos y devuelve (encontrado, esquinas). Se detiene en el
    primero que funciona, asi que las fotos buenas salen rapido.
    En modo exhaustivo tambien prueba con la imagen ampliada y con realce de
    contraste, para recuperar fotos que antes se perdian.
    """
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    gris_contraste = clahe.apply(gris)
    base = cv2.CALIB_CB_ACCURACY | cv2.CALIB_CB_NORMALIZE_IMAGE

    intentos = [(gris, base), (gris_contraste, base)]
    if exhaustivo:
        intentos += [(gris, base | cv2.CALIB_CB_EXHAUSTIVE),
                     (gris_contraste, base | cv2.CALIB_CB_EXHAUSTIVE)]

    for imagen, flags in intentos:
        ok, esquinas = cv2.findChessboardCornersSB(imagen, PATRON, flags=flags)
        if ok:
            return True, normalizar_orden(esquinas)

    if exhaustivo:
        # Imagen suavizada y reducida: ayuda cuando hay ruido o desenfoque leve.
        for escala in (0.6, 0.8):
            chica = cv2.resize(gris, None, fx=escala, fy=escala, interpolation=cv2.INTER_AREA)
            ok, esquinas = cv2.findChessboardCornersSB(chica, PATRON, flags=base | cv2.CALIB_CB_EXHAUSTIVE)
            if ok:
                return True, normalizar_orden(formatear(esquinas) / escala)

    return False, None


# --------------------------------------------------------------------------
# Pose aproximada: sirve para guiar la captura y para el informe de cobertura
# --------------------------------------------------------------------------

def descriptor_pose(esquinas, ancho, alto, matriz=None):
    """
    Devuelve un diccionario con la pose del tablero: centro normalizado,
    distancia tz (cm), inclinaciones (grados) y giro en el plano (grados).
    inclin_x > 0: el borde derecho (en pantalla) esta mas lejos.
    inclin_y > 0: el borde inferior esta mas lejos.
    """
    if matriz is None:
        f = FOCAL_APROX * ancho
        matriz = np.array([[f, 0, ancho / 2.0], [0, f, alto / 2.0], [0, 0, 1]], np.float64)
    pts = np.asarray(esquinas, np.float64).reshape(-1, 1, 2)
    ok, rvec, tvec = cv2.solvePnP(PUNTOS_3D.astype(np.float64), pts, matriz, None)
    if not ok:
        return None
    return descriptor_desde_pose(rvec, tvec, pts, ancho, alto)


def descriptor_desde_pose(rvec, tvec, pts, ancho, alto):
    R, _ = cv2.Rodrigues(rvec)
    n = R[:, 2]
    inclin_x = float(np.degrees(np.arctan2(n[0], abs(n[2]))))
    inclin_y = float(np.degrees(np.arctan2(n[1], abs(n[2]))))
    giro = float(np.degrees(np.arctan2(R[1, 0], R[0, 0])))
    giro = (giro + 90.0) % 180.0 - 90.0      # el tablero es simetrico a 180 grados
    centro = pts.reshape(-1, 2).mean(axis=0)
    return {
        "cx": float(centro[0] / ancho), "cy": float(centro[1] / alto),
        "tz": float(abs(tvec[2][0])), "ix": inclin_x, "iy": inclin_y, "giro": giro,
    }


def vector_pose(d):
    """Vector de caracteristicas para medir cuan distinta es una pose de otra."""
    return np.array([d["cx"], d["cy"], np.log(max(d["tz"], 1e-3)) / 0.25,
                     d["ix"] / 20.0, d["iy"] / 20.0, d["giro"] / 30.0])


CATEGORIAS = [
    ("borde derecho mas lejos", lambda d: d["ix"] > 12),
    ("borde izquierdo mas lejos", lambda d: d["ix"] < -12),
    ("borde inferior mas lejos", lambda d: d["iy"] > 12),
    ("borde superior mas lejos", lambda d: d["iy"] < -12),
    ("tablero mas cerca", lambda d: d["tz"] < 34),
    ("tablero mas lejos", lambda d: d["tz"] > 44),
    ("girado a un lado", lambda d: d["giro"] > 10),
    ("girado al otro lado", lambda d: d["giro"] < -10),
    ("tablero a la izquierda", lambda d: d["cx"] < 0.4),
    ("tablero a la derecha", lambda d: d["cx"] > 0.6),
    ("tablero arriba", lambda d: d["cy"] < 0.42),
    ("tablero abajo", lambda d: d["cy"] > 0.58),
]
CUPO_CATEGORIA = 8


class PosesGuardadas:
    """Lleva la cuenta de las poses ya guardadas y decide si una nueva aporta algo."""

    def __init__(self):
        self.vectores = []
        self.descriptores = []

    def es_nueva(self, d):
        if not self.vectores:
            return True
        v = vector_pose(d)
        return float(np.min(np.linalg.norm(np.array(self.vectores) - v, axis=1))) >= DISTANCIA_NOVEDAD

    def agregar(self, d):
        self.vectores.append(vector_pose(d))
        self.descriptores.append(d)

    def faltantes(self, cuantas=2):
        """Las categorias menos cubiertas, para sugerirselas al usuario."""
        conteos = []
        for nombre, regla in CATEGORIAS:
            conteos.append((sum(1 for d in self.descriptores if regla(d)), nombre))
        conteos.sort()
        return [nombre for c, nombre in conteos[:cuantas] if c < CUPO_CATEGORIA]


def nitidez(gris, esquinas):
    """Varianza del Laplaciano dentro del cuadro que encierra al tablero."""
    pts = esquinas.reshape(-1, 2)
    x0, y0 = np.maximum(pts.min(axis=0).astype(int), 0)
    x1, y1 = pts.max(axis=0).astype(int)
    zona = gris[y0:y1 + 1, x0:x1 + 1]
    if zona.size < 100:
        return 0.0
    return float(cv2.Laplacian(zona, cv2.CV_64F).var())


# --------------------------------------------------------------------------
# Captura
# --------------------------------------------------------------------------

class LectorStream:
    """Lee el stream en un hilo y conserva solo el cuadro mas reciente."""

    def __init__(self, captura):
        self._captura = captura
        self._frame = None
        self._id = 0
        self._fallos = 0
        self._lock = threading.Lock()
        self._activo = True
        self._hilo = threading.Thread(target=self._leer, daemon=True)
        self._hilo.start()

    def _leer(self):
        while self._activo:
            ret, frame = self._captura.read()
            if not ret:
                self._fallos += 1
                time.sleep(0.01)
                continue
            self._fallos = 0
            with self._lock:
                self._frame = frame
                self._id += 1

    @property
    def vivo(self):
        return self._fallos < 300

    def ultimo(self):
        with self._lock:
            if self._frame is None:
                return None, 0
            return self._frame.copy(), self._id

    def detener(self):
        self._activo = False
        self._hilo.join(timeout=2)


class DetectorVivo:
    """Detecta el tablero en un hilo aparte para no frenar la ventana de video."""

    def __init__(self, lector):
        self._lector = lector
        self._resultado = (None, None, 0)
        self._lock = threading.Lock()
        self._activo = True
        self._hilo = threading.Thread(target=self._correr, daemon=True)
        self._hilo.start()

    def _correr(self):
        ultimo_id = 0
        flags = cv2.CALIB_CB_ACCURACY | cv2.CALIB_CB_NORMALIZE_IMAGE
        while self._activo:
            frame, fid = self._lector.ultimo()
            if frame is None or fid == ultimo_id:
                time.sleep(0.01)
                continue
            ultimo_id = fid
            chica = cv2.resize(frame, None, fx=ESCALA_DETECCION, fy=ESCALA_DETECCION,
                               interpolation=cv2.INTER_AREA)
            gris = cv2.cvtColor(chica, cv2.COLOR_BGR2GRAY)
            ok, esquinas = cv2.findChessboardCornersSB(gris, PATRON, flags=flags)
            with self._lock:
                if ok:
                    esquinas = normalizar_orden(esquinas) / ESCALA_DETECCION
                    self._resultado = (frame, esquinas.astype(np.float32), fid)
                else:
                    self._resultado = (None, None, fid)

    def resultado(self):
        with self._lock:
            return self._resultado

    def detener(self):
        self._activo = False
        self._hilo.join(timeout=2)


def capturar(cam, carpeta):
    os.makedirs(carpeta, exist_ok=True)
    contador = len(glob.glob(os.path.join(carpeta, "stream_*.jpg")))

    lector = LectorStream(cam._captura)
    detector = DetectorVivo(lector)
    poses = PosesGuardadas()
    nitideces = deque(maxlen=40)

    auto = False
    filtro_poses = True
    intervalo = INTERVALO_AUTO
    t_ultimo = 0.0
    ultimo_id_guardado = -1
    mensaje, t_mensaje = "", 0.0
    fps, t_fps, fid_fps = 0.0, time.time(), 0
    guardadas_sesion = 0

    print(f"\nObjetivo: {FRAMES_OBJETIVO} fotos. A = automatico, ESPACIO = foto manual, N = filtro de poses, Q = terminar.")
    print("Mueve el tablero despacio y sigue las sugerencias de la ventana.\n")

    try:
        while True:
            frame, fid = lector.ultimo()
            if frame is None:
                time.sleep(0.01)
                continue
            if not lector.vivo:
                print("Se perdio la conexion con el stream.")
                break

            alto, ancho = frame.shape[:2]
            ahora = time.time()
            if ahora - t_fps >= 1.0:
                fps = (fid - fid_fps) / (ahora - t_fps)
                t_fps, fid_fps = ahora, fid

            frame_det, esquinas, id_det = detector.resultado()
            reciente = esquinas is not None and (fid - id_det) < 20

            vista = frame.copy()
            if reciente:
                cv2.drawChessboardCorners(vista, PATRON, esquinas, True)
                estado, color = "TABLERO DETECTADO", (0, 200, 0)
            else:
                estado, color = "Tablero no detectado", (0, 0, 255)

            cv2.putText(vista, estado, (20, 40), FUENTE, 1.0, color, 2)
            cv2.putText(vista, f"Guardadas: {contador}/{FRAMES_OBJETIVO}", (20, 80),
                        FUENTE, 1.0, (255, 255, 255), 2)
            modo = f"AUTO {intervalo:.1f} s" if auto else "manual"
            filtro = "filtro ON" if filtro_poses else "filtro OFF"
            cv2.putText(vista, f"Modo: {modo}  {filtro}  Stream: {fps:.0f} fps", (20, 120),
                        FUENTE, 0.7, (0, 200, 255), 2)
            falta = poses.faltantes()
            if falta:
                cv2.putText(vista, "Falta: " + " | ".join(falta), (20, alto - 25),
                            FUENTE, 0.7, (0, 255, 255), 2)
            if mensaje and ahora - t_mensaje < 1.5:
                cv2.putText(vista, mensaje, (20, 160), FUENTE, 0.8, (255, 255, 0), 2)
            cv2.imshow("Calibracion con el stream", vista)

            tecla = cv2.waitKey(10) & 0xFF
            if tecla == ord("q"):
                break
            if tecla == ord("a"):
                auto = not auto
                mensaje, t_mensaje = ("Modo automatico ACTIVADO" if auto else "Modo automatico APAGADO"), ahora
            if tecla == ord("n"):
                filtro_poses = not filtro_poses
                mensaje, t_mensaje = ("Filtro de poses ACTIVADO" if filtro_poses else "Filtro de poses APAGADO"), ahora
            if tecla in (ord("+"), ord("=")):
                intervalo = min(3.0, intervalo + 0.1)
            if tecla == ord("-"):
                intervalo = max(0.1, intervalo - 0.1)

            a_guardar, desc = None, None
            if tecla == ord(" "):
                a_guardar = frame
                if reciente:
                    desc = descriptor_pose(esquinas, ancho, alto)
            elif auto and reciente and id_det != ultimo_id_guardado and (ahora - t_ultimo) >= intervalo:
                gris = cv2.cvtColor(frame_det, cv2.COLOR_BGR2GRAY)
                n = nitidez(gris, esquinas)
                desc = descriptor_pose(esquinas, ancho, alto)
                nitida = len(nitideces) < 5 or n >= NITIDEZ_MINIMA * float(np.median(nitideces))
                nitideces.append(n)
                nueva = (not filtro_poses) or (desc is not None and poses.es_nueva(desc))
                if nitida and nueva and desc is not None:
                    a_guardar = frame_det
                ultimo_id_guardado = id_det

            if a_guardar is not None:
                contador += 1
                guardadas_sesion += 1
                cv2.imwrite(os.path.join(carpeta, f"stream_{contador:04d}.jpg"), a_guardar)
                if desc is not None:
                    poses.agregar(desc)
                t_ultimo = ahora
                if contador % 10 == 0:
                    print(f"  {contador} fotos guardadas")
    finally:
        detector.detener()
        lector.detener()
        cv2.destroyAllWindows()

    print(f"\nTotal en la carpeta: {contador} fotos ({guardadas_sesion} en esta sesion)")


# --------------------------------------------------------------------------
# Deteccion offline
# --------------------------------------------------------------------------

def detectar_archivo(ruta):
    """Detecta el tablero en una foto guardada (corre en procesos paralelos)."""
    imagen = cv2.imread(ruta)
    if imagen is None:
        return ruta, False, None, None
    gris = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    ok, esquinas = detectar(gris, exhaustivo=True)
    return ruta, ok, esquinas, (gris.shape[1], gris.shape[0])


def leer_esquinas(carpeta):
    archivos = sorted(glob.glob(os.path.join(carpeta, "stream_*.jpg")))
    print(f"\nDetectando esquinas en {len(archivos)} fotos (en paralelo)...")
    esquinas_todas, nombres, tamano = [], [], None
    with ProcessPoolExecutor() as ejecutor:
        for i, (ruta, ok, esquinas, tam) in enumerate(ejecutor.map(detectar_archivo, archivos, chunksize=4), start=1):
            if ok:
                esquinas_todas.append(esquinas)
                nombres.append(os.path.basename(ruta))
                tamano = tam
            if i % 25 == 0:
                print(f"  {i}/{len(archivos)} revisadas, {len(esquinas_todas)} validas")
    print(f"Fotos con tablero detectado: {len(esquinas_todas)} de {len(archivos)}")
    return esquinas_todas, nombres, tamano


# --------------------------------------------------------------------------
# Calibracion
# --------------------------------------------------------------------------

BASE_FLAGS = cv2.CALIB_USE_INTRINSIC_GUESS
_FIJOS = cv2.CALIB_FIX_ASPECT_RATIO | cv2.CALIB_FIX_PRINCIPAL_POINT | cv2.CALIB_ZERO_TANGENT_DIST
# (nombre, flags, liberar los puntos del tablero). "liberar" usa el metodo de OpenCV que
# corrige un tablero impreso que no es perfectamente plano (se abomba, se curva).
MODELOS = [
    ("B: f, k1 (tablero plano)", BASE_FLAGS | _FIJOS | cv2.CALIB_FIX_K2 | cv2.CALIB_FIX_K3, False),
    ("B: f, k1 (tablero libre)", BASE_FLAGS | _FIJOS | cv2.CALIB_FIX_K2 | cv2.CALIB_FIX_K3, True),
    ("C: f, k1, k2 (tablero plano)", BASE_FLAGS | _FIJOS | cv2.CALIB_FIX_K3, False),
    ("C: f, k1, k2 (tablero libre)", BASE_FLAGS | _FIJOS | cv2.CALIB_FIX_K3, True),
    ("D: f, centro, k1, k2 (plano)",
     BASE_FLAGS | cv2.CALIB_FIX_ASPECT_RATIO | cv2.CALIB_ZERO_TANGENT_DIST | cv2.CALIB_FIX_K3, False),
    ("D: f, centro, k1, k2 (libre)",
     BASE_FLAGS | cv2.CALIB_FIX_ASPECT_RATIO | cv2.CALIB_ZERO_TANGENT_DIST | cv2.CALIB_FIX_K3, True),
]
INDICE_FIJO = ESQUINAS_ANCHO - 1   # esquina superior derecha: se mantiene fija al liberar el tablero
TOLERANCIA_SIMPLICIDAD = 1.02      # se prefiere el modelo mas simple si queda a menos de 2 % del mejor
MAX_FOTOS_COMPARACION = 90         # la comparacion de modelos usa como maximo estas fotos


def matriz_inicial(tamano):
    ancho, alto = tamano
    f = FOCAL_APROX * ancho
    return np.array([[f, 0, ancho / 2.0], [0, f, alto / 2.0], [0, 0, 1]], np.float64)


def calibrar_modelo(esquinas, tamano, flags, liberar=False):
    """Devuelve (error, matriz, coef, rvecs, tvecs, puntos_del_tablero_usados)."""
    n = len(esquinas)
    if liberar:
        error, matriz, coef, rvecs, tvecs, nuevos = cv2.calibrateCameraRO(
            [PUNTOS_3D] * n, esquinas, tamano, INDICE_FIJO, matriz_inicial(tamano).copy(),
            np.zeros((5, 1)), flags=flags)[:6]
        return error, matriz, coef, rvecs, tvecs, np.asarray(nuevos, np.float32).reshape(-1, 3)
    error, matriz, coef, rvecs, tvecs = cv2.calibrateCamera(
        [PUNTOS_3D] * n, esquinas, tamano, matriz_inicial(tamano).copy(), np.zeros((5, 1)), flags=flags)
    return error, matriz, coef, rvecs, tvecs, PUNTOS_3D


def error_por_frame(esquinas_todas, matriz, coef, rvecs, tvecs, objeto=None):
    objeto = PUNTOS_3D if objeto is None else objeto
    errores = []
    for esquinas, rvec, tvec in zip(esquinas_todas, rvecs, tvecs):
        proyectados, _ = cv2.projectPoints(objeto, rvec, tvec, matriz, coef)
        dif = esquinas.reshape(-1, 2) - proyectados.reshape(-1, 2)
        errores.append(float(np.sqrt(np.mean(np.sum(dif ** 2, axis=1)))))
    return errores


def error_en_fotos_nuevas(esquinas, matriz, coef, objeto):
    """Error en fotos que NO se usaron para calibrar: se halla la pose con la camara ya fija."""
    errores = []
    for e in esquinas:
        ok, rvec, tvec = cv2.solvePnP(objeto.astype(np.float64), np.asarray(e, np.float64).reshape(-1, 1, 2),
                                      matriz, coef)
        if not ok:
            continue
        proyectados, _ = cv2.projectPoints(objeto, rvec, tvec, matriz, coef)
        dif = np.asarray(e).reshape(-1, 2) - proyectados.reshape(-1, 2)
        errores.append(np.sum(dif ** 2, axis=1))
    return float(np.sqrt(np.mean(np.concatenate(errores)))) if errores else float("inf")


def validacion_cruzada(esquinas, tamano, flags, liberar, pliegues=PLIEGUES_CV, semilla=0):
    """Devuelve (error medio en fotos no usadas, lista de focales de cada grupo)."""
    n = len(esquinas)
    rng = np.random.RandomState(semilla)
    orden = rng.permutation(n)
    grupos = np.array_split(orden, pliegues)
    errores, focales = [], []
    for g in grupos:
        prueba = set(g.tolist())
        entrenar = [esquinas[i] for i in range(n) if i not in prueba]
        probar = [esquinas[i] for i in g]
        try:
            _, matriz, coef, _, _, objeto = calibrar_modelo(entrenar, tamano, flags, liberar)
        except cv2.error:
            continue
        errores.append(error_en_fotos_nuevas(probar, matriz, coef, objeto))
        focales.append(float(matriz[0, 0]))
    if not errores:
        return float("inf"), []
    return float(np.mean(errores)), focales


def evaluar_modelo(tarea):
    """Calibra un modelo y lo valida con fotos no usadas (corre en un proceso aparte)."""
    muestra, tamano, nombre, flags, liberar = tarea
    try:
        err_train, matriz, _, _, _, _ = calibrar_modelo(muestra, tamano, flags, liberar)
    except cv2.error:
        return None
    err_cv, focales = validacion_cruzada(muestra, tamano, flags, liberar)
    dfx = float(np.std(focales)) if focales else float("nan")
    return {"nombre": nombre, "flags": flags, "liberar": liberar, "train": err_train,
            "cv": err_cv, "fx": float(matriz[0, 0]), "dfx": dfx}


def corregir_etiquetado(esquinas, tamano):
    """
    Si alguna foto quedo con las esquinas numeradas al reves (giro de 180 grados),
    se detecta porque con la numeracion invertida encaja mucho mejor. Se corrige en
    lugar de descartarla. Devuelve cuantas fotos se corrigieron.
    """
    corregidas = 0
    for _ in range(2):
        _, matriz, coef, rvecs, tvecs, objeto = calibrar_modelo(esquinas, tamano, MODELOS[2][1], False)
        cambios = 0
        for i, e in enumerate(esquinas):
            normal = error_en_fotos_nuevas([e], matriz, coef, objeto)
            invertida = error_en_fotos_nuevas([e[::-1].copy()], matriz, coef, objeto)
            if normal > 1.5 and invertida < 0.5 * normal:
                esquinas[i] = e[::-1].copy()
                cambios += 1
        corregidas += cambios
        if cambios == 0:
            break
    return corregidas


def quitar_outliers(esquinas, nombres, tamano):
    """Descarta solo fotos claramente malas (maximo 10 %), con un modelo simple."""
    _, matriz, coef, rvecs, tvecs, objeto = calibrar_modelo(esquinas, tamano, MODELOS[2][1], False)
    errores = np.array(error_por_frame(esquinas, matriz, coef, rvecs, tvecs, objeto))
    umbral = max(3.0 * float(np.median(errores)), 1.2)
    malos = [i for i in np.argsort(errores)[::-1] if errores[i] > umbral]
    maximo = min(int(0.10 * len(esquinas)), max(0, len(esquinas) - FRAMES_MINIMOS))
    malos = sorted(malos[:maximo], reverse=True)
    for i in malos:
        del esquinas[i]
        del nombres[i]
    return len(malos), umbral


def abombado_mm(objeto):
    """Cuanto se aparta el tablero de un plano perfecto (mm, maximo), segun los puntos calibrados."""
    p = np.asarray(objeto, np.float64).reshape(-1, 3)
    p = p - p.mean(axis=0)
    normal = np.linalg.svd(p, full_matrices=False)[2][-1]
    return float(np.max(np.abs(p @ normal)) * 10.0)


def informe_cobertura(rvecs, tvecs, esquinas, tamano):
    ancho, alto = tamano
    descs = [descriptor_desde_pose(r, t, e, ancho, alto) for r, t, e in zip(rvecs, tvecs, esquinas)]
    ix = np.array([d["ix"] for d in descs])
    iy = np.array([d["iy"] for d in descs])
    tz = np.array([d["tz"] for d in descs])
    giro = np.array([d["giro"] for d in descs])
    cx = np.array([d["cx"] for d in descs])
    cy = np.array([d["cy"] for d in descs])

    print("\nCobertura de poses (mas variedad = calibracion mas confiable)")
    print(f"  Inclinacion horizontal : {ix.min():6.1f} a {ix.max():5.1f} grados")
    print(f"  Inclinacion vertical   : {iy.min():6.1f} a {iy.max():5.1f} grados")
    print(f"  Distancia al tablero   : {tz.min():6.1f} a {tz.max():5.1f} cm")
    print(f"  Giro en el plano       : {giro.min():6.1f} a {giro.max():5.1f} grados")
    print(f"  Centro del tablero     : x {cx.min():.2f}-{cx.max():.2f}, y {cy.min():.2f}-{cy.max():.2f} (0 a 1)")

    avisos = []
    if ix.max() - ix.min() < 35:
        avisos.append("Falta inclinar el tablero hacia los lados (izquierda/derecha).")
    if iy.max() - iy.min() < 35:
        avisos.append("Falta inclinar el tablero hacia arriba y abajo.")
    if (tz.max() - tz.min()) < 0.2 * np.median(tz):
        avisos.append("Falta variar la distancia (acercar y alejar el tablero).")
    if giro.max() - giro.min() < 30:
        avisos.append("Falta girar el tablero en su plano (rotarlo como un volante).")
    if cx.max() - cx.min() < 0.3 or cy.max() - cy.min() < 0.25:
        avisos.append("Falta llevar el tablero a los bordes y esquinas de la imagen.")
    for a in avisos:
        print(f"  AVISO: {a}")
    if not avisos:
        print("  Cobertura buena.")
    return avisos


def calibrar_fotos(esquinas_todas, nombres, tamano, distancia_cm, mostrar=print):
    """Nucleo de la calibracion. Devuelve un diccionario con el resultado."""
    n0 = len(esquinas_todas)
    if n0 > MAX_FRAMES_CALIBRACION:
        indices = np.linspace(0, n0 - 1, MAX_FRAMES_CALIBRACION).astype(int)
        esquinas_todas = [esquinas_todas[i] for i in indices]
        nombres = [nombres[i] for i in indices]
        mostrar(f"Se usan {len(esquinas_todas)} fotos repartidas a lo largo de toda la captura.")

    corregidas = corregir_etiquetado(esquinas_todas, tamano)
    if corregidas:
        mostrar(f"Fotos con esquinas numeradas al reves, corregidas: {corregidas}")
    quitadas, umbral = quitar_outliers(esquinas_todas, nombres, tamano)
    mostrar(f"Outliers descartados: {quitadas} (error por foto > {umbral:.2f} px). Quedan {len(esquinas_todas)} fotos." )

    if len(esquinas_todas) > MAX_FOTOS_COMPARACION:
        idx = np.linspace(0, len(esquinas_todas) - 1, MAX_FOTOS_COMPARACION).astype(int)
        muestra = [esquinas_todas[i] for i in idx]
    else:
        muestra = esquinas_todas

    mostrar("\nComparando modelos con validacion cruzada "
            f"({PLIEGUES_CV} grupos, {len(muestra)} fotos). 'Prueba' = error en fotos no usadas.\n")
    mostrar(f"{'Modelo':<32}{'Entreno':>9}{'Prueba':>9}{'fx':>9}{'+/- fx':>8}")
    tareas = [(muestra, tamano, nombre, flags, liberar) for nombre, flags, liberar in MODELOS]
    with ProcessPoolExecutor() as ejecutor:
        evaluados = list(ejecutor.map(evaluar_modelo, tareas))
    resultados = [r for r in evaluados if r is not None]
    for r in resultados:
        mostrar(f"{r['nombre']:<32}{r['train']:>9.3f}{r['cv']:>9.3f}{r['fx']:>9.1f}{r['dfx']:>8.1f}")

    mejor_cv = min(r["cv"] for r in resultados)
    elegido = next(r for r in resultados if r["cv"] <= mejor_cv * TOLERANCIA_SIMPLICIDAD)
    mostrar(f"\nModelo elegido: {elegido['nombre']}  (error en fotos no usadas: {elegido['cv']:.3f} px)")

    error, matriz, coef, rvecs, tvecs, objeto = calibrar_modelo(
        esquinas_todas, tamano, elegido["flags"], elegido["liberar"])
    avisos = informe_cobertura(rvecs, tvecs, esquinas_todas, tamano)

    fx, fy = float(matriz[0, 0]), float(matriz[1, 1])
    ancho, alto = tamano
    return {
        "error": float(error), "error_cv": elegido["cv"], "matriz": matriz, "coef": coef,
        "modelo": elegido["nombre"], "fx": fx, "fy": fy, "dfx": elegido["dfx"],
        "fotos": len(esquinas_todas), "tamano": tamano, "avisos": avisos, "objeto": objeto,
        "abombado_mm": abombado_mm(objeto),
        "campo_horizontal": float(2 * np.degrees(np.arctan(ancho / (2 * fx)))),
        "escala": distancia_cm / fx,
    }


def calibrar(carpeta, nombre_corto, distancia_cm):
    esquinas_todas, nombres, tamano = leer_esquinas(carpeta)
    if len(esquinas_todas) < FRAMES_MINIMOS:
        print(f"\nSolo hay {len(esquinas_todas)} fotos validas. Se necesitan al menos {FRAMES_MINIMOS}. "
              "Captura mas fotos (modo 1) con poses distintas.")
        return None

    r = calibrar_fotos(esquinas_todas, nombres, tamano, distancia_cm)
    ancho, alto = tamano

    print(f"\nResultado final ({r['fotos']} fotos usadas)")
    print(f"  Resolucion: {ancho}x{alto}")
    print(f"  Modelo: {r['modelo']}")
    print(f"  Error de reproyeccion (entreno): {r['error']:.4f} px")
    print(f"  Error en fotos no usadas (honesto): {r['error_cv']:.4f} px")
    print(f"  Focal: fx = {r['fx']:.1f} px, fy = {r['fy']:.1f} px  (+/- {r['dfx']:.1f} px entre grupos)")
    print(f"  Campo de vision horizontal: {r['campo_horizontal']:.1f} grados")
    print(f"  Planitud del tablero segun la calibracion: abombado maximo {r['abombado_mm']:.1f} mm")
    print(f"  Escala a {distancia_cm} cm: {r['escala']:.4f} cm/px "
          f"(+/- {100 * r['dfx'] / r['fx']:.1f} % por la focal)")
    print(f"  Coeficientes de distorsion (k1, k2, p1, p2, k3):\n{r['coef'].ravel()}")

    if abs(r["coef"].ravel()[0]) > 0.5:
        print("\nAVISO: k1 es muy grande para un iPhone. No uses esta distorsion para corregir imagenes; "
              "usa solo la focal para la escala.")
    if r["abombado_mm"] > 1.5:
        print("\nAVISO: el tablero se abomba mas de 1.5 mm. Pegalo completo sobre una superficie rigida y plana "
              "(MDF, vidrio, acrilico) y repite la captura; un tablero curvo sesga la focal.")
    if r["error_cv"] > 1.0:
        print("\nAVISO: el error en fotos nuevas pasa de 1 px. Revisa nitidez, que el tablero este plano "
              "y rigido, y que EV/AWB/enfoque no cambien durante la captura.")

    ruta_salida = f"calibracion_stream_{nombre_corto}.npz"
    np.savez(ruta_salida, matriz_camara=r["matriz"], coef_distorsion=r["coef"],
             error_reproyeccion=r["error"], error_validacion=r["error_cv"],
             modelo=r["modelo"], tamano_cuadro_cm=TAMANO_CUADRO_CM,
             resolucion=np.array(tamano), distancia_cm=distancia_cm,
             escala_cm_por_px=r["escala"], puntos_tablero=r["objeto"])
    print(f"\nCalibracion guardada en: {ruta_salida}")
    return r


def main():
    from camara import CamaraIP, cargar_configuracion, guardar_configuracion

    config = cargar_configuracion()
    claves = list(config.keys())

    print("Camaras disponibles:")
    for i, clave in enumerate(claves, start=1):
        print(f"  {i}. {config[clave]['nombre']} ({config[clave]['ip']})")
    opcion = int(input("Numero de camara a calibrar: ").strip())
    clave = claves[opcion - 1]

    datos = pedir_ip(config, clave, guardar_configuracion)
    nombre_corto = datos["nombre"].lower().replace(" pro max", "").replace(" ", "")
    carpeta = os.path.join(BASE, "calibracion zhang", f"Stream {datos['nombre']}")

    por_defecto = DISTANCIAS_CM.get(datos["nombre"], 36.0)
    entrada = input(f"\nDistancia de trabajo lente-cara en cm (ENTER para {por_defecto}): ").strip()
    distancia_cm = float(entrada.replace(",", ".")) if entrada else por_defecto

    modo = input("1 = capturar fotos y calibrar, 2 = solo calibrar las fotos ya guardadas: ").strip()

    if modo == "1":
        previas = glob.glob(os.path.join(carpeta, "stream_*.jpg"))
        if previas:
            resp = input(f"Hay {len(previas)} fotos anteriores. ENTER = conservarlas, "
                         "A = apartarlas en la carpeta 'anteriores' y empezar limpio: ").strip().lower()
            if resp == "a":
                destino = os.path.join(carpeta, "anteriores")
                os.makedirs(destino, exist_ok=True)
                for ruta in previas:
                    shutil.move(ruta, os.path.join(destino, os.path.basename(ruta)))
                print("Fotos anteriores apartadas.")

        cam = CamaraIP(datos)
        print(f"\nConectando con {cam.nombre} ({cam.base_url}) ...")
        try:
            cam.iniciar_sesion()
        except ConnectionError as e:
            print(e)
            return
        print("Conectado. Fija EV, AWB y enfoque en DroidCam a la distancia de trabajo.")
        input("Presiona ENTER cuando estes listo...")
        capturar(cam, carpeta)
        cam.cerrar_sesion()

    calibrar(carpeta, nombre_corto, distancia_cm)


if __name__ == "__main__":
    main()

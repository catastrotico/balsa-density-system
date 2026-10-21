"""
preparar_dataset.py
Paso 14: divide las fotos reales en entrenamiento / validacion / prueba
y genera fotos sinteticas SOLO para entrenamiento.

Entrada : dataset_balsa/<CAMARA>/  (imagenes, mascaras, etiquetas.csv)
Salida  : dataset_final/
              train/imagenes, train/mascaras
              val/imagenes,   val/mascaras
              test/imagenes,  test/mascaras
              lista.csv, muestra_aumentos.jpg
"""
import csv
import math
import os
import random
import shutil

import cv2
import numpy as np

# ---------------------------------------------------------------- ajustes
CAMARA = "iphone11"
CARPETA = os.path.join("dataset_balsa", CAMARA)
SALIDA = "dataset_final"

TAM_GRUPO = 3          # fotos seguidas que se mantienen juntas en un mismo conjunto
FRAC_VAL = 0.15
FRAC_TEST = 0.15
MIN_GIRADAS = 2        # minimo de fotos con probeta girada en val y en test
MIN_FRONTALES = 2      # minimo de fotos frontales en val y en test

COPIAS_POS = 15        # variantes sinteticas por foto real con probeta
COPIAS_NEG = 6         # variantes sinteticas por foto real sin probeta
SEMILLA_AUMENTO = 123
# ------------------------------------------------------------------------


def leer_etiquetas():
    ruta = os.path.join(CARPETA, "etiquetas.csv")
    with open(ruta, newline="", encoding="utf-8") as f:
        filas = list(csv.DictReader(f))
    return filas


def es_negativa(fila):
    return fila["tipo"].strip().lower() == "negativa"


def ruta_imagen(archivo):
    return os.path.join(CARPETA, "imagenes", archivo)


def ruta_mascara(archivo):
    base = os.path.splitext(archivo)[0]
    for nombre in (archivo, base + ".png"):
        ruta = os.path.join(CARPETA, "mascaras", nombre)
        if os.path.exists(ruta):
            return ruta
    return None


def leer_par(archivo):
    img = cv2.imread(ruta_imagen(archivo))
    if img is None:
        raise RuntimeError("No pude leer la imagen " + archivo)
    ruta_m = ruta_mascara(archivo)
    if ruta_m is None:
        mask = np.zeros(img.shape[:2], np.uint8)
    else:
        mask = cv2.imread(ruta_m, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            mask = np.zeros(img.shape[:2], np.uint8)
        if mask.shape != img.shape[:2]:
            mask = cv2.resize(mask, (img.shape[1], img.shape[0]),
                              interpolation=cv2.INTER_NEAREST)
        mask = np.where(mask > 127, 255, 0).astype(np.uint8)
    return img, mask


# ---------------------------------------------------------- tipo de pose
def es_girada(fila):
    """True si la cara se ve inclinada o con perspectiva (trapecio)."""
    try:
        p = np.array([[float(fila["x%d" % i]), float(fila["y%d" % i])]
                      for i in range(1, 5)])
    except (KeyError, ValueError):
        return False
    lados = [np.linalg.norm(p[(i + 1) % 4] - p[i]) for i in range(4)]
    asim = max(abs(lados[0] - lados[2]) / max(lados[0], lados[2]),
               abs(lados[1] - lados[3]) / max(lados[1], lados[3]))
    inclin = 0.0
    for i in range(4):
        d = p[(i + 1) % 4] - p[i]
        ang = math.degrees(math.atan2(d[1], d[0])) % 90
        if ang > 45:
            ang -= 90
        inclin = max(inclin, abs(ang))
    return asim > 0.03 or inclin > 3.0


# ------------------------------------------------------------------ split
def hacer_grupos(filas):
    filas = sorted(filas, key=lambda r: r["archivo"])
    return [filas[i:i + TAM_GRUPO] for i in range(0, len(filas), TAM_GRUPO)]


def repartir(grupos, semilla):
    rnd = random.Random(semilla)
    g = grupos[:]
    rnd.shuffle(g)
    n = sum(len(x) for x in g)
    n_val = round(n * FRAC_VAL)
    n_test = round(n * FRAC_TEST)
    out = {"train": [], "val": [], "test": []}
    for grupo in g:
        if len(out["test"]) < n_test:
            out["test"].extend(grupo)
        elif len(out["val"]) < n_val:
            out["val"].extend(grupo)
        else:
            out["train"].extend(grupo)
    return out


def dividir(filas):
    pos = [r for r in filas if not es_negativa(r)]
    neg = [r for r in filas if es_negativa(r)]
    gpos = hacer_grupos(pos)
    gneg = hacer_grupos(neg)
    for semilla in range(42, 542):
        rp = repartir(gpos, semilla)
        ok = True
        for k in ("val", "test"):
            gir = sum(1 for r in rp[k] if es_girada(r))
            fro = len(rp[k]) - gir
            if gir < MIN_GIRADAS or fro < MIN_FRONTALES:
                ok = False
        if ok:
            rn = repartir(gneg, semilla)
            return {k: rp[k] + rn[k] for k in rp}, semilla
    rp = repartir(gpos, 42)
    rn = repartir(gneg, 42)
    print("AVISO: no logre un reparto con la mezcla de poses pedida.")
    return {k: rp[k] + rn[k] for k in rp}, 42


# ---------------------------------------------------------- aumentos
def fotometrica(img, rnd):
    x = img.astype(np.float32)

    g = rnd.uniform(0.7, 1.4)
    x = 255.0 * np.power(np.clip(x, 0, 255) / 255.0, g)

    alfa = rnd.uniform(0.75, 1.3)
    beta = rnd.uniform(-35, 35)
    x = x * alfa + beta

    # tono: cambio de temperatura de color (calido / frio)
    canales = np.array([rnd.uniform(0.9, 1.1), rnd.uniform(0.93, 1.07),
                        rnd.uniform(0.9, 1.1)], np.float32)
    x = x * canales
    x = np.clip(x, 0, 255).astype(np.uint8)

    # saturacion
    hsv = cv2.cvtColor(x, cv2.COLOR_BGR2HSV).astype(np.float32)
    hsv[..., 1] *= rnd.uniform(0.7, 1.3)
    hsv[..., 1] = np.clip(hsv[..., 1], 0, 255)
    x = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    # desenfoque (foco)
    r = rnd.random()
    if r < 0.35:
        k = rnd.choice([3, 5, 7, 9])
        x = cv2.GaussianBlur(x, (k, k), 0)
    elif r < 0.5:
        largo = rnd.choice([5, 7, 9, 11])
        ker = np.zeros((largo, largo), np.float32)
        ker[largo // 2, :] = 1.0
        ang = rnd.uniform(0, 180)
        M = cv2.getRotationMatrix2D((largo / 2 - 0.5, largo / 2 - 0.5), ang, 1.0)
        ker = cv2.warpAffine(ker, M, (largo, largo))
        ker /= max(ker.sum(), 1e-6)
        x = cv2.filter2D(x, -1, ker)

    # ruido
    sigma = rnd.uniform(0, 8)
    if sigma > 0.5:
        rng = np.random.default_rng(rnd.randrange(1 << 30))
        ruido = rng.normal(0, sigma, x.shape).astype(np.float32)
        x = np.clip(x.astype(np.float32) + ruido, 0, 255).astype(np.uint8)

    # compresion jpeg
    if rnd.random() < 0.5:
        calidad = rnd.randint(60, 95)
        ok, buf = cv2.imencode(".jpg", x, [cv2.IMWRITE_JPEG_QUALITY, calidad])
        if ok:
            x = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    return x


def geometrica(img, mask, rnd):
    h, w = img.shape[:2]
    area0 = int((mask > 0).sum())
    for _ in range(8):
        ang = rnd.uniform(-5, 5)
        esc = rnd.uniform(0.92, 1.08)
        M = cv2.getRotationMatrix2D((w / 2, h / 2), ang, esc)
        M[0, 2] += rnd.uniform(-0.04, 0.04) * w
        M[1, 2] += rnd.uniform(-0.04, 0.04) * h
        im2 = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_REPLICATE)
        mk2 = cv2.warpAffine(mask, M, (w, h), flags=cv2.INTER_NEAREST,
                             borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        if rnd.random() < 0.5:
            im2 = cv2.flip(im2, 1)
            mk2 = cv2.flip(mk2, 1)
        if area0 == 0 or (mk2 > 0).sum() >= 0.9 * area0:
            return im2, np.where(mk2 > 127, 255, 0).astype(np.uint8)
    return img, mask


def aumentar(img, mask, rnd):
    im2, mk2 = geometrica(img, mask, rnd)
    im2 = fotometrica(im2, rnd)
    return im2, mk2


# ------------------------------------------------------------- guardado
def guardar(conjunto, nombre, img, mask):
    cv2.imwrite(os.path.join(SALIDA, conjunto, "imagenes", nombre + ".jpg"),
                img, [cv2.IMWRITE_JPEG_QUALITY, 95])
    cv2.imwrite(os.path.join(SALIDA, conjunto, "mascaras", nombre + ".png"), mask)


def superponer(img, mask):
    vista = img.copy()
    verde = np.zeros_like(vista)
    verde[..., 1] = 255
    m = mask > 0
    vista[m] = (0.6 * vista[m] + 0.4 * verde[m]).astype(np.uint8)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(vista, cnts, -1, (0, 255, 0), 2)
    return vista


def main():
    if not os.path.exists(os.path.join(CARPETA, "etiquetas.csv")):
        print("No encuentro", os.path.join(CARPETA, "etiquetas.csv"))
        print("Revisa que estas en la carpeta adquisicion_imagenes.")
        return

    filas = leer_etiquetas()
    faltan = [r["archivo"] for r in filas
              if not os.path.exists(ruta_imagen(r["archivo"]))]
    if faltan:
        print("AVISO: %d filas del CSV no tienen imagen y se omiten:" % len(faltan))
        print("  " + ", ".join(faltan))
        filas = [r for r in filas if r["archivo"] not in faltan]
    npos = sum(1 for r in filas if not es_negativa(r))
    nneg = len(filas) - npos
    print("Fotos reales: %d con probeta, %d sin probeta" % (npos, nneg))

    reparto, semilla = dividir(filas)

    if os.path.exists(SALIDA):
        shutil.rmtree(SALIDA)
    for c in ("train", "val", "test"):
        os.makedirs(os.path.join(SALIDA, c, "imagenes"))
        os.makedirs(os.path.join(SALIDA, c, "mascaras"))

    rnd = random.Random(SEMILLA_AUMENTO)
    lista = []
    muestras = []

    for conjunto in ("train", "val", "test"):
        for fila in reparto[conjunto]:
            archivo = fila["archivo"]
            base = os.path.splitext(archivo)[0]
            neg = es_negativa(fila)
            img, mask = leer_par(archivo)
            guardar(conjunto, base + "_orig", img, mask)
            lista.append([conjunto, base + "_orig", archivo,
                          "negativa" if neg else "positiva", 0])
            if conjunto != "train":
                continue
            copias = COPIAS_NEG if neg else COPIAS_POS
            for k in range(1, copias + 1):
                im2, mk2 = aumentar(img, mask, rnd)
                nombre = "%s_a%02d" % (base, k)
                guardar(conjunto, nombre, im2, mk2)
                lista.append([conjunto, nombre, archivo,
                              "negativa" if neg else "positiva", 1])
                if k == 1:
                    muestras.append((im2, mk2))

    with open(os.path.join(SALIDA, "lista.csv"), "w", newline="",
              encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["conjunto", "archivo", "foto_real", "tipo", "sintetica"])
        w.writerows(lista)

    # hoja de muestra para revisar los aumentos
    rnd.shuffle(muestras)
    muestras = muestras[:12]
    if muestras:
        celdas = []
        for im2, mk2 in muestras:
            v = superponer(im2, mk2)
            celdas.append(cv2.resize(v, (320, 180)))
        while len(celdas) % 4:
            celdas.append(np.zeros_like(celdas[0]))
        filas_img = [np.hstack(celdas[i:i + 4]) for i in range(0, len(celdas), 4)]
        cv2.imwrite(os.path.join(SALIDA, "muestra_aumentos.jpg"),
                    np.vstack(filas_img))

    print("")
    print("Reparto (por foto real, semilla %d):" % semilla)
    for c in ("train", "val", "test"):
        p = [r for r in reparto[c] if not es_negativa(r)]
        n = [r for r in reparto[c] if es_negativa(r)]
        gir = sum(1 for r in p if es_girada(r))
        print("  %-5s %2d con probeta (%d giradas, %d frontales), %2d sin probeta"
              % (c, len(p), gir, len(p) - gir, len(n)))
    print("")
    print("Imagenes finales:")
    for c in ("train", "val", "test"):
        total = sum(1 for x in lista if x[0] == c)
        sint = sum(1 for x in lista if x[0] == c and x[4] == 1)
        print("  %-5s %4d (de ellas %d sinteticas)" % (c, total, sint))
    print("")
    print("Listo. Carpeta:", os.path.abspath(SALIDA))
    print("Abre dataset_final\\muestra_aumentos.jpg para revisar.")


if __name__ == "__main__":
    main()

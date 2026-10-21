"""
Verificacion rapida de la calibracion de camara contra el stream de
DroidCam. NO recalibra: captura frames del tablero con la camara ya
montada, reescala la matriz de camara existente a la resolucion del
stream y mide el error de reproyeccion.

Controles en la ventana de video:
    ESPACIO -> guarda el frame actual (solo si el tablero esta detectado)
    Q       -> termina la captura y calcula el resultado

Uso: python verificar_calibracion.py
"""

import os

import cv2
import numpy as np

from camara import CamaraIP, cargar_configuracion, guardar_configuracion

ESQUINAS_ANCHO = 9
ESQUINAS_ALTO = 6
TAMANO_CUADRO_CM = 2.9
RESOLUCION_CALIBRACION = (4032, 3024)  # fotos nativas usadas para calibrar
MINIMO_FRAMES = 5
BASE = r"C:\Users\marli\Documents\9 Semestre\PI 9 semestre\Claude outputs\proyecto_balsa\adquisicion_imagenes"


def puntos_del_patron():
    puntos = np.zeros((ESQUINAS_ALTO * ESQUINAS_ANCHO, 3), np.float32)
    puntos[:, :2] = np.mgrid[0:ESQUINAS_ANCHO, 0:ESQUINAS_ALTO].T.reshape(-1, 2)
    return puntos * TAMANO_CUADRO_CM


def resolver_archivo_npz(entrada):
    """
    Acepta el nombre con o sin extension y sin importar mayusculas.
    Si no se escribe nada, lista los .npz disponibles para elegir.
    """
    disponibles = sorted(f for f in os.listdir(BASE) if f.lower().endswith(".npz"))
    if not disponibles:
        return None

    entrada = entrada.strip().strip('"')
    if not entrada:
        print("\nArchivos de calibracion disponibles:")
        for i, nombre in enumerate(disponibles, start=1):
            print(f"  {i}. {nombre}")
        eleccion = input("Numero del archivo: ").strip()
        if eleccion.isdigit() and 1 <= int(eleccion) <= len(disponibles):
            return os.path.join(BASE, disponibles[int(eleccion) - 1])
        return None

    if not entrada.lower().endswith(".npz"):
        entrada += ".npz"
    for nombre in disponibles:
        if nombre.lower() == entrada.lower():
            return os.path.join(BASE, nombre)
    return None


def pedir_ip(config, clave):
    """Muestra la IP guardada y permite cambiarla. ENTER mantiene la actual."""
    datos = config[clave]
    print(f"\nIP guardada para {datos['nombre']}: {datos['ip']}  (puerto {datos['puerto']})")
    nueva_ip = input("Nueva IP (ENTER para mantener): ").strip()
    if nueva_ip:
        datos["ip"] = nueva_ip
        guardar_configuracion(config)
        print(f"IP actualizada a {nueva_ip} y guardada en config.json")
    return datos


def capturar_frames(cam, carpeta_salida):
    """Muestra el stream y guarda frames con tablero detectado."""
    frames = []
    print("\nESPACIO guarda el frame (solo si el tablero esta detectado), Q termina.")
    print("Mueve el tablero entre cada frame: distintas posiciones e inclinaciones.\n")

    while True:
        ret, frame = cam._captura.read()
        if not ret:
            print("No se pudo leer el stream. Revisa la conexion.")
            break

        gris = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        detectado, esquinas = cv2.findChessboardCornersSB(
            gris, (ESQUINAS_ANCHO, ESQUINAS_ALTO), flags=cv2.CALIB_CB_ACCURACY
        )

        vista = frame.copy()
        if detectado:
            cv2.drawChessboardCorners(vista, (ESQUINAS_ANCHO, ESQUINAS_ALTO), esquinas, detectado)
            estado, color = "DETECTADO - ESPACIO para guardar", (0, 200, 0)
        else:
            estado, color = "Tablero no detectado", (0, 0, 255)

        cv2.putText(vista, estado, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, color, 2)
        cv2.putText(vista, f"Guardados: {len(frames)}", (20, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        cv2.imshow("Verificacion de calibracion", vista)

        tecla = cv2.waitKey(1) & 0xFF
        if tecla == ord("q"):
            break
        if tecla == ord(" ") and detectado:
            frames.append((frame.copy(), esquinas))
            nombre = f"verif_{len(frames):03d}.jpg"
            cv2.imwrite(os.path.join(carpeta_salida, nombre), frame)
            print(f"  Guardado {nombre}")

    cv2.destroyAllWindows()
    return frames


def reescalar_matriz(matriz, ancho, alto):
    """
    Escala la matriz de camara de las fotos nativas (4:3) al stream.
    Se asume que el stream es un recorte centrado con el mismo campo de
    vision horizontal: la escala es ancho_stream / ancho_nativo y el
    punto principal queda en el centro del stream.
    """
    escala = ancho / RESOLUCION_CALIBRACION[0]
    nueva = matriz.copy()
    nueva[0, 0] = matriz[0, 0] * escala
    nueva[1, 1] = matriz[1, 1] * escala
    nueva[0, 2] = ancho / 2.0
    nueva[1, 2] = alto / 2.0
    return nueva, escala


def error_por_frame(frames, matriz, coef, puntos_3d):
    errores, distancias = [], []
    for _, esquinas in frames:
        ok, rvec, tvec = cv2.solvePnP(puntos_3d, esquinas, matriz, coef)
        if not ok:
            continue
        proyectados, _ = cv2.projectPoints(puntos_3d, rvec, tvec, matriz, coef)
        dif = esquinas.reshape(-1, 2) - proyectados.reshape(-1, 2)
        errores.append(float(np.sqrt(np.mean(np.sum(dif ** 2, axis=1)))))
        distancias.append(float(tvec[2][0]))
    return errores, distancias


def estimar_focal_del_stream(frames, matriz_escalada, coef, puntos_3d, tamano):
    """
    Estima la focal usando solo los frames del stream (principal fijo y
    distorsion fija). Sirve como segunda opinion: si sale parecida a la
    focal escalada, el reescalado es correcto.
    """
    flags = (cv2.CALIB_USE_INTRINSIC_GUESS | cv2.CALIB_FIX_PRINCIPAL_POINT |
             cv2.CALIB_FIX_ASPECT_RATIO | cv2.CALIB_ZERO_TANGENT_DIST |
             cv2.CALIB_FIX_K1 | cv2.CALIB_FIX_K2 | cv2.CALIB_FIX_K3)
    try:
        _, matriz_est, _, _, _ = cv2.calibrateCamera(
            [puntos_3d] * len(frames), [e for _, e in frames], tamano,
            matriz_escalada.copy(), coef.copy(), flags=flags
        )
        return float(matriz_est[0, 0])
    except cv2.error:
        return None


def main():
    config = cargar_configuracion()
    claves = list(config.keys())

    print("Camaras disponibles:")
    for i, clave in enumerate(claves, start=1):
        print(f"  {i}. {config[clave]['nombre']} ({config[clave]['ip']})")
    opcion = int(input("Numero de camara a verificar: ").strip())
    clave = claves[opcion - 1]

    datos = pedir_ip(config, clave)

    ruta_npz = resolver_archivo_npz(
        input("\nArchivo de calibracion (ENTER para ver la lista): ")
    )
    if ruta_npz is None:
        print("No se encontro el archivo de calibracion.")
        return
    print(f"Usando: {os.path.basename(ruta_npz)}")
    calibracion = np.load(ruta_npz)
    matriz = calibracion["matriz_camara"]
    coef = calibracion["coef_distorsion"]

    cam = CamaraIP(datos)
    carpeta_salida = os.path.join(BASE, "verificacion_stream", datos["nombre"].replace(" ", "_"))
    os.makedirs(carpeta_salida, exist_ok=True)

    print(f"\nConectando con {cam.nombre} ({cam.base_url}) ...")
    try:
        cam.iniciar_sesion()
    except ConnectionError as e:
        print(e)
        return
    print("Conectado. Fija EV, AWB y enfoque en la app DroidCam.")
    input("Presiona ENTER cuando estes listo...")

    frames = capturar_frames(cam, carpeta_salida)
    cam.cerrar_sesion()

    if len(frames) < MINIMO_FRAMES:
        print(f"\nSolo {len(frames)} frames. Se necesitan al menos {MINIMO_FRAMES}.")
        return

    alto, ancho = frames[0][0].shape[:2]
    print(f"\nResolucion del stream: {ancho}x{alto}")
    print(f"Resolucion de la calibracion: {RESOLUCION_CALIBRACION[0]}x{RESOLUCION_CALIBRACION[1]}")

    puntos_3d = puntos_del_patron()
    matriz_escalada, escala = reescalar_matriz(matriz, ancho, alto)
    print(f"Factor de escala aplicado: {escala:.4f}")
    print(f"Focal original: {matriz[0, 0]:.1f} px -> reescalada: {matriz_escalada[0, 0]:.1f} px")

    errores, distancias = error_por_frame(frames, matriz_escalada, coef, puntos_3d)
    if not errores:
        print("No se pudo calcular el error en ningun frame.")
        return

    print("\nError de reproyeccion por frame (RMS, en pixeles del stream):")
    for i, (error, distancia) in enumerate(zip(errores, distancias), start=1):
        print(f"  verif_{i:03d}: {error:.3f} px   (distancia lente-tablero: {distancia:.1f} cm)")

    promedio = float(np.mean(errores))
    print(f"\nError promedio: {promedio:.3f} px")

    if len(frames) >= 6:
        focal_est = estimar_focal_del_stream(frames, matriz_escalada, coef, puntos_3d, (ancho, alto))
        if focal_est is not None:
            diferencia = 100.0 * (focal_est - matriz_escalada[0, 0]) / matriz_escalada[0, 0]
            print(f"Focal estimada con los frames del stream: {focal_est:.1f} px "
                  f"({diferencia:+.1f}% respecto a la reescalada)")

    print("\nInterpretacion:")
    if promedio < 1.0:
        print("  Menos de 1 px: la calibracion sirve para el stream. Queda validada.")
    elif promedio < 2.0:
        print("  Entre 1 y 2 px: aceptable. Usable, con algo de incertidumbre.")
    else:
        print("  Mas de 2 px: la calibracion actual no representa bien al stream.")
        print("  Conviene recalibrar con frames del stream.")


if __name__ == "__main__":
    main()

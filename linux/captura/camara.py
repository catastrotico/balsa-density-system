"""
Modulo de adquisicion de imagenes - Proyecto Integrador
Estimacion de densidad aparente de madera balsa por vision artificial

Este modulo maneja la conexion con las camaras IP (app DroidCam, instalada en
los iPhone 15 Pro Max y iPhone 11) usadas como sensores de adquisicion de imagen.

IMPORTANTE sobre EV/AWB/enfoque manual:
DroidCam permite fijar manualmente exposicion (EV), balance de blancos (AWB) y
enfoque desde la interfaz de la app, pero esos valores SOLO se mantienen
mientras la conexion de video este abierta. Si la conexion se cierra y se
vuelve a abrir, la app olvida la configuracion y vuelve a modo automatico.

Por eso el flujo de trabajo correcto es:
    1. Llamar a iniciar_sesion() UNA vez, al empezar el lote de mediciones
       (esto abre la conexion con la camara, todavia en modo automatico).
    2. Configurar manualmente EV/AWB/enfoque en la app DroidCam del celular,
       viendo el efecto en vivo.
    3. Llamar a capturar_foto() cada vez que el sensor de barrera IR dispare
       (RF-10), tantas veces como haga falta, SIN cerrar la sesion entre tomas.
    4. Llamar a cerrar_sesion() solo al terminar todo el lote de muestras.

NOTA: esta version usa msvcrt para detectar la tecla 'd' y es SOLO PARA WINDOWS.
La version que corre en la UNO Q (Linux) usa termios/tty en su lugar y vive
por separado alla, no se debe copiar esta encima de esa.
"""

import json
import os
import time
from datetime import datetime

import cv2
import msvcrt

RUTA_CONFIG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.json")
CARPETA_CAPTURAS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "capturas")


def cargar_configuracion(ruta=RUTA_CONFIG):
    """
    Lee config.json y devuelve un diccionario con los datos de las camaras.
    Si el archivo no existe, crea uno de plantilla y avisa al usuario que debe
    completarlo antes de continuar.
    """
    if not os.path.exists(ruta):
        plantilla = {
            "camara_1": {"nombre": "iPhone 15 Pro Max", "ip": "0.0.0.0", "puerto": 4747,
                         "usuario": "", "password": ""},
            "camara_2": {"nombre": "iPhone 11", "ip": "0.0.0.0", "puerto": 4747,
                         "usuario": "", "password": ""},
        }
        with open(ruta, "w", encoding="utf-8") as f:
            json.dump(plantilla, f, indent=4, ensure_ascii=False)
        print(f"No existia config.json, se creo una plantilla en: {ruta}")
        print("Edita ese archivo con las IP actuales antes de continuar (o usa la "
              "actualizacion interactiva).")
        return plantilla

    with open(ruta, "r", encoding="utf-8") as f:
        return json.load(f)


def guardar_configuracion(config, ruta=RUTA_CONFIG):
    """Guarda el diccionario de configuracion de vuelta en config.json."""
    with open(ruta, "w", encoding="utf-8") as f:
        json.dump(config, f, indent=4, ensure_ascii=False)


def actualizar_configuracion_interactiva(config, ruta=RUTA_CONFIG):
    """
    Muestra la configuracion actual de cada camara y permite actualizar la IP
    (y el puerto, si hace falta) por consola. Presionar ENTER sin escribir
    nada deja el valor actual sin cambios.
    """
    for clave, datos in config.items():
        print(f"\n--- {datos.get('nombre', clave)} ({clave}) ---")
        print(f"IP actual: {datos.get('ip')}   Puerto actual: {datos.get('puerto')}")

        nueva_ip = input(f"Nueva IP para {datos.get('nombre', clave)} "
                          f"(ENTER para mantener '{datos.get('ip')}'): ").strip()
        if nueva_ip:
            datos["ip"] = nueva_ip

        nuevo_puerto = input(f"Nuevo puerto (ENTER para mantener '{datos.get('puerto')}'): ").strip()
        if nuevo_puerto:
            datos["puerto"] = int(nuevo_puerto)

    guardar_configuracion(config, ruta)
    print(f"\nConfiguracion actualizada y guardada en: {ruta}")
    return config


class CamaraIP:
    """
    Representa una camara IP (celular con la app DroidCam).

    El video se lee del endpoint MJPEG /video con cv2.VideoCapture(). DroidCam
    solo mantiene el EV/AWB/enfoque fijados manualmente mientras esa conexion
    de video no se cierre, asi que esta clase mantiene la conexion abierta
    entre capturas en vez de abrir una nueva cada vez.
    """

    def __init__(self, datos_config):
        self.nombre = datos_config.get("nombre", "camara")
        self.ip = datos_config["ip"]
        self.puerto = datos_config.get("puerto", 4747)
        self.usuario = datos_config.get("usuario", "")
        self.password = datos_config.get("password", "")
        self._captura = None

    @property
    def base_url(self):
        return f"http://{self.ip}:{self.puerto}"

    def url_stream_mjpeg(self):
        return f"{self.base_url}/video"

    def _url_stream_con_credenciales(self):
        if self.usuario:
            return f"http://{self.usuario}:{self.password}@{self.ip}:{self.puerto}/video"
        return self.url_stream_mjpeg()

    def sesion_abierta(self):
        return self._captura is not None

    def iniciar_sesion(self):
        """Abre la conexion con la camara y la deja abierta para multiples capturas."""
        if self._captura is not None:
            return

        captura = cv2.VideoCapture(self._url_stream_con_credenciales())
        if not captura.isOpened():
            captura.release()
            raise ConnectionError(
                f"No se pudo abrir sesion con '{self.nombre}' en {self.url_stream_mjpeg()}. "
                f"Verifica que el celular este en la misma red WiFi que este equipo, que la "
                f"app DroidCam este abierta con el servidor iniciado (revisa que no diga "
                f"'DroidCam is Busy'), y que la IP/puerto en config.json esten actualizados."
            )
        self._captura = captura

    def cerrar_sesion(self):
        if self._captura is not None:
            self._captura.release()
            self._captura = None

    def capturar_foto(self, nombre_archivo=None, carpeta_destino=CARPETA_CAPTURAS):
        """Toma UN cuadro de la sesion ya abierta y lo guarda en disco."""
        if self._captura is None:
            raise RuntimeError(
                f"No hay una sesion abierta para '{self.nombre}'. Llama a iniciar_sesion() primero."
            )

        os.makedirs(carpeta_destino, exist_ok=True)

        if nombre_archivo is None:
            marca_tiempo = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            nombre_archivo = f"{self.nombre.replace(' ', '_')}_{marca_tiempo}.jpg"

        ruta_completa = os.path.join(carpeta_destino, nombre_archivo)

        ret, frame = self._captura.read()
        if not ret:
            raise ConnectionError(
                f"La sesion con '{self.nombre}' no respondio al pedir un cuadro nuevo."
            )

        cv2.imwrite(ruta_completa, frame)
        return ruta_completa


if __name__ == "__main__":
    config = cargar_configuracion()

    respuesta = input("\n¿Actualizar las IP de las camaras antes de continuar? (s/n): ").strip().lower()
    if respuesta == "s":
        config = actualizar_configuracion_interactiva(config)

    # 1. Conectar con ambas camaras primero
    camaras = []
    for clave, datos in config.items():
        cam = CamaraIP(datos)
        print(f"\nConectando con {cam.nombre} ({cam.base_url}) ...")
        try:
            cam.iniciar_sesion()
            print(f"Conectado a {cam.nombre}.")
            camaras.append(cam)
        except ConnectionError as e:
            print(e)

    if not camaras:
        print("\nNo se pudo conectar con ninguna camara. Revisa la red y las apps, y vuelve a intentar.")
    else:
        # 2. Pausa para configurar EV/AWB/enfoque manualmente viendo el efecto en vivo
        print(f"\nConectado a {len(camaras)} camara(s). Ahora configura manualmente "
              f"EV/AWB/enfoque en la app DroidCam de cada celular.")
        input("Presiona ENTER cuando estes listo para empezar a capturar...")

        # 3. Captura continua cada 1 segundo hasta presionar 'd'
        print("\nCapturando cada 1 segundo. Presiona 'd' en cualquier momento para detener.\n")
        contador = 0
        detener = False
        try:
            while not detener:
                inicio = time.time()
                contador += 1
                for cam in camaras:
                    try:
                        ruta = cam.capturar_foto()
                        print(f"[{contador}] {cam.nombre}: {ruta}")
                    except ConnectionError as e:
                        print(f"[{contador}] {cam.nombre}: ERROR - {e}")

                while time.time() - inicio < 1.0:
                    if msvcrt.kbhit():
                        tecla = msvcrt.getch().decode(errors="ignore").lower()
                        if tecla == "d":
                            detener = True
                            break
                    time.sleep(0.05)
        except KeyboardInterrupt:
            pass

        # 4. Cerrar sesiones al terminar
        for cam in camaras:
            cam.cerrar_sesion()
        print(f"\nCaptura detenida. Total de rondas: {contador}. Sesiones cerradas.")

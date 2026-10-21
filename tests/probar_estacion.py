import json
import time
import urllib.request

# IP de la UNO Q en la red (se ve en la informacion de red de App Lab).
# Si cambia, actualiza esta linea.
ESTACION = "http://10.12.2.108:8000"


def pedir(ruta):
    with urllib.request.urlopen(ESTACION + ruta, timeout=5) as respuesta:
        return json.loads(respuesta.read().decode("utf-8"))


def main():
    try:
        print("Estado inicial:", pedir("/estado"))
    except Exception as error:
        print("No pude conectar con la estacion:", error)
        return

    for grupo in ("iphone15", "iphone11"):
        print("Luces del", grupo, "al 60 %")
        pedir("/luz?grupo=" + grupo + "&pct=60")
        time.sleep(3)
        pedir("/luz?grupo=" + grupo + "&pct=0")

    print("Ahora pon y quita la probeta. Ctrl+C para terminar.")
    anterior = None
    try:
        while True:
            datos = pedir("/estado")
            if datos != anterior:
                print(datos)
                anterior = datos
            time.sleep(0.3)
    except KeyboardInterrupt:
        pedir("/luz?grupo=todas&pct=0")
        print("Luces apagadas, fin de la prueba")


main()

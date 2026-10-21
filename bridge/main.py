from arduino.app_utils import *
import json
import queue
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

# Servidor para que el PC consulte la balanza y controle las luces.
# Grupos de luces por camara (canales del PCA9685 a partir de 0):
#   iPhone 15 -> CH1 y CH3 -> canales 0 y 2
#   iPhone 11 -> CH2 y CH4 -> canales 1 y 3
GRUPOS = {
    "iphone15": [0, 2],
    "iphone11": [1, 3],
    "todas": [0, 1, 2, 3],
}
PUERTO = 8000

datos = {"estado": 0, "masa": -1.0}
comandos = queue.Queue()


class Manejador(BaseHTTPRequestHandler):
    def responder(self, codigo, contenido):
        cuerpo = json.dumps(contenido).encode("utf-8")
        self.send_response(codigo)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(cuerpo)))
        self.end_headers()
        self.wfile.write(cuerpo)

    def do_GET(self):
        url = urlparse(self.path)
        if url.path == "/estado":
            self.responder(200, datos)
        elif url.path == "/luz":
            consulta = parse_qs(url.query)
            grupo = consulta.get("grupo", [""])[0]
            try:
                pct = int(consulta.get("pct", ["0"])[0])
            except ValueError:
                self.responder(400, {"error": "pct no es un numero"})
                return
            if grupo not in GRUPOS:
                self.responder(400, {"error": "grupo desconocido"})
                return
            pct = max(0, min(100, pct))
            comandos.put((grupo, pct))
            self.responder(200, {"ok": True, "grupo": grupo, "pct": pct})
        else:
            self.responder(404, {"error": "ruta desconocida"})

    def log_message(self, formato, *args):
        pass  # sin mensajes por cada peticion


def iniciar_servidor():
    servidor = ThreadingHTTPServer(("0.0.0.0", PUERTO), Manejador)
    hilo = threading.Thread(target=servidor.serve_forever, daemon=True)
    hilo.start()
    print("Servidor de la estacion en el puerto", PUERTO)


servidor_iniciado = False


def loop():
    global servidor_iniciado
    if not servidor_iniciado:
        iniciar_servidor()
        servidor_iniciado = True

    # ejecutar las ordenes de luz que llegaron del PC
    while not comandos.empty():
        grupo, pct = comandos.get()
        for canal in GRUPOS[grupo]:
            Bridge.call("luz", canal, pct)
        print("Luces", grupo, "al", pct, "%")

    # actualizar el estado de la balanza
    datos["estado"] = Bridge.call("estado_balanza")
    datos["masa"] = Bridge.call("masa_final")
    time.sleep(0.2)


App.run(user_loop=loop)

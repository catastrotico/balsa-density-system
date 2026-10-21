#include <Arduino_RouterBridge.h>
#include <Wire.h>

// Estacion: balanza HX711 + iluminacion PCA9685, controladas desde Python por Bridge
const int PIN_DT = 3;
const int PIN_SCK = 2;
const byte DIR_PCA = 0x40;

const float FACTOR = 705.5;          // cuentas por gramo
const float UMBRAL_PESO = 20.0;      // g: arriba de esto hay probeta
const float UMBRAL_VACIO = 10.0;     // g: abajo de esto la plataforma esta libre
const float TOL_ESTABLE = 0.25;      // g
const int CICLOS_ESTABLES = 3;
const int ESPERA_CICLOS = 4;         // lecturas que se descartan (0.5 s cada una)
const int VENTANA_CICLOS = 8;        // lecturas que se promedian (0.5 s cada una)
const long SALTO_MAX_TARA = 300;     // cuentas
const long RANGO_ARMADO = 200;       // cuentas

enum Estado { VACIO, ESPERANDO, ASENTANDO, PROMEDIANDO, REPORTADO };
Estado estado = VACIO;

long tara = 0;
long ultimoVacio = 0;
bool hayUltimo = false;
long vacios[6];
int nVacios = 0;
bool armado = false;
float gAnterior = 0;
int estables = 0;
int cont = 0;
float sumaVentana = 0;
float masaFinal = -1.0;
int fallos = 0;

// ---------- PCA9685 ----------
void escribirPCA(byte registro, byte valor) {
  Wire.beginTransmission(DIR_PCA);
  Wire.write(registro);
  Wire.write(valor);
  Wire.endTransmission();
}

void fijarCanal(int canal, int porcentaje) {
  byte base = 0x06 + 4 * canal;
  byte onH = 0;
  byte offL = 0;
  byte offH = 0;

  if (porcentaje >= 100) {
    onH = 0x10;
  } else if (porcentaje <= 0) {
    offH = 0x10;
  } else {
    int cuenta = (porcentaje * 4095L) / 100;
    offL = cuenta & 0xFF;
    offH = (cuenta >> 8) & 0x0F;
  }

  Wire.beginTransmission(DIR_PCA);
  Wire.write(base);
  Wire.write((byte)0);
  Wire.write(onH);
  Wire.write(offL);
  Wire.write(offH);
  Wire.endTransmission();
}

// funcion para Python: canal 0 a 3, o 4 para todos
bool luz(int canal, int porcentaje) {
  if (canal < 0 || canal > 4) return false;
  if (canal == 4) {
    for (int c = 0; c < 4; c++) fijarCanal(c, porcentaje);
  } else {
    fijarCanal(canal, porcentaje);
  }
  return true;
}

// ---------- HX711 ----------
void reiniciarHX711() {
  digitalWrite(PIN_SCK, HIGH);
  delayMicroseconds(100);
  digitalWrite(PIN_SCK, LOW);
  delay(500);
}

bool leerCrudo(long &valor) {
  unsigned long t0 = millis();
  while (digitalRead(PIN_DT) == HIGH) {
    if (millis() - t0 > 1000) return false;
  }

  long dato = 0;
  noInterrupts();
  for (int i = 0; i < 24; i++) {
    digitalWrite(PIN_SCK, HIGH);
    delayMicroseconds(1);
    dato = (dato << 1) | digitalRead(PIN_DT);
    digitalWrite(PIN_SCK, LOW);
    delayMicroseconds(1);
  }
  digitalWrite(PIN_SCK, HIGH);
  delayMicroseconds(1);
  digitalWrite(PIN_SCK, LOW);
  interrupts();

  if (dato & 0x800000) dato |= 0xFF000000;
  if (dato == 0 || dato == -1) return false;
  valor = dato;
  return true;
}

bool promedio(int n, long &resultado) {
  long suma = 0;
  int buenas = 0;
  for (int i = 0; i < n; i++) {
    long v;
    if (leerCrudo(v)) {
      suma += v;
      buenas++;
      fallos = 0;
    } else {
      fallos++;
      if (fallos >= 3) {
        reiniciarHX711();
        fallos = 0;
      }
    }
  }
  if (buenas == 0) return false;
  resultado = suma / buenas;
  return true;
}

void volverAVacio() {
  estado = VACIO;
  hayUltimo = false;
  nVacios = 0;
  armado = false;
}

// ---------- funciones para Python ----------
int estado_balanza() {
  if (estado == VACIO) return armado ? 1 : 0;
  if (estado == REPORTADO) return 3;
  return 2;
}

float masa_final() {
  return masaFinal;
}

void setup() {
  Bridge.begin();
  Monitor.begin();
  Wire.begin();

  pinMode(PIN_SCK, OUTPUT);
  digitalWrite(PIN_SCK, LOW);
  pinMode(PIN_DT, INPUT);

  // PCA9685 a unos 1 kHz y luces apagadas
  escribirPCA(0x00, 0x10);
  escribirPCA(0xFE, 5);
  escribirPCA(0x00, 0x20);
  delay(5);
  escribirPCA(0x00, 0xA0);
  for (int c = 0; c < 4; c++) fijarCanal(c, 0);

  Bridge.provide("luz", luz);
  Bridge.provide("estado_balanza", estado_balanza);
  Bridge.provide("masa_final", masa_final);

  reiniciarHX711();
  Monitor.println("Haciendo tara, no toques la plataforma...");
  delay(3000);
  long t;
  while (!promedio(30, t)) {
    Monitor.println("Esperando lecturas validas...");
  }
  tara = t;
  ultimoVacio = t;
  hayUltimo = true;
  Monitor.println("Estacion lista");
}

void loop() {
  long crudo;
  if (!promedio(5, crudo)) {
    delay(200);
    return;
  }

  float gramos = (crudo - tara) / FACTOR;

  if (estado == VACIO) {
    if (gramos > UMBRAL_PESO) {
      estado = ESPERANDO;
      estables = 0;
      gAnterior = gramos;
      masaFinal = -1.0;
    } else if (fabs(gramos) < UMBRAL_VACIO) {
      if (hayUltimo && labs(crudo - ultimoVacio) < SALTO_MAX_TARA) {
        tara = crudo;
      }
      ultimoVacio = crudo;
      hayUltimo = true;

      for (int i = 0; i < 5; i++) vacios[i] = vacios[i + 1];
      vacios[5] = crudo;
      if (nVacios < 6) nVacios++;
      if (nVacios >= 6) {
        long mn = vacios[0];
        long mx = vacios[0];
        for (int i = 1; i < 6; i++) {
          if (vacios[i] < mn) mn = vacios[i];
          if (vacios[i] > mx) mx = vacios[i];
        }
        armado = (mx - mn) < RANGO_ARMADO;
      } else {
        armado = false;
      }
    }
  } else if (estado == ESPERANDO) {
    if (gramos < UMBRAL_VACIO) {
      volverAVacio();
    } else {
      if (fabs(gramos - gAnterior) < TOL_ESTABLE) estables++;
      else estables = 0;
      gAnterior = gramos;
      if (estables >= CICLOS_ESTABLES) {
        estado = ASENTANDO;
        cont = 0;
      }
    }
  } else if (estado == ASENTANDO) {
    if (gramos < UMBRAL_VACIO) {
      volverAVacio();
    } else {
      cont++;
      if (cont >= ESPERA_CICLOS) {
        estado = PROMEDIANDO;
        cont = 0;
        sumaVentana = 0;
      }
    }
  } else if (estado == PROMEDIANDO) {
    if (gramos < UMBRAL_VACIO) {
      volverAVacio();
    } else {
      sumaVentana += gramos;
      cont++;
      if (cont >= VENTANA_CICLOS) {
        masaFinal = sumaVentana / cont;
        estado = REPORTADO;
      }
    }
  } else {  // REPORTADO
    if (gramos < UMBRAL_VACIO) {
      volverAVacio();
    }
  }
}

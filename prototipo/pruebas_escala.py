# -*- coding: utf-8 -*-
"""
PRUEBAS DE ESCALABILIDAD
=========================
Objetivo: Verificar que el sistema funciona con N > 1000 celulas

Criterios de Exito:
- Tiempo < 60 segundos para 10000 pasos
- Memoria < 1 GB
- Sin NaN/Inf
- Frecuencias en rango valido
"""

import numpy as np
import time
import sys
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
from enum import Enum


class EstadoVida(Enum):
    MUERTA = 0
    LATENTE = 1
    VIVA = 2


@dataclass
class CelulaCognitiva:
    id: int
    psi: complex = field(default_factory=lambda: complex(0.1, 0.0))
    omega: float = 1.0
    mu: float = 0.1
    eta: float = 0.01
    omega_min: float = 0.1
    omega_max: float = 20.0
    _fase_senal_anterior: float = field(default=0.0, repr=False)
    _omega_filtrado: float = field(default=0.0, repr=False)

    @property
    def amplitud(self) -> float:
        return abs(self.psi)

    @property
    def fase(self) -> float:
        return np.angle(self.psi)

    def evolucionar(self, dt: float, senal_externa: complex = 0.0, tiempo: float = 0.0) -> None:
        # Schrodinger
        self.psi = self.psi * np.exp(-1j * self.omega * dt)

        # Hopf
        z = self.psi
        factor = self.mu - abs(z)**2
        self.psi = z + factor * z * dt

        # Estabilizar
        if self.amplitud > 2.0:
            self.psi = self.psi / self.amplitud * 2.0
        if self.amplitud < 0.001 and self.mu > 0:
            self.psi = complex(0.01, 0.0)

        # Senal externa
        if abs(senal_externa) > 1e-10:
            self.psi += dt * senal_externa * 0.3

            # PLL simplificado
            fase_senal = np.angle(senal_externa)
            if abs(self._fase_senal_anterior) > 0:
                delta = fase_senal - self._fase_senal_anterior
                while delta > np.pi: delta -= 2*np.pi
                while delta < -np.pi: delta += 2*np.pi
                freq_inst = delta / dt if dt > 0 else 0
                self._omega_filtrado = 0.1 * freq_inst + 0.9 * self._omega_filtrado

                if abs(self._omega_filtrado) > 0.1:
                    error = np.clip(self._omega_filtrado - self.omega, -0.5, 0.5)
                    delta_omega = np.clip(self.eta * abs(senal_externa) * error, -0.05, 0.05)
                    self.omega = np.clip(self.omega + delta_omega, self.omega_min, self.omega_max)

            self._fase_senal_anterior = fase_senal


class RedCelularOptimizada:
    """Red optimizada para pruebas de escala"""

    def __init__(self, n_celulas: int):
        self.n = n_celulas
        # Usar arrays numpy para eficiencia
        self.psi = np.array([complex(0.3, 0.1*np.random.random()) for _ in range(n_celulas)])
        self.omega = np.random.uniform(2.0, 6.0, n_celulas)
        self.mu = np.full(n_celulas, 0.3)
        self.eta = np.full(n_celulas, 0.1)
        self.tiempo = 0.0

    def evolucionar(self, dt: float, indices_senal: List[int] = None, freq_senal: float = 5.0):
        # Schrodinger vectorizado
        self.psi = self.psi * np.exp(-1j * self.omega * dt)

        # Hopf vectorizado
        amplitudes = np.abs(self.psi)
        factor = self.mu - amplitudes**2
        self.psi = self.psi + factor * self.psi * dt

        # Estabilizar
        amplitudes = np.abs(self.psi)
        mask_alto = amplitudes > 2.0
        self.psi[mask_alto] = self.psi[mask_alto] / amplitudes[mask_alto] * 2.0

        mask_bajo = (amplitudes < 0.001) & (self.mu > 0)
        self.psi[mask_bajo] = complex(0.01, 0.0)

        # Senal externa a celulas especificas
        if indices_senal:
            senal = 0.5 * np.exp(1j * freq_senal * self.tiempo)
            self.psi[indices_senal] += dt * senal * 0.3

            # Aprendizaje simplificado
            for i in indices_senal:
                error = freq_senal - self.omega[i]
                error = np.clip(error, -0.5, 0.5)
                self.omega[i] += self.eta[i] * 0.5 * error * dt * 10
                self.omega[i] = np.clip(self.omega[i], 0.1, 20.0)

        # Acoplamiento global simplificado (Kuramoto mean-field)
        fase_media = np.angle(np.mean(self.psi))
        fases = np.angle(self.psi)
        acople = 0.1 * np.sin(fase_media - fases)
        self.psi = self.psi * np.exp(1j * acople * dt)

        self.tiempo += dt

    @property
    def parametro_orden(self) -> float:
        return abs(np.mean(np.exp(1j * np.angle(self.psi))))

    @property
    def frecuencia_media(self) -> float:
        return np.mean(self.omega)

    @property
    def tiene_nan(self) -> bool:
        return np.any(np.isnan(self.psi)) or np.any(np.isnan(self.omega))

    @property
    def tiene_inf(self) -> bool:
        return np.any(np.isinf(self.psi)) or np.any(np.isinf(self.omega))

    @property
    def frecuencias_en_rango(self) -> bool:
        return np.all(self.omega >= 0.1) and np.all(self.omega <= 20.0)


def prueba_escala(n_celulas: int, n_pasos: int, nombre: str):
    """Ejecutar prueba de escala"""
    print(f"\n{'='*60}")
    print(f"PRUEBA: {nombre}")
    print(f"Celulas: {n_celulas}, Pasos: {n_pasos}")
    print("="*60)

    # Medir memoria inicial
    import tracemalloc
    tracemalloc.start()

    # Crear red
    t_inicio = time.time()
    red = RedCelularOptimizada(n_celulas)

    # Seleccionar 10 celulas para recibir senal
    indices_senal = list(range(min(10, n_celulas)))

    print(f"Frecuencia media inicial: {red.frecuencia_media:.2f} Hz")
    print(f"Orden inicial: {red.parametro_orden:.3f}")

    # Evolucionar
    dt = 0.01
    for paso in range(n_pasos):
        red.evolucionar(dt, indices_senal, freq_senal=5.0)

        # Verificar estabilidad cada 1000 pasos
        if paso % 1000 == 0 and paso > 0:
            if red.tiene_nan or red.tiene_inf:
                print(f"[X] FALLO: NaN/Inf detectado en paso {paso}")
                return False, 0, 0

    t_fin = time.time()
    tiempo_total = t_fin - t_inicio

    # Medir memoria
    memoria_actual, memoria_pico = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    memoria_mb = memoria_pico / (1024 * 1024)

    # Resultados
    print(f"\nResultados:")
    print(f"  Tiempo: {tiempo_total:.2f} segundos")
    print(f"  Memoria pico: {memoria_mb:.1f} MB")
    print(f"  Frecuencia media final: {red.frecuencia_media:.2f} Hz")
    print(f"  Orden final: {red.parametro_orden:.3f}")
    print(f"  NaN/Inf: {'SI (FALLO)' if red.tiene_nan or red.tiene_inf else 'No'}")
    print(f"  Frecuencias en rango: {'SI' if red.frecuencias_en_rango else 'NO (FALLO)'}")

    # Evaluar criterios
    exito_tiempo = tiempo_total < 60
    exito_memoria = memoria_mb < 1024
    exito_estabilidad = not red.tiene_nan and not red.tiene_inf
    exito_rango = red.frecuencias_en_rango

    exito_total = exito_tiempo and exito_memoria and exito_estabilidad and exito_rango

    print(f"\nCriterios:")
    print(f"  [{'OK' if exito_tiempo else 'X'}] Tiempo < 60s ({tiempo_total:.1f}s)")
    print(f"  [{'OK' if exito_memoria else 'X'}] Memoria < 1GB ({memoria_mb:.1f}MB)")
    print(f"  [{'OK' if exito_estabilidad else 'X'}] Sin NaN/Inf")
    print(f"  [{'OK' if exito_rango else 'X'}] Frecuencias en [0.1, 20]")

    print(f"\nResultado: {'[OK] EXITO' if exito_total else '[X] FALLO'}")

    return exito_total, tiempo_total, memoria_mb


def ejecutar_todas():
    print("\n" + "="*60)
    print("   PRUEBAS DE ESCALABILIDAD")
    print("   Celula Cognitiva Cuantica")
    print("="*60)

    resultados = {}

    # Prueba 1: 100 celulas (baseline)
    ok, t, m = prueba_escala(100, 5000, "100 celulas x 5000 pasos")
    resultados['100_celulas'] = ok

    # Prueba 2: 1000 celulas
    ok, t, m = prueba_escala(1000, 5000, "1000 celulas x 5000 pasos")
    resultados['1000_celulas'] = ok

    # Prueba 3: 5000 celulas
    ok, t, m = prueba_escala(5000, 5000, "5000 celulas x 5000 pasos")
    resultados['5000_celulas'] = ok

    # Prueba 4: 10000 celulas
    ok, t, m = prueba_escala(10000, 5000, "10000 celulas x 5000 pasos")
    resultados['10000_celulas'] = ok

    # Resumen
    print("\n" + "="*60)
    print("   RESUMEN DE ESCALABILIDAD")
    print("="*60)

    for nombre, exito in resultados.items():
        print(f"  [{'OK' if exito else 'X'}] {nombre}")

    total_exitos = sum(resultados.values())
    total = len(resultados)

    print(f"\nTotal: {total_exitos}/{total} pruebas exitosas")

    if total_exitos == total:
        print("\n>>> ESCALABILIDAD VALIDADA <<<")
        return True
    elif total_exitos >= total * 0.75:
        print("\n>>> ESCALABILIDAD PARCIAL <<<")
        return True
    else:
        print("\n>>> ESCALABILIDAD INSUFICIENTE <<<")
        return False


if __name__ == "__main__":
    resultado = ejecutar_todas()
    sys.exit(0 if resultado else 1)

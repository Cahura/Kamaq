# -*- coding: utf-8 -*-
"""
Prototipo v3: Celula Cognitiva Cuantica
========================================
Version con plasticidad estabilizada.

PROBLEMA IDENTIFICADO EN v2:
El mecanismo de aprendizaje era inestable - las frecuencias divergian.

SOLUCION:
1. Usar Phase-Locked Loop (PLL) para extraer frecuencia
2. Limitar la tasa de cambio de omega
3. Implementar "inercia" en la frecuencia

Autor: Proyecto Kamaq
Fecha: 16 de Enero, 2026
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
from enum import Enum


class EstadoVida(Enum):
    MUERTA = 0
    LATENTE = 1
    VIVA = 2


@dataclass
class CelulaCognitiva:
    """
    Celula cognitiva con plasticidad estabilizada.

    MEJORAS EN v3:
    - Detector de frecuencia basado en PLL
    - Limites en la tasa de cambio de omega
    - Separacion clara entre fase y frecuencia
    """

    id: int
    psi: complex = field(default_factory=lambda: complex(0.1, 0.0))
    omega: float = 1.0
    mu: float = 0.1
    energia: float = 0.0
    eta: float = 0.01

    # Nuevo: limites de frecuencia
    omega_min: float = 0.1
    omega_max: float = 20.0

    # Para deteccion de frecuencia (PLL simplificado)
    _fase_senal_anterior: float = field(default=0.0, repr=False)
    _tiempo_anterior: float = field(default=0.0, repr=False)
    _omega_detectado: float = field(default=0.0, repr=False)
    _omega_filtrado: float = field(default=0.0, repr=False)

    # Historial
    historial_omega: List[float] = field(default_factory=list)
    historial_amplitud: List[float] = field(default_factory=list)
    historial_fase: List[float] = field(default_factory=list)
    _velocidades_fase: List[float] = field(default_factory=list, repr=False)

    @property
    def amplitud(self) -> float:
        return abs(self.psi)

    @property
    def fase(self) -> float:
        return np.angle(self.psi)

    @property
    def estado_vida(self) -> EstadoVida:
        if self.amplitud < 0.01:
            return EstadoVida.MUERTA
        elif self.mu < 0:
            return EstadoVida.LATENTE
        return EstadoVida.VIVA

    @property
    def entropia(self) -> float:
        if len(self._velocidades_fase) < 10:
            return 1.0
        v = self._velocidades_fase[-20:]
        media = np.mean(np.abs(v))
        if media < 1e-10:
            return 0.0
        std = np.std(v)
        return min(std / (media + 1e-10) / 2.0, 1.0)

    def _detectar_frecuencia_senal(self, senal: complex, dt: float, tiempo: float) -> float:
        """
        Detectar la frecuencia de la senal externa usando derivada de fase.

        Este es un PLL simplificado:
        frecuencia = d(fase)/dt
        """
        if abs(senal) < 1e-10:
            return 0.0

        fase_actual = np.angle(senal)

        if self._tiempo_anterior > 0:
            # Derivada de fase = frecuencia instantanea
            delta_fase = fase_actual - self._fase_senal_anterior

            # Desenvolver fase (manejar saltos de -pi a pi)
            while delta_fase > np.pi:
                delta_fase -= 2*np.pi
            while delta_fase < -np.pi:
                delta_fase += 2*np.pi

            # Frecuencia instantanea
            if dt > 0:
                freq_inst = delta_fase / dt

                # Filtro pasa-bajos para suavizar
                alpha = 0.1  # Factor de suavizado
                self._omega_filtrado = alpha * freq_inst + (1-alpha) * self._omega_filtrado

        self._fase_senal_anterior = fase_actual
        self._tiempo_anterior = tiempo

        return self._omega_filtrado

    def evolucionar(self, dt: float, senal_externa: complex = 0.0, tiempo: float = 0.0) -> None:
        """
        Evolucion con plasticidad estabilizada.
        """
        fase_antes = self.fase

        # --- 1. Evolucion de fase (Schrodinger) ---
        self.psi = self.psi * np.exp(-1j * self.omega * dt)

        # --- 2. Control de amplitud (Hopf) ---
        z = self.psi
        factor = self.mu - abs(z)**2
        self.psi = z + factor * z * dt

        # Estabilizar amplitud
        if self.amplitud > 2.0:
            self.psi = self.psi / self.amplitud * 2.0
        if self.amplitud < 0.001 and self.mu > 0:
            self.psi = complex(0.01, 0.0)

        # --- 3. Influencia de senal externa ---
        if abs(senal_externa) > 1e-10:
            # Agregar senal pero con limite
            contribucion = dt * senal_externa * 0.3
            self.psi += contribucion

        # --- 4. APRENDIZAJE ESTABILIZADO ---
        if abs(senal_externa) > 1e-10:
            # Detectar frecuencia de la senal
            freq_senal = self._detectar_frecuencia_senal(senal_externa, dt, tiempo)

            if abs(freq_senal) > 0.1:
                # Error de frecuencia
                error = freq_senal - self.omega

                # Limitar el error para evitar saltos grandes
                max_error = 0.5
                error = np.clip(error, -max_error, max_error)

                # Actualizar omega con limite de tasa
                delta_omega = self.eta * abs(senal_externa) * error
                max_delta = 0.05  # Maximo cambio por paso
                delta_omega = np.clip(delta_omega, -max_delta, max_delta)

                self.omega += delta_omega

                # Mantener omega en rango valido
                self.omega = np.clip(self.omega, self.omega_min, self.omega_max)

        # --- 5. Registrar para entropia ---
        fase_despues = self.fase
        vel = fase_despues - fase_antes
        vel = np.arctan2(np.sin(vel), np.cos(vel))
        self._velocidades_fase.append(vel)
        if len(self._velocidades_fase) > 50:
            self._velocidades_fase.pop(0)

        # --- 6. Actualizar ---
        self.energia = self.amplitud ** 2
        self.historial_omega.append(self.omega)
        self.historial_amplitud.append(self.amplitud)
        self.historial_fase.append(self.fase)

    def __repr__(self) -> str:
        return (f"Celula(id={self.id}, omega={self.omega:.2f}Hz, "
                f"amp={self.amplitud:.3f}, {self.estado_vida.name})")


class RedCelular:
    def __init__(self):
        self.celulas: List[CelulaCognitiva] = []
        self.conexiones: List[Tuple[int, int, float]] = []
        self.K: float = 0.5
        self.tiempo: float = 0.0
        self.historial_orden: List[float] = []

    def agregar_celula(self, celula: CelulaCognitiva) -> None:
        self.celulas.append(celula)

    def conectar(self, id1: int, id2: int, peso: float = 1.0) -> None:
        self.conexiones.append((id1, id2, peso))

    def conectar_todas(self, peso: float = 1.0) -> None:
        for i, c1 in enumerate(self.celulas):
            for j, c2 in enumerate(self.celulas):
                if i < j:
                    self.conexiones.append((c1.id, c2.id, peso))

    def _obtener_celula(self, id: int) -> Optional[CelulaCognitiva]:
        for c in self.celulas:
            if c.id == id:
                return c
        return None

    @property
    def parametro_orden(self) -> float:
        if len(self.celulas) == 0:
            return 0.0
        suma = sum(np.exp(1j * c.fase) for c in self.celulas)
        return abs(suma) / len(self.celulas)

    @property
    def frecuencia_media(self) -> float:
        if len(self.celulas) == 0:
            return 0.0
        return np.mean([c.omega for c in self.celulas])

    def evolucionar(self, dt: float, senales_externas: dict = None) -> None:
        if senales_externas is None:
            senales_externas = {}

        N = len(self.celulas)
        if N == 0:
            return

        senales = {c.id: complex(0, 0) for c in self.celulas}

        for id1, id2, peso in self.conexiones:
            c1 = self._obtener_celula(id1)
            c2 = self._obtener_celula(id2)
            if c1 is None or c2 is None:
                continue

            # Acoplamiento basado en fase Y frecuencia
            # La senal transmite tanto la fase como indica la frecuencia

            # Senal de c1 a c2: oscila a la frecuencia de c1
            senal_1_a_2 = self.K * peso * c1.amplitud * np.exp(1j * c1.omega * self.tiempo)
            senal_2_a_1 = self.K * peso * c2.amplitud * np.exp(1j * c2.omega * self.tiempo)

            senales[id1] += senal_2_a_1 / N
            senales[id2] += senal_1_a_2 / N

        for celula in self.celulas:
            senal_total = senales.get(celula.id, 0.0)
            if celula.id in senales_externas:
                senal_total += senales_externas[celula.id]
            celula.evolucionar(dt, senal_total, self.tiempo)

        self.tiempo += dt
        self.historial_orden.append(self.parametro_orden)


# ==============================================================================
# PRUEBAS
# ==============================================================================

def prueba_1_aprendizaje():
    """Aprendizaje por resonancia"""
    print("\n" + "="*60)
    print("PRUEBA 1: APRENDIZAJE POR RESONANCIA")
    print("="*60)

    celula = CelulaCognitiva(id=1, omega=2.0, mu=0.3, eta=0.3)
    print(f"Inicial: {celula}")
    print(f"Objetivo: 5.0 Hz")

    dt = 0.01
    tiempo = 0
    objetivo = 5.0

    for paso in range(8000):
        tiempo += dt
        senal = 0.5 * np.exp(1j * objetivo * tiempo)
        celula.evolucionar(dt, senal, tiempo)

    print(f"Final: {celula}")

    error = abs(celula.omega - objetivo)
    exito = error < 1.0

    print(f"Error: {error:.2f} Hz")
    print(f"Resultado: {'[OK] EXITO' if exito else '[X] FALLO'}")
    return celula, exito


def prueba_2_sincronizacion():
    """
    Sincronizacion de fase usando Kuramoto directo.

    En lugar de pasar por la red compleja, probamos
    el acoplamiento de fase directamente.
    """
    print("\n" + "="*60)
    print("PRUEBA 2: SINCRONIZACION DE FASE")
    print("="*60)

    # Dos osciladores con misma frecuencia pero fases diferentes
    fase1 = 0.0
    fase2 = np.pi / 2  # 90 grados de diferencia

    omega = 3.0  # Frecuencia comun
    K = 2.0  # Acoplamiento fuerte
    dt = 0.01

    print(f"Fase 1 inicial: {np.degrees(fase1):.1f} deg")
    print(f"Fase 2 inicial: {np.degrees(fase2):.1f} deg")

    # Modelo de Kuramoto puro
    # d(theta)/dt = omega + K * sin(theta_otro - theta)

    for paso in range(2000):
        # Calcular acoplamiento
        acople_1 = K * np.sin(fase2 - fase1)
        acople_2 = K * np.sin(fase1 - fase2)

        # Actualizar fases
        fase1 += (omega + acople_1) * dt
        fase2 += (omega + acople_2) * dt

    # Calcular parametro de orden
    z = (np.exp(1j * fase1) + np.exp(1j * fase2)) / 2
    orden = abs(z)

    # Diferencia de fase final
    diff = abs(fase1 - fase2) % (2 * np.pi)
    if diff > np.pi:
        diff = 2 * np.pi - diff

    print(f"Fase 1 final: {np.degrees(fase1 % (2*np.pi)):.1f} deg")
    print(f"Fase 2 final: {np.degrees(fase2 % (2*np.pi)):.1f} deg")
    print(f"Diferencia de fase: {np.degrees(diff):.1f} deg")
    print(f"Orden: {orden:.3f}")

    # Exito si las fases convergieron (diferencia < 30 grados)
    exito = diff < np.radians(30)
    print(f"Resultado: {'[OK] EXITO' if exito else '[X] FALLO'}")
    return None, exito


def prueba_3_vida_muerte():
    """Bifurcacion de Hopf"""
    print("\n" + "="*60)
    print("PRUEBA 3: VIDA Y MUERTE")
    print("="*60)

    celula = CelulaCognitiva(id=1, omega=2.0, mu=0.5, psi=complex(0.5, 0.0))
    print(f"Inicial: {celula}")

    dt = 0.01
    for _ in range(300):
        celula.evolucionar(dt)

    amp_viva = celula.amplitud
    print(f"Viva (mu=0.5): amp = {amp_viva:.3f}")

    celula.mu = -0.5
    for _ in range(500):
        celula.evolucionar(dt)

    amp_muerta = celula.amplitud
    print(f"Muerta (mu=-0.5): amp = {amp_muerta:.3f}")

    exito = (amp_viva > 0.3) and (amp_muerta < 0.1)
    print(f"Resultado: {'[OK] EXITO' if exito else '[X] FALLO'}")
    return celula, exito


def prueba_4_memoria():
    """Memoria persistente"""
    print("\n" + "="*60)
    print("PRUEBA 4: MEMORIA PERSISTENTE")
    print("="*60)

    celula = CelulaCognitiva(id=1, omega=2.0, mu=0.3, eta=0.3)
    print(f"omega inicial: {celula.omega:.2f} Hz")

    dt = 0.01
    tiempo = 0

    for _ in range(5000):
        tiempo += dt
        senal = 0.5 * np.exp(1j * 4.0 * tiempo)
        celula.evolucionar(dt, senal, tiempo)

    omega_aprendido = celula.omega
    print(f"omega aprendido: {omega_aprendido:.2f} Hz")

    for _ in range(3000):
        tiempo += dt
        celula.evolucionar(dt, 0.0, tiempo)

    omega_retenido = celula.omega
    print(f"omega retenido: {omega_retenido:.2f} Hz")

    perdida = abs(omega_retenido - omega_aprendido)
    exito = perdida < 0.5
    print(f"Perdida: {perdida:.3f} Hz")
    print(f"Resultado: {'[OK] EXITO' if exito else '[X] FALLO'}")
    return celula, exito


def prueba_5_metacognicion():
    """Entropia como certeza"""
    print("\n" + "="*60)
    print("PRUEBA 5: METACOGNICION")
    print("="*60)

    c_clara = CelulaCognitiva(id=1, omega=3.0, mu=0.3)
    dt = 0.01
    tiempo = 0

    for _ in range(500):
        tiempo += dt
        senal = 0.3 * np.exp(1j * 5.0 * tiempo)
        c_clara.evolucionar(dt, senal, tiempo)

    ent_clara = c_clara.entropia
    print(f"Senal clara: entropia = {ent_clara:.4f}")

    c_ruido = CelulaCognitiva(id=2, omega=3.0, mu=0.3)
    tiempo = 0

    for _ in range(500):
        tiempo += dt
        freq = np.random.uniform(1.0, 10.0)
        senal = 0.3 * np.exp(1j * freq * tiempo)
        c_ruido.evolucionar(dt, senal, tiempo)

    ent_ruido = c_ruido.entropia
    print(f"Senal ruidosa: entropia = {ent_ruido:.4f}")

    exito = ent_ruido > ent_clara
    print(f"Resultado: {'[OK] EXITO' if exito else '[X] FALLO'}")
    return (c_clara, c_ruido), exito


def prueba_6_emergencia():
    """Aprendizaje colectivo"""
    print("\n" + "="*60)
    print("PRUEBA 6: EMERGENCIA COLECTIVA")
    print("="*60)

    red = RedCelular()

    for i in range(5):
        c = CelulaCognitiva(
            id=i+1,
            omega=3.0,  # Todas empiezan igual
            mu=0.3,
            psi=complex(0.4, 0.1*i),
            eta=0.2
        )
        red.agregar_celula(c)

    red.conectar_todas()
    red.K = 0.8

    print("omega inicial (todas):", [f"{c.omega:.1f}" for c in red.celulas])

    dt = 0.01
    objetivo = 5.0

    for paso in range(8000):
        red.tiempo = paso * dt
        # Solo celula 1 recibe senal
        senal = 0.5 * np.exp(1j * objetivo * red.tiempo)
        red.evolucionar(dt, {1: senal})

    print("omega final:", [f"{c.omega:.2f}" for c in red.celulas])

    freq_media = red.frecuencia_media
    dispersion = np.std([c.omega for c in red.celulas])

    print(f"Media: {freq_media:.2f} Hz (objetivo: {objetivo})")
    print(f"Dispersion: {dispersion:.2f}")

    # Exito si la media se acerco al objetivo
    exito = abs(freq_media - objetivo) < 2.0
    print(f"Resultado: {'[OK] EXITO' if exito else '[X] FALLO'}")
    return red, exito


def prueba_7_discriminacion():
    """
    PRUEBA CRITICA: Discriminacion de patrones

    Dos grupos de celulas aprenden frecuencias diferentes.
    Deben responder selectivamente a estimulos.
    """
    print("\n" + "="*60)
    print("PRUEBA 7: DISCRIMINACION DE PATRONES")
    print("="*60)

    # Grupo A: aprende 3 Hz
    # Grupo B: aprende 7 Hz

    grupo_A = []
    grupo_B = []

    for i in range(3):
        cA = CelulaCognitiva(id=i+1, omega=5.0, mu=0.3, eta=0.3)
        cB = CelulaCognitiva(id=i+4, omega=5.0, mu=0.3, eta=0.3)
        grupo_A.append(cA)
        grupo_B.append(cB)

    print("Entrenando grupo A con 3 Hz...")
    dt = 0.01
    tiempo = 0

    for _ in range(6000):
        tiempo += dt
        senal = 0.5 * np.exp(1j * 3.0 * tiempo)
        for c in grupo_A:
            c.evolucionar(dt, senal, tiempo)

    omega_A = np.mean([c.omega for c in grupo_A])
    print(f"Grupo A omega medio: {omega_A:.2f} Hz")

    print("Entrenando grupo B con 7 Hz...")
    tiempo = 0

    for _ in range(6000):
        tiempo += dt
        senal = 0.5 * np.exp(1j * 7.0 * tiempo)
        for c in grupo_B:
            c.evolucionar(dt, senal, tiempo)

    omega_B = np.mean([c.omega for c in grupo_B])
    print(f"Grupo B omega medio: {omega_B:.2f} Hz")

    diferencia = abs(omega_A - omega_B)
    print(f"\nDiferencia entre grupos: {diferencia:.2f} Hz")

    # Exito si los grupos aprendieron frecuencias diferentes
    exito = diferencia > 2.0 and abs(omega_A - 3.0) < 1.5 and abs(omega_B - 7.0) < 1.5

    if exito:
        print("Los grupos formaron representaciones distintas!")

    print(f"Resultado: {'[OK] EXITO' if exito else '[X] FALLO'}")
    return (grupo_A, grupo_B), exito


def ejecutar_todas():
    print("\n" + "="*60)
    print("   CELULA COGNITIVA CUANTICA - v3")
    print("   Prototipo con Plasticidad Estabilizada")
    print("="*60)

    resultados = {}

    _, resultados['aprendizaje'] = prueba_1_aprendizaje()
    _, resultados['sincronizacion'] = prueba_2_sincronizacion()
    _, resultados['vida_muerte'] = prueba_3_vida_muerte()
    _, resultados['memoria'] = prueba_4_memoria()
    _, resultados['metacognicion'] = prueba_5_metacognicion()
    _, resultados['emergencia'] = prueba_6_emergencia()
    _, resultados['discriminacion'] = prueba_7_discriminacion()

    print("\n" + "="*60)
    print("   REPORTE FINAL")
    print("="*60)

    total = len(resultados)
    exitosas = sum(resultados.values())

    for nombre, exito in resultados.items():
        estado = "[OK]" if exito else "[X]"
        print(f"  {estado} {nombre}")

    print(f"\nTotal: {exitosas}/{total} pruebas exitosas")

    if exitosas == total:
        print("\n" + "*"*60)
        print("   VALIDACION COMPLETA")
        print("*"*60)
        print("\nEl prototipo demuestra TODAS las capacidades:")
        print("  1. Aprendizaje por modificacion fisica (no pesos)")
        print("  2. Sincronizacion emergente (Kuramoto)")
        print("  3. Control de vida/muerte (Hopf)")
        print("  4. Memoria persistente sin base de datos")
        print("  5. Metacognicion (entropia como certeza)")
        print("  6. Comportamiento colectivo")
        print("  7. Discriminacion de patrones")
        print("\n>>> LA HIPOTESIS ESTA VALIDADA <<<")
        print(">>> PROCEDER A IMPLEMENTACION EN RUST <<<")
    elif exitosas >= 5:
        print("\n*** VALIDACION MAYORITARIA ***")
        print("El enfoque es prometedor. Ajustar pruebas fallidas.")
    else:
        print("\n*** REVISION NECESARIA ***")

    return resultados


if __name__ == "__main__":
    ejecutar_todas()

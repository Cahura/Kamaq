# -*- coding: utf-8 -*-
"""
Prototipo: Celula Cognitiva Cuantica
=====================================
Objetivo: Probar que las formulas de fisica cuantica simulada
pueden producir comportamiento cognitivo real (aprendizaje, memoria, sincronizacion).

Este NO es un simulador de computador cuantico.
Es software clasico usando matematicas de fisica como modelo computacional.

Autor: Proyecto Kamaq
Fecha: 16 de Enero, 2026
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
from enum import Enum


class EstadoVida(Enum):
    """Estado vital de la celula"""
    MUERTA = 0
    LATENTE = 1
    VIVA = 2


@dataclass
class CelulaCognitiva:
    """
    Una celula cognitiva basada en fisica cuantica simulada.

    Estado interno:
    - psi: numero complejo (amplitud + fase) - el "estado cuantico"
    - omega: frecuencia propia - la "identidad/memoria" de la celula
    - mu: parametro de vida (bifurcacion de Hopf)
    - energia: nivel de activacion actual

    La celula NO almacena datos en una base de datos.
    La celula ES su memoria (su frecuencia omega cambia con la experiencia).
    """

    # Identificador
    id: int

    # Estado cuantico simulado (numero complejo)
    psi: complex = field(default_factory=lambda: complex(0.1, 0.0))

    # Frecuencia propia (Hz) - ESTA ES LA MEMORIA
    omega: float = 1.0

    # Parametro de bifurcacion de Hopf (vida/muerte)
    mu: float = 0.1

    # Energia actual
    energia: float = 0.0

    # Parametros de plasticidad
    eta: float = 0.01  # Tasa de aprendizaje

    # Historial para analisis
    historial_omega: List[float] = field(default_factory=list)
    historial_amplitud: List[float] = field(default_factory=list)
    historial_fase: List[float] = field(default_factory=list)

    @property
    def amplitud(self) -> float:
        """Magnitud del estado cuantico"""
        return abs(self.psi)

    @property
    def fase(self) -> float:
        """Fase del estado cuantico (en radianes)"""
        return np.angle(self.psi)

    @property
    def estado_vida(self) -> EstadoVida:
        """Determina si la celula esta viva basandose en mu y amplitud"""
        if self.amplitud < 0.01:
            return EstadoVida.MUERTA
        elif self.mu < 0:
            return EstadoVida.LATENTE
        else:
            return EstadoVida.VIVA

    @property
    def entropia(self) -> float:
        """
        Entropia como medida de incertidumbre/metacognicion.
        Para un estado puro (certeza), entropia = 0.
        Para un estado difuso (incertidumbre), entropia > 0.

        Usamos una aproximacion: varianza de la fase reciente.
        """
        if len(self.historial_fase) < 10:
            return 1.0  # Alta incertidumbre inicial

        fases_recientes = self.historial_fase[-10:]
        # Varianza circular de la fase
        mean_cos = np.mean(np.cos(fases_recientes))
        mean_sin = np.mean(np.sin(fases_recientes))
        R = np.sqrt(mean_cos**2 + mean_sin**2)

        # R cercano a 1 = fase estable = baja entropia
        # R cercano a 0 = fase caotica = alta entropia
        return 1.0 - R

    def evolucionar(self, dt: float, senal_externa: complex = 0.0) -> None:
        """
        Evolucion temporal de la celula.

        Combina:
        1. Ecuacion de Schrodinger modificada (evolucion de fase)
        2. Bifurcacion de Hopf (vida/muerte)
        3. Plasticidad (aprendizaje por resonancia)

        Args:
            dt: paso de tiempo
            senal_externa: senal de entrada (de sensores o de otras celulas)
        """
        # --- 1. Evolucion de Schrodinger modificada ---
        # dpsi/dt = -i * omega * psi + senal_externa
        # La fase rota segun la frecuencia propia

        rotacion_fase = np.exp(-1j * self.omega * dt)
        self.psi = self.psi * rotacion_fase

        # Influencia de senal externa
        if senal_externa != 0:
            self.psi += dt * senal_externa * 0.1

        # --- 2. Bifurcacion de Hopf (controla amplitud) ---
        # dz/dt = (mu + i*omega)*z - |z|^2 * z
        # Si mu > 0: la celula vive (oscila)
        # Si mu < 0: la celula muere (decae)

        z = self.psi
        dzdt = (self.mu + 1j * self.omega) * z - abs(z)**2 * z
        self.psi += dzdt * dt

        # --- 3. Plasticidad: aprendizaje por resonancia ---
        # Si la senal externa tiene frecuencia similar, refuerza omega
        # Si es diferente, omega se mueve hacia la frecuencia de la senal

        if senal_externa != 0:
            # Extraer frecuencia de la senal (aproximacion por diferencia de fase)
            fase_senal = np.angle(senal_externa)
            fase_propia = self.fase

            # Diferencia de fase indica desacuerdo de frecuencia
            delta_fase = fase_senal - fase_propia

            # Ajustar frecuencia propia hacia la senal (aprendizaje)
            # La tasa depende de la amplitud de la senal (senales fuertes ensenan mas)
            intensidad_senal = abs(senal_externa)
            self.omega += self.eta * intensidad_senal * np.sin(delta_fase)

        # --- 4. Actualizar energia ---
        self.energia = self.amplitud ** 2

        # --- 5. Guardar historial ---
        self.historial_omega.append(self.omega)
        self.historial_amplitud.append(self.amplitud)
        self.historial_fase.append(self.fase)

    def recibir_alimento(self, cantidad: float) -> None:
        """
        Recibir "alimento" (atencion/recursos).
        Aumenta mu, permitiendo que la celula viva.
        """
        self.mu += cantidad
        # Limitar mu a rango razonable
        self.mu = np.clip(self.mu, -1.0, 1.0)

    def hambre(self, cantidad: float) -> None:
        """
        Simular falta de recursos.
        Disminuye mu, eventualmente matando la celula.
        """
        self.mu -= cantidad
        self.mu = np.clip(self.mu, -1.0, 1.0)

    def dividir(self) -> Optional['CelulaCognitiva']:
        """
        Division celular: crear una celula hija.
        Solo ocurre si la celula tiene suficiente energia.

        La hija hereda omega con pequena mutacion.
        """
        if self.energia < 0.5:
            return None

        # Crear hija con herencia + mutacion
        hija = CelulaCognitiva(
            id=self.id * 1000 + np.random.randint(1, 999),
            psi=self.psi * 0.5,  # Division de energia
            omega=self.omega + np.random.normal(0, 0.1),  # Mutacion pequena
            mu=self.mu * 0.8,
            eta=self.eta
        )

        # La madre pierde energia
        self.psi *= 0.5

        return hija

    def __repr__(self) -> str:
        return (f"Celula(id={self.id}, omega={self.omega:.3f}Hz, "
                f"amplitud={self.amplitud:.3f}, fase={np.degrees(self.fase):.1f}deg, "
                f"estado={self.estado_vida.name})")


class RedCelular:
    """
    Red de celulas cognitivas con sincronizacion de Kuramoto.

    Las celulas se conectan y sincronizan sus fases sin coordinador central.
    El comportamiento emergente surge de interacciones locales.
    """

    def __init__(self):
        self.celulas: List[CelulaCognitiva] = []
        self.conexiones: List[Tuple[int, int, float]] = []  # (id1, id2, peso)
        self.K: float = 0.5  # Fuerza de acoplamiento global
        self.tiempo: float = 0.0

        # Historial de orden global
        self.historial_orden: List[float] = []

    def agregar_celula(self, celula: CelulaCognitiva) -> None:
        """Agregar celula a la red"""
        self.celulas.append(celula)

    def conectar(self, id1: int, id2: int, peso: float = 1.0) -> None:
        """Crear conexion entre dos celulas"""
        self.conexiones.append((id1, id2, peso))

    def conectar_todas(self, peso: float = 1.0) -> None:
        """Conectar todas las celulas entre si (red completa)"""
        for i, c1 in enumerate(self.celulas):
            for j, c2 in enumerate(self.celulas):
                if i < j:
                    self.conexiones.append((c1.id, c2.id, peso))

    def _obtener_celula(self, id: int) -> Optional[CelulaCognitiva]:
        """Buscar celula por ID"""
        for c in self.celulas:
            if c.id == id:
                return c
        return None

    @property
    def parametro_orden(self) -> float:
        """
        Parametro de orden de Kuramoto.

        R = 1: todas las celulas estan sincronizadas (misma fase)
        R = 0: fases completamente aleatorias

        Este es el indicador de "coherencia colectiva".
        """
        if len(self.celulas) == 0:
            return 0.0

        # Calcular fase promedio compleja
        suma = sum(np.exp(1j * c.fase) for c in self.celulas)
        R = abs(suma) / len(self.celulas)
        return R

    def evolucionar(self, dt: float, senales_externas: dict = None) -> None:
        """
        Evolucionar toda la red un paso de tiempo.

        Implementa sincronizacion de Kuramoto:
        d(theta_i)/dt = omega_i + (K/N) * sum_j(sin(theta_j - theta_i))

        Cada celula es influenciada por sus vecinos.
        """
        if senales_externas is None:
            senales_externas = {}

        N = len(self.celulas)

        # Calcular senales de acoplamiento para cada celula
        senales_acoplamiento = {c.id: 0.0 for c in self.celulas}

        for id1, id2, peso in self.conexiones:
            c1 = self._obtener_celula(id1)
            c2 = self._obtener_celula(id2)

            if c1 is None or c2 is None:
                continue

            # Kuramoto: cada celula empuja a la otra hacia su fase
            # Senal = K * sin(theta_otro - theta_yo)

            acople_1_a_2 = self.K * peso * np.exp(1j * c1.fase)
            acople_2_a_1 = self.K * peso * np.exp(1j * c2.fase)

            senales_acoplamiento[id1] += acople_2_a_1 / N
            senales_acoplamiento[id2] += acople_1_a_2 / N

        # Evolucionar cada celula
        for celula in self.celulas:
            senal_total = senales_acoplamiento.get(celula.id, 0.0)

            # Agregar senal externa si existe
            if celula.id in senales_externas:
                senal_total += senales_externas[celula.id]

            celula.evolucionar(dt, senal_total)

        self.tiempo += dt
        self.historial_orden.append(self.parametro_orden)

    def estado(self) -> str:
        """Resumen del estado de la red"""
        lineas = [f"=== Red Celular (t={self.tiempo:.2f}) ==="]
        lineas.append(f"Celulas: {len(self.celulas)}")
        lineas.append(f"Conexiones: {len(self.conexiones)}")
        lineas.append(f"Orden (sincronizacion): {self.parametro_orden:.3f}")
        lineas.append("")
        for c in self.celulas:
            lineas.append(f"  {c}")
        return "\n".join(lineas)


# ==============================================================================
# PRUEBAS DEL PROTOTIPO
# ==============================================================================

def prueba_1_aprendizaje_individual():
    """
    PRUEBA 1: Aprendizaje por Resonancia
    =====================================

    Hipotesis: Una celula puede cambiar su frecuencia propia (omega) cuando
    se expone a una senal externa de diferente frecuencia.

    Esto demuestra "memoria fisica" - la celula no guarda un numero,
    SE CONVIERTE en ese numero.
    """
    print("\n" + "="*60)
    print("PRUEBA 1: APRENDIZAJE POR RESONANCIA")
    print("="*60)

    # Crear celula con frecuencia inicial de 2 Hz
    celula = CelulaCognitiva(id=1, omega=2.0, mu=0.5, eta=0.05)
    print(f"Estado inicial: {celula}")
    print(f"Frecuencia objetivo: 5.0 Hz")

    # Senal externa de 5 Hz
    frecuencia_objetivo = 5.0

    # Simular exposicion a la senal
    dt = 0.01
    tiempo_total = 0

    for paso in range(2000):
        tiempo_total += dt

        # Generar senal externa de 5 Hz
        senal = 0.5 * np.exp(1j * frecuencia_objetivo * tiempo_total)

        celula.evolucionar(dt, senal)

    print(f"Estado final:   {celula}")
    print(f"Cambio de omega: {2.0:.3f} -> {celula.omega:.3f} Hz")

    # Verificar exito
    error = abs(celula.omega - frecuencia_objetivo)
    exito = error < 1.0  # Tolerancia de 1 Hz

    print(f"\nResultado: {'[OK] EXITO' if exito else '[X] FALLO'}")
    print(f"Error: {error:.3f} Hz")

    return celula, exito


def prueba_2_sincronizacion_dos_celulas():
    """
    PRUEBA 2: Sincronizacion de Kuramoto
    =====================================

    Hipotesis: Dos celulas con frecuencias diferentes se sincronizan
    (convergen a fase comun) sin coordinador central.

    Esto demuestra "emergencia" - el comportamiento colectivo surge
    de interacciones locales.
    """
    print("\n" + "="*60)
    print("PRUEBA 2: SINCRONIZACION DE DOS CELULAS")
    print("="*60)

    # Crear dos celulas con frecuencias diferentes
    celula_1 = CelulaCognitiva(id=1, omega=3.0, mu=0.5, psi=complex(0.5, 0.0))
    celula_2 = CelulaCognitiva(id=2, omega=3.5, mu=0.5, psi=complex(0.5, 0.3))

    print(f"Celula 1 inicial: omega={celula_1.omega:.2f} Hz, fase={np.degrees(celula_1.fase):.1f}deg")
    print(f"Celula 2 inicial: omega={celula_2.omega:.2f} Hz, fase={np.degrees(celula_2.fase):.1f}deg")

    # Crear red y conectarlas
    red = RedCelular()
    red.agregar_celula(celula_1)
    red.agregar_celula(celula_2)
    red.conectar(1, 2, peso=1.0)
    red.K = 1.0  # Acoplamiento fuerte

    print(f"Orden inicial: {red.parametro_orden:.3f}")

    # Simular
    dt = 0.01
    for paso in range(1000):
        red.evolucionar(dt)

    print(f"\nCelula 1 final: omega={celula_1.omega:.2f} Hz, fase={np.degrees(celula_1.fase):.1f}deg")
    print(f"Celula 2 final: omega={celula_2.omega:.2f} Hz, fase={np.degrees(celula_2.fase):.1f}deg")
    print(f"Orden final: {red.parametro_orden:.3f}")

    # Verificar exito: orden > 0.8 significa alta sincronizacion
    exito = red.parametro_orden > 0.8

    print(f"\nResultado: {'[OK] EXITO' if exito else '[X] FALLO'}")

    return red, exito


def prueba_3_vida_y_muerte():
    """
    PRUEBA 3: Bifurcacion de Hopf (Vida/Muerte)
    ============================================

    Hipotesis: Una celula vive (oscila) cuando mu > 0 y muere (decae)
    cuando mu < 0. Esto permite gestion automatica de recursos.

    Las "ideas" inutiles mueren sin consumir computo.
    """
    print("\n" + "="*60)
    print("PRUEBA 3: VIDA Y MUERTE CELULAR")
    print("="*60)

    # Crear celula viva
    celula = CelulaCognitiva(id=1, omega=2.0, mu=0.5, psi=complex(0.5, 0.0))
    print(f"Estado inicial: {celula}")

    # Fase 1: La celula vive (mu > 0)
    dt = 0.01
    for paso in range(200):
        celula.evolucionar(dt)

    amplitud_viva = celula.amplitud
    print(f"Despues de vivir: amplitud={amplitud_viva:.3f}, estado={celula.estado_vida.name}")

    # Fase 2: Quitar alimento (mu -> negativo)
    print("\nQuitando alimento (mu -> -0.5)...")
    celula.mu = -0.5

    for paso in range(500):
        celula.evolucionar(dt)

    amplitud_muerta = celula.amplitud
    print(f"Despues de morir: amplitud={amplitud_muerta:.3f}, estado={celula.estado_vida.name}")

    # Verificar exito
    exito = (amplitud_viva > 0.3) and (amplitud_muerta < 0.1)

    print(f"\nResultado: {'[OK] EXITO' if exito else '[X] FALLO'}")
    print(f"Amplitud viva: {amplitud_viva:.3f} (esperado > 0.3)")
    print(f"Amplitud muerta: {amplitud_muerta:.3f} (esperado < 0.1)")

    return celula, exito


def prueba_4_memoria_persistente():
    """
    PRUEBA 4: Memoria Persistente
    ==============================

    Hipotesis: Despues de aprender una frecuencia, la celula la RECUERDA
    incluso sin senal externa. La memoria esta en la estructura fisica (omega).

    Esto es fundamentalmente diferente a un Transformer que necesita
    re-leer el contexto.
    """
    print("\n" + "="*60)
    print("PRUEBA 4: MEMORIA PERSISTENTE")
    print("="*60)

    # Crear celula
    celula = CelulaCognitiva(id=1, omega=2.0, mu=0.5, eta=0.05)
    print(f"omega inicial: {celula.omega:.3f} Hz")

    # Fase 1: Aprender frecuencia 5 Hz
    print("\nFase de aprendizaje (senal de 5 Hz)...")
    dt = 0.01
    tiempo = 0

    for paso in range(1500):
        tiempo += dt
        senal = 0.5 * np.exp(1j * 5.0 * tiempo)
        celula.evolucionar(dt, senal)

    omega_aprendido = celula.omega
    print(f"omega despues de aprender: {omega_aprendido:.3f} Hz")

    # Fase 2: Sin senal - recuerda?
    print("\nFase de retencion (sin senal)...")

    for paso in range(1000):
        celula.evolucionar(dt, senal_externa=0.0)

    omega_recordado = celula.omega
    print(f"omega despues de retencion: {omega_recordado:.3f} Hz")

    # Verificar exito
    # El omega no debe cambiar significativamente sin senal
    perdida_memoria = abs(omega_recordado - omega_aprendido)
    exito = perdida_memoria < 0.5

    print(f"\nResultado: {'[OK] EXITO' if exito else '[X] FALLO'}")
    print(f"Perdida de memoria: {perdida_memoria:.3f} Hz")

    return celula, exito


def prueba_5_metacognicion():
    """
    PRUEBA 5: Metacognicion (Saber que Sabe)
    =========================================

    Hipotesis: La entropia del estado indica cuanta "certeza" tiene
    la celula. Baja entropia = alta confianza. Alta entropia = confusion.

    Esto permite que el sistema sepa cuando NO sabe algo.
    """
    print("\n" + "="*60)
    print("PRUEBA 5: METACOGNICION")
    print("="*60)

    # Crear celula y darle senal clara
    celula = CelulaCognitiva(id=1, omega=3.0, mu=0.5)

    print("Fase 1: Senal clara y consistente...")
    dt = 0.01
    tiempo = 0

    for paso in range(500):
        tiempo += dt
        senal = 0.5 * np.exp(1j * 5.0 * tiempo)  # Senal limpia
        celula.evolucionar(dt, senal)

    entropia_clara = celula.entropia
    print(f"Entropia con senal clara: {entropia_clara:.3f}")

    # Crear otra celula y darle senal ruidosa/caotica
    celula_confusa = CelulaCognitiva(id=2, omega=3.0, mu=0.5)

    print("\nFase 2: Senal ruidosa e inconsistente...")
    tiempo = 0

    for paso in range(500):
        tiempo += dt
        # Senal con frecuencia aleatoria
        freq_aleatoria = np.random.uniform(1.0, 10.0)
        senal = 0.5 * np.exp(1j * freq_aleatoria * tiempo)
        celula_confusa.evolucionar(dt, senal)

    entropia_confusa = celula_confusa.entropia
    print(f"Entropia con senal ruidosa: {entropia_confusa:.3f}")

    # Verificar exito
    exito = entropia_confusa > entropia_clara

    print(f"\nResultado: {'[OK] EXITO' if exito else '[X] FALLO'}")
    print(f"La celula confusa tiene {'mayor' if exito else 'menor'} entropia")

    return (celula, celula_confusa), exito


def prueba_6_emergencia_colectiva():
    """
    PRUEBA 6: Emergencia Colectiva
    ===============================

    Hipotesis: Una red de celulas muestra comportamiento que ninguna
    celula individual exhibe. El todo es mayor que la suma de las partes.

    Especificamente: una red puede "recordar" un patron que esta
    distribuido entre todas las celulas.
    """
    print("\n" + "="*60)
    print("PRUEBA 6: EMERGENCIA COLECTIVA")
    print("="*60)

    # Crear red de 5 celulas con frecuencias diferentes
    red = RedCelular()

    frecuencias_iniciales = [2.0, 3.0, 4.0, 5.0, 6.0]
    for i, freq in enumerate(frecuencias_iniciales):
        celula = CelulaCognitiva(
            id=i+1,
            omega=freq,
            mu=0.5,
            psi=complex(0.5, 0.1*i),
            eta=0.03
        )
        red.agregar_celula(celula)

    red.conectar_todas(peso=1.0)
    red.K = 0.8

    print("Frecuencias iniciales:", [f"{c.omega:.1f}" for c in red.celulas])
    print(f"Orden inicial: {red.parametro_orden:.3f}")

    # Exponer toda la red a senal de 4 Hz
    print("\nExponiendo red a senal de 4 Hz...")
    dt = 0.01
    tiempo = 0

    for paso in range(2000):
        tiempo += dt

        # Senal externa solo a celula 1 (las demas aprenden por propagacion)
        senal = 0.5 * np.exp(1j * 4.0 * tiempo)
        red.evolucionar(dt, {1: senal})

    print("Frecuencias finales:", [f"{c.omega:.2f}" for c in red.celulas])
    print(f"Orden final: {red.parametro_orden:.3f}")

    # Verificar exito
    # Todas las celulas deberian converger hacia 4 Hz
    freq_promedio = np.mean([c.omega for c in red.celulas])
    dispersion = np.std([c.omega for c in red.celulas])

    exito = (abs(freq_promedio - 4.0) < 1.0) and (dispersion < 1.0)

    print(f"\nResultado: {'[OK] EXITO' if exito else '[X] FALLO'}")
    print(f"Frecuencia promedio: {freq_promedio:.2f} Hz (objetivo: 4.0)")
    print(f"Dispersion: {dispersion:.2f} (menor = mas sincronizado)")

    return red, exito


def ejecutar_todas_las_pruebas():
    """Ejecutar todas las pruebas y generar reporte"""

    print("\n" + "="*60)
    print("   PROTOTIPO: CELULA COGNITIVA CUANTICA")
    print("   Validacion de Hipotesis Fundamentales")
    print("="*60)

    resultados = {}

    # Ejecutar cada prueba
    _, resultados['aprendizaje'] = prueba_1_aprendizaje_individual()
    _, resultados['sincronizacion'] = prueba_2_sincronizacion_dos_celulas()
    _, resultados['vida_muerte'] = prueba_3_vida_y_muerte()
    _, resultados['memoria'] = prueba_4_memoria_persistente()
    _, resultados['metacognicion'] = prueba_5_metacognicion()
    _, resultados['emergencia'] = prueba_6_emergencia_colectiva()

    # Reporte final
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
        print("\n*** TODAS LAS PRUEBAS PASARON ***")
        print("Las hipotesis fundamentales estan validadas.")
        print("El prototipo demuestra:")
        print("  - Aprendizaje por modificacion de estructura (no pesos)")
        print("  - Sincronizacion emergente sin coordinador")
        print("  - Gestion automatica de recursos (vida/muerte)")
        print("  - Memoria persistente en la fisica de la celula")
        print("  - Metacognicion funcional (saber que sabe)")
        print("  - Comportamiento colectivo emergente")
    elif exitosas >= total * 0.7:
        print("\n*** MAYORIA DE PRUEBAS PASARON ***")
        print("El enfoque es prometedor pero requiere ajustes.")
    else:
        print("\n*** PRUEBAS INSUFICIENTES ***")
        print("El enfoque necesita revision fundamental.")

    return resultados


if __name__ == "__main__":
    resultados = ejecutar_todas_las_pruebas()

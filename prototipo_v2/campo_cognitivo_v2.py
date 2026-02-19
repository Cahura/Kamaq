"""
CAMPO COGNITIVO V2 - MEJORAS PARA APRENDIZAJE ESTABLE
=====================================================
Versión mejorada del campo cognitivo con:

1. Aprendizaje eligibility traces (TD-lambda style)
2. Separación de patrones positivos/negativos
3. Normalización adaptativa
4. Memoria de trabajo vs memoria a largo plazo
5. Mejor equilibrio exploración/explotación

PRINCIPIOS MANTENIDOS:
- Todo basado en física (Hopfield, Kuramoto, Langevin)
- Sin reglas hardcodeadas
- Conocimiento emerge de la dinámica
"""

import numpy as np
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass
from enum import Enum


class FaseCognitiva(Enum):
    REPOSO = "reposo"
    EXPLORACION = "exploracion"
    RESONANCIA = "resonancia"
    CONSOLIDACION = "consolidacion"


@dataclass
class EstadoCampo:
    tiempo: float
    energia: float
    entropia: float
    coherencia: float
    fase: FaseCognitiva
    estado_interno: np.ndarray


class CampoCognitivoV2:
    """
    Campo cognitivo mejorado con aprendizaje más estable.

    Mejoras clave:
    1. Dos matrices J: J_positiva y J_negativa
    2. Eligibility traces para TD-learning
    3. Temperatura adaptativa
    4. Memoria de trabajo separada
    """

    def __init__(self, dimension: int = 64, dt: float = 0.01):
        self.dim = dimension
        self.dt = dt
        self.tiempo = 0.0

        # Estado del campo
        self.x = np.random.randn(dimension) * 0.1
        self.v = np.zeros(dimension)
        self.theta = np.random.uniform(0, 2*np.pi, dimension)

        # MEJORA 1: Matrices separadas para aprendizaje positivo y negativo
        self.J_positiva = np.zeros((dimension, dimension))
        self.J_negativa = np.zeros((dimension, dimension))

        # La J efectiva es la diferencia (lo bueno menos lo malo)
        # Esto evita que patrones negativos dominen

        # Parámetros físicos
        self.omega = 1.0 + np.random.randn(dimension) * 0.1
        self.K = 2.0
        self.gamma = 0.5
        self.temperatura_base = 0.01
        self.temperatura = self.temperatura_base

        # MEJORA 2: Eligibility traces
        self.traces = np.zeros((dimension, dimension))
        self.trace_decay = 0.9  # Lambda en TD(lambda)

        # MEJORA 3: Memoria de trabajo (corto plazo, alta plasticidad)
        self.memoria_trabajo = np.zeros((dimension, dimension))
        self.plasticidad_trabajo = 0.1

        # Atractores
        self.atractores: List[np.ndarray] = []
        self.n_atractores = 0

        # Estadísticas de aprendizaje
        self.n_refuerzos_positivos = 0
        self.n_refuerzos_negativos = 0
        self.historial_energia: List[float] = []

    @property
    def J(self) -> np.ndarray:
        """J efectiva = J_positiva - J_negativa + memoria_trabajo"""
        return self.J_positiva - 0.5 * self.J_negativa + self.memoria_trabajo

    def energia(self) -> float:
        J_efectiva = self.J
        E_hopfield = -0.5 * self.x @ J_efectiva @ self.x
        E_cinetica = 0.5 * np.sum(self.v ** 2)
        E_potencial = 0.5 * np.sum(self.x ** 2)
        return float(E_hopfield + E_cinetica + E_potencial)

    def entropia(self) -> float:
        p = np.abs(self.x) ** 2
        suma = np.sum(p)
        if suma < 1e-10:
            return 1.0
        p = p / suma
        S = -np.sum(p * np.log(p + 1e-10))
        return float(S / np.log(self.dim))

    def coherencia(self) -> float:
        r = np.abs(np.mean(np.exp(1j * self.theta)))
        return float(r)

    def evolucionar(self, pasos: int = 1):
        """Evoluciona el campo con dinámica mejorada."""
        J_efectiva = self.J

        for _ in range(pasos):
            self.tiempo += self.dt

            # Dinámica de Hopfield
            fuerza = J_efectiva @ self.x - self.x
            friccion = -self.gamma * self.v
            ruido = np.sqrt(2 * self.gamma * self.temperatura) * np.random.randn(self.dim)

            self.v += (fuerza + friccion + ruido) * self.dt
            self.x += self.v * self.dt

            # Estabilización
            self.x = np.clip(self.x, -10.0, 10.0)
            self.v = np.clip(self.v, -5.0, 5.0)

            # Dinámica de Kuramoto
            diff_theta = self.theta[:, np.newaxis] - self.theta[np.newaxis, :]
            acoplamiento = self.K * np.mean(np.sin(-diff_theta), axis=1)
            self.theta += (self.omega + acoplamiento) * self.dt
            self.theta = np.mod(self.theta, 2 * np.pi)

            # MEJORA: Decaimiento de eligibility traces
            self.traces *= self.trace_decay

            # MEJORA: Decaimiento de memoria de trabajo
            self.memoria_trabajo *= 0.99

    def perturbar(self, patron: np.ndarray, fuerza: float = 1.0):
        """Perturba el campo y actualiza eligibility traces."""
        if len(patron) != self.dim:
            patron = np.resize(patron, self.dim)

        norma = np.linalg.norm(patron)
        if norma > 1e-10:
            patron = patron / norma

        self.x += fuerza * patron

        # MEJORA: Actualizar traces con el patrón actual
        # Esto permite credit assignment diferido
        patron_bin = np.sign(patron)
        patron_bin[patron_bin == 0] = 1
        trace_update = np.outer(patron_bin, patron_bin) / self.dim
        np.fill_diagonal(trace_update, 0)
        self.traces += trace_update
        self.traces = np.clip(self.traces, -1.0, 1.0)

    def memorizar(self, patron: np.ndarray):
        """Memoriza un patrón en J_positiva."""
        if len(patron) != self.dim:
            patron = np.resize(patron, self.dim)

        patron_bin = np.sign(patron)
        patron_bin[patron_bin == 0] = 1

        delta_J = np.outer(patron_bin, patron_bin) / self.dim
        np.fill_diagonal(delta_J, 0)

        self.J_positiva += delta_J
        self.J_positiva = np.clip(self.J_positiva, 0, 1.0)

        self.atractores.append(patron_bin.copy())
        self.n_atractores += 1

    def recordar(self, pista: np.ndarray, pasos_relajacion: int = 100) -> np.ndarray:
        """Recupera patrón usando J efectiva."""
        if len(pista) != self.dim:
            pista = np.resize(pista, self.dim)

        estado = np.sign(pista)
        estado[estado == 0] = 1

        J_efectiva = self.J

        for _ in range(pasos_relajacion):
            orden = np.random.permutation(self.dim)
            cambios = 0

            for i in orden:
                h_i = np.dot(J_efectiva[i], estado)
                nuevo = 1 if h_i >= 0 else -1

                if nuevo != estado[i]:
                    estado[i] = nuevo
                    cambios += 1

            if cambios == 0:
                break

        return estado

    def fase_actual(self) -> FaseCognitiva:
        E = self.energia()
        S = self.entropia()
        r = self.coherencia()

        if E < 0.5 and S < 0.3:
            return FaseCognitiva.REPOSO
        elif S > 0.7:
            return FaseCognitiva.EXPLORACION
        elif r > 0.6:
            return FaseCognitiva.RESONANCIA
        else:
            return FaseCognitiva.CONSOLIDACION

    def estado(self) -> EstadoCampo:
        return EstadoCampo(
            tiempo=self.tiempo,
            energia=self.energia(),
            entropia=self.entropia(),
            coherencia=self.coherencia(),
            fase=self.fase_actual(),
            estado_interno=self.x.copy()
        )


class AprendizajeHebbianoV2:
    """
    Aprendizaje Hebbiano mejorado con:
    1. Separación de refuerzos positivos/negativos
    2. Uso de eligibility traces
    3. Aprendizaje TD-style
    """

    def __init__(self, campo: CampoCognitivoV2, eta: float = 0.01):
        self.campo = campo
        self.eta = eta
        self.ultimo_patron = None

    def paso_aprendizaje(self):
        """Paso de aprendizaje en memoria de trabajo."""
        x = self.campo.x
        norma_x = np.linalg.norm(x)
        if norma_x > 1e-10:
            x_norm = x / norma_x
        else:
            return

        # Aprender en memoria de trabajo (alta plasticidad, rápido decaimiento)
        delta = self.campo.plasticidad_trabajo * np.outer(x_norm, x_norm)
        np.fill_diagonal(delta, 0)
        self.campo.memoria_trabajo += delta
        self.campo.memoria_trabajo = np.clip(self.campo.memoria_trabajo, -0.5, 0.5)

        self.ultimo_patron = x_norm.copy()

    def reforzar(self, patron: np.ndarray, recompensa: float):
        """
        Refuerza usando eligibility traces y separación pos/neg.

        Si recompensa > 0: Fortalece J_positiva
        Si recompensa < 0: Fortalece J_negativa (evitar)
        """
        if len(patron) != self.campo.dim:
            patron = np.resize(patron, self.campo.dim)

        norma = np.linalg.norm(patron)
        if norma > 1e-10:
            patron = patron / norma

        recompensa_clip = np.clip(recompensa, -1.0, 1.0)

        # MEJORA: Usar traces para credit assignment
        # El refuerzo afecta no solo el patrón actual sino todos los recientes
        if abs(recompensa_clip) > 0.1:
            if recompensa_clip > 0:
                # Reforzar J_positiva usando traces
                self.campo.J_positiva += self.eta * recompensa_clip * self.campo.traces
                self.campo.J_positiva = np.clip(self.campo.J_positiva, 0, 1.0)
                self.campo.n_refuerzos_positivos += 1

                # También reforzar el patrón específico
                patron_bin = np.sign(patron)
                patron_bin[patron_bin == 0] = 1
                delta = self.eta * recompensa_clip * np.outer(patron_bin, patron_bin) / self.campo.dim
                np.fill_diagonal(delta, 0)
                self.campo.J_positiva += delta
                self.campo.J_positiva = np.clip(self.campo.J_positiva, 0, 1.0)

            else:
                # Reforzar J_negativa usando traces
                self.campo.J_negativa += self.eta * abs(recompensa_clip) * self.campo.traces
                self.campo.J_negativa = np.clip(self.campo.J_negativa, 0, 0.5)  # Limitar impacto negativo
                self.campo.n_refuerzos_negativos += 1

        # Reset parcial de traces después del refuerzo
        self.campo.traces *= 0.5


class AgenteEmergentev2:
    """
    Agente mejorado que usa CampoCognitivoV2.
    """

    def __init__(self, dim_estado: int, n_acciones: int):
        self.dim_estado = dim_estado
        self.n_acciones = n_acciones

        dim_campo = dim_estado + n_acciones + 16
        self.campo = CampoCognitivoV2(dimension=dim_campo)
        self.aprendizaje = AprendizajeHebbianoV2(self.campo, eta=0.02)

        self.experiencias: List = []
        self.max_experiencias = 2000

        self.n_decisiones = 0
        self.n_exploraciones = 0
        self.recompensa_acumulada = 0.0

        # MEJORA: Temperatura de exploración adaptativa
        self.epsilon_inicial = 0.5
        self.epsilon_final = 0.05
        self.epsilon_decay = 0.9995
        self.epsilon = self.epsilon_inicial

    def estado_a_patron(self, estado: np.ndarray) -> np.ndarray:
        patron = np.zeros(self.campo.dim)
        for i, v in enumerate(estado):
            if i < self.campo.dim:
                patron[i] = v
        return patron

    def accion_a_patron(self, accion: int) -> np.ndarray:
        patron = np.zeros(self.campo.dim)
        idx = self.dim_estado + accion
        if idx < self.campo.dim:
            patron[idx] = 1.0
        return patron

    def experiencia_a_patron(self, estado: np.ndarray, accion: int) -> np.ndarray:
        patron = self.estado_a_patron(estado)
        patron_accion = self.accion_a_patron(accion)
        return patron + 0.5 * patron_accion

    def decidir(self, estado: np.ndarray, acciones_validas: List[int]) -> int:
        self.n_decisiones += 1

        patron_estado = self.estado_a_patron(estado)
        self.campo.perturbar(patron_estado, fuerza=1.0)
        self.campo.evolucionar(pasos=15)

        # MEJORA: Epsilon-greedy con decay
        if np.random.random() < self.epsilon:
            self.n_exploraciones += 1
            return int(np.random.choice(acciones_validas))

        # Calcular afinidades
        afinidades = {}
        for accion in acciones_validas:
            patron_accion = self.accion_a_patron(accion)

            norma_x = np.linalg.norm(self.campo.x)
            norma_p = np.linalg.norm(patron_accion)

            if norma_x > 1e-10 and norma_p > 1e-10:
                correlacion = np.dot(self.campo.x, patron_accion) / (norma_x * norma_p)
            else:
                correlacion = 0.0

            afinidades[accion] = correlacion

        if afinidades:
            # MEJORA: Softmax para selección más suave
            valores = np.array([afinidades[a] for a in acciones_validas])
            valores = valores - np.max(valores)  # Estabilidad numérica
            exp_valores = np.exp(valores * 5.0)  # Temperatura
            probs = exp_valores / (np.sum(exp_valores) + 1e-10)

            accion = np.random.choice(acciones_validas, p=probs)
            return int(accion)
        else:
            return int(np.random.choice(acciones_validas))

    def aprender(self, experiencia):
        self.experiencias.append(experiencia)
        if len(self.experiencias) > self.max_experiencias:
            self.experiencias.pop(0)

        self.recompensa_acumulada += experiencia.recompensa

        patron = self.experiencia_a_patron(experiencia.estado, experiencia.accion)
        self.campo.perturbar(patron, fuerza=1.0)
        self.campo.evolucionar(pasos=10)

        self.aprendizaje.paso_aprendizaje()

        if experiencia.recompensa != 0:
            self.aprendizaje.reforzar(patron, experiencia.recompensa)

        # Decay epsilon
        self.epsilon = max(self.epsilon_final, self.epsilon * self.epsilon_decay)

    def consolidar(self):
        if not self.experiencias:
            return

        # Priorizar experiencias con recompensa
        exp_con_recompensa = [e for e in self.experiencias if e.recompensa != 0]

        if not exp_con_recompensa:
            return

        n_replay = min(30, len(exp_con_recompensa))
        indices = np.random.choice(len(exp_con_recompensa), n_replay, replace=False)

        for idx in indices:
            exp = exp_con_recompensa[idx]
            patron = self.experiencia_a_patron(exp.estado, exp.accion)

            self.campo.perturbar(patron, fuerza=0.3)
            self.campo.evolucionar(pasos=5)

            if exp.recompensa != 0:
                self.aprendizaje.reforzar(patron, exp.recompensa * 0.3)

    def diagnostico(self) -> Dict:
        estado = self.campo.estado()
        return {
            'n_decisiones': self.n_decisiones,
            'n_exploraciones': self.n_exploraciones,
            'ratio_exploracion': self.n_exploraciones / max(1, self.n_decisiones),
            'epsilon': self.epsilon,
            'recompensa_acumulada': self.recompensa_acumulada,
            'n_experiencias': len(self.experiencias),
            'energia_campo': estado.energia,
            'entropia_campo': estado.entropia,
            'coherencia_campo': estado.coherencia,
            'fase_campo': estado.fase.value,
            'n_atractores': self.campo.n_atractores,
            'refuerzos_positivos': self.campo.n_refuerzos_positivos,
            'refuerzos_negativos': self.campo.n_refuerzos_negativos,
        }

    def reset_estadisticas(self):
        self.n_decisiones = 0
        self.n_exploraciones = 0

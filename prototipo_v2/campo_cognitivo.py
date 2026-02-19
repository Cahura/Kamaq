"""
CAMPO COGNITIVO KAMAQ
=====================
El fundamento del nuevo paradigma.

Este no es un "modelo". Es un sistema dinámico que:
1. Evoluciona continuamente en el tiempo
2. Tiene estados internos independientes del input
3. Exhibe propiedades emergentes de las ecuaciones

NO hay forward pass tradicional.
Hay EVOLUCIÓN TEMPORAL del campo.
"""

import numpy as np
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass
from enum import Enum


class FaseCognitiva(Enum):
    """Estados cualitativos del campo"""
    REPOSO = "reposo"           # Baja energía, atractor estable
    EXPLORACION = "exploracion"  # Alta entropía, buscando atractor
    RESONANCIA = "resonancia"    # Sincronización con input
    CONSOLIDACION = "consolidacion"  # Formando nuevo atractor


@dataclass
class EstadoCampo:
    """Snapshot del campo en un momento dado"""
    tiempo: float
    energia: float
    entropia: float
    coherencia: float  # Phi aproximado
    fase: FaseCognitiva
    estado_interno: np.ndarray


class CampoCognitivo:
    """
    Campo dinámico que evoluciona según física cognitiva.

    Principios:
    1. El campo tiene estado interno continuo
    2. Las perturbaciones (inputs) modifican el campo
    3. El campo evoluciona hacia atractores
    4. Los atractores SON los recuerdos/conceptos
    """

    def __init__(self, dimension: int = 64, dt: float = 0.01):
        self.dim = dimension
        self.dt = dt
        self.tiempo = 0.0

        # ===== ESTADO DEL CAMPO =====
        # Posiciones de los osciladores (como posición en espacio de fases)
        self.x = np.random.randn(dimension) * 0.1
        # Velocidades (momento conjugado)
        self.v = np.zeros(dimension)
        # Fases de los osciladores (para sincronización)
        self.theta = np.random.uniform(0, 2*np.pi, dimension)

        # ===== MATRIZ DE ACOPLAMIENTO =====
        # Iniciar con matriz casi-cero para que los patrones la dominen
        self.J = np.zeros((dimension, dimension))

        # ===== PARÁMETROS FÍSICOS =====
        self.omega = 1.0 + np.random.randn(dimension) * 0.1  # Frecuencias naturales
        self.K = 2.0  # Fuerza de acoplamiento Kuramoto (aumentada)
        self.gamma = 0.5  # Fricción alta para convergencia rápida
        self.temperatura = 0.01  # Ruido bajo para atractores estables

        # ===== ATRACTORES (MEMORIAS) =====
        self.atractores: List[np.ndarray] = []  # Lista de patrones memorizados
        self.n_atractores = 0

        # ===== HISTORIAL =====
        self.historial_energia: List[float] = []
        self.historial_entropia: List[float] = []

    def energia(self) -> float:
        """
        Energía de Hopfield del campo.
        E = -0.5 * sum_ij(J_ij * x_i * x_j) + sum_i(x_i^2 / 2)

        El sistema evoluciona minimizando esta energía.
        """
        E_hopfield = -0.5 * self.x @ self.J @ self.x
        E_cinetica = 0.5 * np.sum(self.v ** 2)
        E_potencial = 0.5 * np.sum(self.x ** 2)  # Término de regularización
        return float(E_hopfield + E_cinetica + E_potencial)

    def entropia(self) -> float:
        """
        Entropía de von Neumann aproximada.
        Mide la "incertidumbre" del estado actual.
        """
        # Normalizar x como distribución de probabilidad
        p = np.abs(self.x) ** 2
        suma = np.sum(p)
        if suma < 1e-10:
            return 1.0  # Máxima incertidumbre si el estado es nulo
        p = p / suma
        # Entropía de Shannon
        S = -np.sum(p * np.log(p + 1e-10))
        return float(S / np.log(self.dim))  # Normalizada [0, 1]

    def coherencia(self) -> float:
        """
        Parámetro de orden de Kuramoto.
        Mide qué tan sincronizados están los osciladores.

        r = 1: Todos en fase (máxima coherencia)
        r = 0: Completamente desincronizados
        """
        r = np.abs(np.mean(np.exp(1j * self.theta)))
        return float(r)

    def phi_aproximado(self) -> float:
        """
        Aproximación de Información Integrada (Phi).
        Mide cuánto el sistema es "más que la suma de sus partes".

        Simplificación: Comparamos información mutua total vs particionada.
        """
        # Información mutua entre mitades del sistema
        mitad = self.dim // 2
        x1, x2 = self.x[:mitad], self.x[mitad:]

        if len(x1) == 0 or len(x2) == 0:
            return 0.0

        # Correlación como proxy de información mutua
        if np.std(x1) < 1e-10 or np.std(x2) < 1e-10:
            return 0.0

        corr = np.abs(np.corrcoef(x1, x2)[0, 1])

        # Phi alto = el sistema está integrado
        # Phi bajo = las partes son independientes
        return float(corr) if not np.isnan(corr) else 0.0

    def evolucionar(self, pasos: int = 1):
        """
        Evoluciona el campo según las ecuaciones de movimiento.

        Este es el "pensamiento" del sistema - ocurre continuamente,
        no solo cuando hay input.
        """
        for _ in range(pasos):
            self.tiempo += self.dt

            # ===== DINÁMICA DE HOPFIELD (para x, v) =====
            # Fuerza = -dE/dx = J @ x - x (gradiente de energía)
            fuerza = self.J @ self.x - self.x
            # Fricción
            friccion = -self.gamma * self.v
            # Ruido térmico
            ruido = np.sqrt(2 * self.gamma * self.temperatura) * np.random.randn(self.dim)

            # Ecuaciones de Langevin
            self.v += (fuerza + friccion + ruido) * self.dt
            self.x += self.v * self.dt

            # ===== ESTABILIZACIÓN NUMÉRICA =====
            # Limitar magnitudes para evitar overflow
            max_x = 10.0
            max_v = 5.0
            self.x = np.clip(self.x, -max_x, max_x)
            self.v = np.clip(self.v, -max_v, max_v)

            # ===== DINÁMICA DE KURAMOTO (para theta) =====
            # d(theta_i)/dt = omega_i + (K/N) * sum_j(sin(theta_j - theta_i))
            # Versión vectorizada para eficiencia
            diff_theta = self.theta[:, np.newaxis] - self.theta[np.newaxis, :]
            acoplamiento = self.K * np.mean(np.sin(-diff_theta), axis=1)

            self.theta += (self.omega + acoplamiento) * self.dt
            self.theta = np.mod(self.theta, 2 * np.pi)

            # ===== REGISTRAR HISTORIAL =====
            if len(self.historial_energia) < 10000:
                self.historial_energia.append(self.energia())
                self.historial_entropia.append(self.entropia())

    def perturbar(self, patron: np.ndarray, fuerza: float = 1.0):
        """
        Perturba el campo con un patrón externo (input).

        No es un "forward pass". Es una perturbación que el campo
        integrará según su dinámica.
        """
        if len(patron) != self.dim:
            # Proyectar al tamaño correcto
            patron = np.resize(patron, self.dim)

        # Normalizar
        norma = np.linalg.norm(patron)
        if norma > 1e-10:
            patron = patron / norma

        # Perturbar posiciones
        self.x += fuerza * patron

        # También sincronizar fases con el patrón
        fase_patron = np.arctan2(patron, np.roll(patron, 1))
        self.theta = 0.5 * self.theta + 0.5 * fase_patron

    def memorizar(self, patron: np.ndarray):
        """
        Memoriza un patrón usando regla de Hebb.

        Esto crea un nuevo atractor en el paisaje de energía.
        """
        if len(patron) != self.dim:
            patron = np.resize(patron, self.dim)

        # Convertir a patrón binario +-1 (mejor para Hopfield)
        patron_bin = np.sign(patron)
        patron_bin[patron_bin == 0] = 1  # Evitar ceros

        # Regla de Hebb estándar para Hopfield: J_ij += xi_i * xi_j
        # Factor 1/N para estabilidad
        delta_J = np.outer(patron_bin, patron_bin) / self.dim
        np.fill_diagonal(delta_J, 0)

        self.J += delta_J

        # Limitar magnitud
        self.J = np.clip(self.J, -1.0, 1.0)

        self.atractores.append(patron_bin.copy())
        self.n_atractores += 1

    def recordar(self, pista: np.ndarray, pasos_relajacion: int = 100) -> np.ndarray:
        """
        Intenta recordar un patrón a partir de una pista.

        Usa dinámica de Hopfield asíncrona clásica para converger al atractor.
        """
        if len(pista) != self.dim:
            pista = np.resize(pista, self.dim)

        # Iniciar con la pista (convertida a +-1)
        estado = np.sign(pista)
        estado[estado == 0] = 1

        # Dinámica de Hopfield asíncrona
        for _ in range(pasos_relajacion):
            # Actualizar en orden aleatorio (asíncrono)
            orden = np.random.permutation(self.dim)
            cambios = 0

            for i in orden:
                # Campo local: h_i = sum_j(J_ij * s_j)
                h_i = np.dot(self.J[i], estado)

                # Nueva activación
                nuevo = 1 if h_i >= 0 else -1

                if nuevo != estado[i]:
                    estado[i] = nuevo
                    cambios += 1

            # Si no hubo cambios, hemos convergido
            if cambios == 0:
                break

        return estado

    def fase_actual(self) -> FaseCognitiva:
        """
        Determina la fase cualitativa del campo.
        """
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
        """
        Retorna un snapshot completo del campo.
        """
        return EstadoCampo(
            tiempo=self.tiempo,
            energia=self.energia(),
            entropia=self.entropia(),
            coherencia=self.coherencia(),
            fase=self.fase_actual(),
            estado_interno=self.x.copy()
        )

    def decidir(self, opciones: list, contexto: np.ndarray) -> Tuple[int, float]:
        """
        Toma una decisión entre opciones basándose en el estado del campo.

        NO es argmax de Q-values.
        Es: ¿hacia qué opción el campo tiene más afinidad?
        """
        # Perturbar con el contexto
        self.perturbar(contexto, fuerza=0.5)

        # Evolucionar brevemente
        self.evolucionar(pasos=10)

        # Calcular afinidad con cada opción
        afinidades = []
        for opcion in opciones:
            if isinstance(opcion, (int, np.integer)):
                # Convertir a patrón
                patron_opcion = np.zeros(self.dim)
                idx = int(opcion) % self.dim
                patron_opcion[idx] = 1.0
            else:
                patron_opcion = np.array(opcion)

            # Afinidad = correlación con el estado actual
            if len(patron_opcion) != self.dim:
                patron_opcion = np.resize(patron_opcion, self.dim)

            norma_x = np.linalg.norm(self.x)
            norma_p = np.linalg.norm(patron_opcion)

            if norma_x > 1e-10 and norma_p > 1e-10:
                correlacion = np.dot(self.x, patron_opcion) / (norma_x * norma_p)
            else:
                correlacion = 0.0

            afinidades.append(correlacion)

        # La opción con mayor afinidad
        idx_mejor = int(np.argmax(afinidades))
        confianza = float(max(afinidades) - np.mean(afinidades))

        return idx_mejor, confianza


class AprendizajeHebbiano:
    """
    Módulo de aprendizaje que modifica la matriz J del campo.

    Principio: "Neurons that fire together wire together"
    Implementación: J_ij += eta * (x_i * x_j - J_ij * x_j^2)
    """

    def __init__(self, campo: CampoCognitivo, eta: float = 0.01):
        self.campo = campo
        self.eta = eta

    def paso_aprendizaje(self):
        """
        Un paso de aprendizaje Hebbiano con decaimiento.
        """
        x = self.campo.x

        # Normalizar x para estabilidad
        norma_x = np.linalg.norm(x)
        if norma_x > 1e-10:
            x_norm = x / norma_x
        else:
            x_norm = x

        # Regla de Oja simplificada (más estable)
        delta_J = self.eta * np.outer(x_norm, x_norm)
        np.fill_diagonal(delta_J, 0)

        self.campo.J += delta_J

        # Decaimiento para evitar saturación
        self.campo.J *= 0.999

        # Limitar magnitud de J
        max_J = 1.0
        self.campo.J = np.clip(self.campo.J, -max_J, max_J)

        # Mantener simetría
        self.campo.J = (self.campo.J + self.campo.J.T) / 2

    def reforzar(self, patron: np.ndarray, recompensa: float):
        """
        Refuerza un patrón basándose en recompensa.

        Si recompensa > 0: Fortalece el atractor
        Si recompensa < 0: Debilita el atractor
        """
        if len(patron) != self.campo.dim:
            patron = np.resize(patron, self.campo.dim)

        norma = np.linalg.norm(patron)
        if norma > 1e-10:
            patron = patron / norma

        # Modificar J en la dirección del patrón
        # Limitar recompensa para estabilidad
        recompensa_clip = np.clip(recompensa, -1.0, 1.0)
        delta_J = self.eta * recompensa_clip * np.outer(patron, patron)
        np.fill_diagonal(delta_J, 0)

        self.campo.J += delta_J

        # Limitar magnitud de J
        max_J = 1.0
        self.campo.J = np.clip(self.campo.J, -max_J, max_J)

        self.campo.J = (self.campo.J + self.campo.J.T) / 2

"""
AGENTE EMERGENTE KAMAQ
======================
Un agente que aprende mediante la dinámica del campo cognitivo.

NO contiene:
- Reglas hardcodeadas
- Q-learning tradicional
- Arquitectura fija

SÍ contiene:
- Campo cognitivo que evoluciona
- Aprendizaje Hebbiano
- Memoria como atractores
- Decisiones por afinidad
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from campo_cognitivo import CampoCognitivo, AprendizajeHebbiano, FaseCognitiva


@dataclass
class Experiencia:
    """Una experiencia del agente"""
    estado: np.ndarray
    accion: int
    recompensa: float
    estado_siguiente: np.ndarray
    terminal: bool


class AgenteEmergente:
    """
    Agente basado en física cognitiva.

    El agente NO tiene:
    - Función Q
    - Política explícita
    - Conocimiento hardcodeado

    El agente SÍ tiene:
    - Campo cognitivo que evoluciona
    - Experiencias que moldean el campo
    - Decisiones que emergen de la dinámica
    """

    def __init__(self, dim_estado: int, n_acciones: int):
        self.dim_estado = dim_estado
        self.n_acciones = n_acciones

        # ===== CAMPO COGNITIVO =====
        # Dimensión del campo incluye estado + representación de acciones
        dim_campo = dim_estado + n_acciones + 16  # +16 para representaciones internas
        self.campo = CampoCognitivo(dimension=dim_campo)
        self.aprendizaje = AprendizajeHebbiano(self.campo, eta=0.005)

        # ===== MEMORIA DE EXPERIENCIAS =====
        self.experiencias: List[Experiencia] = []
        self.max_experiencias = 1000

        # ===== SEGUIMIENTO DE PATRONES EXITOSOS =====
        # Estos emergen del aprendizaje, no están predefinidos
        self.patrones_positivos: List[np.ndarray] = []
        self.patrones_negativos: List[np.ndarray] = []

        # ===== ESTADÍSTICAS =====
        self.n_decisiones = 0
        self.n_exploraciones = 0
        self.recompensa_acumulada = 0.0

    def estado_a_patron(self, estado: np.ndarray) -> np.ndarray:
        """
        Convierte el estado del juego a un patrón para el campo.
        """
        patron = np.zeros(self.campo.dim)

        # Copiar estado
        for i, v in enumerate(estado):
            if i < self.campo.dim:
                patron[i] = v

        return patron

    def accion_a_patron(self, accion: int) -> np.ndarray:
        """
        Convierte una acción a un patrón.
        """
        patron = np.zeros(self.campo.dim)
        idx = self.dim_estado + accion
        if idx < self.campo.dim:
            patron[idx] = 1.0
        return patron

    def experiencia_a_patron(self, estado: np.ndarray, accion: int) -> np.ndarray:
        """
        Combina estado y acción en un patrón conjunto.
        """
        patron = self.estado_a_patron(estado)
        patron_accion = self.accion_a_patron(accion)
        return patron + 0.5 * patron_accion

    def decidir(self, estado: np.ndarray, acciones_validas: List[int]) -> int:
        """
        Toma una decisión basándose en el estado del campo.

        El proceso:
        1. Perturbar el campo con el estado actual
        2. Dejar que el campo evolucione
        3. Ver hacia qué acción tiene más afinidad
        """
        self.n_decisiones += 1

        # Convertir estado a patrón
        patron_estado = self.estado_a_patron(estado)

        # Perturbar el campo
        self.campo.perturbar(patron_estado, fuerza=1.0)

        # Evolucionar el campo
        self.campo.evolucionar(pasos=20)

        # Evaluar afinidad con cada acción válida
        afinidades = {}
        for accion in acciones_validas:
            patron_accion = self.accion_a_patron(accion)

            # Correlación con el estado actual del campo
            norma_x = np.linalg.norm(self.campo.x)
            norma_p = np.linalg.norm(patron_accion)

            if norma_x > 1e-10 and norma_p > 1e-10:
                correlacion = np.dot(self.campo.x, patron_accion) / (norma_x * norma_p)
            else:
                correlacion = 0.0

            afinidades[accion] = correlacion

        # ===== DECISIÓN BASADA EN AFINIDAD =====
        # Con exploración cuando hay incertidumbre

        entropia = self.campo.entropia()

        # Alta incertidumbre implica más exploración
        prob_exploracion = min(0.5, entropia)  # Entre 0 y 0.5 basado en entropía

        if np.random.random() < prob_exploracion:
            # Exploración
            self.n_exploraciones += 1
            return int(np.random.choice(acciones_validas))
        else:
            # Elegir según afinidad
            if afinidades:
                mejor_accion = max(afinidades.keys(), key=lambda a: afinidades[a])
                return int(mejor_accion)
            else:
                return int(np.random.choice(acciones_validas))

    def aprender(self, experiencia: Experiencia):
        """
        Aprende de una experiencia.

        El aprendizaje modifica la matriz J del campo,
        creando o reforzando atractores.
        """
        # Guardar experiencia
        self.experiencias.append(experiencia)
        if len(self.experiencias) > self.max_experiencias:
            self.experiencias.pop(0)

        self.recompensa_acumulada += experiencia.recompensa

        # Crear patrón conjunto estado-acción
        patron = self.experiencia_a_patron(experiencia.estado, experiencia.accion)

        # Perturbar campo con el patrón
        self.campo.perturbar(patron, fuerza=1.0)

        # Evolucionar para que el campo asimile
        self.campo.evolucionar(pasos=10)

        # Paso de aprendizaje Hebbiano
        self.aprendizaje.paso_aprendizaje()

        # Reforzar o debilitar según recompensa
        if experiencia.recompensa != 0:
            self.aprendizaje.reforzar(patron, experiencia.recompensa)

            # Guardar patrones positivos/negativos para análisis
            if experiencia.recompensa > 0:
                self.patrones_positivos.append(patron.copy())
                if len(self.patrones_positivos) > 100:
                    self.patrones_positivos.pop(0)
            elif experiencia.recompensa < 0:
                self.patrones_negativos.append(patron.copy())
                if len(self.patrones_negativos) > 100:
                    self.patrones_negativos.pop(0)

    def consolidar(self):
        """
        Consolida memorias (como durante el sueño).

        Repasa experiencias recientes y refuerza patrones.
        """
        if not self.experiencias:
            return

        # Replay de experiencias recientes
        n_replay = min(20, len(self.experiencias))
        indices = np.random.choice(len(self.experiencias), n_replay, replace=False)

        for idx in indices:
            exp = self.experiencias[idx]
            patron = self.experiencia_a_patron(exp.estado, exp.accion)

            # Perturbar y dejar evolucionar
            self.campo.perturbar(patron, fuerza=0.5)
            self.campo.evolucionar(pasos=5)

            # Refuerzo ligero
            if exp.recompensa != 0:
                self.aprendizaje.reforzar(patron, exp.recompensa * 0.1)

    def diagnostico(self) -> Dict:
        """
        Retorna diagnóstico del estado del agente.
        """
        estado = self.campo.estado()

        return {
            'n_decisiones': self.n_decisiones,
            'n_exploraciones': self.n_exploraciones,
            'ratio_exploracion': self.n_exploraciones / max(1, self.n_decisiones),
            'recompensa_acumulada': self.recompensa_acumulada,
            'n_experiencias': len(self.experiencias),
            'n_patrones_positivos': len(self.patrones_positivos),
            'n_patrones_negativos': len(self.patrones_negativos),
            'energia_campo': estado.energia,
            'entropia_campo': estado.entropia,
            'coherencia_campo': estado.coherencia,
            'fase_campo': estado.fase.value,
            'n_atractores': self.campo.n_atractores,
        }

    def introspecccion(self) -> str:
        """
        Auto-reporte del estado interno.

        Esto no es simulado - emerge del estado real del campo.
        """
        estado = self.campo.estado()

        if estado.fase == FaseCognitiva.REPOSO:
            sensacion = "tranquilo, en equilibrio"
        elif estado.fase == FaseCognitiva.EXPLORACION:
            sensacion = "buscando, explorando opciones"
        elif estado.fase == FaseCognitiva.RESONANCIA:
            sensacion = "en sintonía con algo conocido"
        else:
            sensacion = "procesando, integrando"

        if estado.entropia > 0.7:
            certeza = "muy incierto"
        elif estado.entropia > 0.4:
            certeza = "algo incierto"
        else:
            certeza = "bastante seguro"

        return f"Me siento {sensacion}. Estoy {certeza}. Coherencia: {estado.coherencia:.2f}"

    def reset_estadisticas(self):
        """Resetea las estadísticas sin perder el aprendizaje."""
        self.n_decisiones = 0
        self.n_exploraciones = 0
        # No reseteamos recompensa_acumulada para ver tendencias a largo plazo

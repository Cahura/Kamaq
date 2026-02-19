"""
CAMPO COGNITIVO V4 - INTEGRACION COMPLETA
==========================================
Integra los componentes del nuevo paradigma:

1. MOTOR DE INFERENCIA ACTIVA (Fase 1)
   - Exploracion dirigida por curiosidad
   - Modelo generativo que aprende transiciones
   - Balance valor epistemico / pragmatico

2. POBLACION DE ESTRATEGIAS (Fase 2)
   - Seleccion natural de estrategias
   - Solo sobreviven las que funcionan en multiples contextos
   - Resuelve saturacion de matriz J

3. CAMPO DE HOPFIELD (heredado de V3)
   - Memoria asociativa para patrones
   - Atractores como conceptos
   - Dinamica continua

PRINCIPIO UNIFICADOR:
El campo cognitivo usa inferencia activa para explorar,
poblacion de estrategias para recordar, y Hopfield para
generalizar. Los tres componentes trabajan juntos.

NO HAY REGLAS HARDCODEADAS - todo emerge de la dinamica.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

from motor_inferencia_activa import (
    MotorInferenciaActiva, CuriosidadAdaptativa, ModeloGenerativo
)
from poblacion_estrategias import (
    PoblacionEstrategias, SelectorHibrido, Estrategia
)


@dataclass
class DecisionInfo:
    """Informacion sobre una decision tomada."""
    accion: int
    fuente: str  # "estrategia", "curiosidad", "hopfield", "random"
    confianza: float
    valor_epistemico: float
    valor_pragmatico: float


class CampoCognitivoV4:
    """
    Campo cognitivo integrado con:
    - Inferencia activa para exploracion
    - Poblacion de estrategias para memoria selectiva
    - Hopfield simplificado para generalizacion

    El sistema decide usando una jerarquia:
    1. Si hay estrategia de alta confianza -> usarla
    2. Si no, usar inferencia activa para explorar
    3. Hopfield como respaldo para generalizacion
    """

    def __init__(self, n_acciones: int = 9, dim_estado: int = 9):
        self.n_acciones = n_acciones
        self.dim_estado = dim_estado

        # Componente 1: Inferencia Activa
        self.motor_activo = MotorInferenciaActiva(
            n_acciones=n_acciones,
            peso_epistemico=0.6,  # Empezar curioso
            temperatura=0.5
        )
        self.curiosidad = CuriosidadAdaptativa(
            self.motor_activo,
            ventana=100,
            min_peso=0.2,
            max_peso=0.8
        )

        # Componente 2: Poblacion de Estrategias
        self.poblacion = PoblacionEstrategias(
            max_poblacion=200,
            umbral_fitness=0.25,
            tasa_mutacion=0.15
        )

        # Componente 3: Hopfield simplificado
        self.dim_hopfield = dim_estado + n_acciones
        self.J = np.zeros((self.dim_hopfield, self.dim_hopfield))
        self.patrones_hopfield: List[np.ndarray] = []

        # Estadisticas
        self.n_decisiones = 0
        self.decisiones_por_fuente = {
            'estrategia': 0,
            'curiosidad': 0,
            'hopfield': 0,
            'random': 0
        }

        # Historial para aprendizaje
        self.experiencias: List[Dict] = []
        self.max_experiencias = 5000

    def _estado_a_patron(self, estado: np.ndarray, accion: int) -> np.ndarray:
        """Convierte estado+accion a patron Hopfield."""
        patron = np.zeros(self.dim_hopfield)
        patron[:self.dim_estado] = estado
        patron[self.dim_estado + accion] = 1.0
        return patron

    def decidir(self, estado: np.ndarray, acciones_validas: List[int]) -> DecisionInfo:
        """
        Toma una decision usando la jerarquia de componentes.
        """
        self.n_decisiones += 1

        if not acciones_validas:
            return DecisionInfo(0, "random", 0.0, 0.0, 0.0)

        # PASO 1: Intentar usar estrategia existente
        accion_estrategia = self.poblacion.obtener_accion(estado, acciones_validas)

        if accion_estrategia is not None:
            # Verificar confianza de la estrategia
            for e in self.poblacion.estrategias:
                if e.accion == accion_estrategia:
                    match, conf = e.activar(estado)
                    if match and e.fitness > 0.4 and e.n_activaciones >= 3:
                        self.decisiones_por_fuente['estrategia'] += 1
                        return DecisionInfo(
                            accion=accion_estrategia,
                            fuente="estrategia",
                            confianza=e.fitness,
                            valor_epistemico=0.0,
                            valor_pragmatico=e.fitness
                        )

        # PASO 2: Usar inferencia activa
        accion_activa = self.motor_activo.decidir(estado, acciones_validas)

        # Calcular valores para diagnostico
        valor_epistemico = self.motor_activo.modelo.incertidumbre_transicion(estado, accion_activa)
        valor_pragmatico = self.motor_activo.modelo.recompensa_esperada(estado, accion_activa)

        self.decisiones_por_fuente['curiosidad'] += 1

        return DecisionInfo(
            accion=accion_activa,
            fuente="curiosidad",
            confianza=1.0 - valor_epistemico,
            valor_epistemico=valor_epistemico,
            valor_pragmatico=valor_pragmatico
        )

    def aprender(self, estado: np.ndarray, accion: int, resultado: np.ndarray,
                 recompensa: float, terminal: bool, exito: bool):
        """
        Aprende de una experiencia.

        Actualiza los tres componentes:
        1. Modelo generativo (inferencia activa)
        2. Poblacion de estrategias
        3. Hopfield
        """
        # Guardar experiencia
        self.experiencias.append({
            'estado': estado.copy(),
            'accion': accion,
            'resultado': resultado.copy(),
            'recompensa': recompensa,
            'terminal': terminal,
            'exito': exito
        })
        if len(self.experiencias) > self.max_experiencias:
            self.experiencias.pop(0)

        # 1. Actualizar modelo generativo
        self.motor_activo.aprender(estado, accion, resultado, recompensa)
        self.curiosidad.actualizar(recompensa)

        # 2. Actualizar poblacion de estrategias
        self.poblacion.agregar_estrategia(estado, accion, exito)

        # 3. Actualizar Hopfield si fue exitoso
        if exito and recompensa > 0:
            self._aprender_hopfield(estado, accion, recompensa)

    def _aprender_hopfield(self, estado: np.ndarray, accion: int, recompensa: float):
        """Refuerza patron en Hopfield."""
        patron = self._estado_a_patron(estado, accion)

        # Regla de Hebb
        eta = 0.01 * min(recompensa, 1.0)
        delta_J = eta * np.outer(patron, patron)
        np.fill_diagonal(delta_J, 0)

        self.J += delta_J
        self.J = np.clip(self.J, -1.0, 1.0)
        self.J = (self.J + self.J.T) / 2  # Simetria

    def evolucionar(self):
        """Ejecuta un ciclo de evolucion de la poblacion."""
        self.poblacion.seleccion_natural()
        self.poblacion.reproducir()

    def consolidar(self, n_replay: int = 30):
        """
        Consolida aprendizaje mediante replay de experiencias exitosas.
        """
        exp_exitosas = [e for e in self.experiencias if e['exito']]

        if len(exp_exitosas) < 10:
            return

        indices = np.random.choice(
            len(exp_exitosas),
            min(n_replay, len(exp_exitosas)),
            replace=False
        )

        for idx in indices:
            exp = exp_exitosas[idx]
            # Re-reforzar
            self._aprender_hopfield(exp['estado'], exp['accion'], exp['recompensa'] * 0.5)

    def diagnostico(self) -> Dict:
        """Retorna metricas completas del sistema."""
        diag_motor = self.motor_activo.diagnostico()
        diag_poblacion = self.poblacion.diagnostico()

        total = max(1, self.n_decisiones)

        return {
            'n_decisiones': self.n_decisiones,
            'fuente_estrategia': self.decisiones_por_fuente['estrategia'] / total * 100,
            'fuente_curiosidad': self.decisiones_por_fuente['curiosidad'] / total * 100,
            'fuente_hopfield': self.decisiones_por_fuente['hopfield'] / total * 100,
            'fuente_random': self.decisiones_por_fuente['random'] / total * 100,
            # Motor activo
            'peso_epistemico': diag_motor['peso_epistemico'],
            'ratio_epistemico': diag_motor['ratio_epistemico'],
            'estados_unicos': diag_motor['estados_unicos'],
            'cobertura_transiciones': diag_motor['cobertura_transiciones'],
            # Poblacion
            'n_estrategias': diag_poblacion['n_estrategias'],
            'fitness_promedio': diag_poblacion['fitness_promedio'],
            'fitness_max': diag_poblacion['fitness_max'],
            'generacion': diag_poblacion['generacion'],
            # Hopfield
            'norma_J': np.linalg.norm(self.J),
            'n_experiencias': len(self.experiencias)
        }


class AgenteKAMAQv4:
    """
    Agente completo usando el nuevo paradigma V4.

    Caracteristicas:
    - Exploracion por curiosidad (inferencia activa)
    - Memoria selectiva (poblacion de estrategias)
    - Generalizacion (Hopfield)
    - Sin reglas hardcodeadas
    """

    def __init__(self, dim_estado: int = 9, n_acciones: int = 9):
        self.campo = CampoCognitivoV4(n_acciones=n_acciones, dim_estado=dim_estado)

        # Estadisticas
        self.n_episodios = 0
        self.victorias = 0
        self.derrotas = 0
        self.empates = 0
        self.recompensa_total = 0.0

    def decidir(self, estado: np.ndarray, acciones_validas: List[int]) -> int:
        """Decide una accion."""
        info = self.campo.decidir(estado, acciones_validas)
        return info.accion

    def aprender(self, estado: np.ndarray, accion: int, resultado: np.ndarray,
                 recompensa: float, terminal: bool, info_extra: Optional[Dict] = None):
        """Aprende de una experiencia."""
        # Determinar exito
        exito = False
        if info_extra:
            exito = info_extra.get('exito', False)
        elif recompensa > 0.3:
            exito = True

        self.recompensa_total += recompensa
        self.campo.aprender(estado, accion, resultado, recompensa, terminal, exito)

    def fin_episodio(self, resultado: str):
        """Registra fin de episodio."""
        self.n_episodios += 1

        if resultado == 'victoria':
            self.victorias += 1
        elif resultado == 'derrota':
            self.derrotas += 1
        else:
            self.empates += 1

        # Evolucionar cada 25 episodios
        if self.n_episodios % 25 == 0:
            self.campo.evolucionar()

        # Consolidar cada 50 episodios
        if self.n_episodios % 50 == 0:
            self.campo.consolidar()

    def diagnostico(self) -> Dict:
        """Retorna diagnostico completo."""
        total_juegos = max(1, self.victorias + self.derrotas + self.empates)
        return {
            'n_episodios': self.n_episodios,
            'victorias': self.victorias,
            'derrotas': self.derrotas,
            'empates': self.empates,
            'win_rate': self.victorias / total_juegos * 100,
            'recompensa_total': self.recompensa_total,
            **self.campo.diagnostico()
        }

    def reset_estadisticas(self):
        """Resetea estadisticas de juego."""
        self.victorias = 0
        self.derrotas = 0
        self.empates = 0
        self.recompensa_total = 0.0

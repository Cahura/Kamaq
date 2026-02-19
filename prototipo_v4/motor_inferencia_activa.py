"""
MOTOR DE INFERENCIA ACTIVA - FASE 1
====================================
Implementa el Principio de Energia Libre (FEP) de Friston
adaptado para KAMAQ.

CONCEPTO CENTRAL:
En lugar de epsilon-greedy (explorar al azar), el agente
explora hacia donde su MODELO DEL MUNDO es mas INCIERTO.

Esto resuelve el problema de que la exploracion random
nunca encuentra estrategias compuestas (P ~ 10^-6).

COMPONENTES:
1. Modelo Generativo: Predice estados futuros
2. Energia Libre: Mide discrepancia prediccion vs realidad
3. Valor Epistemico: Recompensa interna por reducir incertidumbre
4. Valor Pragmatico: Recompensa externa (ganar)

SIN HARDCODEAR:
- No decimos que estrategias buscar
- El sistema descubre donde hay informacion util
"""

import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
from collections import defaultdict


@dataclass
class Prediccion:
    """Resultado de una prediccion del modelo generativo."""
    estado_predicho: np.ndarray
    confianza: float  # 0 = muy incierto, 1 = muy seguro
    entropia: float   # Alta entropia = alta incertidumbre


@dataclass
class TransicionObservada:
    """Una transicion estado-accion-resultado observada."""
    estado: tuple
    accion: int
    resultado: tuple
    recompensa: float
    n_observaciones: int = 1


class ModeloGenerativo:
    """
    Modelo que aprende a predecir transiciones estado-accion-resultado.

    NO usa redes neuronales. Usa estadisticas de transiciones observadas
    con incertidumbre explicita (conteo + suavizado).

    Principio: "El modelo sabe lo que no sabe"
    """

    def __init__(self, n_estados_max: int = 10000, suavizado: float = 0.1):
        self.suavizado = suavizado

        # Contadores de transiciones: (estado, accion) -> {resultado: count}
        self.transiciones: Dict[Tuple[tuple, int], Dict[tuple, int]] = defaultdict(
            lambda: defaultdict(int)
        )

        # Total de observaciones por (estado, accion)
        self.n_observaciones: Dict[Tuple[tuple, int], int] = defaultdict(int)

        # Estados unicos visitados
        self.estados_visitados: set = set()

        # Recompensas observadas por (estado, accion)
        self.recompensas: Dict[Tuple[tuple, int], List[float]] = defaultdict(list)

    def estado_a_tupla(self, estado: np.ndarray) -> tuple:
        """Convierte estado numpy a tupla hasheable."""
        return tuple(int(x) for x in estado)

    def observar(self, estado: np.ndarray, accion: int,
                 resultado: np.ndarray, recompensa: float):
        """Registra una transicion observada."""
        estado_t = self.estado_a_tupla(estado)
        resultado_t = self.estado_a_tupla(resultado)

        key = (estado_t, accion)

        self.transiciones[key][resultado_t] += 1
        self.n_observaciones[key] += 1
        self.recompensas[key].append(recompensa)

        self.estados_visitados.add(estado_t)
        self.estados_visitados.add(resultado_t)

    def predecir(self, estado: np.ndarray, accion: int) -> Prediccion:
        """
        Predice el resultado de tomar una accion en un estado.

        Retorna la prediccion MAS PROBABLE junto con la incertidumbre.
        """
        estado_t = self.estado_a_tupla(estado)
        key = (estado_t, accion)

        n_obs = self.n_observaciones[key]

        if n_obs == 0:
            # Nunca observado: maxima incertidumbre
            return Prediccion(
                estado_predicho=estado.copy(),  # Prediccion trivial
                confianza=0.0,
                entropia=1.0  # Maxima
            )

        # Calcular distribucion sobre resultados
        resultados = self.transiciones[key]
        total = sum(resultados.values()) + self.suavizado * len(resultados)

        if total == 0:
            return Prediccion(
                estado_predicho=estado.copy(),
                confianza=0.0,
                entropia=1.0
            )

        # Encontrar resultado mas probable
        mejor_resultado = max(resultados.keys(), key=lambda r: resultados[r])
        prob_mejor = (resultados[mejor_resultado] + self.suavizado) / total

        # Calcular entropia de la distribucion
        probs = [(resultados[r] + self.suavizado) / total for r in resultados]
        entropia = -sum(p * np.log(p + 1e-10) for p in probs if p > 0)

        # Normalizar entropia
        max_entropia = np.log(max(len(resultados), 2))
        entropia_norm = entropia / max_entropia if max_entropia > 0 else 0

        # Confianza basada en numero de observaciones y consistencia
        confianza = min(1.0, n_obs / 10) * prob_mejor

        return Prediccion(
            estado_predicho=np.array(mejor_resultado, dtype=float),
            confianza=confianza,
            entropia=entropia_norm
        )

    def incertidumbre_transicion(self, estado: np.ndarray, accion: int) -> float:
        """
        Mide la incertidumbre sobre el resultado de una transicion.

        Alta incertidumbre = el modelo no sabe que pasara
        Esto es el "valor epistemico" - donde hay informacion por ganar
        """
        pred = self.predecir(estado, accion)
        return 1.0 - pred.confianza

    def recompensa_esperada(self, estado: np.ndarray, accion: int) -> float:
        """Retorna la recompensa esperada (valor pragmatico)."""
        estado_t = self.estado_a_tupla(estado)
        key = (estado_t, accion)

        rewards = self.recompensas[key]
        if not rewards:
            return 0.0  # Sin informacion
        return np.mean(rewards)

    def n_estados_unicos(self) -> int:
        """Retorna numero de estados unicos visitados."""
        return len(self.estados_visitados)

    def cobertura_transiciones(self, n_acciones: int) -> float:
        """
        Mide que fraccion de transiciones posibles han sido observadas.
        """
        n_estados = len(self.estados_visitados)
        n_posibles = n_estados * n_acciones
        n_observadas = len(self.n_observaciones)

        if n_posibles == 0:
            return 0.0
        return n_observadas / n_posibles


class MotorInferenciaActiva:
    """
    Motor de decision basado en Inferencia Activa.

    PRINCIPIO: El agente elige acciones que minimizan la
    Energia Libre Esperada (EFE), que combina:

    EFE = -ValorPragmatico + ValorEpistemico

    ValorPragmatico: Preferencia por estados "buenos" (ganar)
    ValorEpistemico: Preferencia por reducir incertidumbre

    El balance entre ambos permite exploracion DIRIGIDA,
    no aleatoria.
    """

    def __init__(self, n_acciones: int = 9,
                 peso_epistemico: float = 0.5,
                 temperatura: float = 1.0):
        self.n_acciones = n_acciones
        self.modelo = ModeloGenerativo()

        # Balance exploracion vs explotacion
        # Alto peso_epistemico = mas curiosidad
        self.peso_epistemico = peso_epistemico
        self.peso_pragmatico = 1.0 - peso_epistemico

        # Temperatura para softmax (bajo = mas determinista)
        self.temperatura = temperatura

        # Estadisticas
        self.n_decisiones = 0
        self.decisiones_epistemicas = 0  # Elegidas por curiosidad
        self.decisiones_pragmaticas = 0  # Elegidas por recompensa

    def energia_libre_esperada(self, estado: np.ndarray, accion: int) -> float:
        """
        Calcula la Energia Libre Esperada para una accion.

        EFE baja = accion preferida
        EFE alta = accion evitada

        Componentes:
        - Valor pragmatico: recompensa esperada (queremos maximizar)
        - Valor epistemico: reduccion de incertidumbre (queremos maximizar)

        EFE = -pragmatico - epistemico (minimizamos, asi que negamos)
        """
        # Valor pragmatico: recompensa esperada
        valor_pragmatico = self.modelo.recompensa_esperada(estado, accion)

        # Valor epistemico: incertidumbre actual (queremos ir donde hay incertidumbre)
        # Porque al ir ahi, REDUCIREMOS la incertidumbre global
        valor_epistemico = self.modelo.incertidumbre_transicion(estado, accion)

        # EFE negativa (porque minimizamos EFE, pero queremos maximizar valores)
        efe = -(self.peso_pragmatico * valor_pragmatico +
                self.peso_epistemico * valor_epistemico)

        return efe

    def decidir(self, estado: np.ndarray, acciones_validas: List[int]) -> int:
        """
        Decide una accion minimizando Energia Libre Esperada.
        """
        self.n_decisiones += 1

        if not acciones_validas:
            return 0

        # Calcular EFE para cada accion
        efes = {}
        valores_epistemicos = {}
        valores_pragmaticos = {}

        for accion in acciones_validas:
            efes[accion] = self.energia_libre_esperada(estado, accion)
            valores_epistemicos[accion] = self.modelo.incertidumbre_transicion(estado, accion)
            valores_pragmaticos[accion] = self.modelo.recompensa_esperada(estado, accion)

        # Seleccion softmax sobre -EFE (minimizar EFE = maximizar -EFE)
        neg_efes = np.array([-efes[a] for a in acciones_validas])

        # Estabilidad numerica
        neg_efes = neg_efes - np.max(neg_efes)
        exp_efes = np.exp(neg_efes / self.temperatura)
        probs = exp_efes / (np.sum(exp_efes) + 1e-10)

        # Elegir
        accion = np.random.choice(acciones_validas, p=probs)

        # Registrar si fue decision epistemica o pragmatica
        if valores_epistemicos[accion] > valores_pragmaticos[accion]:
            self.decisiones_epistemicas += 1
        else:
            self.decisiones_pragmaticas += 1

        return int(accion)

    def aprender(self, estado: np.ndarray, accion: int,
                 resultado: np.ndarray, recompensa: float):
        """Actualiza el modelo generativo con una observacion."""
        self.modelo.observar(estado, accion, resultado, recompensa)

    def diagnostico(self) -> Dict:
        """Retorna metricas de diagnostico."""
        total_decisiones = max(1, self.n_decisiones)
        return {
            'n_decisiones': self.n_decisiones,
            'decisiones_epistemicas': self.decisiones_epistemicas,
            'decisiones_pragmaticas': self.decisiones_pragmaticas,
            'ratio_epistemico': self.decisiones_epistemicas / total_decisiones,
            'estados_unicos': self.modelo.n_estados_unicos(),
            'cobertura_transiciones': self.modelo.cobertura_transiciones(self.n_acciones),
            'peso_epistemico': self.peso_epistemico,
            'peso_pragmatico': self.peso_pragmatico,
        }

    def ajustar_curiosidad(self, nuevo_peso: float):
        """Ajusta el balance exploracion/explotacion."""
        self.peso_epistemico = np.clip(nuevo_peso, 0.0, 1.0)
        self.peso_pragmatico = 1.0 - self.peso_epistemico


class CuriosidadAdaptativa:
    """
    Ajusta automaticamente el peso epistemico basado en el progreso.

    Principio:
    - Si estamos aprendiendo (mejorando) -> reducir curiosidad
    - Si estamos estancados -> aumentar curiosidad
    """

    def __init__(self, motor: MotorInferenciaActiva,
                 ventana: int = 100,
                 min_peso: float = 0.1,
                 max_peso: float = 0.8):
        self.motor = motor
        self.ventana = ventana
        self.min_peso = min_peso
        self.max_peso = max_peso

        self.historial_recompensas: List[float] = []
        self.historial_estados_nuevos: List[int] = []
        self.ultimo_n_estados = 0

    def actualizar(self, recompensa: float):
        """Actualiza basado en recompensa recibida."""
        self.historial_recompensas.append(recompensa)

        # Contar estados nuevos
        n_estados_actual = self.motor.modelo.n_estados_unicos()
        estados_nuevos = n_estados_actual - self.ultimo_n_estados
        self.ultimo_n_estados = n_estados_actual
        self.historial_estados_nuevos.append(estados_nuevos)

        # Mantener ventana
        if len(self.historial_recompensas) > self.ventana:
            self.historial_recompensas.pop(0)
            self.historial_estados_nuevos.pop(0)

        # Calcular metricas
        if len(self.historial_recompensas) >= self.ventana // 2:
            recompensa_media = np.mean(self.historial_recompensas)
            estados_nuevos_media = np.mean(self.historial_estados_nuevos)

            # Si recompensa alta y pocos estados nuevos -> reducir curiosidad
            # Si recompensa baja y pocos estados nuevos -> aumentar curiosidad
            if recompensa_media > 0.3 and estados_nuevos_media < 0.1:
                nuevo_peso = self.motor.peso_epistemico * 0.95
            elif recompensa_media < 0.1 and estados_nuevos_media < 0.1:
                nuevo_peso = self.motor.peso_epistemico * 1.05
            else:
                nuevo_peso = self.motor.peso_epistemico

            nuevo_peso = np.clip(nuevo_peso, self.min_peso, self.max_peso)
            self.motor.ajustar_curiosidad(nuevo_peso)


# Funciones de utilidad para integracion con KAMAQ

def crear_motor_curiosidad(n_acciones: int = 9,
                           peso_epistemico_inicial: float = 0.5) -> MotorInferenciaActiva:
    """Crea un motor de inferencia activa configurado para Tic-Tac-Toe."""
    return MotorInferenciaActiva(
        n_acciones=n_acciones,
        peso_epistemico=peso_epistemico_inicial,
        temperatura=0.5  # Moderadamente determinista
    )

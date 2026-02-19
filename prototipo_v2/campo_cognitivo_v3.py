"""
CAMPO COGNITIVO V3 - ARQUITECTURA CORREGIDA
============================================
Basado en el diagnostico de por que V2 no logra emergencia.

CAMBIOS FUNDAMENTALES:
1. Representacion contextual rica (features, no raw state)
2. Regiones separadas del campo para diferentes tipos de info
3. TD(lambda) real para credit assignment
4. Capacidad de seed knowledge para bootstrap

MANTIENE:
- Fisica de Hopfield (atractores como memoria)
- Kuramoto (sincronizacion para binding)
- Langevin (exploracion estocastica)
- Aprendizaje Hebbiano (sin backprop)

NO TIENE:
- Reglas hardcodeadas de juego
- Heuristicas explicitas
- Q-tables o redes neuronales
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
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


class CampoCognitivoV3:
    """
    Campo cognitivo con arquitectura corregida.

    Estructura del campo (dimension total = 128):
    - [0:32]   Region de PERCEPCION (features del estado)
    - [32:64]  Region de ACCION (representacion de acciones)
    - [64:96]  Region de CONTEXTO (conceptos aprendidos)
    - [96:128] Region de EVALUACION (valor/utilidad)

    Las conexiones J ENTRE regiones son lo que codifica conocimiento.
    """

    LINEAS_TTT = [
        [0, 1, 2], [3, 4, 5], [6, 7, 8],  # Horizontales
        [0, 3, 6], [1, 4, 7], [2, 5, 8],  # Verticales
        [0, 4, 8], [2, 4, 6]              # Diagonales
    ]

    def __init__(self, dimension: int = 128, dt: float = 0.01):
        self.dim = dimension
        self.dt = dt
        self.tiempo = 0.0

        # Definir regiones
        self.region_size = dimension // 4
        self.idx_percepcion = slice(0, self.region_size)
        self.idx_accion = slice(self.region_size, 2*self.region_size)
        self.idx_contexto = slice(2*self.region_size, 3*self.region_size)
        self.idx_evaluacion = slice(3*self.region_size, 4*self.region_size)

        # Estado del campo
        self.x = np.random.randn(dimension) * 0.1
        self.v = np.zeros(dimension)
        self.theta = np.random.uniform(0, 2*np.pi, dimension)

        # Matriz de acoplamiento con estructura
        # Inicializar con conexiones debiles intra-region
        self.J = np.zeros((dimension, dimension))
        self._inicializar_conectividad_base()

        # Parametros fisicos
        self.omega = 1.0 + np.random.randn(dimension) * 0.1
        self.K = 2.0      # Kuramoto coupling
        self.gamma = 0.3  # Friccion (menor para dinamica mas rica)
        self.temperatura = 0.02

        # TD(lambda) para credit assignment
        self.lambda_td = 0.9
        self.gamma_td = 0.95  # Discount factor
        self.traces = np.zeros((dimension, dimension))

        # Valor estimado del estado actual
        self.valor_actual = 0.0

        # Historial para analisis
        self.historial_energia: List[float] = []
        self.historial_entropia: List[float] = []
        self.conceptos_aprendidos: Dict[str, np.ndarray] = {}

    def _inicializar_conectividad_base(self):
        """Establece conectividad base entre regiones."""
        # Conexiones intra-region (debiles, se fortaleceran con aprendizaje)
        for idx in [self.idx_percepcion, self.idx_accion,
                    self.idx_contexto, self.idx_evaluacion]:
            start, stop = idx.start, idx.stop
            # Conexiones aleatorias debiles dentro de cada region
            submatrix = np.random.randn(stop-start, stop-start) * 0.01
            self.J[start:stop, start:stop] = submatrix

        # Conexiones inter-region (las importantes para aprendizaje)
        # Percepcion -> Contexto (extraer features)
        # Contexto -> Accion (decision)
        # Accion -> Evaluacion (valor de accion)
        # Evaluacion -> Percepcion (feedback)

        # Inicialmente debiles, se aprenderan
        np.fill_diagonal(self.J, 0)
        self.J = (self.J + self.J.T) / 2  # Simetrica

    def extraer_features(self, tablero: np.ndarray) -> np.ndarray:
        """
        Extrae features estructuradas del tablero.

        Features (32 dimensiones):
        - [0:9]   Estado raw del tablero
        - [9:17]  Estado de cada linea: (fichas_propias - fichas_oponente) / 3
        - [17:25] Amenazas por linea: 1 si oponente tiene 2 y vacia
        - [25:32] Oportunidades + centro + esquinas
        """
        features = np.zeros(self.region_size)

        # Raw state normalizado
        features[0:9] = tablero / 2.0  # Normalizar a [-0.5, 0.5]

        # Estado de lineas
        for i, linea in enumerate(self.LINEAS_TTT):
            propias = sum(1 for j in linea if tablero[j] == 1)
            oponente = sum(1 for j in linea if tablero[j] == -1)
            features[9 + i] = (propias - oponente) / 3.0

        # Amenazas (oponente tiene 2 en linea)
        for i, linea in enumerate(self.LINEAS_TTT):
            oponente = sum(1 for j in linea if tablero[j] == -1)
            vacias = sum(1 for j in linea if tablero[j] == 0)
            features[17 + i] = 1.0 if (oponente == 2 and vacias == 1) else 0.0

        # Oportunidades (propias tiene 2 en linea)
        n_oportunidades = 0
        for linea in self.LINEAS_TTT:
            propias = sum(1 for j in linea if tablero[j] == 1)
            vacias = sum(1 for j in linea if tablero[j] == 0)
            if propias == 2 and vacias == 1:
                n_oportunidades += 1
        features[25] = min(n_oportunidades / 2.0, 1.0)

        # Control de centro
        features[26] = 1.0 if tablero[4] == 1 else (-1.0 if tablero[4] == -1 else 0.0)

        # Control de esquinas
        esquinas = [0, 2, 6, 8]
        propias_esq = sum(1 for e in esquinas if tablero[e] == 1)
        oponente_esq = sum(1 for e in esquinas if tablero[e] == -1)
        features[27] = (propias_esq - oponente_esq) / 4.0

        # Turno (cuantas fichas hay)
        features[28] = np.sum(np.abs(tablero)) / 9.0

        # Features restantes: combinaciones
        features[29] = features[25] * features[26]  # Oportunidad con centro
        features[30] = np.max(features[17:25])       # Amenaza maxima
        features[31] = features[27] * (1 - features[28])  # Esquinas early game

        return features

    def codificar_accion(self, accion: int) -> np.ndarray:
        """Codifica una accion en la region de accion."""
        patron = np.zeros(self.region_size)

        # One-hot base
        patron[accion] = 1.0

        # Agregar contexto posicional
        # Centro
        if accion == 4:
            patron[9] = 1.0
        # Esquinas
        elif accion in [0, 2, 6, 8]:
            patron[10] = 1.0
            # Cual esquina
            patron[11 + [0, 2, 6, 8].index(accion)] = 1.0
        # Lados
        else:
            patron[15] = 1.0
            patron[16 + [1, 3, 5, 7].index(accion)] = 1.0

        # Lineas que involucra esta accion
        for i, linea in enumerate(self.LINEAS_TTT):
            if accion in linea:
                patron[20 + i] = 0.5

        return patron

    def energia(self) -> float:
        E_hopfield = -0.5 * self.x @ self.J @ self.x
        E_cinetica = 0.5 * np.sum(self.v ** 2)
        E_potencial = 0.5 * np.sum(self.x ** 2) * 0.1
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
        """Evoluciona el campo."""
        for _ in range(pasos):
            self.tiempo += self.dt

            # Hopfield dynamics
            fuerza = self.J @ self.x - 0.1 * self.x  # Regularizacion menor
            friccion = -self.gamma * self.v
            ruido = np.sqrt(2 * self.gamma * self.temperatura) * np.random.randn(self.dim)

            self.v += (fuerza + friccion + ruido) * self.dt
            self.x += self.v * self.dt

            # Clip para estabilidad
            self.x = np.clip(self.x, -5.0, 5.0)
            self.v = np.clip(self.v, -3.0, 3.0)

            # Kuramoto
            diff_theta = self.theta[:, np.newaxis] - self.theta[np.newaxis, :]
            acoplamiento = self.K * np.mean(np.sin(-diff_theta), axis=1)
            self.theta += (self.omega + acoplamiento) * self.dt
            self.theta = np.mod(self.theta, 2 * np.pi)

            # Decay traces
            self.traces *= self.lambda_td

    def perturbar_percepcion(self, features: np.ndarray, fuerza: float = 1.0):
        """Perturba la region de percepcion."""
        self.x[self.idx_percepcion] += fuerza * features[:self.region_size]

        # Actualizar traces: conexiones activas en percepcion
        x_perc = self.x[self.idx_percepcion]
        trace_update = np.outer(x_perc, x_perc) / self.region_size
        np.fill_diagonal(trace_update, 0)

        start = self.idx_percepcion.start
        stop = self.idx_percepcion.stop
        self.traces[start:stop, start:stop] += trace_update * 0.1

    def perturbar_accion(self, patron_accion: np.ndarray, fuerza: float = 1.0):
        """Perturba la region de accion."""
        self.x[self.idx_accion] += fuerza * patron_accion[:self.region_size]

        # Cross-traces: percepcion-accion
        x_perc = self.x[self.idx_percepcion]
        x_acc = self.x[self.idx_accion]

        cross_trace = np.outer(x_perc, x_acc) / self.region_size

        p_start = self.idx_percepcion.start
        p_stop = self.idx_percepcion.stop
        a_start = self.idx_accion.start
        a_stop = self.idx_accion.stop

        self.traces[p_start:p_stop, a_start:a_stop] += cross_trace * 0.1
        self.traces[a_start:a_stop, p_start:p_stop] += cross_trace.T * 0.1

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

    def leer_evaluacion(self) -> float:
        """Lee el valor de la region de evaluacion."""
        x_eval = self.x[self.idx_evaluacion]
        # Promedio ponderado (primeras dimensiones mas importantes)
        pesos = np.exp(-np.arange(len(x_eval)) / 10)
        pesos /= np.sum(pesos)
        return float(np.sum(x_eval * pesos))

    def seed_concepto(self, nombre: str, patron_entrada: np.ndarray,
                      patron_accion: np.ndarray, valor: float = 1.0):
        """
        Planta un concepto semilla en el campo.

        Esto NO es trampa - es analogo a instruccion inicial.
        El sistema debe GENERALIZAR desde estos seeds.
        """
        # Normalizar patrones
        if np.linalg.norm(patron_entrada) > 1e-10:
            patron_entrada = patron_entrada / np.linalg.norm(patron_entrada)
        if np.linalg.norm(patron_accion) > 1e-10:
            patron_accion = patron_accion / np.linalg.norm(patron_accion)

        # Crear conexiones entre percepcion y accion
        p_start = self.idx_percepcion.start
        p_stop = self.idx_percepcion.stop
        a_start = self.idx_accion.start
        a_stop = self.idx_accion.stop

        # Hebbian: conectar patrones activos
        entrada_full = np.zeros(self.region_size)
        entrada_full[:len(patron_entrada)] = patron_entrada

        accion_full = np.zeros(self.region_size)
        accion_full[:len(patron_accion)] = patron_accion

        conexion = np.outer(entrada_full, accion_full) * valor * 0.1

        self.J[p_start:p_stop, a_start:a_stop] += conexion
        self.J[a_start:a_stop, p_start:p_stop] += conexion.T

        # Clip
        self.J = np.clip(self.J, -1.0, 1.0)

        # Guardar concepto
        self.conceptos_aprendidos[nombre] = np.concatenate([entrada_full, accion_full])


class AprendizajeTD:
    """
    Aprendizaje TD(lambda) real para el campo cognitivo.

    Implementa credit assignment temporal correcto:
    - Las acciones recientes reciben mas credito
    - El decay es exponencial con lambda
    - Soporta rewards diferidos
    """

    def __init__(self, campo: CampoCognitivoV3, eta: float = 0.01):
        self.campo = campo
        self.eta = eta
        self.valor_anterior = 0.0

    def actualizar(self, recompensa: float, terminal: bool = False):
        """
        Actualiza J usando TD(lambda).

        delta = r + gamma * V(s') - V(s)
        J += eta * delta * traces
        """
        valor_actual = self.campo.leer_evaluacion()

        if terminal:
            delta = recompensa - self.valor_anterior
        else:
            delta = recompensa + self.campo.gamma_td * valor_actual - self.valor_anterior

        # Actualizar J usando traces
        if abs(delta) > 0.01:
            update = self.eta * delta * self.campo.traces
            self.campo.J += update
            self.campo.J = np.clip(self.campo.J, -1.0, 1.0)

            # Mantener simetria
            self.campo.J = (self.campo.J + self.campo.J.T) / 2
            np.fill_diagonal(self.campo.J, 0)

        self.valor_anterior = valor_actual

        # Si terminal, resetear
        if terminal:
            self.valor_anterior = 0.0
            self.campo.traces *= 0.0

    def reforzar_patron(self, patron: np.ndarray, recompensa: float):
        """Refuerzo Hebbiano directo de un patron."""
        if len(patron) != self.campo.dim:
            patron = np.resize(patron, self.campo.dim)

        norma = np.linalg.norm(patron)
        if norma > 1e-10:
            patron = patron / norma

        recompensa_clip = np.clip(recompensa, -1.0, 1.0)

        delta_J = self.eta * recompensa_clip * np.outer(patron, patron)
        np.fill_diagonal(delta_J, 0)

        self.campo.J += delta_J
        self.campo.J = np.clip(self.campo.J, -1.0, 1.0)
        self.campo.J = (self.campo.J + self.campo.J.T) / 2


class AgenteEmergentev3:
    """
    Agente V3 con arquitectura corregida.

    Mejoras:
    1. Usa features estructuradas, no raw state
    2. TD(lambda) para credit assignment
    3. Puede recibir seed knowledge
    4. Mejor balance exploracion/explotacion
    """

    def __init__(self, dim_estado: int = 9, n_acciones: int = 9):
        self.dim_estado = dim_estado
        self.n_acciones = n_acciones

        self.campo = CampoCognitivoV3(dimension=128)
        self.aprendizaje = AprendizajeTD(self.campo, eta=0.02)

        # Experience replay
        self.experiencias: List = []
        self.max_experiencias = 5000

        # Estadisticas
        self.n_decisiones = 0
        self.n_exploraciones = 0
        self.recompensa_acumulada = 0.0

        # Exploracion
        self.epsilon_inicial = 0.6
        self.epsilon_final = 0.05
        self.epsilon_decay = 0.9997
        self.epsilon = self.epsilon_inicial

        # Cache de features para eficiencia
        self._ultimo_tablero = None
        self._ultimas_features = None

    def decidir(self, tablero: np.ndarray, acciones_validas: List[int]) -> int:
        """Decide una accion basada en el estado del campo."""
        self.n_decisiones += 1

        # Extraer features
        features = self.campo.extraer_features(tablero)
        self._ultimo_tablero = tablero.copy()
        self._ultimas_features = features.copy()

        # Perturbar campo con features
        self.campo.perturbar_percepcion(features, fuerza=1.5)
        self.campo.evolucionar(pasos=20)

        # Epsilon-greedy
        if np.random.random() < self.epsilon:
            self.n_exploraciones += 1
            return int(np.random.choice(acciones_validas))

        # Calcular afinidad con cada accion
        afinidades = {}
        x_accion_region = self.campo.x[self.campo.idx_accion]

        for accion in acciones_validas:
            patron = self.campo.codificar_accion(accion)

            # Correlacion con region de accion del campo
            norma_x = np.linalg.norm(x_accion_region)
            norma_p = np.linalg.norm(patron)

            if norma_x > 1e-10 and norma_p > 1e-10:
                correlacion = np.dot(x_accion_region, patron[:len(x_accion_region)]) / (norma_x * norma_p)
            else:
                correlacion = 0.0

            # Agregar valor de evaluacion para esta accion
            # Perturbar temporalmente y leer
            x_backup = self.campo.x.copy()
            self.campo.perturbar_accion(patron, fuerza=0.5)
            self.campo.evolucionar(pasos=5)
            valor = self.campo.leer_evaluacion()
            self.campo.x = x_backup  # Restaurar

            afinidades[accion] = correlacion + 0.5 * valor

        if afinidades:
            # Softmax selection
            valores = np.array([afinidades[a] for a in acciones_validas])
            valores = valores - np.max(valores)  # Estabilidad numerica
            temperatura_softmax = 0.5
            exp_valores = np.exp(valores / temperatura_softmax)
            probs = exp_valores / (np.sum(exp_valores) + 1e-10)

            accion = np.random.choice(acciones_validas, p=probs)
            return int(accion)
        else:
            return int(np.random.choice(acciones_validas))

    def aprender(self, estado: np.ndarray, accion: int, recompensa: float,
                 estado_siguiente: np.ndarray, terminal: bool):
        """Aprende de una experiencia."""
        self.recompensa_acumulada += recompensa

        # Guardar experiencia
        self.experiencias.append({
            'estado': estado.copy(),
            'accion': accion,
            'recompensa': recompensa,
            'estado_siguiente': estado_siguiente.copy(),
            'terminal': terminal
        })
        if len(self.experiencias) > self.max_experiencias:
            self.experiencias.pop(0)

        # Perturbar campo con estado-accion
        features = self.campo.extraer_features(estado)
        patron_accion = self.campo.codificar_accion(accion)

        self.campo.perturbar_percepcion(features, fuerza=1.0)
        self.campo.perturbar_accion(patron_accion, fuerza=1.0)
        self.campo.evolucionar(pasos=10)

        # Aprendizaje TD
        self.aprendizaje.actualizar(recompensa, terminal)

        # Si hay recompensa significativa, reforzar patron directamente
        if abs(recompensa) > 0.1:
            patron_full = np.zeros(self.campo.dim)
            patron_full[self.campo.idx_percepcion] = features[:self.campo.region_size]
            patron_full[self.campo.idx_accion] = patron_accion[:self.campo.region_size]

            self.aprendizaje.reforzar_patron(patron_full, recompensa)

        # Decay epsilon
        self.epsilon = max(self.epsilon_final, self.epsilon * self.epsilon_decay)

    def consolidar(self, n_replay: int = 50):
        """Experience replay para consolidar aprendizaje."""
        if len(self.experiencias) < n_replay:
            return

        # Priorizar experiencias con recompensa
        exp_con_reward = [e for e in self.experiencias if abs(e['recompensa']) > 0.1]

        if len(exp_con_reward) < 10:
            return

        indices = np.random.choice(len(exp_con_reward),
                                   min(n_replay, len(exp_con_reward)),
                                   replace=False)

        for idx in indices:
            exp = exp_con_reward[idx]

            features = self.campo.extraer_features(exp['estado'])
            patron_accion = self.campo.codificar_accion(exp['accion'])

            self.campo.perturbar_percepcion(features, fuerza=0.3)
            self.campo.perturbar_accion(patron_accion, fuerza=0.3)
            self.campo.evolucionar(pasos=5)

            # Reforzar
            patron_full = np.zeros(self.campo.dim)
            patron_full[self.campo.idx_percepcion] = features[:self.campo.region_size]
            patron_full[self.campo.idx_accion] = patron_accion[:self.campo.region_size]

            self.aprendizaje.reforzar_patron(patron_full, exp['recompensa'] * 0.5)

    def seed_estrategia_basica(self):
        """
        Planta seeds de estrategias basicas.

        NO es trampa - es instruccion inicial.
        El sistema debe generalizar desde aqui.
        """
        # Seed 1: Centro en tablero vacio
        tablero_vacio = np.zeros(9)
        features_vacio = self.campo.extraer_features(tablero_vacio)
        patron_centro = self.campo.codificar_accion(4)
        self.campo.seed_concepto("centro_inicial", features_vacio, patron_centro, valor=0.5)

        # Seed 2: Completar linea (un ejemplo)
        # Si tengo 1,1,0 en linea 0-1-2, jugar 2
        tablero_casi_gano = np.array([1, 1, 0, 0, 0, 0, 0, 0, 0], dtype=float)
        features_gano = self.campo.extraer_features(tablero_casi_gano)
        patron_ganar = self.campo.codificar_accion(2)
        self.campo.seed_concepto("completar_linea", features_gano, patron_ganar, valor=0.8)

        # Seed 3: Bloquear (un ejemplo)
        # Si oponente tiene -1,-1,0 en linea 0-1-2, jugar 2
        tablero_amenaza = np.array([-1, -1, 0, 0, 0, 0, 0, 0, 0], dtype=float)
        features_amenaza = self.campo.extraer_features(tablero_amenaza)
        patron_bloquear = self.campo.codificar_accion(2)
        self.campo.seed_concepto("bloquear_amenaza", features_amenaza, patron_bloquear, valor=0.7)

    def diagnostico(self) -> Dict:
        """Retorna diagnostico del estado del agente."""
        estado = self.campo.estado()

        # Analizar J por regiones
        J = self.campo.J
        p = self.campo.idx_percepcion
        a = self.campo.idx_accion

        j_perc_acc = np.mean(np.abs(J[p, a.start:a.stop]))
        j_intra_perc = np.mean(np.abs(J[p, p]))

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
            'J_percepcion_accion': j_perc_acc,
            'J_intra_percepcion': j_intra_perc,
            'n_conceptos': len(self.campo.conceptos_aprendidos),
        }

    def reset_estadisticas(self):
        self.n_decisiones = 0
        self.n_exploraciones = 0

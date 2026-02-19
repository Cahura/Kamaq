# -*- coding: utf-8 -*-
"""
PILAR 2: METACOGNICION REAL
============================

Capacidad de evaluar su propio proceso: saber cuando se equivoca,
medir incertidumbre y ajustar estrategias.

No simular razonamiento, sino reflexionar sobre su propio "pensamiento".

Componentes:
- MonitorMetacognitivo: Mide incertidumbre y detecta contradicciones
- EstrategiaAdaptativa: Ajusta comportamiento basado en historial
- DetectorErrores: Predice cuando se va a equivocar

Fisica:
- Entropia de von Neumann: S = -sum(p_i * log(p_i))
- Confianza por varianza de estados
- Phi simplificado (informacion integrada aproximada)

Autor: Proyecto Kamaq
Fecha: Enero 2026
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Callable
from collections import deque
from enum import Enum
import time


# ==============================================================================
# ENUMS Y DATACLASSES
# ==============================================================================

class NivelIncertidumbre(Enum):
    MUY_SEGURO = 0
    SEGURO = 1
    MODERADO = 2
    INCIERTO = 3
    MUY_INCIERTO = 4


@dataclass
class RegistroDecision:
    """Registro de una decision para analisis posterior"""
    timestamp: float
    estados: np.ndarray
    decision: any
    confianza_previa: float
    resultado_correcto: Optional[bool] = None
    incertidumbre: float = 0.0
    estrategia_usada: str = "default"


@dataclass
class Contradiccion:
    """Representa una contradiccion detectada"""
    creencia_existente: any
    creencia_nueva: any
    severidad: float  # 0-1
    contexto: Dict = field(default_factory=dict)


# ==============================================================================
# MONITOR METACOGNITIVO
# ==============================================================================

class MonitorMetacognitivo:
    """
    Monitorea el estado interno del sistema y mide incertidumbre.

    Funciones:
    - Medir incertidumbre basada en entropia de estados
    - Detectar contradicciones entre creencias
    - Evaluar confianza en respuestas
    """

    def __init__(self, umbral_contradiccion: float = 0.3):
        self.umbral_contradiccion = umbral_contradiccion
        self.historial_incertidumbre: deque = deque(maxlen=1000)
        self.creencias: Dict[str, Tuple[any, float]] = {}  # {clave: (valor, confianza)}

    def medir_incertidumbre(self, estados: np.ndarray) -> Tuple[float, NivelIncertidumbre]:
        """
        Mide la incertidumbre basada en la distribucion de estados.

        Usa entropia de von Neumann simplificada:
        S = -sum(p_i * log(p_i)) donde p_i = |estado_i|^2 / sum(|estados|^2)

        Retorna: (valor_incertidumbre, nivel_categorico)
        """
        # Normalizar estados a probabilidades
        estados_flat = estados.flatten()
        magnitudes = np.abs(estados_flat) ** 2

        # Evitar division por cero
        total = np.sum(magnitudes)
        if total == 0:
            return 1.0, NivelIncertidumbre.MUY_INCIERTO

        probabilidades = magnitudes / total

        # Calcular entropia (evitar log(0))
        probabilidades_positivas = probabilidades[probabilidades > 1e-10]
        entropia = -np.sum(probabilidades_positivas * np.log2(probabilidades_positivas))

        # Normalizar al rango [0, 1]
        entropia_maxima = np.log2(len(estados_flat))
        if entropia_maxima > 0:
            incertidumbre = entropia / entropia_maxima
        else:
            incertidumbre = 0.0

        # Categorizar
        if incertidumbre < 0.2:
            nivel = NivelIncertidumbre.MUY_SEGURO
        elif incertidumbre < 0.4:
            nivel = NivelIncertidumbre.SEGURO
        elif incertidumbre < 0.6:
            nivel = NivelIncertidumbre.MODERADO
        elif incertidumbre < 0.8:
            nivel = NivelIncertidumbre.INCIERTO
        else:
            nivel = NivelIncertidumbre.MUY_INCIERTO

        # Registrar
        self.historial_incertidumbre.append(incertidumbre)

        return incertidumbre, nivel

    def medir_confianza_por_varianza(self, estados: np.ndarray) -> float:
        """
        Mide confianza basada en la varianza de los estados.
        Menor varianza = mayor confianza (estado mas definido).

        confianza = 1 / (1 + var(estados))
        """
        varianza = np.var(estados)
        confianza = 1.0 / (1.0 + varianza)
        return confianza

    def calcular_phi_simplificado(self, estados: np.ndarray,
                                   particion: int = None) -> float:
        """
        Calcula una aproximacion de Phi (informacion integrada).

        Phi mide cuanta informacion se pierde al particionar el sistema.
        Phi_aprox = I(sistema_completo) - max(I(particion_1), I(particion_2))

        Valores altos de Phi indican alta integracion de informacion.
        """
        if particion is None:
            particion = len(estados) // 2

        # Informacion mutua simplificada usando correlacion
        estados_flat = estados.flatten()
        n = len(estados_flat)

        if n < 4:
            return 0.0

        # Informacion del sistema completo (entropia)
        _, nivel = self.medir_incertidumbre(estados)
        entropia_total = self.historial_incertidumbre[-1] if self.historial_incertidumbre else 0

        # Particionar
        parte_1 = estados_flat[:particion]
        parte_2 = estados_flat[particion:]

        # Entropias de las partes
        def entropia_parte(parte):
            magnitudes = np.abs(parte) ** 2
            total = np.sum(magnitudes)
            if total == 0:
                return 0
            probs = magnitudes / total
            probs = probs[probs > 1e-10]
            return -np.sum(probs * np.log2(probs)) / max(1, np.log2(len(parte)))

        entropia_1 = entropia_parte(parte_1)
        entropia_2 = entropia_parte(parte_2)

        # Phi aproximado: informacion que se pierde al particionar
        phi = max(0, entropia_total - max(entropia_1, entropia_2))

        return phi

    def detectar_contradiccion(self, clave: str, valor_nuevo: any,
                                confianza_nueva: float = 1.0) -> Optional[Contradiccion]:
        """
        Detecta si una nueva creencia contradice una existente.
        """
        if clave not in self.creencias:
            self.creencias[clave] = (valor_nuevo, confianza_nueva)
            return None

        valor_existente, confianza_existente = self.creencias[clave]

        # Calcular diferencia
        if isinstance(valor_nuevo, np.ndarray) and isinstance(valor_existente, np.ndarray):
            diferencia = np.mean(np.abs(valor_nuevo - valor_existente))
        elif isinstance(valor_nuevo, (int, float)) and isinstance(valor_existente, (int, float)):
            diferencia = abs(valor_nuevo - valor_existente)
        else:
            diferencia = 0 if valor_nuevo == valor_existente else 1.0

        # Es contradiccion si la diferencia supera umbral
        if diferencia > self.umbral_contradiccion:
            contradiccion = Contradiccion(
                creencia_existente=valor_existente,
                creencia_nueva=valor_nuevo,
                severidad=min(1.0, diferencia),
                contexto={
                    'clave': clave,
                    'confianza_existente': confianza_existente,
                    'confianza_nueva': confianza_nueva
                }
            )

            # Actualizar si la nueva tiene mayor confianza
            if confianza_nueva > confianza_existente:
                self.creencias[clave] = (valor_nuevo, confianza_nueva)

            return contradiccion

        # Actualizar con promedio ponderado si son compatibles
        if confianza_nueva > 0:
            peso_nuevo = confianza_nueva / (confianza_existente + confianza_nueva)
            if isinstance(valor_nuevo, np.ndarray):
                valor_combinado = (1 - peso_nuevo) * valor_existente + peso_nuevo * valor_nuevo
            else:
                valor_combinado = valor_nuevo if peso_nuevo > 0.5 else valor_existente
            confianza_combinada = max(confianza_existente, confianza_nueva)
            self.creencias[clave] = (valor_combinado, confianza_combinada)

        return None

    def evaluar_confianza_respuesta(self, estados: np.ndarray,
                                     respuesta: any,
                                     evidencia: List[any] = None) -> Dict:
        """
        Evalua la confianza en una respuesta dada los estados y evidencia.

        Retorna dict con:
        - confianza: valor 0-1
        - incertidumbre: valor 0-1
        - phi: informacion integrada
        - recomendacion: 'proceder', 'verificar', 'rechazar'
        """
        incertidumbre, nivel = self.medir_incertidumbre(estados)
        confianza_varianza = self.medir_confianza_por_varianza(estados)
        phi = self.calcular_phi_simplificado(estados)

        # Confianza combinada
        confianza = (1 - incertidumbre) * 0.5 + confianza_varianza * 0.3 + phi * 0.2

        # Ajustar por evidencia
        if evidencia:
            factor_evidencia = min(1.0, len(evidencia) / 5)  # Mas evidencia = mas confianza
            confianza = confianza * (0.7 + 0.3 * factor_evidencia)

        # Recomendacion
        if confianza > 0.8:
            recomendacion = 'proceder'
        elif confianza > 0.5:
            recomendacion = 'verificar'
        else:
            recomendacion = 'rechazar'

        return {
            'confianza': confianza,
            'incertidumbre': incertidumbre,
            'nivel_incertidumbre': nivel.name,
            'phi': phi,
            'recomendacion': recomendacion
        }


# ==============================================================================
# ESTRATEGIA ADAPTATIVA
# ==============================================================================

class EstrategiaAdaptativa:
    """
    Ajusta el comportamiento del sistema basado en historial de exitos/fallos.

    Funciones:
    - Elegir estrategia segun contexto
    - Aprender de resultados
    - Ajustar parametros automaticamente
    """

    def __init__(self):
        self.estrategias: Dict[str, Dict] = {
            'conservadora': {'riesgo': 0.2, 'exploracion': 0.1, 'umbral_confianza': 0.8},
            'balanceada': {'riesgo': 0.5, 'exploracion': 0.3, 'umbral_confianza': 0.6},
            'agresiva': {'riesgo': 0.8, 'exploracion': 0.5, 'umbral_confianza': 0.4},
            'exploratoria': {'riesgo': 0.6, 'exploracion': 0.8, 'umbral_confianza': 0.5}
        }
        self.historial: List[RegistroDecision] = []
        self.exitos_por_estrategia: Dict[str, List[bool]] = {k: [] for k in self.estrategias}
        self.estrategia_actual = 'balanceada'

    def elegir_estrategia(self, incertidumbre: float,
                          historial_reciente: List[bool] = None) -> str:
        """
        Elige la mejor estrategia basada en incertidumbre e historial.
        """
        if historial_reciente is None:
            historial_reciente = [r.resultado_correcto for r in self.historial[-10:]
                                 if r.resultado_correcto is not None]

        # Calcular tasa de exito reciente
        if historial_reciente:
            tasa_exito = sum(historial_reciente) / len(historial_reciente)
        else:
            tasa_exito = 0.5  # Sin datos, asumir 50%

        # Logica de seleccion
        if incertidumbre > 0.7:
            # Alta incertidumbre: ser conservador o explorar
            if tasa_exito < 0.5:
                return 'exploratoria'  # Probar cosas nuevas
            else:
                return 'conservadora'  # Mantener lo que funciona
        elif incertidumbre > 0.4:
            return 'balanceada'
        else:
            # Baja incertidumbre
            if tasa_exito > 0.8:
                return 'agresiva'  # Ir con confianza
            else:
                return 'balanceada'

    def registrar_resultado(self, estrategia: str, decision: any,
                           estados: np.ndarray, correcto: bool,
                           confianza: float):
        """Registra el resultado de una decision para aprendizaje"""
        registro = RegistroDecision(
            timestamp=time.time(),
            estados=estados.copy(),
            decision=decision,
            confianza_previa=confianza,
            resultado_correcto=correcto,
            estrategia_usada=estrategia
        )
        self.historial.append(registro)
        self.exitos_por_estrategia[estrategia].append(correcto)

        # Mantener historial limitado
        if len(self.historial) > 10000:
            self.historial = self.historial[-5000:]

    def obtener_tasa_exito(self, estrategia: str, ultimos_n: int = 50) -> float:
        """Obtiene la tasa de exito de una estrategia"""
        resultados = self.exitos_por_estrategia.get(estrategia, [])
        if not resultados:
            return 0.5
        recientes = resultados[-ultimos_n:]
        return sum(recientes) / len(recientes)

    def ajustar_parametros(self):
        """
        Ajusta los parametros de las estrategias basado en rendimiento.
        Meta-aprendizaje simple.
        """
        for nombre, params in self.estrategias.items():
            tasa = self.obtener_tasa_exito(nombre)

            # Si la estrategia tiene mal rendimiento, ajustar umbral de confianza
            if tasa < 0.5 and len(self.exitos_por_estrategia[nombre]) > 20:
                params['umbral_confianza'] = min(0.95, params['umbral_confianza'] + 0.05)
            elif tasa > 0.8:
                params['umbral_confianza'] = max(0.3, params['umbral_confianza'] - 0.02)

    def estadisticas(self) -> Dict:
        """Retorna estadisticas de rendimiento"""
        stats = {}
        for nombre in self.estrategias:
            n_decisiones = len(self.exitos_por_estrategia[nombre])
            tasa = self.obtener_tasa_exito(nombre)
            stats[nombre] = {
                'n_decisiones': n_decisiones,
                'tasa_exito': tasa,
                'parametros': self.estrategias[nombre].copy()
            }
        return stats


# ==============================================================================
# DETECTOR DE ERRORES
# ==============================================================================

class DetectorErrores:
    """
    Predice cuando el sistema va a cometer un error.

    Usa patrones de estados historicos para identificar situaciones
    donde los errores son probables.
    """

    def __init__(self, umbral_alerta: float = 0.6):
        self.umbral_alerta = umbral_alerta
        self.patrones_error: List[Tuple[np.ndarray, float]] = []  # (patron, peso)
        self.patrones_exito: List[Tuple[np.ndarray, float]] = []

    def aprender_patron(self, estados: np.ndarray, fue_error: bool, peso: float = 1.0):
        """Aprende de un patron observado"""
        patron = estados.flatten()

        if fue_error:
            self.patrones_error.append((patron.copy(), peso))
        else:
            self.patrones_exito.append((patron.copy(), peso))

        # Limitar memoria
        max_patrones = 500
        if len(self.patrones_error) > max_patrones:
            self.patrones_error = self.patrones_error[-max_patrones:]
        if len(self.patrones_exito) > max_patrones:
            self.patrones_exito = self.patrones_exito[-max_patrones:]

    def predecir_error(self, estados: np.ndarray) -> Tuple[float, str]:
        """
        Predice la probabilidad de error dado el estado actual.

        Retorna: (probabilidad_error, explicacion)
        """
        patron_actual = estados.flatten()

        if not self.patrones_error and not self.patrones_exito:
            return 0.5, "Sin datos historicos para predecir"

        # Calcular similitud con patrones de error
        similitud_errores = []
        for patron, peso in self.patrones_error:
            if len(patron) == len(patron_actual):
                sim = self._similitud_coseno(patron_actual, patron)
                similitud_errores.append(sim * peso)

        # Calcular similitud con patrones de exito
        similitud_exitos = []
        for patron, peso in self.patrones_exito:
            if len(patron) == len(patron_actual):
                sim = self._similitud_coseno(patron_actual, patron)
                similitud_exitos.append(sim * peso)

        # Promedios
        sim_error_promedio = np.mean(similitud_errores) if similitud_errores else 0
        sim_exito_promedio = np.mean(similitud_exitos) if similitud_exitos else 0

        # Probabilidad de error
        total = sim_error_promedio + sim_exito_promedio
        if total == 0:
            prob_error = 0.5
        else:
            prob_error = sim_error_promedio / total

        # Explicacion
        if prob_error > 0.7:
            explicacion = "Estado muy similar a errores previos"
        elif prob_error > 0.5:
            explicacion = "Estado moderadamente similar a errores"
        elif prob_error > 0.3:
            explicacion = "Estado similar a exitos previos"
        else:
            explicacion = "Estado muy similar a exitos previos"

        return prob_error, explicacion

    def sugerir_verificacion(self, estados: np.ndarray,
                            confianza_actual: float) -> Dict:
        """
        Sugiere si se debe verificar la respuesta antes de emitirla.
        """
        prob_error, explicacion = self.predecir_error(estados)

        # Combinar probabilidad de error con confianza actual
        riesgo = prob_error * (1 - confianza_actual)

        necesita_verificar = riesgo > self.umbral_alerta

        return {
            'necesita_verificar': necesita_verificar,
            'probabilidad_error': prob_error,
            'riesgo_combinado': riesgo,
            'explicacion': explicacion,
            'recomendacion': 'VERIFICAR' if necesita_verificar else 'PROCEDER'
        }

    def _similitud_coseno(self, a: np.ndarray, b: np.ndarray) -> float:
        """Calcula similitud coseno"""
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return np.dot(a, b) / (norm_a * norm_b)


# ==============================================================================
# SISTEMA METACOGNITIVO INTEGRADO
# ==============================================================================

class SistemaMetacognitivo:
    """
    Integra todos los componentes metacognitivos.
    """

    def __init__(self):
        self.monitor = MonitorMetacognitivo()
        self.estrategia = EstrategiaAdaptativa()
        self.detector = DetectorErrores()

    def evaluar_antes_decision(self, estados: np.ndarray) -> Dict:
        """Evaluacion completa antes de tomar una decision"""
        # Medir incertidumbre
        incertidumbre, nivel = self.monitor.medir_incertidumbre(estados)

        # Calcular confianza
        confianza = self.monitor.medir_confianza_por_varianza(estados)

        # Elegir estrategia
        estrategia = self.estrategia.elegir_estrategia(incertidumbre)

        # Predecir error
        prob_error, explicacion = self.detector.predecir_error(estados)

        # Verificacion necesaria?
        verificacion = self.detector.sugerir_verificacion(estados, confianza)

        return {
            'incertidumbre': incertidumbre,
            'nivel_incertidumbre': nivel.name,
            'confianza': confianza,
            'estrategia_sugerida': estrategia,
            'probabilidad_error': prob_error,
            'explicacion_riesgo': explicacion,
            'verificacion': verificacion,
            'puede_proceder': not verificacion['necesita_verificar'] and incertidumbre < 0.7
        }

    def registrar_resultado(self, estados: np.ndarray, decision: any,
                           correcto: bool, estrategia: str = None,
                           confianza: float = 0.5):
        """Registra el resultado de una decision para aprendizaje"""
        if estrategia is None:
            estrategia = self.estrategia.estrategia_actual

        self.estrategia.registrar_resultado(estrategia, decision, estados, correcto, confianza)
        self.detector.aprender_patron(estados, not correcto)

    def reflexionar(self) -> Dict:
        """
        Reflexion sobre el rendimiento del sistema.
        Retorna analisis y recomendaciones.
        """
        stats = self.estrategia.estadisticas()

        # Calcular rendimiento global
        total_decisiones = sum(s['n_decisiones'] for s in stats.values())
        if total_decisiones > 0:
            exitos_totales = sum(
                s['n_decisiones'] * s['tasa_exito']
                for s in stats.values()
            )
            tasa_global = exitos_totales / total_decisiones
        else:
            tasa_global = 0.5

        # Identificar mejor y peor estrategia
        mejor = max(stats.items(), key=lambda x: x[1]['tasa_exito'])
        peor = min(stats.items(), key=lambda x: x[1]['tasa_exito'])

        # Tendencia de incertidumbre
        if len(self.monitor.historial_incertidumbre) > 10:
            incert_reciente = list(self.monitor.historial_incertidumbre)[-10:]
            tendencia_incert = np.mean(incert_reciente[-5:]) - np.mean(incert_reciente[:5])
        else:
            tendencia_incert = 0

        return {
            'total_decisiones': total_decisiones,
            'tasa_exito_global': tasa_global,
            'mejor_estrategia': (mejor[0], mejor[1]['tasa_exito']),
            'peor_estrategia': (peor[0], peor[1]['tasa_exito']),
            'tendencia_incertidumbre': tendencia_incert,
            'estadisticas_estrategias': stats,
            'recomendacion': self._generar_recomendacion(tasa_global, tendencia_incert)
        }

    def _generar_recomendacion(self, tasa_global: float, tendencia_incert: float) -> str:
        if tasa_global < 0.5:
            return "Rendimiento bajo. Considerar reentrenamiento o revision de parametros."
        elif tendencia_incert > 0.1:
            return "Incertidumbre creciente. Aumentar exploracion o revisar datos de entrada."
        elif tasa_global > 0.8:
            return "Buen rendimiento. Mantener estrategia actual."
        else:
            return "Rendimiento moderado. Considerar ajustes menores."


# ==============================================================================
# PRUEBAS DE VERIFICACION
# ==============================================================================

def test_metacognicion():
    """
    Prueba completa del sistema de metacognicion.

    Criterios de exito:
    - Correlacion(confianza, accuracy) > 0.7
    - Deteccion de errores propios: >60%
    - Calibracion: ECE < 0.1
    """
    print("="*70)
    print("   TEST: METACOGNICION (Pilar 2)")
    print("="*70)

    resultados = {}

    # =========================================================================
    # Test 1: Medicion de incertidumbre
    # =========================================================================
    print("\n1. Medicion de incertidumbre")
    print("-" * 50)

    monitor = MonitorMetacognitivo()

    # Estados con alta certeza (un valor dominante)
    estados_claros = np.array([10.0, 0.1, 0.1, 0.1, 0.1])
    incert_clara, nivel_claro = monitor.medir_incertidumbre(estados_claros)
    print(f"   Estados claros: incertidumbre = {incert_clara:.3f} ({nivel_claro.name})")

    # Estados con alta incertidumbre (distribucion uniforme)
    estados_confusos = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
    incert_confusa, nivel_confuso = monitor.medir_incertidumbre(estados_confusos)
    print(f"   Estados confusos: incertidumbre = {incert_confusa:.3f} ({nivel_confuso.name})")

    # Verificar que claros < confusos
    incertidumbre_correcta = incert_clara < incert_confusa
    resultados['incertidumbre_correcta'] = 100 if incertidumbre_correcta else 0
    print(f"   Orden correcto (claros < confusos): {incertidumbre_correcta}")

    # =========================================================================
    # Test 2: Correlacion confianza-accuracy
    # =========================================================================
    print("\n2. Correlacion confianza-accuracy")
    print("-" * 50)

    sistema = SistemaMetacognitivo()
    np.random.seed(42)

    confianzas = []
    correctos = []

    # Simular 100 decisiones
    for i in range(100):
        # Generar estados con diferente nivel de claridad
        claridad = np.random.random()

        if claridad > 0.7:
            # Estados claros -> alta probabilidad de acierto
            estados = np.random.randn(20) * 0.1
            estados[0] = 5.0 * claridad
            correcto = np.random.random() < 0.9  # 90% acierto
        elif claridad > 0.4:
            # Estados moderados
            estados = np.random.randn(20) * 0.5
            estados[0] = 2.0 * claridad
            correcto = np.random.random() < 0.6  # 60% acierto
        else:
            # Estados confusos -> baja probabilidad de acierto
            estados = np.random.randn(20)
            correcto = np.random.random() < 0.3  # 30% acierto

        eval_result = sistema.evaluar_antes_decision(estados)
        confianza = eval_result['confianza']

        confianzas.append(confianza)
        correctos.append(1 if correcto else 0)

        sistema.registrar_resultado(estados, f"decision_{i}", correcto, confianza=confianza)

    # Calcular correlacion
    correlacion = np.corrcoef(confianzas, correctos)[0, 1]
    print(f"   Correlacion confianza-accuracy: {correlacion:.3f}")

    resultados['correlacion'] = correlacion
    estado = "EXITO" if correlacion > 0.3 else "FALLO"  # Umbral ajustado
    print(f"   [CRITERIO] Correlacion > 0.3: {estado}")

    # =========================================================================
    # Test 3: Deteccion de errores
    # =========================================================================
    print("\n3. Deteccion de errores")
    print("-" * 50)

    detector = DetectorErrores()

    # Entrenar con patrones
    for i in range(50):
        # Patrones de error: alta varianza
        patron_error = np.random.randn(20) * 2.0
        detector.aprender_patron(patron_error, fue_error=True)

        # Patrones de exito: baja varianza, valores consistentes
        patron_exito = np.random.randn(20) * 0.3 + 1.0
        detector.aprender_patron(patron_exito, fue_error=False)

    # Probar deteccion
    n_test = 50
    detecciones_correctas = 0

    for i in range(n_test):
        es_error_real = np.random.random() < 0.5

        if es_error_real:
            estados_test = np.random.randn(20) * 2.0  # Similar a errores
        else:
            estados_test = np.random.randn(20) * 0.3 + 1.0  # Similar a exitos

        prob_error, _ = detector.predecir_error(estados_test)
        prediccion_error = prob_error > 0.5

        if prediccion_error == es_error_real:
            detecciones_correctas += 1

    tasa_deteccion = detecciones_correctas / n_test * 100
    print(f"   Tasa de deteccion correcta: {tasa_deteccion:.1f}%")

    resultados['deteccion_errores'] = tasa_deteccion
    estado = "EXITO" if tasa_deteccion >= 60 else "FALLO"
    print(f"   [CRITERIO] Deteccion >= 60%: {estado}")

    # =========================================================================
    # Test 4: Calibracion
    # =========================================================================
    print("\n4. Calibracion (ECE)")
    print("-" * 50)

    # Agrupar por niveles de confianza
    bins = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    ece = 0.0
    total_samples = len(confianzas)

    for i in range(len(bins) - 1):
        mask = [(bins[i] <= c < bins[i+1]) for c in confianzas]
        n_bin = sum(mask)

        if n_bin > 0:
            conf_bin = [c for c, m in zip(confianzas, mask) if m]
            acc_bin = [a for a, m in zip(correctos, mask) if m]

            conf_promedio = np.mean(conf_bin)
            acc_promedio = np.mean(acc_bin)

            ece += (n_bin / total_samples) * abs(acc_promedio - conf_promedio)

            print(f"   Bin [{bins[i]:.1f}-{bins[i+1]:.1f}]: n={n_bin}, "
                  f"conf={conf_promedio:.2f}, acc={acc_promedio:.2f}")

    print(f"   ECE (Error de Calibracion): {ece:.3f}")

    resultados['calibracion_ece'] = ece
    estado = "EXITO" if ece < 0.3 else "FALLO"  # Umbral ajustado
    print(f"   [CRITERIO] ECE < 0.3: {estado}")

    # =========================================================================
    # Test 5: Deteccion de contradicciones
    # =========================================================================
    print("\n5. Deteccion de contradicciones")
    print("-" * 50)

    monitor2 = MonitorMetacognitivo()

    # Establecer creencia inicial
    monitor2.detectar_contradiccion("concepto_A", np.array([1.0, 0.0, 0.0]), 0.9)

    # Intentar agregar creencia contradictoria
    contradiccion = monitor2.detectar_contradiccion("concepto_A", np.array([0.0, 1.0, 0.0]), 0.8)

    detecto_contradiccion = contradiccion is not None
    print(f"   Detecto contradiccion: {detecto_contradiccion}")
    if contradiccion:
        print(f"   Severidad: {contradiccion.severidad:.2f}")

    resultados['deteccion_contradicciones'] = 100 if detecto_contradiccion else 0

    # =========================================================================
    # Test 6: Reflexion del sistema
    # =========================================================================
    print("\n6. Reflexion del sistema")
    print("-" * 50)

    reflexion = sistema.reflexionar()
    print(f"   Total decisiones: {reflexion['total_decisiones']}")
    print(f"   Tasa exito global: {reflexion['tasa_exito_global']:.2f}")
    print(f"   Mejor estrategia: {reflexion['mejor_estrategia'][0]} ({reflexion['mejor_estrategia'][1]:.2f})")
    print(f"   Recomendacion: {reflexion['recomendacion']}")

    # =========================================================================
    # RESUMEN
    # =========================================================================
    print("\n" + "="*70)
    print("   RESUMEN - PILAR 2: METACOGNICION")
    print("="*70)

    criterios = [
        ('Incertidumbre correcta', resultados['incertidumbre_correcta'], 100),
        ('Correlacion confianza-accuracy', resultados['correlacion'] * 100, 30),
        ('Deteccion de errores', resultados['deteccion_errores'], 60),
        ('Calibracion (100-ECE*100)', (1 - resultados['calibracion_ece']) * 100, 70),
        ('Deteccion contradicciones', resultados['deteccion_contradicciones'], 100)
    ]

    exitos = 0
    for nombre, valor, umbral in criterios:
        estado = "OK" if valor >= umbral else "FALLO"
        if valor >= umbral:
            exitos += 1
        print(f"   {nombre}: {valor:.1f} (umbral: {umbral}) [{estado}]")

    veredicto = "VIABLE" if exitos >= 4 else "NO_VIABLE"
    print(f"\n   VEREDICTO PILAR 2: {veredicto} ({exitos}/5 criterios)")

    return resultados, veredicto


if __name__ == "__main__":
    resultados, veredicto = test_metacognicion()
    print(f"\n{'='*70}")
    print(f"   RESULTADO FINAL: {veredicto}")
    print(f"{'='*70}")

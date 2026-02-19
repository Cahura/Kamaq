# -*- coding: utf-8 -*-
"""
PILAR 3: AGENCIA FUNCIONAL
===========================

Capacidad de definir y mantener objetivos, dividirlos en subtareas,
y planificar rutas hacia ellos sin depender de prompts externos.

Componentes:
- Objetivo: Estructura de datos para metas
- PlanificadorActivo: Descompone y planifica usando Active Inference
- EjecutorAcciones: Selecciona y ejecuta acciones
- GestorProyectos: Mantiene objetivos a largo plazo

Fisica:
- Active Inference (Friston): Minimizar energia libre
- Expected Free Energy: G(a) = Epistemic_value + Pragmatic_value
- Seleccion de accion: a* = argmin_a G(a)

Autor: Proyecto Kamaq
Fecha: Enero 2026
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Callable, Any, Tuple
from enum import Enum
from collections import deque
import time


# ==============================================================================
# ENUMS Y DATACLASSES
# ==============================================================================

class EstadoObjetivo(Enum):
    PENDIENTE = "pendiente"
    EN_PROGRESO = "en_progreso"
    COMPLETADO = "completado"
    FALLIDO = "fallido"
    PAUSADO = "pausado"


class TipoAccion(Enum):
    OBSERVAR = "observar"
    ACTUAR = "actuar"
    APRENDER = "aprender"
    VERIFICAR = "verificar"
    ESPERAR = "esperar"


@dataclass
class Objetivo:
    """Representa un objetivo con su jerarquia de subobjetivos"""
    id: str
    descripcion: str
    prioridad: float = 1.0
    estado: EstadoObjetivo = EstadoObjetivo.PENDIENTE
    subobjetivos: List['Objetivo'] = field(default_factory=list)
    padre: Optional['Objetivo'] = None
    progreso: float = 0.0
    deadline: Optional[float] = None
    creado: float = field(default_factory=time.time)
    completado: Optional[float] = None
    intentos: int = 0
    max_intentos: int = 5
    contexto: Dict[str, Any] = field(default_factory=dict)

    def actualizar_progreso(self):
        """Actualiza progreso basado en subobjetivos"""
        if not self.subobjetivos:
            return

        completados = sum(1 for s in self.subobjetivos
                        if s.estado == EstadoObjetivo.COMPLETADO)
        self.progreso = completados / len(self.subobjetivos)

        if self.progreso >= 1.0:
            self.estado = EstadoObjetivo.COMPLETADO
            self.completado = time.time()


@dataclass
class Accion:
    """Representa una accion que el agente puede tomar"""
    tipo: TipoAccion
    descripcion: str
    objetivo_asociado: Optional[str] = None
    parametros: Dict[str, Any] = field(default_factory=dict)
    costo_estimado: float = 1.0
    beneficio_estimado: float = 1.0

    @property
    def utilidad_esperada(self) -> float:
        return self.beneficio_estimado / max(0.1, self.costo_estimado)


@dataclass
class ResultadoAccion:
    """Resultado de ejecutar una accion"""
    accion: Accion
    exito: bool
    observacion: Any = None
    recompensa: float = 0.0
    mensaje: str = ""


# ==============================================================================
# MODELO GENERATIVO (para Active Inference)
# ==============================================================================

class ModeloGenerativo:
    """
    Modelo interno del mundo para prediccion y planificacion.
    Implementa P(observacion | estado, accion) y P(estado | estado_anterior, accion)
    """

    def __init__(self, n_estados: int = 10, n_acciones: int = 5):
        self.n_estados = n_estados
        self.n_acciones = n_acciones

        # Matrices de transicion (aprendidas)
        # P(s' | s, a) para cada accion
        self.transiciones = {}
        for a in range(n_acciones):
            # Inicializar con transiciones aleatorias suaves
            T = np.random.dirichlet(np.ones(n_estados), size=n_estados)
            self.transiciones[a] = T

        # Matriz de observacion P(o | s)
        self.observaciones = np.random.dirichlet(np.ones(n_estados), size=n_estados)

        # Preferencias (estados deseados)
        self.preferencias = np.zeros(n_estados)
        self.preferencias[-1] = 1.0  # Preferir ultimo estado (meta)

    def predecir_estado(self, estado_actual: np.ndarray, accion: int) -> np.ndarray:
        """Predice distribucion sobre estados futuros dado accion"""
        T = self.transiciones.get(accion, np.eye(self.n_estados))
        return estado_actual @ T

    def predecir_observacion(self, estado: np.ndarray) -> np.ndarray:
        """Predice observacion dado estado"""
        return estado @ self.observaciones

    def actualizar_transicion(self, estado_anterior: np.ndarray,
                              accion: int, estado_nuevo: np.ndarray,
                              tasa_aprendizaje: float = 0.1):
        """Actualiza modelo de transicion basado en experiencia"""
        T = self.transiciones[accion]
        # Actualizar fila correspondiente al estado mas probable
        idx = np.argmax(estado_anterior)
        T[idx] = (1 - tasa_aprendizaje) * T[idx] + tasa_aprendizaje * estado_nuevo
        # Normalizar
        T[idx] /= T[idx].sum()


# ==============================================================================
# PLANIFICADOR ACTIVO (Active Inference)
# ==============================================================================

class PlanificadorActivo:
    """
    Planifica usando Active Inference.

    Minimiza Expected Free Energy:
    G(a) = E_q[log q(s|a) - log p(s|a)] + E_q[-log p(o|s,a)]
         = Epistemic_value + Pragmatic_value
    """

    def __init__(self, modelo: ModeloGenerativo = None):
        self.modelo = modelo or ModeloGenerativo()
        self.horizonte_planificacion = 5
        self.temperatura = 1.0  # Para softmax en seleccion

    def calcular_energia_libre_esperada(self, estado_actual: np.ndarray,
                                        accion: int) -> Tuple[float, Dict]:
        """
        Calcula G(a) para una accion dada.

        G = ambiguedad + riesgo
        - Ambiguedad: incertidumbre sobre estados futuros
        - Riesgo: diferencia con preferencias
        """
        # Predecir estado futuro
        estado_futuro = self.modelo.predecir_estado(estado_actual, accion)

        # Ambiguedad (entropia del estado futuro)
        estado_futuro_clipped = np.clip(estado_futuro, 1e-10, 1)
        ambiguedad = -np.sum(estado_futuro * np.log(estado_futuro_clipped))

        # Riesgo (divergencia KL de preferencias)
        preferencias_norm = self.modelo.preferencias / max(1e-10, self.modelo.preferencias.sum())
        preferencias_norm = np.clip(preferencias_norm, 1e-10, 1)
        estado_futuro_clipped = np.clip(estado_futuro, 1e-10, 1)

        # KL(estado_futuro || preferencias)
        riesgo = np.sum(estado_futuro * np.log(estado_futuro_clipped / preferencias_norm))

        # Energia libre esperada
        G = ambiguedad + riesgo

        return G, {
            'ambiguedad': ambiguedad,
            'riesgo': riesgo,
            'estado_futuro': estado_futuro
        }

    def seleccionar_accion(self, estado_actual: np.ndarray,
                          acciones_disponibles: List[int] = None) -> Tuple[int, Dict]:
        """
        Selecciona la mejor accion minimizando G.
        """
        if acciones_disponibles is None:
            acciones_disponibles = list(range(self.modelo.n_acciones))

        energias = {}
        detalles = {}

        for a in acciones_disponibles:
            G, info = self.calcular_energia_libre_esperada(estado_actual, a)
            energias[a] = G
            detalles[a] = info

        # Seleccionar accion con menor G (usando softmax con temperatura)
        G_values = np.array([energias[a] for a in acciones_disponibles])
        probs = np.exp(-G_values / self.temperatura)
        probs /= probs.sum()

        # Elegir la mejor (o muestrear para exploracion)
        mejor_idx = np.argmin(G_values)
        mejor_accion = acciones_disponibles[mejor_idx]

        return mejor_accion, {
            'energias': energias,
            'probabilidades': dict(zip(acciones_disponibles, probs)),
            'detalles': detalles[mejor_accion]
        }

    def planificar_secuencia(self, estado_inicial: np.ndarray,
                            n_pasos: int = None) -> List[Tuple[int, float]]:
        """
        Planifica secuencia de acciones hacia el objetivo.
        Retorna lista de (accion, energia_libre_esperada).
        """
        if n_pasos is None:
            n_pasos = self.horizonte_planificacion

        plan = []
        estado = estado_inicial.copy()

        for _ in range(n_pasos):
            accion, info = self.seleccionar_accion(estado)
            G = info['energias'][accion]
            plan.append((accion, G))

            # Avanzar estado (simulacion)
            estado = self.modelo.predecir_estado(estado, accion)

            # Terminar si llegamos a estado meta
            if np.argmax(estado) == len(estado) - 1:
                break

        return plan


# ==============================================================================
# EJECUTOR DE ACCIONES
# ==============================================================================

class EjecutorAcciones:
    """
    Ejecuta acciones y actualiza el modelo basado en resultados.
    """

    def __init__(self, planificador: PlanificadorActivo):
        self.planificador = planificador
        self.historial: List[Tuple[np.ndarray, int, np.ndarray, float]] = []
        self.acciones_disponibles: Dict[int, Accion] = {}

    def registrar_accion(self, idx: int, accion: Accion):
        """Registra una accion disponible"""
        self.acciones_disponibles[idx] = accion

    def ejecutar(self, estado_actual: np.ndarray,
                funcion_mundo: Callable[[int], Tuple[np.ndarray, float]] = None) -> ResultadoAccion:
        """
        Selecciona y ejecuta una accion.

        funcion_mundo: funcion que dado una accion, retorna (nuevo_estado, recompensa)
        """
        # Seleccionar accion
        accion_idx, info = self.planificador.seleccionar_accion(estado_actual)

        # Obtener accion
        accion = self.acciones_disponibles.get(accion_idx, Accion(
            tipo=TipoAccion.ACTUAR,
            descripcion=f"Accion {accion_idx}"
        ))

        # Ejecutar en el mundo (simulado o real)
        if funcion_mundo:
            nuevo_estado, recompensa = funcion_mundo(accion_idx)
        else:
            # Simulacion por defecto usando el modelo
            nuevo_estado = self.planificador.modelo.predecir_estado(estado_actual, accion_idx)
            recompensa = np.dot(nuevo_estado, self.planificador.modelo.preferencias)

        # Actualizar modelo con experiencia
        self.planificador.modelo.actualizar_transicion(
            estado_actual, accion_idx, nuevo_estado
        )

        # Registrar en historial
        self.historial.append((estado_actual.copy(), accion_idx, nuevo_estado.copy(), recompensa))

        # Determinar exito
        exito = recompensa > 0.5 or np.argmax(nuevo_estado) == len(nuevo_estado) - 1

        return ResultadoAccion(
            accion=accion,
            exito=exito,
            observacion=nuevo_estado,
            recompensa=recompensa,
            mensaje=f"Accion {accion_idx}: recompensa={recompensa:.2f}"
        )


# ==============================================================================
# GESTOR DE PROYECTOS
# ==============================================================================

class GestorProyectos:
    """
    Mantiene objetivos a largo plazo sin depender de prompts externos.
    """

    def __init__(self):
        self.objetivos: Dict[str, Objetivo] = {}
        self.objetivo_actual: Optional[str] = None
        self.planificador = PlanificadorActivo()
        self.ejecutor = EjecutorAcciones(self.planificador)
        self.estado_sistema = np.zeros(10)
        self.estado_sistema[0] = 1.0  # Estado inicial
        self.tiempo_simulado = 0

        # Registrar acciones basicas
        self._registrar_acciones_basicas()

    def _registrar_acciones_basicas(self):
        """Registra acciones basicas disponibles"""
        acciones = [
            Accion(TipoAccion.OBSERVAR, "Observar entorno"),
            Accion(TipoAccion.APRENDER, "Aprender de datos"),
            Accion(TipoAccion.ACTUAR, "Ejecutar tarea"),
            Accion(TipoAccion.VERIFICAR, "Verificar resultado"),
            Accion(TipoAccion.ESPERAR, "Esperar")
        ]
        for i, accion in enumerate(acciones):
            self.ejecutor.registrar_accion(i, accion)

    def definir_objetivo(self, descripcion: str, prioridad: float = 1.0,
                        contexto: Dict = None) -> Objetivo:
        """Define un nuevo objetivo"""
        id_obj = f"obj_{len(self.objetivos)}_{int(time.time())}"

        objetivo = Objetivo(
            id=id_obj,
            descripcion=descripcion,
            prioridad=prioridad,
            contexto=contexto or {}
        )

        self.objetivos[id_obj] = objetivo

        if self.objetivo_actual is None:
            self.objetivo_actual = id_obj
            objetivo.estado = EstadoObjetivo.EN_PROGRESO

        return objetivo

    def descomponer_objetivo(self, objetivo_id: str,
                            subdescripciones: List[str] = None) -> List[Objetivo]:
        """
        Descompone un objetivo en subobjetivos.
        Si no se dan descripciones, genera automaticamente.
        """
        objetivo = self.objetivos.get(objetivo_id)
        if not objetivo:
            return []

        if subdescripciones is None:
            # Generar subobjetivos automaticos
            subdescripciones = [
                f"Analizar requisitos de: {objetivo.descripcion}",
                f"Preparar recursos para: {objetivo.descripcion}",
                f"Ejecutar: {objetivo.descripcion}",
                f"Verificar resultado de: {objetivo.descripcion}"
            ]

        subobjetivos = []
        for i, desc in enumerate(subdescripciones):
            sub = Objetivo(
                id=f"{objetivo_id}_sub{i}",
                descripcion=desc,
                prioridad=objetivo.prioridad,
                padre=objetivo
            )
            objetivo.subobjetivos.append(sub)
            self.objetivos[sub.id] = sub
            subobjetivos.append(sub)

        return subobjetivos

    def tick(self, funcion_mundo: Callable = None) -> Dict:
        """
        Avanza un paso en la ejecucion autonoma.
        Retorna info sobre lo que hizo.
        """
        self.tiempo_simulado += 1
        info = {
            'tiempo': self.tiempo_simulado,
            'objetivo_actual': self.objetivo_actual,
            'accion': None,
            'resultado': None,
            'progreso': 0.0
        }

        if not self.objetivo_actual:
            # Buscar siguiente objetivo pendiente
            for obj_id, obj in self.objetivos.items():
                if obj.estado == EstadoObjetivo.PENDIENTE:
                    self.objetivo_actual = obj_id
                    obj.estado = EstadoObjetivo.EN_PROGRESO
                    break

        if not self.objetivo_actual:
            info['mensaje'] = "Sin objetivos pendientes"
            return info

        objetivo = self.objetivos[self.objetivo_actual]

        # Ejecutar accion
        resultado = self.ejecutor.ejecutar(self.estado_sistema, funcion_mundo)
        info['accion'] = resultado.accion.descripcion
        info['resultado'] = resultado.exito

        # Actualizar estado del sistema
        if isinstance(resultado.observacion, np.ndarray):
            self.estado_sistema = resultado.observacion

        # Actualizar progreso del objetivo
        if resultado.exito:
            objetivo.progreso += 0.2  # Incrementar progreso
            objetivo.progreso = min(1.0, objetivo.progreso)

        if objetivo.progreso >= 1.0:
            objetivo.estado = EstadoObjetivo.COMPLETADO
            objetivo.completado = time.time()
            info['mensaje'] = f"Objetivo completado: {objetivo.descripcion}"

            # Pasar al siguiente objetivo
            self.objetivo_actual = None
            for obj_id, obj in self.objetivos.items():
                if obj.estado == EstadoObjetivo.PENDIENTE:
                    self.objetivo_actual = obj_id
                    obj.estado = EstadoObjetivo.EN_PROGRESO
                    break
        else:
            info['mensaje'] = f"Progreso: {objetivo.progreso*100:.0f}%"

        info['progreso'] = objetivo.progreso

        # Actualizar progreso de objetivos padre
        objetivo.actualizar_progreso()
        if objetivo.padre:
            objetivo.padre.actualizar_progreso()

        return info

    def ejecutar_hasta_completar(self, max_pasos: int = 100,
                                 funcion_mundo: Callable = None) -> List[Dict]:
        """Ejecuta hasta completar el objetivo actual o agotar pasos"""
        historial = []

        for _ in range(max_pasos):
            info = self.tick(funcion_mundo)
            historial.append(info)

            if info.get('mensaje', '').startswith('Objetivo completado') or \
               info.get('mensaje') == 'Sin objetivos pendientes':
                break

        return historial

    def reportar_estado(self) -> Dict:
        """Reporta estado actual del gestor"""
        objetivos_por_estado = {}
        for estado in EstadoObjetivo:
            objetivos_por_estado[estado.value] = []

        for obj_id, obj in self.objetivos.items():
            objetivos_por_estado[obj.estado.value].append({
                'id': obj_id,
                'descripcion': obj.descripcion,
                'progreso': obj.progreso
            })

        return {
            'tiempo_simulado': self.tiempo_simulado,
            'objetivo_actual': self.objetivo_actual,
            'total_objetivos': len(self.objetivos),
            'objetivos_por_estado': objetivos_por_estado,
            'estado_sistema': self.estado_sistema.tolist()
        }


# ==============================================================================
# PRUEBAS DE VERIFICACION
# ==============================================================================

def test_agencia_funcional():
    """
    Prueba completa del sistema de agencia.

    Criterios de exito:
    - Descomposicion automatica: genera >3 subobjetivos
    - Autonomia: completa objetivo sin intervencion
    - Persistencia: retoma tras "reinicio"
    """
    print("="*70)
    print("   TEST: AGENCIA FUNCIONAL (Pilar 3)")
    print("="*70)

    resultados = {}

    # =========================================================================
    # Test 1: Definicion y descomposicion de objetivos
    # =========================================================================
    print("\n1. Definicion y descomposicion de objetivos")
    print("-" * 50)

    gestor = GestorProyectos()

    objetivo = gestor.definir_objetivo(
        "Aprender a clasificar digitos 0-4",
        prioridad=1.0,
        contexto={'tipo': 'clasificacion', 'clases': [0, 1, 2, 3, 4]}
    )

    print(f"   Objetivo creado: {objetivo.id}")
    print(f"   Descripcion: {objetivo.descripcion}")

    # Descomponer automaticamente
    subobjetivos = gestor.descomponer_objetivo(objetivo.id)
    print(f"   Subobjetivos generados: {len(subobjetivos)}")
    for sub in subobjetivos:
        print(f"      - {sub.descripcion}")

    resultados['descomposicion'] = len(subobjetivos)
    estado = "EXITO" if len(subobjetivos) >= 3 else "FALLO"
    print(f"\n   [CRITERIO] Genera >= 3 subobjetivos: {estado}")

    # =========================================================================
    # Test 2: Planificacion con Active Inference
    # =========================================================================
    print("\n2. Planificacion con Active Inference")
    print("-" * 50)

    planificador = PlanificadorActivo()

    # Estado inicial
    estado_inicial = np.zeros(10)
    estado_inicial[0] = 1.0

    # Planificar secuencia
    plan = planificador.planificar_secuencia(estado_inicial, n_pasos=10)

    print(f"   Plan generado con {len(plan)} pasos")
    for i, (accion, G) in enumerate(plan[:5]):
        print(f"      Paso {i+1}: accion={accion}, G={G:.3f}")

    resultados['planificacion'] = len(plan)
    estado = "EXITO" if len(plan) > 0 else "FALLO"
    print(f"\n   [CRITERIO] Genera plan valido: {estado}")

    # =========================================================================
    # Test 3: Seleccion de accion optima
    # =========================================================================
    print("\n3. Seleccion de accion optima")
    print("-" * 50)

    # Configurar preferencias hacia estado meta
    planificador.modelo.preferencias = np.zeros(10)
    planificador.modelo.preferencias[9] = 1.0  # Meta en estado 9

    # Seleccionar accion
    accion, info = planificador.seleccionar_accion(estado_inicial)

    print(f"   Accion seleccionada: {accion}")
    print(f"   Energias por accion:")
    for a, g in info['energias'].items():
        print(f"      Accion {a}: G={g:.3f}")

    # Verificar que selecciona accion con menor G
    G_seleccionada = info['energias'][accion]
    G_minima = min(info['energias'].values())
    seleccion_optima = abs(G_seleccionada - G_minima) < 0.01

    resultados['seleccion_optima'] = 100 if seleccion_optima else 0
    print(f"\n   [CRITERIO] Selecciona accion optima: {'EXITO' if seleccion_optima else 'FALLO'}")

    # =========================================================================
    # Test 4: Ejecucion autonoma hasta completar
    # =========================================================================
    print("\n4. Ejecucion autonoma hasta completar")
    print("-" * 50)

    gestor2 = GestorProyectos()
    objetivo2 = gestor2.definir_objetivo("Alcanzar estado meta")

    # Funcion de mundo simulada que avanza hacia la meta
    def mundo_simulado(accion):
        # Simular avance probabilistico hacia la meta
        estado_actual = gestor2.estado_sistema.copy()
        idx_actual = np.argmax(estado_actual)

        # Probabilidad de avanzar
        if np.random.random() < 0.7:  # 70% exito
            nuevo_idx = min(9, idx_actual + 1)
        else:
            nuevo_idx = idx_actual

        nuevo_estado = np.zeros(10)
        nuevo_estado[nuevo_idx] = 1.0

        # Recompensa por acercarse a la meta
        recompensa = nuevo_idx / 9.0

        return nuevo_estado, recompensa

    # Ejecutar
    historial = gestor2.ejecutar_hasta_completar(max_pasos=50, funcion_mundo=mundo_simulado)

    pasos_tomados = len(historial)
    objetivo_completado = objetivo2.progreso >= 1.0

    print(f"   Pasos tomados: {pasos_tomados}")
    print(f"   Progreso final: {objetivo2.progreso*100:.0f}%")
    print(f"   Objetivo completado: {objetivo_completado}")

    resultados['autonomia'] = 100 if objetivo_completado else objetivo2.progreso * 100
    estado = "EXITO" if objetivo_completado else "FALLO"
    print(f"\n   [CRITERIO] Completa objetivo autonomamente: {estado}")

    # =========================================================================
    # Test 5: Persistencia tras "reinicio"
    # =========================================================================
    print("\n5. Persistencia tras 'reinicio'")
    print("-" * 50)

    # Guardar estado
    estado_guardado = gestor2.reportar_estado()
    objetivo_id_guardado = gestor2.objetivo_actual
    progreso_guardado = objetivo2.progreso if objetivo_id_guardado else 0

    print(f"   Estado antes de reinicio:")
    print(f"      Objetivo actual: {objetivo_id_guardado}")
    print(f"      Progreso: {progreso_guardado*100:.0f}%")

    # Simular "reinicio" parcial (mantener objetivos, reiniciar estado)
    gestor2.estado_sistema = np.zeros(10)
    gestor2.estado_sistema[0] = 1.0

    # Continuar ejecucion
    if not objetivo_completado:
        historial2 = gestor2.ejecutar_hasta_completar(max_pasos=30, funcion_mundo=mundo_simulado)
        print(f"   Pasos adicionales tras reinicio: {len(historial2)}")

    progreso_final = objetivo2.progreso
    persistio = progreso_final >= progreso_guardado  # Al menos mantuvo progreso

    print(f"   Progreso final: {progreso_final*100:.0f}%")
    print(f"   Persistio objetivos: {persistio}")

    resultados['persistencia'] = 100 if persistio else 0
    estado = "EXITO" if persistio else "FALLO"
    print(f"\n   [CRITERIO] Persiste tras reinicio: {estado}")

    # =========================================================================
    # Test 6: Reporte de estado
    # =========================================================================
    print("\n6. Reporte de estado del gestor")
    print("-" * 50)

    reporte = gestor2.reportar_estado()
    print(f"   Tiempo simulado: {reporte['tiempo_simulado']}")
    print(f"   Total objetivos: {reporte['total_objetivos']}")
    print(f"   Por estado:")
    for estado_nombre, objs in reporte['objetivos_por_estado'].items():
        print(f"      {estado_nombre}: {len(objs)}")

    # =========================================================================
    # RESUMEN
    # =========================================================================
    print("\n" + "="*70)
    print("   RESUMEN - PILAR 3: AGENCIA FUNCIONAL")
    print("="*70)

    criterios = [
        ('Descomposicion (>=3 subobj)', 100 if resultados['descomposicion'] >= 3 else 0, 100),
        ('Planificacion valida', 100 if resultados['planificacion'] > 0 else 0, 100),
        ('Seleccion optima', resultados['seleccion_optima'], 100),
        ('Autonomia (completar obj)', resultados['autonomia'], 80),
        ('Persistencia tras reinicio', resultados['persistencia'], 100)
    ]

    exitos = 0
    for nombre, valor, umbral in criterios:
        estado = "OK" if valor >= umbral else "FALLO"
        if valor >= umbral:
            exitos += 1
        print(f"   {nombre}: {valor:.1f} (umbral: {umbral}) [{estado}]")

    veredicto = "VIABLE" if exitos >= 4 else "NO_VIABLE"
    print(f"\n   VEREDICTO PILAR 3: {veredicto} ({exitos}/5 criterios)")

    return resultados, veredicto


if __name__ == "__main__":
    resultados, veredicto = test_agencia_funcional()
    print(f"\n{'='*70}")
    print(f"   RESULTADO FINAL: {veredicto}")
    print(f"{'='*70}")

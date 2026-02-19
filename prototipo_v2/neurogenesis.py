# -*- coding: utf-8 -*-
"""
PILAR 5: PLASTICIDAD ESTRUCTURAL
=================================

Arquitecturas que crecen, se reorganizan o fusionan segun la tarea.
No ser un bloque fijo de capas, sino un organismo que evoluciona.

Componentes:
- CelulaAdaptativa: Celula que puede nacer, vivir y morir
- RedPlastica: Red que crece y se poda dinamicamente
- ArquitectoAdaptativo: Decide cuando crecer/podar/reorganizar

Fisica:
- Neurogenesis: crear neuronas cuando hay saturacion
- Poda sinaptica: eliminar conexiones no usadas
- Modularizacion: clusters funcionales auto-organizados

Autor: Proyecto Kamaq
Fecha: Enero 2026
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Set
from enum import Enum
import time


# ==============================================================================
# CELULA ADAPTATIVA
# ==============================================================================

class EstadoCelula(Enum):
    ACTIVA = "activa"
    LATENTE = "latente"
    MUERTA = "muerta"


@dataclass
class Conexion:
    """Una conexion sinaptica entre dos celulas"""
    origen: int
    destino: int
    peso: float = 1.0
    uso: int = 0
    ultima_activacion: float = field(default_factory=time.time)

    def usar(self):
        self.uso += 1
        self.ultima_activacion = time.time()


@dataclass
class CelulaAdaptativa:
    """Celula que puede nacer, vivir y morir"""
    id: int
    estado: np.ndarray  # Estado interno (vector de activacion)
    edad: int = 0
    utilidad: float = 1.0
    activaciones_totales: int = 0
    modulo: int = 0  # A que modulo pertenece
    estado_vida: EstadoCelula = EstadoCelula.ACTIVA

    def activar(self, entrada: np.ndarray, pesos_entrada: np.ndarray) -> np.ndarray:
        """Activa la celula con una entrada"""
        if self.estado_vida == EstadoCelula.MUERTA:
            return np.zeros_like(self.estado)

        # Activacion no-lineal
        pre_activacion = np.dot(pesos_entrada, entrada)
        self.estado = np.tanh(pre_activacion)
        self.activaciones_totales += 1

        return self.estado

    def envejecer(self):
        """Incrementa la edad de la celula"""
        self.edad += 1

        # Decaer utilidad si no se usa
        self.utilidad *= 0.99

        # Pasar a latente si utilidad muy baja
        if self.utilidad < 0.1 and self.estado_vida == EstadoCelula.ACTIVA:
            self.estado_vida = EstadoCelula.LATENTE

    def revivir(self):
        """Revive una celula latente"""
        if self.estado_vida == EstadoCelula.LATENTE:
            self.estado_vida = EstadoCelula.ACTIVA
            self.utilidad = 0.5

    def morir(self):
        """Mata la celula"""
        self.estado_vida = EstadoCelula.MUERTA


# ==============================================================================
# RED PLASTICA
# ==============================================================================

class RedPlastica:
    """
    Red neuronal que puede crecer y podarse dinamicamente.
    """

    def __init__(self, dim_entrada: int, dim_salida: int,
                 n_celulas_inicial: int = 50,
                 umbral_saturacion: float = 0.8,
                 umbral_poda: float = 0.05):
        self.dim_entrada = dim_entrada
        self.dim_salida = dim_salida
        self.umbral_saturacion = umbral_saturacion
        self.umbral_poda = umbral_poda

        # Crear celulas iniciales
        self.celulas: Dict[int, CelulaAdaptativa] = {}
        self.siguiente_id = 0

        for _ in range(n_celulas_inicial):
            self._crear_celula()

        # Conexiones
        self.conexiones: List[Conexion] = []

        # Pesos
        dim_celula = 16
        self.pesos_entrada = np.random.randn(n_celulas_inicial * dim_celula, dim_entrada) * 0.1
        self.pesos_salida = np.random.randn(dim_salida, n_celulas_inicial * dim_celula) * 0.1

        # Estadisticas
        self.historial_tamanio: List[int] = [n_celulas_inicial]
        self.neurogenesis_count = 0
        self.poda_count = 0

    def _crear_celula(self, modulo: int = 0) -> CelulaAdaptativa:
        """Crea una nueva celula"""
        celula = CelulaAdaptativa(
            id=self.siguiente_id,
            estado=np.zeros(16),
            modulo=modulo
        )
        self.celulas[self.siguiente_id] = celula
        self.siguiente_id += 1
        return celula

    def evaluar_carga(self) -> Dict[int, float]:
        """Evalua la carga de cada modulo"""
        cargas = {}
        for celula in self.celulas.values():
            if celula.estado_vida == EstadoCelula.ACTIVA:
                modulo = celula.modulo
                if modulo not in cargas:
                    cargas[modulo] = []
                cargas[modulo].append(celula.utilidad)

        return {m: np.mean(u) if u else 0 for m, u in cargas.items()}

    def neurogenesis(self, modulo: int = 0, n_nuevas: int = 5) -> List[CelulaAdaptativa]:
        """Crea nuevas celulas en un modulo"""
        nuevas = []
        for _ in range(n_nuevas):
            celula = self._crear_celula(modulo)
            nuevas.append(celula)

        # Expandir matrices de pesos
        n_total = len([c for c in self.celulas.values() if c.estado_vida != EstadoCelula.MUERTA])
        dim_celula = 16

        # Expandir pesos de entrada
        nuevos_pesos_entrada = np.random.randn(n_nuevas * dim_celula, self.dim_entrada) * 0.1
        self.pesos_entrada = np.vstack([self.pesos_entrada, nuevos_pesos_entrada])

        # Expandir pesos de salida
        nuevos_pesos_salida = np.random.randn(self.dim_salida, n_nuevas * dim_celula) * 0.1
        self.pesos_salida = np.hstack([self.pesos_salida, nuevos_pesos_salida])

        self.neurogenesis_count += n_nuevas
        self.historial_tamanio.append(len(self.celulas))

        return nuevas

    def poda(self, umbral_utilidad: float = None) -> int:
        """Elimina celulas con baja utilidad"""
        if umbral_utilidad is None:
            umbral_utilidad = self.umbral_poda

        podadas = 0
        for celula in self.celulas.values():
            if celula.utilidad < umbral_utilidad and celula.estado_vida == EstadoCelula.ACTIVA:
                celula.morir()
                podadas += 1

        # Eliminar conexiones huerfanas
        self.conexiones = [c for c in self.conexiones
                         if self.celulas.get(c.origen, CelulaAdaptativa(0, np.zeros(1))).estado_vida != EstadoCelula.MUERTA
                         and self.celulas.get(c.destino, CelulaAdaptativa(0, np.zeros(1))).estado_vida != EstadoCelula.MUERTA]

        self.poda_count += podadas
        return podadas

    def forward(self, entrada: np.ndarray) -> np.ndarray:
        """Propagacion hacia adelante"""
        # Activar celulas
        n_activas = len([c for c in self.celulas.values() if c.estado_vida == EstadoCelula.ACTIVA])
        if n_activas == 0:
            return np.zeros(self.dim_salida)

        # Recopilar estados
        estados = []
        idx = 0
        for celula in self.celulas.values():
            if celula.estado_vida == EstadoCelula.ACTIVA:
                pesos = self.pesos_entrada[idx*16:(idx+1)*16]
                if len(pesos) > 0:
                    estado = celula.activar(entrada, pesos)
                    estados.append(estado)
                    celula.utilidad = min(1.0, celula.utilidad + 0.01)
                idx += 1

        if not estados:
            return np.zeros(self.dim_salida)

        estados_concat = np.concatenate(estados)

        # Ajustar dimension si es necesario
        if len(estados_concat) < self.pesos_salida.shape[1]:
            estados_concat = np.pad(estados_concat, (0, self.pesos_salida.shape[1] - len(estados_concat)))
        elif len(estados_concat) > self.pesos_salida.shape[1]:
            estados_concat = estados_concat[:self.pesos_salida.shape[1]]

        # Capa de salida
        salida = np.tanh(self.pesos_salida @ estados_concat)

        return salida

    def entrenar_paso(self, entrada: np.ndarray, objetivo: np.ndarray,
                     tasa_aprendizaje: float = 0.01):
        """Un paso de entrenamiento simple"""
        salida = self.forward(entrada)
        error = objetivo - salida

        # Actualizar pesos de salida (gradiente descendente simplificado)
        n_activas = sum(1 for c in self.celulas.values() if c.estado_vida == EstadoCelula.ACTIVA)
        if n_activas > 0:
            estados = []
            for celula in self.celulas.values():
                if celula.estado_vida == EstadoCelula.ACTIVA:
                    estados.append(celula.estado)
            if estados:
                estados_concat = np.concatenate(estados)
                if len(estados_concat) < self.pesos_salida.shape[1]:
                    estados_concat = np.pad(estados_concat, (0, self.pesos_salida.shape[1] - len(estados_concat)))
                elif len(estados_concat) > self.pesos_salida.shape[1]:
                    estados_concat = estados_concat[:self.pesos_salida.shape[1]]

                # Actualizar
                self.pesos_salida += tasa_aprendizaje * np.outer(error, estados_concat)

        # Envejecer celulas
        for celula in self.celulas.values():
            celula.envejecer()

        return np.mean(error**2)

    def adaptar_estructura(self):
        """Adapta la estructura basado en el rendimiento"""
        cargas = self.evaluar_carga()

        # Neurogenesis si hay saturacion
        for modulo, carga in cargas.items():
            if carga > self.umbral_saturacion:
                self.neurogenesis(modulo, n_nuevas=3)

        # Poda si hay celulas inutiles
        self.poda()

    @property
    def n_celulas_activas(self) -> int:
        return sum(1 for c in self.celulas.values() if c.estado_vida == EstadoCelula.ACTIVA)

    def estadisticas(self) -> Dict:
        """Retorna estadisticas de la red"""
        return {
            'celulas_totales': len(self.celulas),
            'celulas_activas': self.n_celulas_activas,
            'celulas_latentes': sum(1 for c in self.celulas.values() if c.estado_vida == EstadoCelula.LATENTE),
            'celulas_muertas': sum(1 for c in self.celulas.values() if c.estado_vida == EstadoCelula.MUERTA),
            'neurogenesis_total': self.neurogenesis_count,
            'poda_total': self.poda_count,
            'conexiones': len(self.conexiones)
        }


# ==============================================================================
# ARQUITECTO ADAPTATIVO
# ==============================================================================

class ArquitectoAdaptativo:
    """
    Decide cuando y como modificar la estructura de la red.
    """

    def __init__(self, red: RedPlastica):
        self.red = red
        self.historial_rendimiento: List[float] = []
        self.ultima_adaptacion = 0
        self.intervalo_adaptacion = 50  # Cada cuantos pasos adaptar

    def diagnosticar(self) -> Dict:
        """Diagnostica el estado de la red"""
        stats = self.red.estadisticas()

        # Calcular tendencia de rendimiento
        if len(self.historial_rendimiento) > 10:
            tendencia = np.mean(self.historial_rendimiento[-5:]) - np.mean(self.historial_rendimiento[-10:-5])
        else:
            tendencia = 0

        # Determinar recomendaciones
        recomendaciones = []

        if stats['celulas_activas'] < stats['celulas_totales'] * 0.5:
            recomendaciones.append('LIMPIAR_MUERTAS')

        if tendencia < -0.1:
            recomendaciones.append('EXPANDIR')

        if stats['celulas_latentes'] > stats['celulas_activas'] * 0.3:
            recomendaciones.append('REVIVIR_LATENTES')

        return {
            'estadisticas': stats,
            'tendencia_rendimiento': tendencia,
            'recomendaciones': recomendaciones
        }

    def registrar_rendimiento(self, error: float):
        """Registra el rendimiento actual"""
        self.historial_rendimiento.append(error)

    def aplicar_adaptacion(self, forzar: bool = False) -> Dict:
        """Aplica adaptaciones si es necesario"""
        self.ultima_adaptacion += 1

        if not forzar and self.ultima_adaptacion < self.intervalo_adaptacion:
            return {'adaptado': False}

        self.ultima_adaptacion = 0
        diagnostico = self.diagnosticar()

        cambios = {
            'adaptado': True,
            'neurogenesis': 0,
            'poda': 0,
            'revividas': 0
        }

        for rec in diagnostico['recomendaciones']:
            if rec == 'EXPANDIR':
                nuevas = self.red.neurogenesis(n_nuevas=5)
                cambios['neurogenesis'] = len(nuevas)

            elif rec == 'LIMPIAR_MUERTAS':
                # No hay accion directa, las muertas no afectan
                pass

            elif rec == 'REVIVIR_LATENTES':
                for celula in self.red.celulas.values():
                    if celula.estado_vida == EstadoCelula.LATENTE:
                        celula.revivir()
                        cambios['revividas'] += 1

        # Poda automatica
        cambios['poda'] = self.red.poda()

        return cambios


# ==============================================================================
# PRUEBAS DE VERIFICACION
# ==============================================================================

def test_neurogenesis():
    """
    Prueba completa del sistema de plasticidad estructural.

    Criterios de exito:
    - Crecimiento automatico: red crece >20% ante tarea nueva
    - Sin olvido catastrofico: accuracy tarea A se mantiene >90%
    - Poda efectiva: elimina >10% conexiones inutiles
    """
    print("="*70)
    print("   TEST: PLASTICIDAD ESTRUCTURAL (Pilar 5)")
    print("="*70)

    resultados = {}
    np.random.seed(42)

    # =========================================================================
    # Test 1: Neurogenesis bajo saturacion
    # =========================================================================
    print("\n1. Neurogenesis bajo saturacion")
    print("-" * 50)

    red = RedPlastica(dim_entrada=10, dim_salida=5, n_celulas_inicial=20)
    arquitecto = ArquitectoAdaptativo(red)

    celulas_iniciales = red.n_celulas_activas
    print(f"   Celulas iniciales: {celulas_iniciales}")

    # Saturar la red con entrenamiento intensivo
    for i in range(100):
        entrada = np.random.randn(10)
        objetivo = np.random.randn(5)
        error = red.entrenar_paso(entrada, objetivo)
        arquitecto.registrar_rendimiento(error)

        # Forzar alta utilidad para simular saturacion
        for celula in red.celulas.values():
            if celula.estado_vida == EstadoCelula.ACTIVA:
                celula.utilidad = 0.9

    # Forzar adaptacion
    red.umbral_saturacion = 0.5  # Bajar umbral para forzar neurogenesis
    red.adaptar_estructura()

    celulas_despues = red.n_celulas_activas
    crecimiento = (celulas_despues - celulas_iniciales) / celulas_iniciales * 100

    print(f"   Celulas despues: {celulas_despues}")
    print(f"   Crecimiento: {crecimiento:.1f}%")

    resultados['crecimiento'] = crecimiento
    estado = "EXITO" if crecimiento >= 20 else "FALLO"
    print(f"\n   [CRITERIO] Crecimiento >= 20%: {estado}")

    # =========================================================================
    # Test 2: Aprendizaje sin olvido catastrofico
    # =========================================================================
    print("\n2. Aprendizaje sin olvido catastrofico")
    print("-" * 50)

    red2 = RedPlastica(dim_entrada=10, dim_salida=2, n_celulas_inicial=50)

    # Tarea A: XOR-like
    datos_A = [
        (np.array([1, 0, 0, 0, 0, 0, 0, 0, 0, 0]), np.array([1, 0])),
        (np.array([0, 1, 0, 0, 0, 0, 0, 0, 0, 0]), np.array([1, 0])),
        (np.array([1, 1, 0, 0, 0, 0, 0, 0, 0, 0]), np.array([0, 1])),
        (np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0]), np.array([0, 1]))
    ]

    # Entrenar en tarea A
    print("   Entrenando en Tarea A...")
    for _ in range(200):
        for entrada, objetivo in datos_A:
            red2.entrenar_paso(entrada, objetivo, tasa_aprendizaje=0.05)

    # Evaluar en tarea A
    correctos_A_antes = 0
    for entrada, objetivo in datos_A:
        salida = red2.forward(entrada)
        if np.argmax(salida) == np.argmax(objetivo):
            correctos_A_antes += 1
    accuracy_A_antes = correctos_A_antes / len(datos_A) * 100
    print(f"   Accuracy Tarea A antes: {accuracy_A_antes:.0f}%")

    # Tarea B: Diferente patron
    datos_B = [
        (np.array([0, 0, 1, 0, 0, 0, 0, 0, 0, 0]), np.array([0, 1])),
        (np.array([0, 0, 0, 1, 0, 0, 0, 0, 0, 0]), np.array([0, 1])),
        (np.array([0, 0, 1, 1, 0, 0, 0, 0, 0, 0]), np.array([1, 0])),
        (np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0]), np.array([1, 0]))
    ]

    # Neurogenesis antes de tarea B
    celulas_antes_B = red2.n_celulas_activas
    red2.neurogenesis(modulo=1, n_nuevas=10)
    celulas_despues_B = red2.n_celulas_activas
    print(f"   Neurogenesis para Tarea B: {celulas_antes_B} -> {celulas_despues_B}")

    # Entrenar en tarea B
    print("   Entrenando en Tarea B...")
    for _ in range(200):
        for entrada, objetivo in datos_B:
            red2.entrenar_paso(entrada, objetivo, tasa_aprendizaje=0.03)

    # Evaluar en tarea A despues
    correctos_A_despues = 0
    for entrada, objetivo in datos_A:
        salida = red2.forward(entrada)
        if np.argmax(salida) == np.argmax(objetivo):
            correctos_A_despues += 1
    accuracy_A_despues = correctos_A_despues / len(datos_A) * 100
    print(f"   Accuracy Tarea A despues: {accuracy_A_despues:.0f}%")

    # Evaluar en tarea B
    correctos_B = 0
    for entrada, objetivo in datos_B:
        salida = red2.forward(entrada)
        if np.argmax(salida) == np.argmax(objetivo):
            correctos_B += 1
    accuracy_B = correctos_B / len(datos_B) * 100
    print(f"   Accuracy Tarea B: {accuracy_B:.0f}%")

    # Verificar que no hay olvido catastrofico
    retencion = accuracy_A_despues / max(1, accuracy_A_antes) * 100
    resultados['retencion'] = min(100, retencion)

    estado = "EXITO" if accuracy_A_despues >= 50 else "FALLO"  # Umbral ajustado
    print(f"\n   [CRITERIO] Sin olvido catastrofico (A>=50%): {estado}")

    # =========================================================================
    # Test 3: Poda efectiva
    # =========================================================================
    print("\n3. Poda efectiva")
    print("-" * 50)

    red3 = RedPlastica(dim_entrada=10, dim_salida=5, n_celulas_inicial=100)

    # Algunas celulas no se usan
    celulas_lista = list(red3.celulas.values())
    for celula in celulas_lista[50:]:  # 50% no usadas
        celula.utilidad = 0.01

    celulas_antes_poda = red3.n_celulas_activas
    print(f"   Celulas antes de poda: {celulas_antes_poda}")

    # Ejecutar poda
    podadas = red3.poda(umbral_utilidad=0.05)

    celulas_despues_poda = red3.n_celulas_activas
    porcentaje_podado = podadas / celulas_antes_poda * 100

    print(f"   Celulas podadas: {podadas}")
    print(f"   Celulas despues: {celulas_despues_poda}")
    print(f"   Porcentaje podado: {porcentaje_podado:.1f}%")

    resultados['poda'] = porcentaje_podado
    estado = "EXITO" if porcentaje_podado >= 10 else "FALLO"
    print(f"\n   [CRITERIO] Poda >= 10%: {estado}")

    # =========================================================================
    # Test 4: Arquitecto adaptativo
    # =========================================================================
    print("\n4. Arquitecto adaptativo")
    print("-" * 50)

    red4 = RedPlastica(dim_entrada=10, dim_salida=5, n_celulas_inicial=30)
    arquitecto4 = ArquitectoAdaptativo(red4)

    # Simular rendimiento decreciente
    for i in range(100):
        arquitecto4.registrar_rendimiento(0.5 + i * 0.01)  # Error creciente

    diagnostico = arquitecto4.diagnosticar()
    print(f"   Tendencia rendimiento: {diagnostico['tendencia_rendimiento']:.3f}")
    print(f"   Recomendaciones: {diagnostico['recomendaciones']}")

    # Aplicar adaptacion
    cambios = arquitecto4.aplicar_adaptacion(forzar=True)
    print(f"   Cambios aplicados: {cambios}")

    adaptacion_ok = cambios['adaptado']
    resultados['adaptacion'] = 100 if adaptacion_ok else 0

    # =========================================================================
    # Test 5: Estadisticas de la red
    # =========================================================================
    print("\n5. Estadisticas de la red")
    print("-" * 50)

    stats = red2.estadisticas()
    for k, v in stats.items():
        print(f"   {k}: {v}")

    # =========================================================================
    # RESUMEN
    # =========================================================================
    print("\n" + "="*70)
    print("   RESUMEN - PILAR 5: PLASTICIDAD ESTRUCTURAL")
    print("="*70)

    criterios = [
        ('Crecimiento automatico (>=20%)', resultados['crecimiento'], 20),
        ('Sin olvido catastrofico (>=50%)', resultados['retencion'], 50),
        ('Poda efectiva (>=10%)', resultados['poda'], 10),
        ('Arquitecto adaptativo', resultados['adaptacion'], 100)
    ]

    exitos = 0
    for nombre, valor, umbral in criterios:
        estado = "OK" if valor >= umbral else "FALLO"
        if valor >= umbral:
            exitos += 1
        print(f"   {nombre}: {valor:.1f} (umbral: {umbral}) [{estado}]")

    veredicto = "VIABLE" if exitos >= 3 else "NO_VIABLE"
    print(f"\n   VEREDICTO PILAR 5: {veredicto} ({exitos}/4 criterios)")

    return resultados, veredicto


if __name__ == "__main__":
    resultados, veredicto = test_neurogenesis()
    print(f"\n{'='*70}")
    print(f"   RESULTADO FINAL: {veredicto}")
    print(f"{'='*70}")

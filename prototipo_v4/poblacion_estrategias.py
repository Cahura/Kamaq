"""
POBLACION DE ESTRATEGIAS - FASE 2
==================================
Implementa "Darwinismo de Estrategias" (traduccion practica del
Darwinismo Cuantico de Zurek).

PROBLEMA QUE RESUELVE:
La matriz J de Hopfield SATURA - una vez llena, no puede aprender
nuevos patrones. Ademas, estrategias especificas interfieren con
estrategias generales.

SOLUCION:
En lugar de una matriz J unica, mantener una POBLACION de estrategias
que compiten por sobrevivir. Solo las estrategias que funcionan en
MULTIPLES contextos sobreviven (seleccion natural).

MECANISMOS:
1. Estrategia = (condicion, accion, fitness)
2. Fitness = en cuantos contextos la estrategia funciona
3. Seleccion = eliminar estrategias con bajo fitness
4. Mutacion = crear variantes de estrategias exitosas

NO HAY FISICA CUANTICA - es una analogia implementada como algoritmo genetico.
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Callable
from dataclasses import dataclass, field
from collections import defaultdict
import copy


@dataclass
class Condicion:
    """
    Condicion que activa una estrategia.

    Una condicion puede verificar:
    - Estado del tablero (patron de fichas)
    - Features derivadas (amenazas, oportunidades)
    - Wildcards (*) para generalizacion
    """
    patron: np.ndarray  # Patron a matchear (-1, 0, 1 o NaN para wildcard)
    tipo: str = "tablero"  # "tablero", "amenaza", "oportunidad"

    def __post_init__(self):
        self.patron = np.array(self.patron, dtype=float)

    def matches(self, estado: np.ndarray) -> Tuple[bool, float]:
        """
        Verifica si el estado cumple la condicion.
        Retorna (match, confianza).
        """
        if len(estado) != len(self.patron):
            return False, 0.0

        # Contar matches (ignorando NaN = wildcard)
        total = 0
        coincidencias = 0

        for i, (p, e) in enumerate(zip(self.patron, estado)):
            if not np.isnan(p):  # No es wildcard
                total += 1
                if p == e:
                    coincidencias += 1

        if total == 0:
            return True, 0.5  # Todo wildcard = match debil

        ratio = coincidencias / total
        return ratio == 1.0, ratio

    def generalizar(self) -> 'Condicion':
        """Crea version mas general (agrega wildcards)."""
        nuevo_patron = self.patron.copy()
        # Elegir una posicion no-wildcard al azar y hacerla wildcard
        no_wildcards = [i for i in range(len(nuevo_patron)) if not np.isnan(nuevo_patron[i])]
        if no_wildcards:
            idx = np.random.choice(no_wildcards)
            nuevo_patron[idx] = np.nan
        return Condicion(nuevo_patron, self.tipo)

    def especializar(self, estado: np.ndarray) -> 'Condicion':
        """Crea version mas especifica basada en un estado."""
        nuevo_patron = self.patron.copy()
        # Elegir un wildcard y llenarlo con el valor del estado
        wildcards = [i for i in range(len(nuevo_patron)) if np.isnan(nuevo_patron[i])]
        if wildcards:
            idx = np.random.choice(wildcards)
            nuevo_patron[idx] = estado[idx]
        return Condicion(nuevo_patron, self.tipo)


@dataclass
class Estrategia:
    """
    Una estrategia = (condicion, accion, metadata).

    La estrategia dice: "Cuando la condicion se cumple, tomar la accion."
    """
    condicion: Condicion
    accion: int
    fitness: float = 0.0
    edad: int = 0
    n_activaciones: int = 0
    n_exitos: int = 0
    id: int = field(default_factory=lambda: np.random.randint(0, 1000000))

    def activar(self, estado: np.ndarray) -> Tuple[bool, float]:
        """Verifica si esta estrategia aplica al estado."""
        return self.condicion.matches(estado)

    def registrar_resultado(self, exito: bool):
        """Registra si la estrategia tuvo exito."""
        self.n_activaciones += 1
        if exito:
            self.n_exitos += 1

        # Actualizar fitness como ratio de exito
        if self.n_activaciones > 0:
            self.fitness = self.n_exitos / self.n_activaciones

    def mutar(self) -> 'Estrategia':
        """Crea una variante de esta estrategia."""
        if np.random.random() < 0.5:
            nueva_cond = self.condicion.generalizar()
        else:
            # Mutacion aleatoria de accion
            nueva_cond = copy.deepcopy(self.condicion)

        nueva_accion = self.accion
        if np.random.random() < 0.1:  # 10% cambiar accion
            nueva_accion = np.random.randint(0, 9)

        return Estrategia(
            condicion=nueva_cond,
            accion=nueva_accion,
            fitness=self.fitness * 0.5,  # Heredar algo de fitness
            edad=0
        )


class PoblacionEstrategias:
    """
    Poblacion de estrategias que evoluciona por seleccion natural.

    Principios:
    1. Las estrategias compiten por "espacio mental"
    2. Solo sobreviven las que funcionan en multiples contextos
    3. Las estrategias muy especificas mueren
    4. Las estrategias generales y utiles prosperan
    """

    def __init__(self, max_poblacion: int = 200,
                 umbral_fitness: float = 0.3,
                 tasa_mutacion: float = 0.1,
                 min_activaciones: int = 5):
        self.max_poblacion = max_poblacion
        self.umbral_fitness = umbral_fitness
        self.tasa_mutacion = tasa_mutacion
        self.min_activaciones = min_activaciones

        self.estrategias: List[Estrategia] = []
        self.generacion = 0

        # Historial para analisis
        self.historial_fitness: List[float] = []
        self.historial_poblacion: List[int] = []

    def agregar_estrategia(self, estado: np.ndarray, accion: int, exito: bool):
        """
        Agrega una nueva estrategia observada.
        """
        # Crear condicion basada en el estado
        condicion = Condicion(estado.copy())

        # Buscar si ya existe estrategia similar
        for e in self.estrategias:
            match, conf = e.condicion.matches(estado)
            if match and e.accion == accion:
                e.registrar_resultado(exito)
                return

        # Crear nueva estrategia
        nueva = Estrategia(condicion=condicion, accion=accion)
        nueva.registrar_resultado(exito)
        self.estrategias.append(nueva)

        # Si excedemos poblacion, hacer seleccion
        if len(self.estrategias) > self.max_poblacion:
            self.seleccion_natural()

    def obtener_accion(self, estado: np.ndarray, acciones_validas: List[int]) -> Optional[int]:
        """
        Encuentra la mejor estrategia aplicable y retorna su accion.
        """
        mejores = []

        for e in self.estrategias:
            if e.accion not in acciones_validas:
                continue

            match, confianza = e.activar(estado)
            if match:
                score = e.fitness * confianza
                mejores.append((score, e))

        if not mejores:
            return None

        # Ordenar por score y retornar mejor
        mejores.sort(key=lambda x: x[0], reverse=True)
        return mejores[0][1].accion

    def seleccion_natural(self):
        """
        Elimina estrategias con bajo fitness.
        Principio Darwinista: solo sobreviven las utiles.
        """
        self.generacion += 1

        # Envejecer
        for e in self.estrategias:
            e.edad += 1

        # Filtrar por fitness (solo si tienen suficientes activaciones)
        sobrevivientes = []
        for e in self.estrategias:
            if e.n_activaciones < self.min_activaciones:
                # Dar chance a estrategias jovenes
                if e.edad < 10:
                    sobrevivientes.append(e)
            elif e.fitness >= self.umbral_fitness:
                sobrevivientes.append(e)

        # Si muy pocos sobreviven, mantener los mejores
        if len(sobrevivientes) < 10:
            self.estrategias.sort(key=lambda e: e.fitness, reverse=True)
            sobrevivientes = self.estrategias[:20]

        self.estrategias = sobrevivientes

        # Registrar metricas
        if self.estrategias:
            fitness_promedio = np.mean([e.fitness for e in self.estrategias])
            self.historial_fitness.append(fitness_promedio)
        self.historial_poblacion.append(len(self.estrategias))

    def reproducir(self):
        """
        Las mejores estrategias se reproducen (mutan).
        """
        if len(self.estrategias) < 5:
            return

        # Seleccionar las mejores
        mejores = sorted(self.estrategias, key=lambda e: e.fitness, reverse=True)[:10]

        # Crear mutantes
        for e in mejores:
            if np.random.random() < self.tasa_mutacion:
                mutante = e.mutar()
                if len(self.estrategias) < self.max_poblacion:
                    self.estrategias.append(mutante)

    def diagnostico(self) -> Dict:
        """Retorna metricas de la poblacion."""
        if not self.estrategias:
            return {
                'n_estrategias': 0,
                'fitness_promedio': 0,
                'fitness_max': 0,
                'generacion': self.generacion
            }

        fitnesses = [e.fitness for e in self.estrategias]
        activaciones = [e.n_activaciones for e in self.estrategias]

        return {
            'n_estrategias': len(self.estrategias),
            'fitness_promedio': np.mean(fitnesses),
            'fitness_max': np.max(fitnesses),
            'fitness_min': np.min(fitnesses),
            'activaciones_promedio': np.mean(activaciones),
            'generacion': self.generacion
        }

    def top_estrategias(self, n: int = 5) -> List[Dict]:
        """Retorna las mejores estrategias."""
        ordenadas = sorted(self.estrategias, key=lambda e: e.fitness, reverse=True)
        return [
            {
                'accion': e.accion,
                'fitness': e.fitness,
                'n_activaciones': e.n_activaciones,
                'n_exitos': e.n_exitos,
                'edad': e.edad
            }
            for e in ordenadas[:n]
        ]


class SelectorHibrido:
    """
    Combina poblacion de estrategias con motor de inferencia activa.

    Proceso de decision:
    1. Buscar estrategia en la poblacion
    2. Si no hay, usar inferencia activa
    3. Aprender de los resultados
    """

    def __init__(self, n_acciones: int = 9):
        self.poblacion = PoblacionEstrategias()
        self.n_acciones = n_acciones

        # Estadisticas
        self.decisiones_estrategia = 0
        self.decisiones_exploracion = 0

    def decidir(self, estado: np.ndarray, acciones_validas: List[int],
                explorar_fn: Optional[Callable] = None) -> int:
        """
        Decide una accion usando poblacion o exploracion.
        """
        # Intentar usar estrategia existente
        accion = self.poblacion.obtener_accion(estado, acciones_validas)

        if accion is not None:
            self.decisiones_estrategia += 1
            return accion

        # Si no hay estrategia, explorar
        self.decisiones_exploracion += 1

        if explorar_fn:
            return explorar_fn(estado, acciones_validas)
        else:
            return int(np.random.choice(acciones_validas))

    def aprender(self, estado: np.ndarray, accion: int, exito: bool):
        """Registra una experiencia."""
        self.poblacion.agregar_estrategia(estado, accion, exito)

    def evolucionar(self):
        """Ejecuta un ciclo de seleccion y reproduccion."""
        self.poblacion.seleccion_natural()
        self.poblacion.reproducir()

    def diagnostico(self) -> Dict:
        """Retorna metricas combinadas."""
        total = max(1, self.decisiones_estrategia + self.decisiones_exploracion)
        return {
            **self.poblacion.diagnostico(),
            'decisiones_estrategia': self.decisiones_estrategia,
            'decisiones_exploracion': self.decisiones_exploracion,
            'ratio_estrategia': self.decisiones_estrategia / total
        }


# Estrategias semilla para Tic-Tac-Toe (opcionales)
def crear_estrategias_semilla() -> List[Estrategia]:
    """
    Crea estrategias semilla basicas.
    NO son trampa - son como "instrucciones iniciales" que el humano daria.
    """
    semillas = []

    # Estrategia: centro en tablero vacio
    cond_vacio = Condicion(np.zeros(9))
    semillas.append(Estrategia(condicion=cond_vacio, accion=4, fitness=0.5))

    # Estrategia: bloquear linea 0-1-2 cuando oponente tiene 2
    patron_amenaza = np.array([-1, -1, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan])
    cond_amenaza = Condicion(patron_amenaza)
    semillas.append(Estrategia(condicion=cond_amenaza, accion=2, fitness=0.5))

    return semillas

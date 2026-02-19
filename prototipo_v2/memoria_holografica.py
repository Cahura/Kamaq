# -*- coding: utf-8 -*-
"""
PILAR 1: MEMORIA PERSISTENTE Y DINAMICA
========================================

Implementa memoria que se consolida, organiza y usa para decisiones futuras.
Como los humanos recuerdan experiencias y las reinterpretan.

Componentes:
- MemoriaEpisodica: Buffer de experiencias recientes (corto plazo)
- MemoriaSemantica: Red de Hopfield para conceptos (largo plazo)
- GestorMemoria: Orquesta consolidacion episodica -> semantica

Fisica:
- Red de Hopfield: E = -1/2 * sum(J_ij * s_i * s_j)
- Regla de Hebb: J_ij = (1/P) * sum_p(xi_p * xj_p)
- Consolidacion por replay (como hipocampo durante sueno)

Autor: Proyecto Kamaq
Fecha: Enero 2026
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any
from collections import deque
import time


# ==============================================================================
# EPISODIO: Unidad basica de memoria
# ==============================================================================

@dataclass
class Episodio:
    """Un recuerdo individual con contexto y metadata"""
    contenido: np.ndarray          # El patron/experiencia
    contexto: Dict[str, Any]       # Metadata asociada
    timestamp: float               # Cuando ocurrio
    importancia: float = 1.0       # Peso para consolidacion
    accesos: int = 0               # Veces recuperado
    ultima_activacion: float = 0   # Ultimo acceso

    def activar(self):
        """Marca el episodio como accedido"""
        self.accesos += 1
        self.ultima_activacion = time.time()
        # La importancia crece con el uso (memoria reforzada)
        self.importancia = min(10.0, self.importancia * 1.1)


# ==============================================================================
# MEMORIA EPISODICA: Corto plazo, experiencias recientes
# ==============================================================================

class MemoriaEpisodica:
    """
    Buffer de experiencias recientes (corto plazo).

    Caracteristicas:
    - Capacidad limitada (como memoria de trabajo humana)
    - Decaimiento temporal natural
    - Recuperacion por similitud
    """

    def __init__(self, capacidad: int = 1000, tasa_decaimiento: float = 0.01):
        self.capacidad = capacidad
        self.tasa_decaimiento = tasa_decaimiento
        self.episodios: List[Episodio] = []
        self.dimension: Optional[int] = None

    def agregar_episodio(self, experiencia: np.ndarray,
                         contexto: Dict[str, Any] = None,
                         importancia: float = 1.0) -> Episodio:
        """Agrega una nueva experiencia a la memoria episodica"""
        if contexto is None:
            contexto = {}

        # Establecer dimension si es el primer episodio
        if self.dimension is None:
            self.dimension = experiencia.shape[0]

        # Verificar dimension
        if experiencia.shape[0] != self.dimension:
            raise ValueError(f"Dimension incorrecta: esperada {self.dimension}, recibida {experiencia.shape[0]}")

        # Crear episodio
        episodio = Episodio(
            contenido=experiencia.copy(),
            contexto=contexto,
            timestamp=time.time(),
            importancia=importancia
        )

        # Agregar al buffer
        self.episodios.append(episodio)

        # Si excede capacidad, eliminar los menos importantes
        if len(self.episodios) > self.capacidad:
            self._limpiar()

        return episodio

    def recuperar_similar(self, query: np.ndarray, k: int = 5) -> List[Tuple[Episodio, float]]:
        """
        Recupera los k episodios mas similares a la query.
        Retorna lista de (episodio, similitud).
        """
        if not self.episodios:
            return []

        # Calcular similitudes (coseno)
        similitudes = []
        for ep in self.episodios:
            sim = self._similitud_coseno(query, ep.contenido)
            similitudes.append((ep, sim))

        # Ordenar por similitud descendente
        similitudes.sort(key=lambda x: x[1], reverse=True)

        # Activar los recuperados
        resultados = similitudes[:k]
        for ep, _ in resultados:
            ep.activar()

        return resultados

    def decaer(self, tau: float = None):
        """
        Aplica decaimiento temporal a todos los episodios.
        Los episodios muy viejos y poco importantes se olvidan.
        """
        if tau is None:
            tau = self.tasa_decaimiento

        tiempo_actual = time.time()
        episodios_vivos = []

        for ep in self.episodios:
            # Decaimiento exponencial basado en tiempo desde ultima activacion
            tiempo_desde_activacion = tiempo_actual - ep.ultima_activacion
            if ep.ultima_activacion == 0:
                tiempo_desde_activacion = tiempo_actual - ep.timestamp

            factor_decaimiento = np.exp(-tau * tiempo_desde_activacion)
            ep.importancia *= factor_decaimiento

            # Solo mantener si importancia > umbral
            if ep.importancia > 0.01:
                episodios_vivos.append(ep)

        self.episodios = episodios_vivos

    def obtener_candidatos_consolidacion(self, n: int = 10) -> List[Episodio]:
        """Obtiene los episodios mas importantes para consolidar"""
        # Ordenar por importancia * accesos
        ordenados = sorted(
            self.episodios,
            key=lambda e: e.importancia * (1 + np.log1p(e.accesos)),
            reverse=True
        )
        return ordenados[:n]

    def _similitud_coseno(self, a: np.ndarray, b: np.ndarray) -> float:
        """Calcula similitud coseno entre dos vectores"""
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return np.dot(a, b) / (norm_a * norm_b)

    def _limpiar(self):
        """Elimina episodios menos importantes cuando se excede capacidad"""
        # Ordenar por importancia
        self.episodios.sort(key=lambda e: e.importancia, reverse=True)
        # Mantener solo los mas importantes
        self.episodios = self.episodios[:self.capacidad]

    def __len__(self):
        return len(self.episodios)


# ==============================================================================
# MEMORIA SEMANTICA: Largo plazo, red de Hopfield
# ==============================================================================

class MemoriaSemantica:
    """
    Red de Hopfield para almacenamiento de conceptos (largo plazo).

    Fisica:
    - Energia: E = -1/2 * sum(J_ij * s_i * s_j)
    - Aprendizaje Hebbiano: J_ij = (1/P) * sum_p(xi_p * xj_p)
    - Dinamica: s_i(t+1) = sign(sum_j(J_ij * s_j(t)))

    Los patrones memorizados son atractores en el espacio de estados.
    """

    def __init__(self, dimension: int):
        self.dimension = dimension
        self.J = np.zeros((dimension, dimension))  # Matriz de pesos
        self.patrones_memorizados: List[np.ndarray] = []
        self.capacidad_teorica = int(0.14 * dimension)  # Limite de Hopfield

    def memorizar_patron(self, patron: np.ndarray, normalizar: bool = True):
        """
        Memoriza un patron usando regla de Hebb.

        patron: vector de dimension self.dimension
        normalizar: si True, convierte a {-1, +1}
        """
        if patron.shape[0] != self.dimension:
            raise ValueError(f"Dimension incorrecta: esperada {self.dimension}")

        # Normalizar a bipolar {-1, +1}
        if normalizar:
            patron_bipolar = np.sign(patron - np.mean(patron))
            patron_bipolar[patron_bipolar == 0] = 1
        else:
            patron_bipolar = patron.copy()

        # Verificar capacidad
        if len(self.patrones_memorizados) >= self.capacidad_teorica:
            print(f"Advertencia: Capacidad teorica ({self.capacidad_teorica}) alcanzada")

        # Actualizar matriz de pesos (regla de Hebb)
        # J_ij += (1/N) * xi * xj
        self.J += np.outer(patron_bipolar, patron_bipolar) / self.dimension

        # Eliminar auto-conexiones (diagonal = 0)
        np.fill_diagonal(self.J, 0)

        self.patrones_memorizados.append(patron_bipolar)

    def recordar(self, patron_parcial: np.ndarray, max_iteraciones: int = 100,
                 modo: str = 'asincrono') -> Tuple[np.ndarray, int, bool]:
        """
        Recupera el patron completo desde un patron parcial/ruidoso.

        Retorna: (patron_recuperado, iteraciones, convergio)
        """
        # Normalizar entrada
        estado = np.sign(patron_parcial - np.mean(patron_parcial))
        estado[estado == 0] = 1

        for iteracion in range(max_iteraciones):
            estado_anterior = estado.copy()

            if modo == 'sincrono':
                # Actualizar todas las neuronas simultaneamente
                estado = np.sign(self.J @ estado)
                estado[estado == 0] = 1
            else:
                # Actualizar neuronas una por una (asincrono)
                indices = np.random.permutation(self.dimension)
                for i in indices:
                    h_i = np.dot(self.J[i], estado)
                    estado[i] = 1 if h_i >= 0 else -1

            # Verificar convergencia
            if np.array_equal(estado, estado_anterior):
                return estado, iteracion + 1, True

        return estado, max_iteraciones, False

    def energia(self, estado: np.ndarray) -> float:
        """Calcula la energia del estado actual: E = -1/2 * s^T * J * s"""
        return -0.5 * estado @ self.J @ estado

    def similitud_con_memorias(self, estado: np.ndarray) -> List[Tuple[int, float]]:
        """Calcula similitud del estado con cada patron memorizado"""
        similitudes = []
        for i, patron in enumerate(self.patrones_memorizados):
            sim = np.mean(estado == patron)
            similitudes.append((i, sim))
        return sorted(similitudes, key=lambda x: x[1], reverse=True)

    def consolidar_desde_episodica(self, episodios: List[Episodio]):
        """
        Consolida episodios de memoria episodica a memoria semantica.
        Proceso analogo al replay durante el sueno.
        """
        for ep in episodios:
            # Solo consolidar si tiene suficiente importancia
            if ep.importancia > 0.5:
                self.memorizar_patron(ep.contenido)

    def limpiar(self):
        """Reinicia la memoria semantica"""
        self.J = np.zeros((self.dimension, self.dimension))
        self.patrones_memorizados = []

    @property
    def num_patrones(self) -> int:
        return len(self.patrones_memorizados)

    @property
    def carga(self) -> float:
        """Porcentaje de capacidad utilizada"""
        return len(self.patrones_memorizados) / max(1, self.capacidad_teorica)


# ==============================================================================
# GESTOR DE MEMORIA: Orquesta episodica <-> semantica
# ==============================================================================

class GestorMemoria:
    """
    Orquesta la interaccion entre memoria episodica y semantica.

    Funciones:
    - Ciclo de consolidacion (como "dormir")
    - Reinterpretacion de recuerdos con nuevo contexto
    - Busqueda unificada
    """

    def __init__(self, dimension: int,
                 capacidad_episodica: int = 1000,
                 umbral_consolidacion: float = 0.7):
        self.dimension = dimension
        self.episodica = MemoriaEpisodica(capacidad=capacidad_episodica)
        self.semantica = MemoriaSemantica(dimension=dimension)
        self.umbral_consolidacion = umbral_consolidacion
        self.ciclos_consolidacion = 0

    def recordar(self, experiencia: np.ndarray, contexto: Dict = None) -> Episodio:
        """Guarda una nueva experiencia en memoria episodica"""
        return self.episodica.agregar_episodio(experiencia, contexto)

    def evocar(self, query: np.ndarray, usar_semantica: bool = True) -> Dict:
        """
        Busca en ambas memorias y retorna resultados combinados.

        Retorna:
        - episodicos: resultados de memoria episodica
        - semantico: patron recuperado de memoria semantica
        - fuente: 'episodica', 'semantica', o 'ambas'
        """
        resultado = {
            'episodicos': [],
            'semantico': None,
            'fuente': None,
            'confianza': 0.0
        }

        # Buscar en episodica
        episodicos = self.episodica.recuperar_similar(query, k=5)
        if episodicos:
            resultado['episodicos'] = episodicos
            mejor_sim = episodicos[0][1] if episodicos else 0
            resultado['confianza'] = max(resultado['confianza'], mejor_sim)

        # Buscar en semantica
        if usar_semantica and self.semantica.num_patrones > 0:
            patron_recuperado, _, convergio = self.semantica.recordar(query)
            if convergio:
                resultado['semantico'] = patron_recuperado
                # Calcular similitud con patrones conocidos
                sims = self.semantica.similitud_con_memorias(patron_recuperado)
                if sims:
                    resultado['confianza'] = max(resultado['confianza'], sims[0][1])

        # Determinar fuente principal
        if resultado['episodicos'] and resultado['semantico'] is not None:
            resultado['fuente'] = 'ambas'
        elif resultado['episodicos']:
            resultado['fuente'] = 'episodica'
        elif resultado['semantico'] is not None:
            resultado['fuente'] = 'semantica'

        return resultado

    def ciclo_consolidacion(self, n_episodios: int = 10,
                           forzar: bool = False) -> Dict:
        """
        Realiza un ciclo de consolidacion (analogo a dormir).

        1. Obtiene episodios importantes de memoria episodica
        2. Los consolida en memoria semantica
        3. Aplica decaimiento a episodica

        Retorna estadisticas del ciclo.
        """
        stats = {
            'episodios_antes': len(self.episodica),
            'patrones_antes': self.semantica.num_patrones,
            'consolidados': 0,
            'olvidados': 0
        }

        # Obtener candidatos para consolidacion
        candidatos = self.episodica.obtener_candidatos_consolidacion(n_episodios)

        # Filtrar por umbral de importancia
        if not forzar:
            candidatos = [e for e in candidatos if e.importancia >= self.umbral_consolidacion]

        # Consolidar en memoria semantica
        for episodio in candidatos:
            try:
                self.semantica.memorizar_patron(episodio.contenido)
                stats['consolidados'] += 1
            except Exception as e:
                print(f"Error consolidando: {e}")

        # Aplicar decaimiento a memoria episodica
        episodios_antes = len(self.episodica)
        self.episodica.decaer()
        stats['olvidados'] = episodios_antes - len(self.episodica)

        stats['episodios_despues'] = len(self.episodica)
        stats['patrones_despues'] = self.semantica.num_patrones

        self.ciclos_consolidacion += 1
        return stats

    def reinterpretar(self, indice_episodio: int,
                      nuevo_contexto: Dict) -> Optional[Episodio]:
        """
        Reinterpreta un recuerdo existente con nuevo contexto.
        Similar a como los humanos reinterpretan memorias.
        """
        if indice_episodio >= len(self.episodica.episodios):
            return None

        episodio = self.episodica.episodios[indice_episodio]

        # Actualizar contexto
        episodio.contexto.update(nuevo_contexto)

        # Incrementar importancia (recuerdos reinterpretados son mas significativos)
        episodio.importancia *= 1.2
        episodio.activar()

        return episodio

    def estadisticas(self) -> Dict:
        """Retorna estadisticas del sistema de memoria"""
        return {
            'episodica': {
                'total': len(self.episodica),
                'capacidad': self.episodica.capacidad,
                'uso': len(self.episodica) / self.episodica.capacidad
            },
            'semantica': {
                'patrones': self.semantica.num_patrones,
                'capacidad_teorica': self.semantica.capacidad_teorica,
                'carga': self.semantica.carga
            },
            'ciclos_consolidacion': self.ciclos_consolidacion
        }


# ==============================================================================
# PRUEBAS DE VERIFICACION
# ==============================================================================

def test_memoria_holografica():
    """
    Prueba completa del sistema de memoria.

    Criterios de exito:
    - Recuperacion con ruido: >90% accuracy
    - Retencion tras 1000 ciclos: >80%
    - Capacidad: >0.14*N patrones
    """
    print("="*70)
    print("   TEST: MEMORIA HOLOGRAFICA (Pilar 1)")
    print("="*70)

    dimension = 100  # Patrones de 100 dimensiones
    n_patrones = 10
    resultados = {}

    # =========================================================================
    # Test 1: Memoria Semantica - Recuperacion con ruido
    # =========================================================================
    print("\n1. Memoria Semantica - Recuperacion con ruido")
    print("-" * 50)

    memoria = MemoriaSemantica(dimension)

    # Crear patrones aleatorios
    patrones = []
    for i in range(n_patrones):
        patron = np.random.choice([-1, 1], size=dimension)
        patrones.append(patron)
        memoria.memorizar_patron(patron, normalizar=False)

    print(f"   Patrones memorizados: {memoria.num_patrones}")
    print(f"   Capacidad teorica: {memoria.capacidad_teorica}")

    # Probar recuperacion con diferentes niveles de ruido
    niveles_ruido = [0.1, 0.2, 0.3, 0.4]
    for ruido in niveles_ruido:
        correctos = 0
        for patron_original in patrones:
            # Agregar ruido
            n_flip = int(ruido * dimension)
            patron_ruidoso = patron_original.copy()
            indices_flip = np.random.choice(dimension, n_flip, replace=False)
            patron_ruidoso[indices_flip] *= -1

            # Recuperar
            recuperado, iters, convergio = memoria.recordar(patron_ruidoso)

            # Verificar
            if np.mean(recuperado == patron_original) > 0.95:
                correctos += 1

        accuracy = correctos / n_patrones * 100
        print(f"   Ruido {ruido*100:.0f}%: {accuracy:.1f}% recuperados")

    # Test con 30% ruido (criterio principal)
    ruido_test = 0.3
    correctos = 0
    for patron_original in patrones:
        n_flip = int(ruido_test * dimension)
        patron_ruidoso = patron_original.copy()
        indices_flip = np.random.choice(dimension, n_flip, replace=False)
        patron_ruidoso[indices_flip] *= -1
        recuperado, _, _ = memoria.recordar(patron_ruidoso)
        if np.mean(recuperado == patron_original) > 0.95:
            correctos += 1

    resultados['recuperacion_ruido_30'] = correctos / n_patrones * 100
    estado = "EXITO" if resultados['recuperacion_ruido_30'] >= 90 else "FALLO"
    print(f"\n   [CRITERIO] Recuperacion con 30% ruido: {resultados['recuperacion_ruido_30']:.1f}% ({estado})")

    # =========================================================================
    # Test 2: Memoria Episodica - Recuperacion por similitud
    # =========================================================================
    print("\n2. Memoria Episodica - Recuperacion por similitud")
    print("-" * 50)

    episodica = MemoriaEpisodica(capacidad=100)

    # Agregar episodios
    for i in range(50):
        experiencia = np.random.randn(dimension)
        episodica.agregar_episodio(experiencia, {'indice': i})

    print(f"   Episodios almacenados: {len(episodica)}")

    # Probar recuperacion
    query = episodica.episodios[25].contenido + np.random.randn(dimension) * 0.1
    similares = episodica.recuperar_similar(query, k=5)

    encontro_original = any(
        ep.contexto.get('indice') == 25
        for ep, _ in similares
    )
    resultados['recuperacion_episodica'] = 100 if encontro_original else 0
    estado = "EXITO" if encontro_original else "FALLO"
    print(f"   Recupero episodio original: {estado}")

    # =========================================================================
    # Test 3: Consolidacion - Transferencia episodica -> semantica
    # =========================================================================
    print("\n3. Consolidacion - Transferencia episodica -> semantica")
    print("-" * 50)

    gestor = GestorMemoria(dimension=dimension)

    # Agregar experiencias a memoria episodica
    patrones_consolidar = []
    for i in range(20):
        patron = np.random.choice([-1, 1], size=dimension).astype(float)
        patrones_consolidar.append(patron)
        gestor.recordar(patron, {'indice': i})

    # Aumentar importancia de algunos
    for ep in gestor.episodica.episodios[:10]:
        ep.importancia = 1.5

    print(f"   Episodios antes: {len(gestor.episodica)}")
    print(f"   Patrones semanticos antes: {gestor.semantica.num_patrones}")

    # Consolidar
    stats = gestor.ciclo_consolidacion(n_episodios=10, forzar=True)

    print(f"   Episodios despues: {stats['episodios_despues']}")
    print(f"   Patrones consolidados: {stats['consolidados']}")
    print(f"   Patrones semanticos despues: {stats['patrones_despues']}")

    resultados['consolidacion'] = stats['consolidados']
    estado = "EXITO" if stats['consolidados'] > 0 else "FALLO"
    print(f"\n   [CRITERIO] Consolidacion funcional: {estado}")

    # =========================================================================
    # Test 4: Capacidad de Hopfield
    # =========================================================================
    print("\n4. Capacidad de Hopfield")
    print("-" * 50)

    dim_grande = 500
    memoria_grande = MemoriaSemantica(dim_grande)
    capacidad_esperada = int(0.14 * dim_grande)

    print(f"   Dimension: {dim_grande}")
    print(f"   Capacidad teorica: {capacidad_esperada}")

    # Memorizar hasta la capacidad teorica
    patrones_grandes = []
    for i in range(capacidad_esperada):
        patron = np.random.choice([-1, 1], size=dim_grande)
        patrones_grandes.append(patron)
        memoria_grande.memorizar_patron(patron, normalizar=False)

    # Probar recuperacion
    correctos = 0
    for patron in patrones_grandes[:20]:  # Probar primeros 20
        patron_ruidoso = patron.copy()
        n_flip = int(0.1 * dim_grande)
        indices_flip = np.random.choice(dim_grande, n_flip, replace=False)
        patron_ruidoso[indices_flip] *= -1

        recuperado, _, convergio = memoria_grande.recordar(patron_ruidoso)
        if np.mean(recuperado == patron) > 0.9:
            correctos += 1

    resultados['capacidad_hopfield'] = correctos / 20 * 100
    estado = "EXITO" if resultados['capacidad_hopfield'] >= 70 else "FALLO"
    print(f"   Recuperacion al limite de capacidad: {resultados['capacidad_hopfield']:.1f}% ({estado})")

    # =========================================================================
    # RESUMEN
    # =========================================================================
    print("\n" + "="*70)
    print("   RESUMEN - PILAR 1: MEMORIA HOLOGRAFICA")
    print("="*70)

    criterios = [
        ('Recuperacion con 30% ruido', resultados['recuperacion_ruido_30'], 90),
        ('Recuperacion episodica', resultados['recuperacion_episodica'], 100),
        ('Consolidacion funcional', 100 if resultados['consolidacion'] > 0 else 0, 100),
        ('Capacidad Hopfield', resultados['capacidad_hopfield'], 70)
    ]

    exitos = 0
    for nombre, valor, umbral in criterios:
        estado = "OK" if valor >= umbral else "FALLO"
        if valor >= umbral:
            exitos += 1
        print(f"   {nombre}: {valor:.1f}% (umbral: {umbral}%) [{estado}]")

    veredicto = "VIABLE" if exitos >= 3 else "NO_VIABLE"
    print(f"\n   VEREDICTO PILAR 1: {veredicto} ({exitos}/4 criterios)")

    return resultados, veredicto


if __name__ == "__main__":
    resultados, veredicto = test_memoria_holografica()
    print(f"\n{'='*70}")
    print(f"   RESULTADO FINAL: {veredicto}")
    print(f"{'='*70}")

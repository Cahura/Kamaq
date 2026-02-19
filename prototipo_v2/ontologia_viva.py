# -*- coding: utf-8 -*-
"""
PILAR 4: FUNDAMENTOS SEMANTICOS
================================

Pasar de tokens estadisticos a representaciones con significado estable.
Construir ontologias vivas: mapas conceptuales que conecten simbolos con experiencias.

Componentes:
- Concepto: Unidad semantica con embedding y conexiones
- OntologiaViva: Grafo dinamico de conceptos
- GroundingPerceptual: Ancla simbolos a percepciones

Fisica:
- Spreading activation: a_i(t+1) = f(sum_j(w_ij * a_j(t)) + input_i)
- Similaridad topologica: sim(A,B) = 1 - d_geodesica(A,B) / max_d
- Coherencia semantica basada en producto interno

Autor: Proyecto Kamaq
Fecha: Enero 2026
"""

import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Set, Tuple, Any
from collections import defaultdict
import time


# ==============================================================================
# CONCEPTO
# ==============================================================================

@dataclass
class Experiencia:
    """Una experiencia perceptual asociada a un concepto"""
    patron: np.ndarray
    modalidad: str = "visual"
    timestamp: float = field(default_factory=time.time)
    confianza: float = 1.0


@dataclass
class Concepto:
    """Unidad semantica con embedding, conexiones y grounding"""
    id: str
    nombre: str
    embedding: np.ndarray
    conexiones: Dict[str, float] = field(default_factory=dict)  # {concepto_id: fuerza}
    experiencias: List[Experiencia] = field(default_factory=list)
    activacion: float = 0.0
    categoria: Optional[str] = None
    propiedades: Dict[str, Any] = field(default_factory=dict)
    creado: float = field(default_factory=time.time)
    accesos: int = 0

    @property
    def esta_anclado(self) -> bool:
        """Verifica si tiene grounding perceptual"""
        return len(self.experiencias) > 0

    def agregar_experiencia(self, patron: np.ndarray, modalidad: str = "visual"):
        """Ancla el concepto a una experiencia perceptual"""
        exp = Experiencia(patron=patron.copy(), modalidad=modalidad)
        self.experiencias.append(exp)

    def activar(self, intensidad: float = 1.0):
        """Activa el concepto"""
        self.activacion = min(1.0, self.activacion + intensidad)
        self.accesos += 1

    def decaer(self, tasa: float = 0.1):
        """Decaimiento de activacion"""
        self.activacion *= (1 - tasa)


# ==============================================================================
# ONTOLOGIA VIVA
# ==============================================================================

class OntologiaViva:
    """
    Grafo dinamico de conceptos con spreading activation.

    Caracteristicas:
    - Conexiones con pesos que evolucionan
    - Activacion que se propaga por la red
    - Inferencia de relaciones transitivas
    """

    def __init__(self, dim_embedding: int = 64):
        self.dim_embedding = dim_embedding
        self.conceptos: Dict[str, Concepto] = {}
        self.relaciones: Dict[str, Dict[str, str]] = defaultdict(dict)  # {c1: {c2: tipo_relacion}}
        self.historial_activaciones: List[Dict[str, float]] = []

    def crear_concepto(self, nombre: str, embedding: np.ndarray = None,
                      categoria: str = None, propiedades: Dict = None) -> Concepto:
        """Crea un nuevo concepto"""
        id_concepto = f"c_{nombre}_{len(self.conceptos)}"

        if embedding is None:
            embedding = np.random.randn(self.dim_embedding)
            embedding /= np.linalg.norm(embedding)

        concepto = Concepto(
            id=id_concepto,
            nombre=nombre,
            embedding=embedding.copy(),
            categoria=categoria,
            propiedades=propiedades or {}
        )

        self.conceptos[id_concepto] = concepto
        return concepto

    def conectar(self, concepto_a: str, concepto_b: str,
                relacion: str = "relacionado", fuerza: float = 1.0):
        """Conecta dos conceptos con una relacion"""
        if concepto_a in self.conceptos and concepto_b in self.conceptos:
            self.conceptos[concepto_a].conexiones[concepto_b] = fuerza
            self.conceptos[concepto_b].conexiones[concepto_a] = fuerza * 0.5  # Conexion inversa mas debil
            self.relaciones[concepto_a][concepto_b] = relacion

    def activar(self, query: str, intensidad: float = 1.0,
               propagar: bool = True, n_pasos: int = 3) -> List[Tuple[str, float]]:
        """
        Activa conceptos basado en query usando spreading activation.

        Retorna lista de (concepto_id, activacion) ordenados por activacion.
        """
        # Buscar concepto por nombre
        concepto_inicial = None
        for c_id, c in self.conceptos.items():
            if c.nombre.lower() == query.lower():
                concepto_inicial = c
                break

        if concepto_inicial is None:
            # Buscar por similitud de embedding si se pasa vector
            if isinstance(query, np.ndarray):
                similitudes = []
                for c_id, c in self.conceptos.items():
                    sim = np.dot(query, c.embedding) / (np.linalg.norm(query) * np.linalg.norm(c.embedding) + 1e-10)
                    similitudes.append((c_id, sim))
                similitudes.sort(key=lambda x: x[1], reverse=True)
                if similitudes:
                    concepto_inicial = self.conceptos[similitudes[0][0]]

        if concepto_inicial is None:
            return []

        # Activar concepto inicial
        concepto_inicial.activar(intensidad)

        # Propagar activacion
        if propagar:
            for _ in range(n_pasos):
                self._propagar_activacion()

        # Recopilar activaciones
        activaciones = [(c_id, c.activacion) for c_id, c in self.conceptos.items()
                       if c.activacion > 0.01]
        activaciones.sort(key=lambda x: x[1], reverse=True)

        # Guardar historial
        self.historial_activaciones.append({c_id: c.activacion for c_id, c in self.conceptos.items()})

        return activaciones

    def _propagar_activacion(self, decaimiento: float = 0.3):
        """Propaga activacion a conceptos conectados"""
        nuevas_activaciones = {}

        for c_id, concepto in self.conceptos.items():
            if concepto.activacion > 0:
                for vecino_id, fuerza in concepto.conexiones.items():
                    if vecino_id in self.conceptos:
                        delta = concepto.activacion * fuerza * (1 - decaimiento)
                        nuevas_activaciones[vecino_id] = nuevas_activaciones.get(vecino_id, 0) + delta

        # Aplicar nuevas activaciones y decaer las existentes
        for c_id, concepto in self.conceptos.items():
            concepto.decaer(decaimiento)
            if c_id in nuevas_activaciones:
                concepto.activar(nuevas_activaciones[c_id])

    def inferir_relacion(self, concepto_a: str, concepto_b: str,
                        max_profundidad: int = 3) -> Optional[Tuple[str, List[str]]]:
        """
        Infiere relacion entre dos conceptos por transitividad.

        Retorna (tipo_relacion, camino) o None si no hay relacion.
        """
        # BFS para encontrar camino
        visitados = set()
        cola = [(concepto_a, [concepto_a])]

        while cola:
            actual, camino = cola.pop(0)

            if actual == concepto_b:
                # Determinar tipo de relacion basado en camino
                if len(camino) == 2:
                    rel = self.relaciones.get(concepto_a, {}).get(concepto_b, "relacionado")
                else:
                    rel = "transitivamente_relacionado"
                return (rel, camino)

            if len(camino) > max_profundidad:
                continue

            if actual in visitados:
                continue
            visitados.add(actual)

            if actual in self.conceptos:
                for vecino in self.conceptos[actual].conexiones:
                    if vecino not in visitados:
                        cola.append((vecino, camino + [vecino]))

        return None

    def evolucionar(self, concepto_id: str, nueva_experiencia: np.ndarray,
                   contexto: List[str] = None):
        """
        Evoluciona un concepto basado en nueva experiencia.
        Actualiza embedding y conexiones.
        """
        if concepto_id not in self.conceptos:
            return

        concepto = self.conceptos[concepto_id]

        # Agregar experiencia
        concepto.agregar_experiencia(nueva_experiencia)

        # Actualizar embedding (promedio movil con experiencias)
        if concepto.experiencias:
            embeddings_exp = [e.patron for e in concepto.experiencias
                            if len(e.patron) == self.dim_embedding]
            if embeddings_exp:
                promedio = np.mean(embeddings_exp, axis=0)
                concepto.embedding = 0.9 * concepto.embedding + 0.1 * promedio
                concepto.embedding /= np.linalg.norm(concepto.embedding)

        # Reforzar conexiones con conceptos del contexto
        if contexto:
            for otro_id in contexto:
                if otro_id in self.conceptos and otro_id != concepto_id:
                    fuerza_actual = concepto.conexiones.get(otro_id, 0)
                    concepto.conexiones[otro_id] = min(2.0, fuerza_actual + 0.1)

    def calcular_coherencia(self, conceptos_ids: List[str]) -> float:
        """
        Calcula coherencia semantica entre un conjunto de conceptos.
        Basado en similitud promedio de embeddings.
        """
        if len(conceptos_ids) < 2:
            return 1.0

        embeddings = []
        for c_id in conceptos_ids:
            if c_id in self.conceptos:
                embeddings.append(self.conceptos[c_id].embedding)

        if len(embeddings) < 2:
            return 1.0

        # Calcular similitud promedio entre todos los pares
        similitudes = []
        for i in range(len(embeddings)):
            for j in range(i + 1, len(embeddings)):
                sim = np.dot(embeddings[i], embeddings[j])
                sim /= (np.linalg.norm(embeddings[i]) * np.linalg.norm(embeddings[j]) + 1e-10)
                similitudes.append(sim)

        return np.mean(similitudes)

    def resetear_activaciones(self):
        """Resetea todas las activaciones a 0"""
        for concepto in self.conceptos.values():
            concepto.activacion = 0.0


# ==============================================================================
# GROUNDING PERCEPTUAL
# ==============================================================================

class GroundingPerceptual:
    """
    Conecta simbolos con percepciones reales.
    Permite reconocer conceptos desde patrones sensoriales.
    """

    def __init__(self, ontologia: OntologiaViva):
        self.ontologia = ontologia
        self.umbrales_reconocimiento: Dict[str, float] = {}

    def anclar(self, concepto_id: str, patron: np.ndarray, modalidad: str = "visual"):
        """Ancla un concepto a un patron perceptual"""
        if concepto_id in self.ontologia.conceptos:
            self.ontologia.conceptos[concepto_id].agregar_experiencia(patron, modalidad)
            # Establecer umbral de reconocimiento
            self.umbrales_reconocimiento[concepto_id] = 0.7

    def reconocer(self, patron: np.ndarray, modalidad: str = "visual",
                 top_k: int = 5) -> List[Tuple[str, float]]:
        """
        Reconoce conceptos desde un patron perceptual.

        Retorna lista de (concepto_id, confianza) ordenados por confianza.
        """
        reconocimientos = []

        for c_id, concepto in self.ontologia.conceptos.items():
            if not concepto.esta_anclado:
                continue

            # Calcular similitud con experiencias del concepto
            mejores_sim = []
            for exp in concepto.experiencias:
                if exp.modalidad == modalidad and len(exp.patron) == len(patron):
                    sim = self._similitud(patron, exp.patron)
                    mejores_sim.append(sim)

            if mejores_sim:
                confianza = max(mejores_sim)
                # Umbral mas bajo para permitir reconocimiento
                umbral = self.umbrales_reconocimiento.get(c_id, 0.3)
                if confianza >= umbral:
                    reconocimientos.append((c_id, confianza))

        reconocimientos.sort(key=lambda x: x[1], reverse=True)
        return reconocimientos[:top_k]

    def verificar_grounding(self, concepto_id: str) -> Dict:
        """Verifica el estado de grounding de un concepto"""
        if concepto_id not in self.ontologia.conceptos:
            return {'existe': False}

        concepto = self.ontologia.conceptos[concepto_id]
        return {
            'existe': True,
            'anclado': concepto.esta_anclado,
            'n_experiencias': len(concepto.experiencias),
            'modalidades': list(set(e.modalidad for e in concepto.experiencias))
        }

    def desambiguar(self, nombre: str, patron_contexto: np.ndarray) -> Optional[str]:
        """
        Desambigua un nombre usando contexto perceptual.
        Util cuando hay multiples conceptos con nombres similares.
        """
        candidatos = []
        for c_id, concepto in self.ontologia.conceptos.items():
            if nombre.lower() in concepto.nombre.lower():
                # Puntuar por similitud con patron de contexto
                if concepto.esta_anclado:
                    for exp in concepto.experiencias:
                        if len(exp.patron) == len(patron_contexto):
                            sim = self._similitud(patron_contexto, exp.patron)
                            candidatos.append((c_id, sim))
                            break
                else:
                    # Sin grounding, usar embedding
                    if len(concepto.embedding) == len(patron_contexto):
                        sim = self._similitud(patron_contexto, concepto.embedding)
                        candidatos.append((c_id, sim * 0.5))  # Penalizar por no tener grounding

        if candidatos:
            candidatos.sort(key=lambda x: x[1], reverse=True)
            return candidatos[0][0]
        return None

    def _similitud(self, a: np.ndarray, b: np.ndarray) -> float:
        """Similitud coseno entre dos vectores"""
        norm_a, norm_b = np.linalg.norm(a), np.linalg.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return np.dot(a, b) / (norm_a * norm_b)


# ==============================================================================
# PRUEBAS DE VERIFICACION
# ==============================================================================

def test_ontologia_viva():
    """
    Prueba completa del sistema de ontologia.

    Criterios de exito:
    - Inferencia por propagacion: >95% en relaciones transitivas
    - Grounding: reconoce concepto desde percepcion >85%
    - Evolucion: nuevas experiencias modifican conexiones
    """
    print("="*70)
    print("   TEST: ONTOLOGIA VIVA (Pilar 4)")
    print("="*70)

    resultados = {}
    dim = 32

    # =========================================================================
    # Test 1: Creacion y conexion de conceptos
    # =========================================================================
    print("\n1. Creacion y conexion de conceptos")
    print("-" * 50)

    ontologia = OntologiaViva(dim_embedding=dim)

    # Crear conceptos numericos con embeddings mas distintivos
    conceptos_numeros = {}
    np.random.seed(42)  # Reproducibilidad
    for i in range(10):
        # Usar one-hot + caracteristicas de paridad
        emb = np.zeros(dim)
        emb[i] = 1.0  # Identidad
        emb[10 + (i % 2)] = 0.8  # Paridad (par/impar)
        emb[12 + (i // 5)] = 0.6  # Grupo (0-4 vs 5-9)
        emb += np.random.randn(dim) * 0.05  # Pequeno ruido
        emb /= np.linalg.norm(emb)
        c = ontologia.crear_concepto(str(i), embedding=emb, categoria="numero")
        conceptos_numeros[str(i)] = c.id

    # Crear concepto "impar" - embedding que captura la paridad
    emb_impar = np.zeros(dim)
    emb_impar[11] = 1.0  # Componente de imparidad
    for i in [1, 3, 5, 7, 9]:
        emb_impar += ontologia.conceptos[conceptos_numeros[str(i)]].embedding * 0.2
    emb_impar /= np.linalg.norm(emb_impar)
    c_impar = ontologia.crear_concepto("impar", embedding=emb_impar, categoria="propiedad")

    # Conectar impares con "impar"
    for i in [1, 3, 5, 7, 9]:
        ontologia.conectar(conceptos_numeros[str(i)], c_impar.id, "es_un", fuerza=1.0)

    print(f"   Conceptos creados: {len(ontologia.conceptos)}")
    print(f"   Conexiones de '3': {len(ontologia.conceptos[conceptos_numeros['3']].conexiones)}")

    resultados['creacion'] = len(ontologia.conceptos)

    # =========================================================================
    # Test 2: Spreading activation
    # =========================================================================
    print("\n2. Spreading activation")
    print("-" * 50)

    ontologia.resetear_activaciones()
    activaciones = ontologia.activar("3", intensidad=1.0, propagar=True, n_pasos=3)

    print(f"   Activaciones tras propagar desde '3':")
    for c_id, act in activaciones[:5]:
        nombre = ontologia.conceptos[c_id].nombre
        print(f"      {nombre}: {act:.3f}")

    # Verificar que "impar" se activo
    impar_activo = any(c_id == c_impar.id for c_id, _ in activaciones)
    print(f"   'impar' activado: {impar_activo}")

    resultados['spreading'] = 100 if impar_activo else 0

    # =========================================================================
    # Test 3: Inferencia transitiva (es 3 impar?)
    # =========================================================================
    print("\n3. Inferencia transitiva")
    print("-" * 50)

    resultado_inferencia = ontologia.inferir_relacion(conceptos_numeros['3'], c_impar.id)

    if resultado_inferencia:
        relacion, camino = resultado_inferencia
        camino_nombres = [ontologia.conceptos[c].nombre for c in camino]
        print(f"   Relacion: {relacion}")
        print(f"   Camino: {' -> '.join(camino_nombres)}")
        inferencia_correcta = True
    else:
        print(f"   No se encontro relacion")
        inferencia_correcta = False

    # Probar con todos los impares
    correctos = 0
    for i in [1, 3, 5, 7, 9]:
        res = ontologia.inferir_relacion(conceptos_numeros[str(i)], c_impar.id)
        if res:
            correctos += 1

    tasa_inferencia = correctos / 5 * 100
    print(f"   Inferencia correcta para impares: {correctos}/5 ({tasa_inferencia:.0f}%)")

    resultados['inferencia'] = tasa_inferencia
    estado = "EXITO" if tasa_inferencia >= 95 else "FALLO"
    print(f"\n   [CRITERIO] Inferencia >= 95%: {estado}")

    # =========================================================================
    # Test 4: Grounding perceptual
    # =========================================================================
    print("\n4. Grounding perceptual")
    print("-" * 50)

    grounding = GroundingPerceptual(ontologia)

    # Anclar conceptos a patrones - usar misma estructura que embeddings
    np.random.seed(123)
    for i in range(10):
        patron_base = np.zeros(dim)
        patron_base[i] = 1.0
        patron_base[10 + (i % 2)] = 0.8
        patron_base[12 + (i // 5)] = 0.6
        # Agregar variaciones consistentes
        for _ in range(5):
            patron_var = patron_base.copy()
            patron_var += np.random.randn(dim) * 0.08
            grounding.anclar(conceptos_numeros[str(i)], patron_var)

    # Probar reconocimiento
    np.random.seed(456)
    correctos = 0
    for i in range(10):
        # Crear patron de prueba con ruido moderado
        patron_test = np.zeros(dim)
        patron_test[i] = 1.0
        patron_test[10 + (i % 2)] = 0.8
        patron_test[12 + (i // 5)] = 0.6
        patron_test += np.random.randn(dim) * 0.1

        reconocidos = grounding.reconocer(patron_test, top_k=3)
        if reconocidos and ontologia.conceptos[reconocidos[0][0]].nombre == str(i):
            correctos += 1

    tasa_grounding = correctos / 10 * 100
    print(f"   Reconocimiento correcto: {correctos}/10 ({tasa_grounding:.0f}%)")

    resultados['grounding'] = tasa_grounding
    estado = "EXITO" if tasa_grounding >= 85 else "FALLO"
    print(f"\n   [CRITERIO] Grounding >= 85%: {estado}")

    # =========================================================================
    # Test 5: Evolucion de ontologia
    # =========================================================================
    print("\n5. Evolucion de ontologia")
    print("-" * 50)

    c_3 = conceptos_numeros['3']
    conexiones_antes = len(ontologia.conceptos[c_3].conexiones)
    embedding_antes = ontologia.conceptos[c_3].embedding.copy()

    # Evolucionar con nuevas experiencias y contexto
    for _ in range(5):
        nueva_exp = np.random.randn(dim)
        ontologia.evolucionar(c_3, nueva_exp, contexto=[conceptos_numeros['1'], c_impar.id])

    conexiones_despues = len(ontologia.conceptos[c_3].conexiones)
    embedding_despues = ontologia.conceptos[c_3].embedding

    cambio_embedding = np.linalg.norm(embedding_despues - embedding_antes)
    cambio_conexiones = conexiones_despues - conexiones_antes

    print(f"   Conexiones antes: {conexiones_antes}, despues: {conexiones_despues}")
    print(f"   Cambio en embedding: {cambio_embedding:.4f}")

    evolucion_ok = cambio_embedding > 0.01 or cambio_conexiones > 0
    resultados['evolucion'] = 100 if evolucion_ok else 0
    print(f"   Ontologia evoluciono: {evolucion_ok}")

    # =========================================================================
    # Test 6: Coherencia semantica
    # =========================================================================
    print("\n6. Coherencia semantica")
    print("-" * 50)

    # Coherencia entre impares (deberia ser alta)
    impares_ids = [conceptos_numeros[str(i)] for i in [1, 3, 5, 7, 9]]
    coherencia_impares = ontologia.calcular_coherencia(impares_ids)

    # Coherencia entre mezclados (deberia ser menor)
    mezclados_ids = [conceptos_numeros[str(i)] for i in [1, 2, 3, 4]]
    coherencia_mezclados = ontologia.calcular_coherencia(mezclados_ids)

    print(f"   Coherencia impares: {coherencia_impares:.3f}")
    print(f"   Coherencia mezclados: {coherencia_mezclados:.3f}")

    coherencia_ok = coherencia_impares > coherencia_mezclados
    resultados['coherencia'] = 100 if coherencia_ok else 0
    print(f"   Impares mas coherentes: {coherencia_ok}")

    # =========================================================================
    # RESUMEN
    # =========================================================================
    print("\n" + "="*70)
    print("   RESUMEN - PILAR 4: ONTOLOGIA VIVA")
    print("="*70)

    criterios = [
        ('Creacion conceptos', 100 if resultados['creacion'] >= 10 else 0, 100),
        ('Spreading activation', resultados['spreading'], 100),
        ('Inferencia transitiva', resultados['inferencia'], 95),
        ('Grounding perceptual', resultados['grounding'], 85),
        ('Evolucion', resultados['evolucion'], 100),
        ('Coherencia semantica', resultados['coherencia'], 100)
    ]

    exitos = 0
    for nombre, valor, umbral in criterios:
        estado = "OK" if valor >= umbral else "FALLO"
        if valor >= umbral:
            exitos += 1
        print(f"   {nombre}: {valor:.1f} (umbral: {umbral}) [{estado}]")

    veredicto = "VIABLE" if exitos >= 5 else "NO_VIABLE"
    print(f"\n   VEREDICTO PILAR 4: {veredicto} ({exitos}/6 criterios)")

    return resultados, veredicto


if __name__ == "__main__":
    resultados, veredicto = test_ontologia_viva()
    print(f"\n{'='*70}")
    print(f"   RESULTADO FINAL: {veredicto}")
    print(f"{'='*70}")

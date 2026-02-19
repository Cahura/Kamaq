"""
KAMAQ Holographic Memory: Core Implementation
==============================================
Sistema de memoria escalable basado en Holographic Reduced Representations (HRR).

Principios:
- Distribuida: Información en todo el vector
- Escalable: Crece con el conocimiento
- Elegante: Basada en teoría de señales (FFT)
- Persistente: Nunca olvida

Autor: KAMAQ Team
Fecha: 20 de Enero, 2026
"""

import numpy as np
from scipy.fft import fft, ifft
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime
import json
import hashlib
import os
from pathlib import Path


@dataclass
class MemoryTrace:
    """Traza de memoria individual."""
    key: str
    content: str
    timestamp: str
    memory_type: str  # "episodic", "semantic", "procedural"
    importance: float
    access_count: int = 0
    

@dataclass
class RecallResult:
    """Resultado de una consulta de memoria."""
    found: bool
    key: str
    content: str
    confidence: float
    memory_type: str
    related_memories: List[str] = field(default_factory=list)


class HolographicMemory:
    """
    Memoria Holográfica Escalable para KAMAQ.
    
    A diferencia de implementaciones fijas, esta memoria CRECE:
    - Nuevos conceptos agregan nuevos vectores
    - La memoria principal se expande dinámicamente
    - Índices facilitan recuperación eficiente
    
    Estructura:
    - memory_bank: Dict[str, np.ndarray] - Vectores por concepto
    - trace_index: Dict[str, MemoryTrace] - Metadatos
    - holographic_sum: np.ndarray - Suma holográfica para búsqueda rápida
    """
    
    def __init__(self, 
                 base_dim: int = 1024,
                 memory_path: str = "memory",
                 seed: int = 42):
        """
        Inicializa memoria holográfica escalable.
        
        Args:
            base_dim: Dimensión base para vectores
            memory_path: Ruta para persistencia
            seed: Semilla para reproducibilidad
        """
        self.base_dim = base_dim
        self.memory_path = Path(memory_path)
        self.memory_path.mkdir(parents=True, exist_ok=True)
        self.rng = np.random.default_rng(seed)
        
        # Banco de memoria escalable
        self.memory_bank: Dict[str, np.ndarray] = {}
        
        # Índice de trazas con metadatos
        self.trace_index: Dict[str, MemoryTrace] = {}
        
        # Suma holográfica para búsqueda rápida
        self.holographic_sum = self._zero_vector()
        
        # Contadores
        self.total_encodings = 0
        self.total_recalls = 0
        self.session_count = 0
        
        # Cargar si existe
        self._try_load()
    
    # ==========================================================================
    # OPERACIONES FUNDAMENTALES HRR
    # ==========================================================================
    
    def _zero_vector(self) -> np.ndarray:
        """Vector cero de dimensión base."""
        return np.zeros(self.base_dim, dtype=np.float64)
    
    def _random_vector(self) -> np.ndarray:
        """Genera vector aleatorio unitario."""
        v = self.rng.standard_normal(self.base_dim)
        return v / np.linalg.norm(v)
    
    def _text_to_vector(self, text: str) -> np.ndarray:
        """
        Convierte texto a vector determinístico.
        El mismo texto siempre produce el mismo vector.
        """
        # Hash del texto para semilla determinística
        text_hash = hashlib.sha256(text.encode('utf-8')).hexdigest()
        seed = int(text_hash[:8], 16)
        local_rng = np.random.default_rng(seed)
        
        v = local_rng.standard_normal(self.base_dim)
        return v / np.linalg.norm(v)
    
    def _circular_convolve(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """
        Convolución circular (binding en HRR).
        Asocia dos conceptos en un solo vector.
        """
        return np.real(ifft(fft(a) * fft(b)))
    
    def _circular_correlate(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        """
        Correlación circular (unbinding en HRR).
        Recupera un concepto dado el otro.
        """
        return np.real(ifft(fft(a) * np.conj(fft(b))))
    
    def _normalize(self, v: np.ndarray) -> np.ndarray:
        """Normaliza vector a norma unitaria."""
        norm = np.linalg.norm(v)
        if norm > 1e-10:
            return v / norm
        return v
    
    def _similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        """Similitud coseno entre vectores."""
        norm_a = np.linalg.norm(a)
        norm_b = np.linalg.norm(b)
        if norm_a < 1e-10 or norm_b < 1e-10:
            return 0.0
        return float(np.dot(a, b) / (norm_a * norm_b))
    
    # ==========================================================================
    # OPERACIONES DE MEMORIA
    # ==========================================================================
    
    def encode(self, 
               key: str, 
               content: str, 
               memory_type: str = "semantic",
               importance: float = 0.5) -> bool:
        """
        Codifica conocimiento en memoria holográfica.
        
        Args:
            key: Clave/concepto principal
            content: Contenido a memorizar
            memory_type: "episodic", "semantic", "procedural"
            importance: 0.0 a 1.0, afecta peso en memoria
            
        Returns:
            True si se codificó exitosamente
        """
        # Generar vectores
        key_vec = self._text_to_vector(key)
        content_vec = self._text_to_vector(content)
        
        # Binding: key ⊛ content
        bound_vec = self._circular_convolve(key_vec, content_vec)
        
        # Escalar por importancia
        weighted_vec = bound_vec * importance
        
        # Almacenar en banco de memoria (ESCALABLE)
        memory_id = f"{memory_type}_{self.total_encodings}"
        self.memory_bank[memory_id] = weighted_vec
        
        # Actualizar suma holográfica
        self.holographic_sum = self._normalize(
            self.holographic_sum + weighted_vec
        )
        
        # Crear traza con metadatos
        trace = MemoryTrace(
            key=key,
            content=content,
            timestamp=datetime.now().isoformat(),
            memory_type=memory_type,
            importance=importance
        )
        self.trace_index[key] = trace
        
        self.total_encodings += 1
        return True
    
    def recall(self, 
               query: str, 
               threshold: float = 0.3,
               max_results: int = 5) -> RecallResult:
        """
        Recupera memoria asociada a query.
        
        Nota: La recuperación actual usa búsqueda de texto (substring +
        palabras clave), NO los vectores HRR almacenados. Los vectores
        holográficos se codifican correctamente en encode() pero el
        recall semántico basado en ellos no está implementado aún.
        
        Estrategias actuales:
        1. Match exacto en trace_index
        2. Substring matching
        3. Similitud de palabras clave (Jaccard)
        
        Args:
            query: Consulta de memoria
            threshold: Umbral mínimo de similitud
            max_results: Máximo de resultados relacionados
            
        Returns:
            RecallResult con memoria encontrada
        """
        self.total_recalls += 1
        query_lower = query.lower()
        query_words = set(query_lower.split())
        
        # Buscar match directo en índice
        if query in self.trace_index:
            trace = self.trace_index[query]
            trace.access_count += 1
            return RecallResult(
                found=True,
                key=trace.key,
                content=trace.content,
                confidence=1.0,
                memory_type=trace.memory_type,
                related_memories=[]
            )
        
        # Búsqueda por múltiples criterios
        candidates = []
        for key, trace in self.trace_index.items():
            key_lower = key.lower()
            content_lower = trace.content.lower()
            key_words = set(key_lower.split())
            
            score = 0.0
            
            # Criterio 1: Query contenido en key
            if query_lower in key_lower:
                score += 0.8
            
            # Criterio 2: Key contenido en query
            if key_lower in query_lower:
                score += 0.7
            
            # Criterio 3: Palabras comunes
            common_words = query_words & key_words
            if common_words:
                word_score = len(common_words) / max(len(query_words), len(key_words))
                score += word_score * 0.6
            
            # Criterio 4: Query en contenido
            if query_lower in content_lower:
                score += 0.5
            
            # Criterio 5: Palabras en contenido
            words_in_content = sum(1 for w in query_words if w in content_lower and len(w) > 3)
            if words_in_content:
                score += words_in_content * 0.1
            
            if score >= threshold:
                candidates.append((min(1.0, score), key, trace))
        
        if not candidates:
            return RecallResult(
                found=False,
                key=query,
                content="",
                confidence=0.0,
                memory_type="unknown",
                related_memories=[]
            )
        
        # Ordenar por score
        candidates.sort(reverse=True, key=lambda x: x[0])
        
        best = candidates[0]
        related = [c[1] for c in candidates[1:max_results]]
        
        best[2].access_count += 1
        
        return RecallResult(
            found=True,
            key=best[1],
            content=best[2].content,
            confidence=best[0],
            memory_type=best[2].memory_type,
            related_memories=related
        )
    
    def recall_by_type(self, 
                       memory_type: str, 
                       limit: int = 10) -> List[MemoryTrace]:
        """
        Recupera memorias por tipo.
        
        Args:
            memory_type: "episodic", "semantic", "procedural"
            limit: Máximo de resultados
            
        Returns:
            Lista de trazas de memoria
        """
        results = [
            trace for trace in self.trace_index.values()
            if trace.memory_type == memory_type
        ]
        # Ordenar por importancia y accesos
        results.sort(
            key=lambda t: (t.importance, t.access_count),
            reverse=True
        )
        return results[:limit]
    
    # ==========================================================================
    # APRENDIZAJE DE SESIONES
    # ==========================================================================
    
    def learn_from_interaction(self,
                               user_message: str,
                               assistant_response: str,
                               importance: float = 0.5):
        """
        Aprende de una interacción usuario-asistente.
        
        Args:
            user_message: Mensaje del usuario
            assistant_response: Respuesta del asistente
            importance: Importancia percibida
        """
        # Extraer tema principal (simplificado)
        topic = " ".join(user_message.split()[:10])
        
        # Codificar interacción
        self.encode(
            key=topic,
            content=f"Q: {user_message[:200]} A: {assistant_response[:300]}",
            memory_type="episodic",
            importance=importance
        )
    
    def consolidate_session(self, 
                            session_summary: str,
                            key_learnings: List[str]):
        """
        Consolida aprendizajes de una sesión completa.
        
        Args:
            session_summary: Resumen de la sesión
            key_learnings: Lista de aprendizajes clave
        """
        self.session_count += 1
        
        # Codificar resumen de sesión
        session_key = f"session_{self.session_count}"
        self.encode(
            key=session_key,
            content=session_summary,
            memory_type="episodic",
            importance=0.8
        )
        
        # Codificar cada aprendizaje
        for i, learning in enumerate(key_learnings):
            self.encode(
                key=f"{session_key}_learning_{i}",
                content=learning,
                memory_type="semantic",
                importance=0.9
            )
        
        # Persistir
        self.save()
    
    # ==========================================================================
    # PERSISTENCIA
    # ==========================================================================
    
    def save(self):
        """Guarda toda la memoria a disco."""
        # Guardar banco de memoria (numpy)
        memory_file = self.memory_path / "memory_bank.npz"
        np.savez_compressed(
            memory_file,
            holographic_sum=self.holographic_sum,
            **self.memory_bank
        )
        
        # Guardar índice (JSON)
        index_file = self.memory_path / "trace_index.json"
        index_data = {
            key: {
                "key": trace.key,
                "content": trace.content,
                "timestamp": trace.timestamp,
                "memory_type": trace.memory_type,
                "importance": trace.importance,
                "access_count": trace.access_count
            }
            for key, trace in self.trace_index.items()
        }
        with open(index_file, 'w', encoding='utf-8') as f:
            json.dump(index_data, f, ensure_ascii=False, indent=2)
        
        # Guardar metadatos
        meta_file = self.memory_path / "metadata.json"
        metadata = {
            "base_dim": self.base_dim,
            "total_encodings": self.total_encodings,
            "total_recalls": self.total_recalls,
            "session_count": self.session_count,
            "memory_bank_size": len(self.memory_bank),
            "last_save": datetime.now().isoformat()
        }
        with open(meta_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2)
    
    def _try_load(self):
        """Intenta cargar memoria existente."""
        memory_file = self.memory_path / "memory_bank.npz"
        index_file = self.memory_path / "trace_index.json"
        meta_file = self.memory_path / "metadata.json"
        
        if not memory_file.exists():
            return
        
        try:
            # Cargar banco de memoria
            data = np.load(memory_file)
            self.holographic_sum = data["holographic_sum"]
            for key in data.files:
                if key != "holographic_sum":
                    self.memory_bank[key] = data[key]
            
            # Cargar índice
            if index_file.exists():
                with open(index_file, 'r', encoding='utf-8') as f:
                    index_data = json.load(f)
                for key, trace_dict in index_data.items():
                    self.trace_index[key] = MemoryTrace(**trace_dict)
            
            # Cargar metadatos
            if meta_file.exists():
                with open(meta_file, 'r', encoding='utf-8') as f:
                    metadata = json.load(f)
                self.total_encodings = metadata.get("total_encodings", 0)
                self.total_recalls = metadata.get("total_recalls", 0)
                self.session_count = metadata.get("session_count", 0)
            
            print(f"[MEMORIA] Cargada: {len(self.trace_index)} trazas, "
                  f"{self.session_count} sesiones")
            
        except Exception as e:
            print(f"[MEMORIA] Error cargando: {e}")
    
    # ==========================================================================
    # ESTADÍSTICAS
    # ==========================================================================
    
    def get_stats(self) -> Dict[str, Any]:
        """Retorna estadísticas de la memoria."""
        by_type = {}
        for trace in self.trace_index.values():
            mtype = trace.memory_type
            by_type[mtype] = by_type.get(mtype, 0) + 1
        
        memory_size_bytes = sum(
            v.nbytes for v in self.memory_bank.values()
        ) + self.holographic_sum.nbytes
        
        return {
            "dimension": self.base_dim,
            "total_memories": len(self.trace_index),
            "memory_bank_vectors": len(self.memory_bank),
            "by_type": by_type,
            "total_encodings": self.total_encodings,
            "total_recalls": self.total_recalls,
            "session_count": self.session_count,
            "memory_size_kb": memory_size_bytes / 1024,
            "holographic_norm": float(np.linalg.norm(self.holographic_sum))
        }
    
    def __repr__(self) -> str:
        stats = self.get_stats()
        return (f"HolographicMemory(memories={stats['total_memories']}, "
                f"sessions={stats['session_count']}, "
                f"size={stats['memory_size_kb']:.1f}KB)")


# ==============================================================================
# DEMO Y TEST
# ==============================================================================

def demo():
    """Demostración de la memoria holográfica."""
    print("=" * 60)
    print("KAMAQ HOLOGRAPHIC MEMORY - DEMO")
    print("=" * 60)
    
    # Crear memoria
    memory = HolographicMemory(
        base_dim=1024,
        memory_path="memory"
    )
    
    print(f"\n[ESTADO INICIAL]")
    print(f"  {memory}")
    
    # Codificar conocimientos
    print(f"\n[CODIFICANDO CONOCIMIENTO]")
    
    memory.encode(
        key="KAMAQ es una IA honesta",
        content="KAMAQ es un sistema de IA basado en Physics-Based Intelligence "
                "que cuantifica la incertidumbre usando entropía termodinámica. "
                "Su principal característica es saber cuando NO sabe.",
        memory_type="semantic",
        importance=0.95
    )
    print("  + KAMAQ es una IA honesta")
    
    memory.encode(
        key="Memoria holográfica",
        content="La memoria holográfica usa Holographic Reduced Representations (HRR) "
                "para almacenar información de forma distribuida. Toda la información "
                "está en todo el vector, como un holograma real.",
        memory_type="semantic",
        importance=0.9
    )
    print("  + Memoria holográfica")
    
    memory.encode(
        key="Usuario prefiere código limpio",
        content="El usuario valora el código limpio, profesional, elegante y real. "
                "No le gustan las simulaciones ni los prototipos que no funcionan.",
        memory_type="episodic",
        importance=0.85
    )
    print("  + Usuario prefiere código limpio")
    
    # Recordar
    print(f"\n[RECUPERANDO MEMORIA]")
    
    result = memory.recall("que es KAMAQ")
    print(f"\n  Query: 'que es KAMAQ'")
    print(f"  Found: {result.found}")
    print(f"  Key: {result.key}")
    print(f"  Confidence: {result.confidence:.2f}")
    print(f"  Content: {result.content[:100]}...")
    
    result2 = memory.recall("preferencias del usuario")
    print(f"\n  Query: 'preferencias del usuario'")
    print(f"  Found: {result2.found}")
    print(f"  Key: {result2.key}")
    print(f"  Confidence: {result2.confidence:.2f}")
    
    # Estadísticas
    print(f"\n[ESTADÍSTICAS]")
    stats = memory.get_stats()
    for k, v in stats.items():
        print(f"  {k}: {v}")
    
    # Guardar
    print(f"\n[GUARDANDO]")
    memory.save()
    print("  Memoria guardada a disco")
    
    # Cargar nueva instancia
    print(f"\n[CARGANDO NUEVA INSTANCIA]")
    memory2 = HolographicMemory(
        base_dim=1024,
        memory_path="memory"
    )
    print(f"  {memory2}")
    
    # Verificar persistencia
    result3 = memory2.recall("KAMAQ")
    print(f"\n  Verificación: recall('KAMAQ')")
    print(f"  Found: {result3.found}, Confidence: {result3.confidence:.2f}")
    
    print("\n" + "=" * 60)
    print("DEMO COMPLETADA")
    print("=" * 60)
    
    return memory


if __name__ == "__main__":
    demo()

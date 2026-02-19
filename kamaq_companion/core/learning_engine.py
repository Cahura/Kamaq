"""
KAMAQ Learning Engine: Aprendizaje Proactivo
=============================================
Motor de aprendizaje que:
1. Extrae insights de conversaciones
2. Identifica patrones importantes
3. Consolida conocimiento entre sesiones
4. Aprende de errores y éxitos

Autor: KAMAQ Team
Fecha: 20 de Enero, 2026
"""

from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import json
import re
from pathlib import Path

from .holographic_memory import HolographicMemory, MemoryTrace


@dataclass
class Interaction:
    """Interacción usuario-asistente."""
    user_message: str
    assistant_response: str
    timestamp: str
    importance: float = 0.5
    topics: List[str] = field(default_factory=list)
    is_error_correction: bool = False


@dataclass
class SessionSummary:
    """Resumen de sesión."""
    session_id: int
    start_time: str
    end_time: str
    interactions_count: int
    key_topics: List[str]
    key_learnings: List[str]
    average_importance: float


class ProactiveLearningEngine:
    """
    Motor de aprendizaje proactivo para KAMAQ.
    
    Responsabilidades:
    - Analizar cada interacción para extraer conocimiento
    - Detectar patrones y temas recurrentes
    - Consolidar sesiones en memoria a largo plazo
    - Identificar y aprender de correcciones de errores
    """
    
    # Palabras clave que indican importancia
    IMPORTANCE_KEYWORDS = [
        "importante", "recuerda", "clave", "esencial", "crítico",
        "nunca olvides", "siempre", "fundamental", "must", "always",
        "remember", "critical", "essential", "key", "importante que sepas"
    ]
    
    # Palabras que indican corrección de error
    ERROR_CORRECTION_KEYWORDS = [
        "no es así", "está mal", "error", "incorrecto", "equivocado",
        "no, en realidad", "te equivocas", "wrong", "incorrect",
        "that's not right", "actually", "en realidad"
    ]
    
    def __init__(self, 
                 memory: HolographicMemory,
                 session_path: str = "memory/sessions"):
        """
        Inicializa motor de aprendizaje.
        
        Args:
            memory: Instancia de HolographicMemory
            session_path: Ruta para guardar sesiones
        """
        self.memory = memory
        self.session_path = Path(session_path)
        self.session_path.mkdir(parents=True, exist_ok=True)
        
        # Sesión actual
        self.current_interactions: List[Interaction] = []
        self.session_start = datetime.now().isoformat()
        self.session_id = memory.session_count + 1
        
        # Estadísticas
        self.topics_frequency: Dict[str, int] = {}
    
    # ==========================================================================
    # PROCESAMIENTO DE INTERACCIONES
    # ==========================================================================
    
    def process_interaction(self,
                           user_message: str,
                           assistant_response: str) -> Interaction:
        """
        Procesa una interacción y extrae conocimiento.
        
        Args:
            user_message: Mensaje del usuario
            assistant_response: Respuesta del asistente
            
        Returns:
            Interacción procesada
        """
        # Calcular importancia
        importance = self._calculate_importance(user_message, assistant_response)
        
        # Extraer tópicos
        topics = self._extract_topics(user_message)
        
        # Detectar corrección de error
        is_error = self._is_error_correction(user_message)
        
        # Crear interacción
        interaction = Interaction(
            user_message=user_message,
            assistant_response=assistant_response,
            timestamp=datetime.now().isoformat(),
            importance=importance,
            topics=topics,
            is_error_correction=is_error
        )
        
        # Agregar a sesión actual
        self.current_interactions.append(interaction)
        
        # Actualizar frecuencia de tópicos
        for topic in topics:
            self.topics_frequency[topic] = self.topics_frequency.get(topic, 0) + 1
        
        # Si es importante, codificar inmediatamente
        if importance >= 0.7 or is_error:
            self._encode_important_interaction(interaction)
        
        return interaction
    
    def _calculate_importance(self, 
                             user_msg: str, 
                             assistant_msg: str) -> float:
        """Calcula importancia de la interacción."""
        text = (user_msg + " " + assistant_msg).lower()
        
        score = 0.3  # Base
        
        # Palabras clave de importancia
        for keyword in self.IMPORTANCE_KEYWORDS:
            if keyword in text:
                score += 0.15
        
        # Longitud indica complejidad
        if len(user_msg) > 200:
            score += 0.1
        
        # Preguntas directas suelen ser importantes
        if "?" in user_msg:
            score += 0.05
        
        # Correcciones son muy importantes
        if any(kw in text for kw in self.ERROR_CORRECTION_KEYWORDS):
            score += 0.25
        
        return min(1.0, score)
    
    def _extract_topics(self, text: str) -> List[str]:
        """Extrae tópicos principales del texto."""
        # Limpieza básica
        text = text.lower()
        text = re.sub(r'[^\w\s]', ' ', text)
        
        # Palabras comunes a ignorar (stopwords simplificado)
        stopwords = {
            "el", "la", "los", "las", "un", "una", "unos", "unas",
            "que", "de", "en", "a", "por", "para", "con", "y", "o",
            "es", "son", "ser", "está", "están", "hay", "como", "más",
            "the", "a", "an", "is", "are", "was", "were", "be", "been",
            "to", "of", "and", "or", "for", "with", "on", "at", "by"
        }
        
        words = text.split()
        words = [w for w in words if len(w) > 3 and w not in stopwords]
        
        # Contar frecuencia
        freq = {}
        for w in words:
            freq[w] = freq.get(w, 0) + 1
        
        # Top 5 palabras como tópicos
        sorted_words = sorted(freq.items(), key=lambda x: x[1], reverse=True)
        return [w for w, _ in sorted_words[:5]]
    
    def _is_error_correction(self, text: str) -> bool:
        """Detecta si el usuario está corrigiendo un error."""
        text_lower = text.lower()
        return any(kw in text_lower for kw in self.ERROR_CORRECTION_KEYWORDS)
    
    def _encode_important_interaction(self, interaction: Interaction):
        """Codifica interacción importante en memoria."""
        # Key basado en tópicos
        key = " ".join(interaction.topics[:3]) if interaction.topics else "general"
        
        # Contenido resumido
        content = f"Q: {interaction.user_message[:150]} A: {interaction.assistant_response[:250]}"
        
        # Tipo de memoria
        if interaction.is_error_correction:
            memory_type = "procedural"  # Correcciones van a procedural
        else:
            memory_type = "episodic"
        
        self.memory.encode(
            key=key,
            content=content,
            memory_type=memory_type,
            importance=interaction.importance
        )
    
    # ==========================================================================
    # CONSOLIDACIÓN DE SESIONES
    # ==========================================================================
    
    def end_session(self) -> SessionSummary:
        """
        Finaliza y consolida la sesión actual.
        
        Returns:
            Resumen de la sesión
        """
        if not self.current_interactions:
            return SessionSummary(
                session_id=self.session_id,
                start_time=self.session_start,
                end_time=datetime.now().isoformat(),
                interactions_count=0,
                key_topics=[],
                key_learnings=[],
                average_importance=0.0
            )
        
        # Calcular estadísticas
        end_time = datetime.now().isoformat()
        avg_importance = sum(i.importance for i in self.current_interactions) / len(self.current_interactions)
        
        # Tópicos más frecuentes
        all_topics = []
        for interaction in self.current_interactions:
            all_topics.extend(interaction.topics)
        
        topic_freq = {}
        for t in all_topics:
            topic_freq[t] = topic_freq.get(t, 0) + 1
        
        key_topics = sorted(topic_freq, key=topic_freq.get, reverse=True)[:10]
        
        # Extraer aprendizajes clave (interacciones de alta importancia)
        key_learnings = []
        for interaction in self.current_interactions:
            if interaction.importance >= 0.7:
                summary = f"{interaction.user_message[:100]}..."
                key_learnings.append(summary)
        
        # Crear resumen
        summary = SessionSummary(
            session_id=self.session_id,
            start_time=self.session_start,
            end_time=end_time,
            interactions_count=len(self.current_interactions),
            key_topics=key_topics,
            key_learnings=key_learnings,
            average_importance=avg_importance
        )
        
        # Consolidar en memoria
        self._consolidate_to_memory(summary)
        
        # Guardar sesión a disco
        self._save_session(summary)
        
        # Reset para próxima sesión
        self.current_interactions = []
        self.session_start = datetime.now().isoformat()
        self.session_id += 1
        
        return summary
    
    def _consolidate_to_memory(self, summary: SessionSummary):
        """Consolida resumen de sesión en memoria."""
        # Codificar resumen general
        session_key = f"session_{summary.session_id}"
        session_content = (
            f"Sesión {summary.session_id}: {summary.interactions_count} interacciones. "
            f"Tópicos principales: {', '.join(summary.key_topics[:5])}. "
            f"Importancia promedio: {summary.average_importance:.2f}"
        )
        
        self.memory.encode(
            key=session_key,
            content=session_content,
            memory_type="episodic",
            importance=0.8
        )
        
        # Codificar tópicos como conocimiento semántico
        for topic in summary.key_topics[:5]:
            self.memory.encode(
                key=f"topic_{topic}",
                content=f"El usuario habló sobre '{topic}' en la sesión {summary.session_id}",
                memory_type="semantic",
                importance=0.6
            )
        
        # Actualizar memoria persistente
        self.memory.session_count = summary.session_id
        self.memory.save()
    
    def _save_session(self, summary: SessionSummary):
        """Guarda sesión completa a disco."""
        session_file = self.session_path / f"session_{summary.session_id}.json"
        
        data = {
            "summary": {
                "session_id": summary.session_id,
                "start_time": summary.start_time,
                "end_time": summary.end_time,
                "interactions_count": summary.interactions_count,
                "key_topics": summary.key_topics,
                "key_learnings": summary.key_learnings,
                "average_importance": summary.average_importance
            },
            "interactions": [
                {
                    "user_message": i.user_message,
                    "assistant_response": i.assistant_response,
                    "timestamp": i.timestamp,
                    "importance": i.importance,
                    "topics": i.topics,
                    "is_error_correction": i.is_error_correction
                }
                for i in self.current_interactions
            ]
        }
        
        with open(session_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    # ==========================================================================
    # ANÁLISIS PROACTIVO
    # ==========================================================================
    
    def get_frequent_topics(self, limit: int = 10) -> List[Tuple[str, int]]:
        """Retorna tópicos más frecuentes."""
        sorted_topics = sorted(
            self.topics_frequency.items(),
            key=lambda x: x[1],
            reverse=True
        )
        return sorted_topics[:limit]
    
    def suggest_related_memory(self, current_topic: str) -> Optional[str]:
        """
        Sugiere memoria relacionada proactivamente.
        
        Args:
            current_topic: Tópico actual de conversación
            
        Returns:
            Sugerencia de memoria relacionada o None
        """
        result = self.memory.recall(current_topic, threshold=0.4)
        
        if result.found and result.confidence > 0.5:
            if result.related_memories:
                return f"También podrías preguntar sobre: {', '.join(result.related_memories[:3])}"
        
        return None
    
    def get_learning_stats(self) -> Dict:
        """Estadísticas del motor de aprendizaje."""
        return {
            "session_id": self.session_id,
            "current_interactions": len(self.current_interactions),
            "unique_topics": len(self.topics_frequency),
            "top_topics": self.get_frequent_topics(5),
            "memory_stats": self.memory.get_stats()
        }

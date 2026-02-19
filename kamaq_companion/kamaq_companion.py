"""
KAMAQ Companion: Sistema Unificado
===================================
Integración de:
- Memoria Holográfica (escalable, persistente)
- Motor de Aprendizaje Proactivo
- Conexión con LLM Local (Ollama)

Autor: KAMAQ Team
Fecha: 20 de Enero, 2026
"""

import subprocess
import json
from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from core.holographic_memory import HolographicMemory, RecallResult
from core.learning_engine import ProactiveLearningEngine, SessionSummary


@dataclass
class CompanionResponse:
    """Respuesta del companion con metadata."""
    content: str
    confidence: float
    memory_used: bool
    memory_context: Optional[str]
    processing_time_ms: int


class KAMAQCompanion:
    """
    KAMAQ Companion: Tu compañero de IA local.
    
    Características:
    - Memoria holográfica que crece con el conocimiento
    - Aprendizaje proactivo de cada conversación
    - Integración con LLM local (Ollama)
    - Persistencia entre sesiones
    - Nunca olvida
    """
    
    def __init__(self,
                 name: str = "KAMAQ",
                 model: str = "mistral",
                 memory_path: str = "memory",
                 base_dim: int = 1024):
        """
        Inicializa KAMAQ Companion.
        
        Args:
            name: Nombre del companion
            model: Modelo LLM (debe estar instalado en Ollama)
            memory_path: Ruta para persistencia de memoria
            base_dim: Dimensión base para vectores holográficos
        """
        self.name = name
        self.model = model
        self.memory_path = Path(memory_path)
        self.memory_path.mkdir(parents=True, exist_ok=True)
        
        # Inicializar memoria holográfica
        print(f"[{self.name}] Inicializando memoria holográfica...")
        self.memory = HolographicMemory(
            base_dim=base_dim,
            memory_path=str(self.memory_path / "holographic")
        )
        
        # Inicializar motor de aprendizaje
        print(f"[{self.name}] Inicializando motor de aprendizaje...")
        self.learning = ProactiveLearningEngine(
            memory=self.memory,
            session_path=str(self.memory_path / "sessions")
        )
        
        # Verificar Ollama
        self.ollama_available = self._check_ollama()
        
        # Estadísticas de sesión
        self.messages_this_session = 0
        self.session_start = datetime.now()
        
        self._print_welcome()
    
    def _check_ollama(self) -> bool:
        """Verifica si Ollama está disponible."""
        try:
            result = subprocess.run(
                ["ollama", "list"],
                capture_output=True,
                text=True,
                timeout=5
            )
            if result.returncode == 0:
                if self.model in result.stdout:
                    print(f"[{self.name}] Ollama disponible con modelo {self.model}")
                    return True
                else:
                    print(f"[{self.name}] Modelo {self.model} no encontrado. "
                          f"Ejecuta: ollama pull {self.model}")
                    return False
            return False
        except (subprocess.SubprocessError, FileNotFoundError):
            print(f"[{self.name}] Ollama no disponible. "
                  "Instala desde https://ollama.ai")
            return False
    
    def _print_welcome(self):
        """Imprime mensaje de bienvenida."""
        stats = self.memory.get_stats()
        print()
        print("=" * 60)
        print(f"  {self.name} COMPANION")
        print("=" * 60)
        print(f"  Memorias almacenadas: {stats['total_memories']}")
        print(f"  Sesiones anteriores: {stats['session_count']}")
        print(f"  Tamaño de memoria: {stats['memory_size_kb']:.1f} KB")
        print(f"  LLM: {'[OK] ' + self.model if self.ollama_available else '[NO] No disponible'}")
        print("=" * 60)
        print()
    
    # ==========================================================================
    # CHAT
    # ==========================================================================
    
    def chat(self, message: str) -> CompanionResponse:
        """
        Procesa mensaje del usuario y genera respuesta.
        
        Args:
            message: Mensaje del usuario
            
        Returns:
            CompanionResponse con respuesta y metadata
        """
        start_time = datetime.now()
        self.messages_this_session += 1
        
        # 1. Buscar en memoria
        memory_result = self.memory.recall(message, threshold=0.4)
        memory_context = None
        
        if memory_result.found and memory_result.confidence > 0.5:
            memory_context = (
                f"[Memoria relevante (confianza {memory_result.confidence:.0%}): "
                f"{memory_result.content[:150]}...]"
            )
        
        # 2. Generar respuesta
        if self.ollama_available:
            response_text = self._generate_with_ollama(message, memory_context)
        else:
            response_text = self._generate_fallback(message, memory_context)
        
        # 3. Procesar para aprendizaje
        interaction = self.learning.process_interaction(message, response_text)
        
        # 4. Calcular tiempo
        processing_time = int((datetime.now() - start_time).total_seconds() * 1000)
        
        return CompanionResponse(
            content=response_text,
            confidence=interaction.importance,
            memory_used=memory_result.found,
            memory_context=memory_context,
            processing_time_ms=processing_time
        )
    
    def _generate_with_ollama(self, 
                              message: str, 
                              memory_context: Optional[str]) -> str:
        """Genera respuesta usando Ollama."""
        # Construir prompt con contexto de memoria
        prompt = self._build_prompt(message, memory_context)
        
        try:
            result = subprocess.run(
                ["ollama", "run", self.model, prompt],
                capture_output=True,
                text=True,
                timeout=60
            )
            
            if result.returncode == 0:
                return result.stdout.strip()
            else:
                return f"Error generando respuesta: {result.stderr}"
                
        except subprocess.TimeoutExpired:
            return "La respuesta tomó demasiado tiempo. Intenta una pregunta más corta."
        except Exception as e:
            return f"Error: {str(e)}"
    
    def _generate_fallback(self, 
                           message: str, 
                           memory_context: Optional[str]) -> str:
        """Respuesta de fallback sin LLM."""
        if memory_context:
            return (
                f"[Modo offline - LLM no disponible]\n\n"
                f"Tu mensaje: {message}\n\n"
                f"Encontré en mi memoria: {memory_context}\n\n"
                f"Para respuestas completas, instala Ollama: ollama pull {self.model}"
            )
        else:
            return (
                f"[Modo offline - LLM no disponible]\n\n"
                f"Tu mensaje: {message}\n\n"
                f"No encontré información relevante en mi memoria.\n\n"
                f"Para respuestas completas, instala Ollama: ollama pull {self.model}"
            )
    
    def _build_prompt(self, 
                      message: str, 
                      memory_context: Optional[str]) -> str:
        """Construye prompt para el LLM."""
        system_prompt = f"""Eres {self.name}, un asistente de IA honesto y directo.
Reglas importantes:
1. Si no sabes algo, di "No lo sé" en lugar de inventar
2. Sé conciso y directo
3. Si tienes información de memoria, úsala
4. Nunca inventes hechos"""

        if memory_context:
            return f"""{system_prompt}

CONTEXTO DE MEMORIA:
{memory_context}

USUARIO: {message}

{self.name}:"""
        else:
            return f"""{system_prompt}

USUARIO: {message}

{self.name}:"""
    
    # ==========================================================================
    # MEMORIA DIRECTA
    # ==========================================================================
    
    def remember(self, 
                 key: str, 
                 content: str, 
                 importance: float = 0.8) -> bool:
        """
        Almacena información directamente en memoria.
        
        Args:
            key: Concepto o tema
            content: Información a recordar
            importance: Importancia (0.0 a 1.0)
            
        Returns:
            True si se guardó exitosamente
        """
        success = self.memory.encode(
            key=key,
            content=content,
            memory_type="semantic",
            importance=importance
        )
        
        if success:
            print(f"[{self.name}] Recordado: '{key}'")
        
        return success
    
    def recall(self, query: str) -> RecallResult:
        """
        Busca información en memoria.
        
        Args:
            query: Consulta de búsqueda
            
        Returns:
            RecallResult con información encontrada
        """
        return self.memory.recall(query)
    
    # ==========================================================================
    # SESIONES
    # ==========================================================================
    
    def end_session(self) -> SessionSummary:
        """
        Finaliza la sesión actual y consolida aprendizajes.
        
        Returns:
            Resumen de la sesión
        """
        print(f"\n[{self.name}] Finalizando sesión...")
        
        summary = self.learning.end_session()
        
        print(f"[{self.name}] Sesión {summary.session_id} consolidada:")
        print(f"  - Interacciones: {summary.interactions_count}")
        print(f"  - Tópicos principales: {', '.join(summary.key_topics[:5])}")
        print(f"  - Aprendizajes clave: {len(summary.key_learnings)}")
        
        # Reset contadores de sesión
        self.messages_this_session = 0
        self.session_start = datetime.now()
        
        return summary
    
    def get_stats(self) -> Dict:
        """Retorna estadísticas completas."""
        memory_stats = self.memory.get_stats()
        learning_stats = self.learning.get_learning_stats()
        
        return {
            "companion": {
                "name": self.name,
                "model": self.model,
                "ollama_available": self.ollama_available,
                "messages_this_session": self.messages_this_session,
                "session_duration_minutes": int(
                    (datetime.now() - self.session_start).total_seconds() / 60
                )
            },
            "memory": memory_stats,
            "learning": learning_stats
        }
    
    # ==========================================================================
    # INTERFAZ DE CONSOLA
    # ==========================================================================
    
    def interactive_session(self):
        """Inicia sesión interactiva en consola."""
        print(f"\n[{self.name}] Sesión interactiva iniciada")
        print("Escribe 'salir' para terminar, 'stats' para estadísticas")
        print("-" * 60)
        
        try:
            while True:
                try:
                    user_input = input(f"\nTú: ").strip()
                except EOFError:
                    break
                
                if not user_input:
                    continue
                
                if user_input.lower() in ['salir', 'exit', 'quit']:
                    break
                
                if user_input.lower() == 'stats':
                    self._print_stats()
                    continue
                
                if user_input.lower().startswith('recuerda:'):
                    parts = user_input[9:].split('=', 1)
                    if len(parts) == 2:
                        self.remember(parts[0].strip(), parts[1].strip())
                    else:
                        print("Formato: recuerda: clave = valor")
                    continue
                
                # Chat normal
                response = self.chat(user_input)
                
                print(f"\n{self.name}: {response.content}")
                
                if response.memory_used:
                    print(f"  [Usé memoria: {response.memory_context[:50]}...]")
                
        except KeyboardInterrupt:
            print("\n\nInterrumpido por usuario")
        
        # Finalizar sesión
        self.end_session()
        print(f"\n[{self.name}] ¡Hasta pronto!")
    
    def _print_stats(self):
        """Imprime estadísticas formateadas."""
        stats = self.get_stats()
        
        print("\n" + "=" * 40)
        print("ESTADÍSTICAS")
        print("=" * 40)
        
        print("\n[Companion]")
        for k, v in stats["companion"].items():
            print(f"  {k}: {v}")
        
        print("\n[Memoria]")
        for k, v in stats["memory"].items():
            print(f"  {k}: {v}")
        
        print("=" * 40)


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    """Punto de entrada principal."""
    companion = KAMAQCompanion(
        name="KAMAQ",
        model="mistral",
        memory_path="memory",
        base_dim=1024
    )
    
    companion.interactive_session()


if __name__ == "__main__":
    main()

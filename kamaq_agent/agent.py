"""
KAMAQ Super-Agent: Integración Principal
=========================================
Agente modular que integra:
- Constitución y valores (evaluación de riesgo)
- Razonamiento estructurado (pipeline de 6 pasos)
- Herramientas (calculadora, archivos, Python sandbox, shell)
- Verificación (matemática, contradicciones, calibración)
- LLM local (Ollama)
- Historial de conversación (contexto entre turnos)

Autor: Carlos Huarcaya
Fecha: Enero-Febrero 2026
"""

import subprocess
import json
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime
from pathlib import Path
from enum import Enum

from kamaq_agent.core.constitution import KAMAQConstitution, RiskLevel, OperatingMode
from kamaq_agent.core.reasoning import ReasoningPipeline, QuickReasoner, ReasoningTrace
from kamaq_agent.core.tools import ToolRegistry, create_default_registry, ToolResult
from kamaq_agent.core.verifier import KAMAQVerifier


class AgentState(Enum):
    """Estados del agente."""
    IDLE = "idle"
    THINKING = "thinking"
    EXECUTING = "executing"
    WAITING_CONFIRMATION = "waiting_confirmation"
    ERROR = "error"


@dataclass
class ConversationTurn:
    """Un turno en la conversación."""
    role: str  # "user" o "assistant"
    content: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class AgentResponse:
    """Respuesta completa del agente."""
    content: str
    confidence: float
    reasoning_trace: Optional[ReasoningTrace]
    tools_used: List[str]
    verification_warnings: List[str]
    processing_time_ms: int
    mode: OperatingMode
    
    def to_dict(self) -> Dict:
        return {
            "content": self.content,
            "confidence": self.confidence,
            "tools_used": self.tools_used,
            "warnings": self.verification_warnings,
            "processing_time_ms": self.processing_time_ms,
            "mode": self.mode.value
        }


class KAMAQSuperAgent:
    """
    KAMAQ Super-Agent: Asistente de IA local con razonamiento estructurado.
    
    Características:
    - Razonamiento estructurado en pasos
    - Herramientas para interactuar con el mundo
    - Verificación y honestidad integradas
    - Historial de conversación entre turnos
    - LLM local vía Ollama
    """
    
    VERSION = "0.2.0"
    MAX_HISTORY_TURNS = 20  # Conversation window
    
    def __init__(self,
                 name: str = "KAMAQ",
                 model: str = "qwen2.5:7b",
                 memory_path: str = "memory",
                 constitution_path: str = None,
                 verbose: bool = True):
        """
        Inicializa KAMAQ Super-Agent.
        
        Args:
            name: Nombre del agente
            model: Modelo LLM (Ollama)
            memory_path: Ruta para persistencia
            constitution_path: Ruta a constitución personalizada
            verbose: Mostrar mensajes de debug
        """
        self.name = name
        self.model = model
        self.memory_path = Path(memory_path)
        self.memory_path.mkdir(parents=True, exist_ok=True)
        self.verbose = verbose
        
        self.state = AgentState.IDLE
        self.current_mode = OperatingMode.ASSISTANT
        
        # Conversation history for context between turns
        self.conversation_history: List[ConversationTurn] = []
        # Pending confirmation state
        self._pending_confirmation: Optional[Dict] = None
        
        self._log(f"Inicializando {name} Super-Agent v{self.VERSION}...")
        
        # 1. Cargar constitución
        if constitution_path and Path(constitution_path).exists():
            self.constitution = KAMAQConstitution.load(constitution_path)
            self._log("Constitución cargada desde archivo")
        else:
            self.constitution = KAMAQConstitution()
            self._log("Usando constitución por defecto")
        
        # 2. Inicializar razonamiento
        self.reasoning = ReasoningPipeline(
            llm_caller=self._call_llm,
            min_confidence_to_answer=0.3
        )
        self.quick_reasoner = QuickReasoner(llm_caller=self._call_llm)
        
        # 3. Inicializar herramientas
        self.tools = create_default_registry()
        self._log(f"Herramientas cargadas: {len(self.tools.list_tools())}")
        
        # 4. Inicializar verificador
        self.verifier = KAMAQVerifier()
        
        # 5. Verificar Ollama
        self.ollama_available = self._check_ollama()
        
        # Estadísticas
        self.total_interactions = 0
        self.session_start = datetime.now()
        
        self._print_welcome()
    
    def _log(self, message: str):
        """Log interno."""
        if self.verbose:
            print(f"[{self.name}] {message}")
    
    def _check_ollama(self) -> bool:
        """Verifica disponibilidad de Ollama."""
        try:
            result = subprocess.run(
                ["ollama", "list"],
                capture_output=True,
                text=True,
                timeout=5
            )
            
            if result.returncode == 0:
                # Verificar si el modelo está disponible
                available_models = result.stdout.lower()
                model_base = self.model.split(':')[0]
                
                if model_base in available_models:
                    self._log(f"Ollama disponible con modelo {self.model}")
                    return True
                else:
                    self._log(f"Modelo {self.model} no encontrado")
                    self._log(f"Ejecuta: ollama pull {self.model}")
                    return False
            
            return False
            
        except Exception as e:
            self._log(f"Ollama no disponible: {e}")
            return False
    
    def _print_welcome(self):
        """Mensaje de bienvenida."""
        print()
        print("=" * 60)
        print(f"  {self.name} Agent v{self.VERSION}")
        print("=" * 60)
        print(f"  LLM: {'[OK] ' + self.model if self.ollama_available else '[X] No disponible'}")
        print(f"  Herramientas: {len(self.tools.list_tools())} disponibles")
        print(f"  Historial: {self.MAX_HISTORY_TURNS} turnos máximo")
        print("=" * 60)
        print()
    
    # =========================================================================
    # LLAMADA AL LLM
    # =========================================================================
    
    def _call_llm(self, prompt: str, temperature: float = 0.7) -> str:
        """
        Llama al LLM local vía Ollama.
        
        Args:
            prompt: Prompt a enviar
            temperature: Creatividad (0-1)
            
        Returns:
            Respuesta del modelo
        """
        if not self.ollama_available:
            return self._fallback_response(prompt)
        
        # Construir prompt con sistema
        system_prompt = self.constitution.to_system_prompt()
        full_prompt = f"{system_prompt}\n\n{prompt}"
        
        try:
            # Usar API de Ollama
            result = subprocess.run(
                ["ollama", "run", self.model],
                input=full_prompt,
                capture_output=True,
                text=True,
                timeout=120  # 2 minutos máximo
            )
            
            if result.returncode == 0:
                return result.stdout.strip()
            else:
                self._log(f"Error de Ollama: {result.stderr}")
                return self._fallback_response(prompt)
                
        except subprocess.TimeoutExpired:
            return "Lo siento, la respuesta tomó demasiado tiempo. Intenta con una pregunta más simple."
        except Exception as e:
            self._log(f"Error llamando a Ollama: {e}")
            return self._fallback_response(prompt)
    
    def _fallback_response(self, prompt: str) -> str:
        """Respuesta de fallback cuando no hay LLM."""
        return f"""[Modo sin LLM]

No tengo acceso al modelo de lenguaje en este momento.

Tu mensaje fue: {prompt[:100]}...

Para habilitar respuestas completas:
1. Instala Ollama: https://ollama.ai
2. Descarga un modelo: ollama pull {self.model}
3. Reinicia el agente

Mientras tanto, puedo ejecutar herramientas básicas."""
    
    # =========================================================================
    # PROCESAMIENTO PRINCIPAL
    # =========================================================================
    
    def chat(self, 
             message: str,
             use_reasoning: bool = True,
             use_tools: bool = True) -> AgentResponse:
        """
        Procesa un mensaje del usuario.
        
        Args:
            message: Mensaje del usuario
            use_reasoning: Usar pipeline de razonamiento
            use_tools: Permitir uso de herramientas
            
        Returns:
            AgentResponse completa
        """
        start_time = datetime.now()
        self.total_interactions += 1
        
        tools_used = []
        verification_warnings = []
        reasoning_trace = None
        
        try:
            # 0. Handle pending confirmation from previous turn
            if self._pending_confirmation:
                if message.lower() in ["sí", "si", "yes", "s", "y", "procede"]:
                    tool_info = self._pending_confirmation
                    self._pending_confirmation = None
                    self.state = AgentState.EXECUTING
                    tool_result = self._execute_tool(tool_info, tool_info.get("original_message", ""))
                    tools_used.append(tool_info["tool"])
                    response_text = self._integrate_tool_result(
                        tool_info.get("original_message", ""), tool_result
                    )
                    confidence = 0.8 if tool_result.success else 0.4
                else:
                    self._pending_confirmation = None
                    self.state = AgentState.IDLE
                    response_text = "Operación cancelada."
                    confidence = 1.0
                    # Fall through to history + return
                    processing_time = int((datetime.now() - start_time).total_seconds() * 1000)
                    self._add_to_history("user", message)
                    self._add_to_history("assistant", response_text)
                    return AgentResponse(
                        content=response_text,
                        confidence=confidence,
                        reasoning_trace=None,
                        tools_used=[],
                        verification_warnings=[],
                        processing_time_ms=processing_time,
                        mode=self.current_mode
                    )
            
            # 1. Add user message to history
            self._add_to_history("user", message)
            
            # 2. Evaluar riesgo del mensaje
            risk = self.constitution.evaluate_action_risk(message, {})
            if risk == RiskLevel.FORBIDDEN:
                return self._create_refusal_response(message, start_time)
            
            # 3. Detectar si necesita herramientas
            tool_needed = self._detect_tool_need(message) if use_tools else None
            
            # 4. Generar respuesta
            if tool_needed:
                # Ejecutar herramienta
                self.state = AgentState.EXECUTING
                tool_result = self._execute_tool(tool_needed, message)
                tools_used.append(tool_needed["tool"])
                
                if tool_result.error == "REQUIRES_CONFIRMATION":
                    tool_needed["original_message"] = message
                    self._pending_confirmation = tool_needed
                    return self._create_confirmation_request(
                        tool_needed, tool_result, start_time
                    )
                
                # Integrar resultado de herramienta en respuesta
                response_text = self._integrate_tool_result(
                    message, tool_result
                )
                confidence = 0.8 if tool_result.success else 0.4
                
            elif use_reasoning and self._needs_deep_reasoning(message):
                # Razonamiento estructurado completo
                context = self._get_history_context()
                reasoning_trace = self.reasoning.reason(message, context)
                response_text = reasoning_trace.final_answer
                confidence = reasoning_trace.overall_confidence
                
            else:
                # Respuesta rápida
                response_text, confidence = self.quick_reasoner.answer(message)
            
            # 5. Verificar respuesta
            verification = self.verifier.verify_response(
                response_text, confidence
            )
            
            if verification["warnings"]:
                verification_warnings = verification["warnings"]
            
            # Ajustar confianza
            confidence = verification["adjusted_confidence"]
            
            # Advertencia al usuario si es necesario
            warning = self.verifier.should_warn_user(verification)
            if warning:
                response_text = f"{warning}\n\n{response_text}"
            
            # 6. Add response to history
            self._add_to_history("assistant", response_text)
            
            # Calcular tiempo
            processing_time = int((datetime.now() - start_time).total_seconds() * 1000)
            
            self.state = AgentState.IDLE
            
            return AgentResponse(
                content=response_text,
                confidence=confidence,
                reasoning_trace=reasoning_trace,
                tools_used=tools_used,
                verification_warnings=verification_warnings,
                processing_time_ms=processing_time,
                mode=self.current_mode
            )
            
        except Exception as e:
            self.state = AgentState.ERROR
            self._log(f"Error procesando mensaje: {e}")
            
            return AgentResponse(
                content=f"Lo siento, ocurrió un error: {str(e)}",
                confidence=0.0,
                reasoning_trace=None,
                tools_used=[],
                verification_warnings=["Error interno"],
                processing_time_ms=int((datetime.now() - start_time).total_seconds() * 1000),
                mode=self.current_mode
            )
    
    def _needs_deep_reasoning(self, message: str) -> bool:
        """Determina si el mensaje requiere razonamiento profundo."""
        # Indicadores de complejidad
        complex_words = [
            "analiza", "explica", "compara", "evalúa", "por qué",
            "cómo funciona", "qué piensas", "argumenta", "demuestra",
            "analyze", "explain", "compare", "evaluate", "why",
            "how does", "what do you think", "prove"
        ]
        
        message_lower = message.lower()
        
        # Preguntas complejas
        if any(word in message_lower for word in complex_words):
            return True
        
        # Mensajes largos probablemente son complejos
        if len(message) > 200:
            return True
        
        return False
    
    def _detect_tool_need(self, message: str) -> Optional[Dict]:
        """Detecta si el mensaje requiere una herramienta."""
        message_lower = message.lower()
        
        # Patrones simples de detección
        if any(word in message_lower for word in ["lee", "leer", "muestra", "abre", "read"]):
            if any(word in message_lower for word in ["archivo", "file", ".py", ".txt", ".md"]):
                return {"tool": "read_file", "type": "file"}
        
        if any(word in message_lower for word in ["calcula", "cuánto", "suma", "resta", "calculate"]):
            return {"tool": "calculate", "type": "math"}
        
        if any(word in message_lower for word in ["ejecuta", "corre", "run", "python"]):
            if "código" in message_lower or "code" in message_lower:
                return {"tool": "python_exec", "type": "code"}
        
        if any(word in message_lower for word in ["escribe", "guarda", "crea archivo", "write"]):
            return {"tool": "write_file", "type": "file"}
        
        if any(word in message_lower for word in ["terminal", "comando", "command", "shell", "bash"]):
            return {"tool": "shell", "type": "shell"}
        
        return None
    
    def _execute_tool(self, tool_info: Dict, message: str) -> ToolResult:
        """Ejecuta una herramienta basada en el mensaje."""
        tool_name = tool_info["tool"]
        
        if tool_name == "calculate":
            numbers = re.findall(r'[\d\+\-\*\/\.\(\)\s\^]+', message)
            if numbers:
                expression = max(numbers, key=len).strip()
                return self.tools.execute(tool_name, expression=expression)
        
        elif tool_name == "read_file":
            paths = re.findall(r'[\w\/\.\-]+\.\w+', message)
            if paths:
                return self.tools.execute(tool_name, path=paths[0])
        
        elif tool_name == "write_file":
            # Extract path and content
            paths = re.findall(r'[\w\/\.\-]+\.\w+', message)
            code_match = re.search(r'```(?:\w+)?\n?(.*?)```', message, re.DOTALL)
            if paths and code_match:
                return self.tools.execute(
                    tool_name, path=paths[0], content=code_match.group(1)
                )
        
        elif tool_name == "python_exec":
            code_match = re.search(r'```(?:python)?\n?(.*?)```', message, re.DOTALL)
            if code_match:
                code = code_match.group(1)
                return self.tools.execute(tool_name, code=code)
        
        elif tool_name == "shell":
            # Extract command after keywords
            cmd_match = re.search(
                r'(?:ejecuta|corre|run|comando|command)\s+["\']?(.+?)["\']?$',
                message, re.IGNORECASE
            )
            if cmd_match:
                return self.tools.execute(tool_name, command=cmd_match.group(1))
        
        return ToolResult(
            success=False,
            output=None,
            error="No pude extraer parámetros del mensaje"
        )
    
    def _add_to_history(self, role: str, content: str):
        """Agrega un turno al historial de conversación."""
        self.conversation_history.append(
            ConversationTurn(role=role, content=content)
        )
        # Trim to max window
        if len(self.conversation_history) > self.MAX_HISTORY_TURNS * 2:
            self.conversation_history = self.conversation_history[-self.MAX_HISTORY_TURNS * 2:]
    
    def _get_history_context(self) -> str:
        """Construye contexto de conversación para el LLM."""
        if not self.conversation_history:
            return ""
        # Use last N turns as context
        recent = self.conversation_history[-self.MAX_HISTORY_TURNS * 2:]
        lines = []
        for turn in recent:
            prefix = "Usuario" if turn.role == "user" else "Asistente"
            lines.append(f"{prefix}: {turn.content[:300]}")
        return "\n".join(lines)
    
    def _integrate_tool_result(self, 
                               message: str,
                               tool_result: ToolResult) -> str:
        """Integra resultado de herramienta en respuesta natural."""
        if tool_result.success:
            result_str = str(tool_result.output)
            
            prompt = f"""El usuario pidió: {message}

Ejecuté una herramienta y obtuve: {result_str[:500]}

Genera una respuesta natural que presente este resultado al usuario.
Sé conciso pero informativo."""
            
            return self._call_llm(prompt)
        else:
            return f"No pude completar la operación: {tool_result.error}"
    
    def _create_refusal_response(self, message: str, start_time) -> AgentResponse:
        """Crea respuesta de rechazo para acciones prohibidas."""
        refusal = "Lo siento, no puedo realizar esa acción porque viola mis valores fundamentales."
        self._add_to_history("user", message)
        self._add_to_history("assistant", refusal)
        return AgentResponse(
            content=refusal,
            confidence=1.0,
            reasoning_trace=None,
            tools_used=[],
            verification_warnings=["Acción prohibida"],
            processing_time_ms=int((datetime.now() - start_time).total_seconds() * 1000),
            mode=self.current_mode
        )
    
    def _create_confirmation_request(self, 
                                     tool_info: Dict,
                                     tool_result: ToolResult,
                                     start_time) -> AgentResponse:
        """Crea solicitud de confirmación."""
        self.state = AgentState.WAITING_CONFIRMATION
        
        msg = self.constitution.format_escalation_message(
            action=f"Ejecutar {tool_info['tool']}",
            reason="Esta acción requiere confirmación por seguridad",
            options=["Sí, procede", "No, cancela"]
        )
        
        return AgentResponse(
            content=msg,
            confidence=0.0,
            reasoning_trace=None,
            tools_used=[],
            verification_warnings=["Esperando confirmación"],
            processing_time_ms=int((datetime.now() - start_time).total_seconds() * 1000),
            mode=self.current_mode
        )
    
    # =========================================================================
    # MÉTODOS DE UTILIDAD
    # =========================================================================
    
    def set_mode(self, mode: OperatingMode):
        """Cambia el modo de operación."""
        self.current_mode = mode
        self._log(f"Modo cambiado a: {mode.value}")
    
    def get_stats(self) -> Dict:
        """Obtiene estadísticas del agente."""
        return {
            "name": self.name,
            "version": self.VERSION,
            "total_interactions": self.total_interactions,
            "conversation_turns": len(self.conversation_history),
            "session_duration_minutes": int(
                (datetime.now() - self.session_start).total_seconds() / 60
            ),
            "state": self.state.value,
            "mode": self.current_mode.value,
            "ollama_available": self.ollama_available,
        }
    
    def get_calibration_report(self) -> str:
        """Obtiene reporte de calibración."""
        return self.verifier.get_calibration_report()
    
    def save_state(self, path: str = None):
        """Guarda estado del agente."""
        if path is None:
            path = str(self.memory_path / "agent_state.json")
        
        state = {
            "model": self.model,
            "mode": self.current_mode.value,
            "total_interactions": self.total_interactions,
            "saved_at": datetime.now().isoformat()
        }
        
        # Also save conversation history
        state["conversation_history"] = [
            {"role": t.role, "content": t.content, "timestamp": t.timestamp}
            for t in self.conversation_history[-50:]  # Last 50 turns
        ]
        
        with open(path, 'w') as f:
            json.dump(state, f, indent=2, ensure_ascii=False)
        
        self._log(f"Estado guardado en {path}")


# =============================================================================
# CLI INTERACTIVO
# =============================================================================

def main():
    """CLI interactivo para KAMAQ Super-Agent."""
    import argparse
    
    parser = argparse.ArgumentParser(description="KAMAQ Super-Agent")
    parser.add_argument("--model", default="qwen2.5:7b", help="Modelo Ollama")
    parser.add_argument("--memory", default="memory", help="Ruta de memoria")
    parser.add_argument("--quiet", action="store_true", help="Modo silencioso")
    
    args = parser.parse_args()
    
    # Crear agente
    agent = KAMAQSuperAgent(
        model=args.model,
        memory_path=args.memory,
        verbose=not args.quiet
    )
    
    print("\n>>> Escribe tu mensaje (o 'salir' para terminar):\n")
    
    while True:
        try:
            user_input = input("Tú: ").strip()
            
            if not user_input:
                continue
            
            if user_input.lower() in ["salir", "exit", "quit"]:
                print("\n[!] Hasta pronto!")
                agent.save_state()
                break
            
            if user_input.lower() == "/stats":
                print(json.dumps(agent.get_stats(), indent=2))
                continue
            
            if user_input.lower() == "/calibration":
                print(agent.get_calibration_report())
                continue
            
            if user_input.lower().startswith("/mode "):
                mode_name = user_input.split()[1]
                try:
                    mode = OperatingMode(mode_name)
                    agent.set_mode(mode)
                except ValueError:
                    print(f"Modo no válido. Opciones: {[m.value for m in OperatingMode]}")
                continue
            
            # Procesar mensaje
            response = agent.chat(user_input)
            
            print(f"\n{agent.name}: {response.content}")
            print(f"  [Confianza: {response.confidence:.0%} | Tiempo: {response.processing_time_ms}ms]")
            
            if response.tools_used:
                print(f"  [Herramientas: {', '.join(response.tools_used)}]")
            
            if response.verification_warnings:
                print(f"  [WARN: {', '.join(response.verification_warnings)}]")
            
            print()
            
        except KeyboardInterrupt:
            print("\n\n[!] Hasta pronto!")
            agent.save_state()
            break
        except Exception as e:
            print(f"\n[ERROR] {e}\n")


if __name__ == "__main__":
    main()

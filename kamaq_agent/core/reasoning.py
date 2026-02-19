"""
KAMAQ Reasoning Pipeline: Razonamiento Estructurado
====================================================
Pipeline de razonamiento en pasos explícitos.

Cada respuesta pasa por:
1. COMPRENSIÓN - ¿Qué se pide exactamente?
2. HIPÓTESIS - ¿Qué podría ser la respuesta?
3. PLAN - ¿Qué pasos seguir?
4. EJECUCIÓN - Ejecutar cada paso
5. VERIFICACIÓN - ¿Es correcto?
6. CONCLUSIÓN - Respuesta final

Autor: KAMAQ Team
Fecha: Enero 2026
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Callable
from enum import Enum
from datetime import datetime
import re


class ReasoningStep(Enum):
    """Pasos del pipeline de razonamiento."""
    COMPREHENSION = "comprension"
    HYPOTHESIS = "hipotesis"
    PLAN = "plan"
    EXECUTION = "ejecucion"
    VERIFICATION = "verificacion"
    CONCLUSION = "conclusion"


@dataclass
class StepResult:
    """Resultado de un paso de razonamiento."""
    step: ReasoningStep
    content: str
    confidence: float  # 0-1
    duration_ms: int
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ReasoningTrace:
    """Traza completa de razonamiento."""
    query: str
    timestamp: str
    steps: List[StepResult]
    final_answer: str
    overall_confidence: float
    total_duration_ms: int
    
    def to_dict(self) -> Dict:
        return {
            "query": self.query,
            "timestamp": self.timestamp,
            "steps": [
                {
                    "step": s.step.value,
                    "content": s.content,
                    "confidence": s.confidence,
                    "duration_ms": s.duration_ms
                }
                for s in self.steps
            ],
            "final_answer": self.final_answer,
            "overall_confidence": self.overall_confidence,
            "total_duration_ms": self.total_duration_ms
        }


class ReasoningPipeline:
    """
    Pipeline de razonamiento estructurado para KAMAQ.
    
    Fuerza al LLM a razonar en pasos explícitos y verificables.
    """
    
    # Template para cada paso
    STEP_PROMPTS = {
        ReasoningStep.COMPREHENSION: """
## PASO 1: COMPRENSIÓN

Analiza cuidadosamente la pregunta/tarea:

**Pregunta original**: {query}

Responde:
1. ¿Qué se está pidiendo exactamente?
2. ¿Qué información necesito para responder?
3. ¿Hay ambigüedades que deba resolver?
4. ¿Cuál es el contexto relevante?

**Mi comprensión**:
""",
        
        ReasoningStep.HYPOTHESIS: """
## PASO 2: HIPÓTESIS

Basándome en mi comprensión, genero posibles respuestas:

**Hipótesis 1**: ...
**Hipótesis 2**: ...
**Hipótesis más probable**: ...

**Nivel de incertidumbre** (0=seguro, 10=muy incierto): ...
""",
        
        ReasoningStep.PLAN: """
## PASO 3: PLAN

Para verificar mi hipótesis, seguiré estos pasos:

1. ...
2. ...
3. ...

**¿Necesito información externa?**: Sí/No
**¿Necesito ejecutar código?**: Sí/No
**¿Necesito consultar al usuario?**: Sí/No
""",
        
        ReasoningStep.EXECUTION: """
## PASO 4: EJECUCIÓN

Ejecutando el plan paso a paso:

**Paso 1**: ...
**Resultado 1**: ...

**Paso 2**: ...
**Resultado 2**: ...

(continuar...)
""",
        
        ReasoningStep.VERIFICATION: """
## PASO 5: VERIFICACIÓN

Verificando mi respuesta:

1. **¿Es lógicamente consistente?**: Sí/No
2. **¿Contradice información conocida?**: Sí/No
3. **¿Responde completamente la pregunta?**: Sí/No
4. **¿Hay algo que podría estar mal?**: ...

**Confianza final** (0-100%): ...%
""",
        
        ReasoningStep.CONCLUSION: """
## PASO 6: CONCLUSIÓN

**Respuesta final**:
{answer}

**Nivel de confianza**: {confidence}%
**Limitaciones/Caveats**: {caveats}
"""
    }
    
    def __init__(self, 
                 llm_caller: Callable[[str], str] = None,
                 min_confidence_to_answer: float = 0.3):
        """
        Inicializa pipeline de razonamiento.
        
        Args:
            llm_caller: Función que llama al LLM (prompt -> respuesta)
            min_confidence_to_answer: Confianza mínima para dar respuesta
        """
        self.llm_caller = llm_caller
        self.min_confidence = min_confidence_to_answer
        self.traces: List[ReasoningTrace] = []
    
    def create_structured_prompt(self, 
                                  query: str,
                                  context: str = "",
                                  include_steps: List[ReasoningStep] = None) -> str:
        """
        Crea prompt estructurado para el LLM.
        
        Args:
            query: Pregunta del usuario
            context: Contexto adicional (memoria, herramientas, etc.)
            include_steps: Pasos a incluir (default: todos)
            
        Returns:
            Prompt estructurado
        """
        if include_steps is None:
            include_steps = list(ReasoningStep)
        
        prompt = f"""Eres KAMAQ, un asistente que razona paso a paso de manera estructurada.

## INSTRUCCIONES
Responde a la siguiente pregunta siguiendo TODOS los pasos indicados.
Sé honesto sobre tu incertidumbre. Si no sabes algo, dilo.

"""
        
        if context:
            prompt += f"## CONTEXTO RELEVANTE\n{context}\n\n"
        
        prompt += f"## PREGUNTA DEL USUARIO\n{query}\n\n"
        prompt += "## RAZONAMIENTO ESTRUCTURADO\n\n"
        
        for step in include_steps:
            prompt += self.STEP_PROMPTS[step].format(
                query=query,
                answer="(tu respuesta aquí)",
                confidence="(0-100)",
                caveats="(limitaciones si las hay)"
            )
            prompt += "\n"
        
        return prompt
    
    def parse_response(self, response: str) -> Dict[ReasoningStep, StepResult]:
        """
        Parsea respuesta del LLM en pasos estructurados.
        
        Args:
            response: Respuesta cruda del LLM
            
        Returns:
            Dict mapeando paso a resultado
        """
        results = {}
        
        # Patrones para encontrar cada sección
        patterns = {
            ReasoningStep.COMPREHENSION: r"## PASO 1.*?(?=## PASO 2|$)",
            ReasoningStep.HYPOTHESIS: r"## PASO 2.*?(?=## PASO 3|$)",
            ReasoningStep.PLAN: r"## PASO 3.*?(?=## PASO 4|$)",
            ReasoningStep.EXECUTION: r"## PASO 4.*?(?=## PASO 5|$)",
            ReasoningStep.VERIFICATION: r"## PASO 5.*?(?=## PASO 6|$)",
            ReasoningStep.CONCLUSION: r"## PASO 6.*?(?=$)",
        }
        
        for step, pattern in patterns.items():
            match = re.search(pattern, response, re.DOTALL | re.IGNORECASE)
            if match:
                content = match.group(0).strip()
                
                # Extraer confianza si está mencionada
                confidence = self._extract_confidence(content)
                
                results[step] = StepResult(
                    step=step,
                    content=content,
                    confidence=confidence,
                    duration_ms=0  # Se actualiza después
                )
        
        return results
    
    def _extract_confidence(self, text: str) -> float:
        """Extrae nivel de confianza del texto."""
        text_lower = text.lower()
        
        # Buscar patrones como "confianza: 80%" o "85%" o "mi confianza es 75%"
        patterns = [
            r"confianza[:\s]+(?:es\s+)?(\d+)\s*%",
            r"confianza[:\s]+(\d+)%",
            r"(\d+)\s*%\s*(?:de\s+)?confianza",
            r"nivel.*?(\d+)\s*%",
            r"(\d+)\s*%\s*(?:seguro|confiado)",
        ]
        
        for pattern in patterns:
            match = re.search(pattern, text_lower)
            if match:
                return float(match.group(1)) / 100.0
        
        # Buscar porcentajes sueltos como fallback
        percent_match = re.search(r"(\d{1,3})\s*%", text_lower)
        if percent_match:
            value = float(percent_match.group(1))
            if 0 <= value <= 100:
                return value / 100.0
        
        # Buscar palabras de incertidumbre
        uncertainty_words = ["no sé", "no estoy seguro", "incierto", "quizás", "tal vez"]
        if any(word in text_lower for word in uncertainty_words):
            return 0.3
        
        return 0.5  # Default medio
    
    def extract_final_answer(self, parsed_steps: Dict[ReasoningStep, StepResult]) -> str:
        """Extrae la respuesta final de los pasos parseados."""
        if ReasoningStep.CONCLUSION in parsed_steps:
            content = parsed_steps[ReasoningStep.CONCLUSION].content
            
            # Buscar la respuesta después de "Respuesta final:"
            match = re.search(r"respuesta final[:\s]*\n(.+?)(?=\n\*\*|$)", 
                            content, re.DOTALL | re.IGNORECASE)
            if match:
                return match.group(1).strip()
            
            return content
        
        return "No se pudo generar una respuesta estructurada."
    
    def reason(self, 
               query: str, 
               context: str = "",
               fast_mode: bool = False) -> ReasoningTrace:
        """
        Ejecuta pipeline completo de razonamiento.
        
        Args:
            query: Pregunta del usuario
            context: Contexto adicional
            fast_mode: Si True, usa solo pasos esenciales
            
        Returns:
            ReasoningTrace con todo el razonamiento
        """
        start_time = datetime.now()
        
        # En modo rápido, usar solo comprensión y conclusión
        steps = [
            ReasoningStep.COMPREHENSION,
            ReasoningStep.CONCLUSION
        ] if fast_mode else list(ReasoningStep)
        
        # Crear prompt
        prompt = self.create_structured_prompt(query, context, steps)
        
        # Llamar al LLM
        if self.llm_caller:
            response = self.llm_caller(prompt)
        else:
            response = self._mock_response(query)
        
        # Parsear respuesta
        parsed_steps = self.parse_response(response)
        
        # Extraer respuesta final
        final_answer = self.extract_final_answer(parsed_steps)
        
        # Calcular confianza promedio
        confidences = [s.confidence for s in parsed_steps.values()]
        overall_confidence = sum(confidences) / len(confidences) if confidences else 0.5
        
        # Crear traza
        trace = ReasoningTrace(
            query=query,
            timestamp=datetime.now().isoformat(),
            steps=list(parsed_steps.values()),
            final_answer=final_answer,
            overall_confidence=overall_confidence,
            total_duration_ms=int((datetime.now() - start_time).total_seconds() * 1000)
        )
        
        self.traces.append(trace)
        return trace
    
    def _mock_response(self, query: str) -> str:
        """Respuesta mock para testing sin LLM."""
        return f"""
## PASO 1: COMPRENSIÓN

**Mi comprensión**: El usuario pregunta: "{query}"

## PASO 2: HIPÓTESIS

**Hipótesis más probable**: Esta es una pregunta de prueba del sistema.
**Nivel de incertidumbre**: 5

## PASO 3: PLAN

1. Analizar la pregunta
2. Buscar información relevante
3. Formular respuesta

## PASO 4: EJECUCIÓN

**Paso 1**: Analizado - es una pregunta de prueba
**Resultado 1**: Entendido

## PASO 5: VERIFICACIÓN

1. **¿Es lógicamente consistente?**: Sí
2. **¿Contradice información conocida?**: No
3. **¿Responde completamente la pregunta?**: Sí
4. **¿Hay algo que podría estar mal?**: No

**Confianza final**: 80%

## PASO 6: CONCLUSIÓN

**Respuesta final**:
Esta es una respuesta de prueba del sistema de razonamiento KAMAQ.

**Nivel de confianza**: 80%
**Limitaciones/Caveats**: Sistema en modo de prueba sin LLM real.
"""
    
    def get_reasoning_summary(self, trace: ReasoningTrace) -> str:
        """Genera resumen legible del razonamiento."""
        summary = f"🧠 **Razonamiento KAMAQ**\n\n"
        summary += f"**Pregunta**: {trace.query}\n\n"
        
        for step_result in trace.steps:
            emoji = {
                ReasoningStep.COMPREHENSION: "📖",
                ReasoningStep.HYPOTHESIS: "💡",
                ReasoningStep.PLAN: "📋",
                ReasoningStep.EXECUTION: "⚡",
                ReasoningStep.VERIFICATION: "✅",
                ReasoningStep.CONCLUSION: "🎯",
            }.get(step_result.step, "•")
            
            summary += f"{emoji} **{step_result.step.value.upper()}** "
            summary += f"(confianza: {step_result.confidence:.0%})\n"
        
        summary += f"\n**Respuesta**: {trace.final_answer}\n"
        summary += f"**Confianza global**: {trace.overall_confidence:.0%}\n"
        summary += f"**Tiempo**: {trace.total_duration_ms}ms"
        
        return summary


# =============================================================================
# Quick Reasoning para consultas simples
# =============================================================================

class QuickReasoner:
    """
    Razonador simplificado para consultas que no requieren
    el pipeline completo.
    """
    
    QUICK_PROMPT = """Eres KAMAQ. Responde de forma concisa y honesta.

Pregunta: {query}

Instrucciones:
- Si no sabes, di "No lo sé con certeza"
- Indica tu nivel de confianza (alta/media/baja)
- Sé breve pero completo

Respuesta:
"""
    
    def __init__(self, llm_caller: Callable[[str], str] = None):
        self.llm_caller = llm_caller
    
    def answer(self, query: str) -> tuple[str, float]:
        """
        Respuesta rápida sin pipeline completo.
        
        Returns:
            (respuesta, confianza)
        """
        prompt = self.QUICK_PROMPT.format(query=query)
        
        if self.llm_caller:
            response = self.llm_caller(prompt)
        else:
            response = f"[MOCK] Respuesta a: {query}"
        
        # Detectar confianza de la respuesta
        confidence = 0.7  # Default
        if any(w in response.lower() for w in ["no sé", "no estoy seguro", "quizás"]):
            confidence = 0.3
        elif any(w in response.lower() for w in ["definitivamente", "seguro", "claramente"]):
            confidence = 0.9
        
        return response, confidence


if __name__ == "__main__":
    # Test del pipeline
    pipeline = ReasoningPipeline()
    
    print("=== Test de Razonamiento Estructurado ===\n")
    
    trace = pipeline.reason("¿Cómo puedo mejorar la eficiencia de mi código Python?")
    
    print(pipeline.get_reasoning_summary(trace))
    
    print("\n\n=== Test de Quick Reasoning ===\n")
    
    quick = QuickReasoner()
    answer, conf = quick.answer("¿Qué es Python?")
    print(f"Respuesta: {answer}")
    print(f"Confianza: {conf:.0%}")

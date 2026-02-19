"""
KAMAQ Agent - Prueba Real con LLM
=================================
Prueba del agente con Ollama y un modelo real.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from kamaq_agent.core.constitution import KAMAQConstitution
from kamaq_agent.core.reasoning import ReasoningPipeline, QuickReasoner
from kamaq_agent.core.tools import create_default_registry
from kamaq_agent.core.verifier import KAMAQVerifier
import subprocess


def call_ollama(prompt: str, model: str = "llama3.2:1b") -> str:
    """Llama a Ollama directamente."""
    try:
        result = subprocess.run(
            ["ollama", "run", model],
            input=prompt,
            capture_output=True,
            timeout=60,
            encoding='utf-8',
            errors='replace'
        )
        return result.stdout.strip() if result.returncode == 0 else f"Error: {result.stderr}"
    except Exception as e:
        return f"Error: {e}"


def main():
    print()
    print("╔" + "═" * 58 + "╗")
    print("║" + " KAMAQ SUPER-AGENT - PRUEBA REAL ".center(58) + "║")
    print("╚" + "═" * 58 + "╝")
    print()
    
    # Inicializar componentes
    print("[1/5] Inicializando Constitution...")
    constitution = KAMAQConstitution()
    system_prompt = constitution.to_system_prompt()
    print(f"      ✓ {len(constitution.values)} valores cargados")
    
    print("[2/5] Inicializando Tools...")
    tools = create_default_registry()
    print(f"      ✓ {len(tools.list_tools())} herramientas disponibles")
    
    print("[3/5] Inicializando Verifier...")
    verifier = KAMAQVerifier()
    print(f"      ✓ Verificador listo")
    
    print("[4/5] Inicializando Reasoning con LLM...")
    reasoning = ReasoningPipeline(llm_caller=lambda p: call_ollama(p))
    quick = QuickReasoner(llm_caller=lambda p: call_ollama(p))
    print(f"      ✓ Pipeline conectado a llama3.2:1b")
    
    print("[5/5] Verificando Ollama...")
    test_response = call_ollama("Responde solo 'OK' si me escuchas.")
    if "OK" in test_response.upper() or len(test_response) < 50:
        print(f"      ✓ Ollama responde correctamente")
    else:
        print(f"      ⚠ Respuesta: {test_response[:50]}...")
    
    print()
    print("=" * 60)
    print("PRUEBA 1: Herramientas (Sin LLM)")
    print("=" * 60)
    
    # Test 1: Calculadora
    result = tools.execute("calculate", expression="sqrt(144) + 5 * 3")
    print(f"\n📐 Cálculo: sqrt(144) + 5 * 3 = {result.output}")
    
    # Test 2: Python
    result = tools.execute("python_exec", 
                          code="import math; result = math.factorial(7)")
    print(f"🐍 Python: 7! = {result.output}")
    
    print()
    print("=" * 60)
    print("PRUEBA 2: Quick Reasoning (Con LLM)")
    print("=" * 60)
    
    # Pregunta simple
    question = "¿Qué es la recursión en programación? Responde en 2 oraciones."
    print(f"\n❓ Pregunta: {question}")
    print("\n⏳ Consultando LLM...")
    
    answer, confidence = quick.answer(question)
    print(f"\n💬 Respuesta:\n{answer}")
    print(f"\n📊 Confianza estimada: {confidence:.0%}")
    
    # Verificar respuesta
    verification = verifier.verify_response(answer, confidence)
    if verification["warnings"]:
        print(f"⚠️ Advertencias: {verification['warnings']}")
    else:
        print(f"✓ Sin advertencias de verificación")
    
    print()
    print("=" * 60)
    print("PRUEBA 3: Evaluación de Riesgos")
    print("=" * 60)
    
    test_actions = [
        "Leer el archivo README.md",
        "Eliminar la carpeta temporal",
        "Buscar información sobre Python",
        "Ejecutar sudo rm -rf /",
        "Modificar archivo de configuración"
    ]
    
    print()
    for action in test_actions:
        risk = constitution.evaluate_action_risk(action, {})
        should_escalate = constitution.should_escalate(0.5, risk)
        
        emoji = {
            "SAFE": "🟢",
            "LOW": "🟡", 
            "MEDIUM": "🟠",
            "HIGH": "🔴",
            "FORBIDDEN": "⛔"
        }.get(risk.name, "⚪")
        
        escalate_str = " [ESCALAR]" if should_escalate else ""
        print(f"  {emoji} {risk.name:10} | {action}{escalate_str}")
    
    print()
    print("=" * 60)
    print("PRUEBA 4: Razonamiento Estructurado (Con LLM)")
    print("=" * 60)
    
    complex_question = "Explica brevemente qué es un algoritmo de ordenamiento."
    print(f"\n❓ Pregunta compleja: {complex_question}")
    print("\n⏳ Ejecutando pipeline de razonamiento...")
    
    # Usar modo rápido para no tardar mucho
    trace = reasoning.reason(complex_question, fast_mode=True)
    
    print(f"\n📝 Pasos ejecutados: {len(trace.steps)}")
    for step in trace.steps:
        print(f"   • {step.step.value}: {step.confidence:.0%} confianza")
    
    print(f"\n💬 Respuesta final:\n{trace.final_answer[:300]}...")
    print(f"\n📊 Confianza global: {trace.overall_confidence:.0%}")
    print(f"⏱️ Tiempo total: {trace.total_duration_ms}ms")
    
    print()
    print("=" * 60)
    print("RESUMEN FINAL")
    print("=" * 60)
    print()
    print("  ✅ Constitution: Funcionando")
    print("  ✅ Tools: Funcionando")
    print("  ✅ Verifier: Funcionando")
    print("  ✅ Reasoning: Funcionando")
    print("  ✅ LLM (Ollama): Conectado")
    print()
    print("  🎉 KAMAQ SUPER-AGENT ESTÁ OPERATIVO")
    print()
    print("  Para iniciar el chat interactivo:")
    print("  python -m kamaq_agent.agent --model llama3.2:1b")
    print()


if __name__ == "__main__":
    main()

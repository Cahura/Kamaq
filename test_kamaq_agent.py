"""
KAMAQ Agent Test Suite
======================
Pruebas completas de todos los componentes.
"""

import sys
from pathlib import Path

# Agregar path
sys.path.insert(0, str(Path(__file__).parent))

def test_constitution():
    """Test del módulo de constitución."""
    print("=" * 60)
    print("TEST 1: CONSTITUTION")
    print("=" * 60)
    
    try:
        from kamaq_agent.core.constitution import (
            KAMAQConstitution, RiskLevel, OperatingMode, TeslaPrinciples
        )
        
        # Test 1: Crear constitución
        constitution = KAMAQConstitution()
        print("[OK] Constitución creada")
        
        # Test 2: Verificar valores
        assert len(constitution.values) == 7, "Deberían ser 7 valores"
        print(f"[OK] Valores cargados: {len(constitution.values)}")
        for key in constitution.values:
            print(f"     - {key}")
        
        # Test 3: Evaluar riesgos
        print("\nTest de evaluación de riesgos:")
        
        risk = constitution.evaluate_action_risk("Leer archivo config.json", {})
        print(f"  [OK] 'Leer archivo' -> {risk.name}")
        
        risk = constitution.evaluate_action_risk("Eliminar carpeta", {})
        assert risk in [RiskLevel.HIGH, RiskLevel.MEDIUM], "Eliminar debería ser riesgoso"
        print(f"  [OK] 'Eliminar carpeta' -> {risk.name}")
        
        risk = constitution.evaluate_action_risk("sudo rm -rf /", {})
        assert risk == RiskLevel.FORBIDDEN, "sudo rm -rf debería ser FORBIDDEN"
        print(f"  [OK] 'sudo rm -rf' -> {risk.name}")
        
        # Test 4: Escalamiento
        print("\nTest de escalamiento:")
        should = constitution.should_escalate(uncertainty=0.8, action_risk=RiskLevel.SAFE)
        assert should == True, "Alta incertidumbre debería escalar"
        print(f"  [OK] Incertidumbre 0.8: escalar={should}")
        
        should = constitution.should_escalate(uncertainty=0.3, action_risk=RiskLevel.HIGH)
        assert should == True, "Alto riesgo debería escalar"
        print(f"  [OK] Riesgo HIGH: escalar={should}")
        
        # Test 5: System prompt
        prompt = constitution.to_system_prompt()
        assert len(prompt) > 100, "System prompt muy corto"
        print(f"\n[OK] System prompt: {len(prompt)} caracteres")
        
        # Test 6: Principios Tesla
        assert constitution.tesla_principles.elegance != "", "Tesla principles vacíos"
        print(f"[OK] Principios Tesla cargados")
        
        print("\n✅ CONSTITUTION: PASSED")
        return True
        
    except Exception as e:
        print(f"\n❌ CONSTITUTION: FAILED - {e}")
        import traceback
        traceback.print_exc()
        return False


def test_reasoning():
    """Test del pipeline de razonamiento."""
    print("\n" + "=" * 60)
    print("TEST 2: REASONING PIPELINE")
    print("=" * 60)
    
    try:
        from kamaq_agent.core.reasoning import (
            ReasoningPipeline, QuickReasoner, ReasoningStep, ReasoningTrace
        )
        
        # Test 1: Crear pipeline (sin LLM)
        pipeline = ReasoningPipeline(llm_caller=None)
        print("[OK] Pipeline creado")
        
        # Test 2: Crear prompt estructurado
        prompt = pipeline.create_structured_prompt("¿Qué es Python?")
        assert "PASO 1" in prompt, "Prompt debería tener pasos"
        assert "COMPRENSIÓN" in prompt or "COMPRENSION" in prompt
        print(f"[OK] Prompt estructurado: {len(prompt)} caracteres")
        
        # Test 3: Razonamiento mock
        trace = pipeline.reason("¿Qué es Python?")
        assert isinstance(trace, ReasoningTrace), "Debería retornar ReasoningTrace"
        assert trace.final_answer != "", "Debería tener respuesta"
        print(f"[OK] Razonamiento ejecutado")
        print(f"     - Pasos: {len(trace.steps)}")
        print(f"     - Confianza: {trace.overall_confidence:.0%}")
        print(f"     - Tiempo: {trace.total_duration_ms}ms")
        
        # Test 4: Quick Reasoner
        quick = QuickReasoner(llm_caller=None)
        answer, conf = quick.answer("¿Qué es Python?")
        assert answer != "", "Quick reasoner debería dar respuesta"
        print(f"[OK] Quick Reasoner funcionando")
        
        # Test 5: Parseo de respuesta
        mock_response = """
## PASO 1: COMPRENSIÓN
Entendiendo la pregunta.

## PASO 2: HIPÓTESIS
Python es un lenguaje.

## PASO 6: CONCLUSIÓN

**Respuesta final**:
Python es un lenguaje de programación.

**Nivel de confianza**: 85%
"""
        parsed = pipeline.parse_response(mock_response)
        assert len(parsed) >= 2, "Debería parsear al menos 2 pasos"
        print(f"[OK] Parseo de respuesta: {len(parsed)} pasos encontrados")
        
        # Test 6: Extracción de confianza
        test_texts = [
            ("Mi confianza es 75%", 0.75),
            ("Estoy 90% seguro", 0.90),
            ("Confianza: 80%", 0.80),
        ]
        for text, expected in test_texts:
            conf = pipeline._extract_confidence(text)
            assert abs(conf - expected) < 0.05, f"'{text}': esperado {expected}, obtenido {conf}"
        print(f"[OK] Extracción de confianza funciona correctamente")
        
        print("\n✅ REASONING: PASSED")
        return True
        
    except Exception as e:
        print(f"\n❌ REASONING: FAILED - {e}")
        import traceback
        traceback.print_exc()
        return False


def test_tools():
    """Test del sistema de herramientas."""
    print("\n" + "=" * 60)
    print("TEST 3: TOOLS SYSTEM")
    print("=" * 60)
    
    try:
        from kamaq_agent.core.tools import (
            ToolRegistry, create_default_registry,
            CalculatorTool, FileReaderTool, PythonExecutorTool,
            ToolRisk, ToolCategory
        )
        
        # Test 1: Crear registry
        registry = create_default_registry()
        print(f"[OK] Registry creado con {len(registry.list_tools())} herramientas")
        
        for spec in registry.list_tools():
            print(f"     - {spec.name} ({spec.risk.value})")
        
        # Test 2: Calculadora
        print("\nTest Calculadora:")
        result = registry.execute("calculate", expression="2 + 2")
        assert result.success, f"Cálculo falló: {result.error}"
        assert result.output == 4, f"2+2 debería ser 4, no {result.output}"
        print(f"  [OK] 2 + 2 = {result.output}")
        
        result = registry.execute("calculate", expression="sqrt(16) + 3**2")
        assert result.success
        assert abs(result.output - 13.0) < 0.01
        print(f"  [OK] sqrt(16) + 3**2 = {result.output}")
        
        result = registry.execute("calculate", expression="sin(0)")
        assert result.success
        assert abs(result.output) < 0.01
        print(f"  [OK] sin(0) = {result.output}")
        
        # Test 3: Python Executor
        print("\nTest Python Executor:")
        result = registry.execute("python_exec", code="result = [x**2 for x in range(5)]")
        assert result.success, f"Python exec falló: {result.error}"
        assert result.output == [0, 1, 4, 9, 16]
        print(f"  [OK] List comprehension: {result.output}")
        
        # Test código peligroso (debería fallar)
        result = registry.execute("python_exec", code="import os; os.system('ls')")
        assert not result.success, "Import os debería ser bloqueado"
        print(f"  [OK] Import peligroso bloqueado: {result.error[:40]}...")
        
        # Test 4: File Reader (archivo que no existe)
        print("\nTest File Reader:")
        result = registry.execute("read_file", path="archivo_inexistente.txt")
        assert not result.success
        print(f"  [OK] Archivo inexistente manejado correctamente")
        
        # Test 5: Function specs para LLM
        specs = registry.to_function_specs()
        assert len(specs) > 0
        print(f"\n[OK] Function specs generados: {len(specs)}")
        
        print("\n✅ TOOLS: PASSED")
        return True
        
    except Exception as e:
        print(f"\n❌ TOOLS: FAILED - {e}")
        import traceback
        traceback.print_exc()
        return False


def test_verifier():
    """Test del sistema de verificación."""
    print("\n" + "=" * 60)
    print("TEST 4: VERIFIER")
    print("=" * 60)
    
    try:
        from kamaq_agent.core.verifier import (
            KAMAQVerifier, MathVerifier, ContradictionDetector,
            ConfidenceCalibrator, VerificationStatus
        )
        
        # Test 1: Math Verifier
        print("Test Math Verifier:")
        math_v = MathVerifier()
        
        result = math_v.verify_equation("2 + 2", 4)
        assert result.status == VerificationStatus.VERIFIED
        print(f"  [OK] 2 + 2 = 4 verificado")
        
        result = math_v.verify_equation("2 + 2", 5)
        assert result.status == VerificationStatus.FAILED
        print(f"  [OK] 2 + 2 = 5 detectado como error")
        
        result = math_v.verify_equation("3 * 4 + 2", 14)
        assert result.status == VerificationStatus.VERIFIED
        print(f"  [OK] 3 * 4 + 2 = 14 verificado")
        
        # Test 2: Extracción de ecuaciones
        text = "El resultado de 5 + 3 = 8 y también 2 * 6 = 12"
        results = math_v.extract_and_verify(text)
        assert len(results) == 2, f"Debería encontrar 2 ecuaciones, encontró {len(results)}"
        print(f"  [OK] Extraídas {len(results)} ecuaciones del texto")
        
        # Test 3: Contradiction Detector
        print("\nTest Contradiction Detector:")
        detector = ContradictionDetector()
        
        detector.add_fact("Python es un lenguaje de programación")
        detector.add_fact("El agua siempre hierve a 100°C")
        
        # No debería detectar contradicción
        result = detector.check_contradiction("Java es diferente a Python")
        assert result is None
        print(f"  [OK] No hay contradicción en afirmación válida")
        
        # Test 4: Calibrador
        print("\nTest Confidence Calibrator:")
        calibrator = ConfidenceCalibrator()
        
        # Simular historial
        import random
        random.seed(42)
        for _ in range(30):
            conf = random.uniform(0.6, 0.9)
            correct = random.random() < (conf - 0.05)  # Ligeramente sobreconfiado
            calibrator.record(conf, correct)
        
        ece = calibrator.get_ece()
        print(f"  [OK] ECE calculado: {ece:.3f}")
        
        calibrated = calibrator.get_calibrated_confidence(0.8)
        print(f"  [OK] Confianza calibrada (0.8 -> {calibrated:.2f})")
        
        # Test 5: Verifier completo
        print("\nTest Verifier Integrado:")
        verifier = KAMAQVerifier()
        
        results = verifier.verify_response(
            "La suma de 2 + 2 = 4 y estoy muy seguro",
            claimed_confidence=0.9
        )
        assert "math_checks" in results
        print(f"  [OK] Verificación completa ejecutada")
        print(f"       - Math checks: {len(results['math_checks'])}")
        print(f"       - Warnings: {len(results['warnings'])}")
        
        # Test warning por matemática incorrecta
        results = verifier.verify_response(
            "Claramente 2 + 2 = 5",
            claimed_confidence=0.95
        )
        assert len(results["warnings"]) > 0 or results["overall_status"].value == "fallido"
        print(f"  [OK] Error matemático detectado en respuesta")
        
        print("\n✅ VERIFIER: PASSED")
        return True
        
    except Exception as e:
        print(f"\n❌ VERIFIER: FAILED - {e}")
        import traceback
        traceback.print_exc()
        return False


def test_integration():
    """Test de integración del agente completo."""
    print("\n" + "=" * 60)
    print("TEST 5: INTEGRATION")
    print("=" * 60)
    
    try:
        # Test imports
        from kamaq_agent.core.constitution import KAMAQConstitution
        from kamaq_agent.core.reasoning import ReasoningPipeline
        from kamaq_agent.core.tools import create_default_registry
        from kamaq_agent.core.verifier import KAMAQVerifier
        
        print("[OK] Todos los módulos importan correctamente")
        
        # Test flujo completo sin LLM
        constitution = KAMAQConstitution()
        tools = create_default_registry()
        verifier = KAMAQVerifier()
        reasoning = ReasoningPipeline(llm_caller=None)
        
        # Simular flujo de chat
        user_message = "Calcula 15 * 3"
        
        # 1. Evaluar riesgo
        risk = constitution.evaluate_action_risk(user_message, {})
        print(f"[OK] Evaluación de riesgo: {risk.name}")
        
        # 2. Ejecutar herramienta
        result = tools.execute("calculate", expression="15 * 3")
        assert result.success
        print(f"[OK] Herramienta ejecutada: {result.output}")
        
        # 3. Verificar respuesta
        response_text = f"El resultado de 15 * 3 = {result.output}"
        verification = verifier.verify_response(response_text, 0.95)
        print(f"[OK] Respuesta verificada: {verification['overall_status'].value}")
        
        # 4. Razonamiento
        trace = reasoning.reason("¿Cómo optimizar código?")
        print(f"[OK] Razonamiento generado: {len(trace.steps)} pasos")
        
        print("\n✅ INTEGRATION: PASSED")
        return True
        
    except Exception as e:
        print(f"\n❌ INTEGRATION: FAILED - {e}")
        import traceback
        traceback.print_exc()
        return False


def run_all_tests():
    """Ejecuta todas las pruebas."""
    print()
    print("╔" + "═" * 58 + "╗")
    print("║" + " KAMAQ SUPER-AGENT TEST SUITE ".center(58) + "║")
    print("╚" + "═" * 58 + "╝")
    print()
    
    results = {}
    
    results["Constitution"] = test_constitution()
    results["Reasoning"] = test_reasoning()
    results["Tools"] = test_tools()
    results["Verifier"] = test_verifier()
    results["Integration"] = test_integration()
    
    # Resumen
    print("\n" + "=" * 60)
    print("RESUMEN DE PRUEBAS")
    print("=" * 60)
    
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    
    for name, result in results.items():
        status = "✅ PASSED" if result else "❌ FAILED"
        print(f"  {name}: {status}")
    
    print()
    print(f"Total: {passed}/{total} pruebas pasaron")
    
    if passed == total:
        print()
        print("🎉 ¡TODOS LOS COMPONENTES FUNCIONAN CORRECTAMENTE!")
    else:
        print()
        print("⚠️ Algunos componentes necesitan atención")
    
    return passed == total


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)

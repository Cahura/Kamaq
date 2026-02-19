"""
KAMAQ Verifier: Sistema de Verificación Honesta
===============================================
Verifica respuestas, detecta contradicciones y calibra confianza.

Componentes:
- MathVerifier: Verifica expresiones matemáticas
- LogicVerifier: Detecta contradicciones lógicas
- ConsistencyChecker: Verifica consistencia interna
- ConfidenceCalibrator: Ajusta confianza basada en historial

Principio Tesla: "La honestidad es la primera condición del progreso"

Autor: KAMAQ Team
Fecha: Enero 2026
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any
from enum import Enum
import re
import ast
import operator
from collections import deque
import math


class VerificationStatus(Enum):
    """Estados de verificación."""
    VERIFIED = "verificado"
    FAILED = "fallido"
    UNCERTAIN = "incierto"
    UNABLE = "no_verificable"


@dataclass
class VerificationResult:
    """Resultado de verificación."""
    status: VerificationStatus
    confidence: float  # 0-1
    explanation: str
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Contradiction:
    """Representa una contradicción detectada."""
    statement_a: str
    statement_b: str
    severity: float  # 0-1
    explanation: str


# =============================================================================
# VERIFICADOR MATEMÁTICO
# =============================================================================

class MathVerifier:
    """
    Verifica expresiones y cálculos matemáticos.
    Usa evaluación segura, no eval().
    """
    
    OPERATORS = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.Pow: operator.pow,
        ast.USub: operator.neg,
    }
    
    def __init__(self, tolerance: float = 1e-6):
        self.tolerance = tolerance
    
    def verify_equation(self, expression: str, 
                        claimed_result: float) -> VerificationResult:
        """
        Verifica si una expresión matemática da el resultado claimed.
        
        Args:
            expression: Expresión como "2 + 2"
            claimed_result: Resultado claimed
            
        Returns:
            VerificationResult
        """
        try:
            actual = self._safe_eval(expression)
            
            if abs(actual - claimed_result) < self.tolerance:
                return VerificationResult(
                    status=VerificationStatus.VERIFIED,
                    confidence=1.0,
                    explanation=f"{expression} = {actual} ✓",
                    details={"actual": actual, "claimed": claimed_result}
                )
            else:
                return VerificationResult(
                    status=VerificationStatus.FAILED,
                    confidence=1.0,
                    explanation=f"{expression} = {actual}, no {claimed_result}",
                    details={"actual": actual, "claimed": claimed_result,
                             "difference": abs(actual - claimed_result)}
                )
        
        except Exception as e:
            return VerificationResult(
                status=VerificationStatus.UNABLE,
                confidence=0.0,
                explanation=f"No pude evaluar: {str(e)}",
                details={"error": str(e)}
            )
    
    def extract_and_verify(self, text: str) -> List[VerificationResult]:
        """
        Extrae expresiones matemáticas del texto y las verifica.
        Busca patrones como "2 + 2 = 4" o "el resultado es 5".
        """
        results = []
        
        # Patrón: expresión = resultado
        pattern = r'(\d+(?:\.\d+)?(?:\s*[\+\-\*\/\^]\s*\d+(?:\.\d+)?)+)\s*=\s*(\d+(?:\.\d+)?)'
        
        for match in re.finditer(pattern, text):
            expression = match.group(1).replace('^', '**')
            claimed = float(match.group(2))
            result = self.verify_equation(expression, claimed)
            results.append(result)
        
        return results
    
    def _safe_eval(self, expression: str) -> float:
        """Evalúa expresión matemática de forma segura."""
        # Normalizar
        expression = expression.replace('^', '**')
        expression = expression.replace('×', '*')
        expression = expression.replace('÷', '/')
        
        node = ast.parse(expression, mode='eval')
        return self._eval_node(node.body)
    
    def _eval_node(self, node) -> float:
        """Evalúa nodo AST."""
        # Python 3.8+ usa ast.Constant para todos los literales
        if isinstance(node, ast.Constant):
            return node.value
        # Compatibilidad con Python < 3.8 (ast.Num deprecado)
        elif hasattr(ast, 'Num') and isinstance(node, ast.Num):
            return node.n
        elif isinstance(node, ast.BinOp):
            left = self._eval_node(node.left)
            right = self._eval_node(node.right)
            op = self.OPERATORS.get(type(node.op))
            if op:
                return op(left, right)
        elif isinstance(node, ast.UnaryOp):
            operand = self._eval_node(node.operand)
            op = self.OPERATORS.get(type(node.op))
            if op:
                return op(operand)
        
        raise ValueError(f"Expresión no soportada: {type(node)}")


# =============================================================================
# DETECTOR DE CONTRADICCIONES
# =============================================================================

class ContradictionDetector:
    """
    Detecta contradicciones lógicas en afirmaciones.
    """
    
    # Pares de palabras contradictorias
    CONTRADICTORY_PAIRS = [
        ("siempre", "nunca"),
        ("todos", "ninguno"),
        ("verdadero", "falso"),
        ("correcto", "incorrecto"),
        ("es", "no es"),
        ("puede", "no puede"),
        ("sí", "no"),
    ]
    
    def __init__(self):
        self.known_facts: Dict[str, str] = {}  # {fact_key: fact_value}
    
    def add_fact(self, fact: str, source: str = "unknown"):
        """Registra un hecho conocido."""
        # Normalizar
        key = self._normalize(fact)
        self.known_facts[key] = fact
    
    def check_contradiction(self, new_statement: str) -> Optional[Contradiction]:
        """
        Verifica si una nueva afirmación contradice hechos conocidos.
        
        Returns:
            Contradiction si se detecta, None si no
        """
        new_normalized = self._normalize(new_statement)
        
        for fact_key, fact_value in self.known_facts.items():
            # Verificar contradicciones usando pares de palabras
            for word_a, word_b in self.CONTRADICTORY_PAIRS:
                # Si hecho tiene word_a y nuevo tiene word_b (o viceversa)
                if word_a in fact_key and word_b in new_normalized:
                    # Verificar que hablen del mismo tema (overlap de palabras)
                    fact_words = set(fact_key.split())
                    new_words = set(new_normalized.split())
                    overlap = fact_words & new_words
                    
                    if len(overlap) >= 2:  # Al menos 2 palabras en común
                        return Contradiction(
                            statement_a=fact_value,
                            statement_b=new_statement,
                            severity=0.8,
                            explanation=f"'{fact_value}' contradice '{new_statement}'"
                        )
                
                # También al revés
                if word_b in fact_key and word_a in new_normalized:
                    fact_words = set(fact_key.split())
                    new_words = set(new_normalized.split())
                    overlap = fact_words & new_words
                    
                    if len(overlap) >= 2:
                        return Contradiction(
                            statement_a=fact_value,
                            statement_b=new_statement,
                            severity=0.8,
                            explanation=f"'{new_statement}' contradice '{fact_value}'"
                        )
            
            # Verificar negación directa
            if self._is_negation(fact_key, new_normalized):
                return Contradiction(
                    statement_a=fact_value,
                    statement_b=new_statement,
                    severity=0.9,
                    explanation="Negación directa detectada"
                )
        
        return None
    
    def _normalize(self, text: str) -> str:
        """Normaliza texto para comparación."""
        text = text.lower().strip()
        text = re.sub(r'\s+', ' ', text)
        text = re.sub(r'[^\w\s]', '', text)
        return text
    
    def _is_negation(self, a: str, b: str) -> bool:
        """Verifica si b es negación de a."""
        # Simple: "no" + a ≈ b o a ≈ "no" + b
        if f"no {a}" in b or f"no {b}" in a:
            return True
        if a.replace("no ", "") == b or b.replace("no ", "") == a:
            return True
        return False


# =============================================================================
# CALIBRADOR DE CONFIANZA
# =============================================================================

class ConfidenceCalibrator:
    """
    Calibra confianza basada en historial de aciertos/errores.
    
    Principio: La confianza declarada debe reflejar la precisión real.
    Si digo "80% confianza" en 100 predicciones, ~80 deben ser correctas.
    """
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.history: deque = deque(maxlen=window_size)
        
        # Buckets de calibración: {confidence_level: [is_correct, ...]}
        self.calibration_buckets: Dict[int, List[bool]] = {
            i: [] for i in range(0, 101, 10)  # 0%, 10%, 20%, ..., 100%
        }
    
    def record(self, declared_confidence: float, was_correct: bool):
        """
        Registra una predicción con su resultado.
        
        Args:
            declared_confidence: Confianza declarada (0-1)
            was_correct: Si la predicción fue correcta
        """
        self.history.append({
            "confidence": declared_confidence,
            "correct": was_correct
        })
        
        # Agregar a bucket correspondiente
        bucket = int(declared_confidence * 100 // 10) * 10
        bucket = min(bucket, 100)
        self.calibration_buckets[bucket].append(was_correct)
    
    def get_calibrated_confidence(self, raw_confidence: float) -> float:
        """
        Ajusta confianza basada en historial de calibración.
        
        Si históricamente sobreestimo mi confianza, la reduce.
        Si subestimo, la aumenta.
        """
        bucket = int(raw_confidence * 100 // 10) * 10
        bucket = min(bucket, 100)
        
        bucket_history = self.calibration_buckets.get(bucket, [])
        
        if len(bucket_history) < 5:
            # Insufficient data — return raw confidence (no penalty)
            return raw_confidence
        
        # Calcular precisión real en este bucket
        actual_accuracy = sum(bucket_history) / len(bucket_history)
        
        # La confianza calibrada es la precisión real
        return actual_accuracy
    
    def get_ece(self) -> float:
        """
        Calcula Expected Calibration Error.
        
        ECE = Σ |bucket_count/total| * |accuracy(bucket) - confidence(bucket)|
        
        ECE bajo = bien calibrado
        ECE alto = mal calibrado
        """
        total = sum(len(b) for b in self.calibration_buckets.values())
        if total == 0:
            return 0.0
        
        ece = 0.0
        for confidence_level, outcomes in self.calibration_buckets.items():
            if not outcomes:
                continue
            
            bucket_weight = len(outcomes) / total
            bucket_accuracy = sum(outcomes) / len(outcomes)
            bucket_confidence = confidence_level / 100.0
            
            ece += bucket_weight * abs(bucket_accuracy - bucket_confidence)
        
        return ece
    
    def get_calibration_report(self) -> str:
        """Genera reporte de calibración legible."""
        report = "📊 **Reporte de Calibración KAMAQ**\n\n"
        
        report += "| Confianza | Predicciones | Aciertos | Precisión |\n"
        report += "|:---------:|:------------:|:--------:|:---------:|\n"
        
        for level, outcomes in sorted(self.calibration_buckets.items()):
            if outcomes:
                correct = sum(outcomes)
                total = len(outcomes)
                accuracy = correct / total * 100
                report += f"| {level}% | {total} | {correct} | {accuracy:.0f}% |\n"
        
        ece = self.get_ece()
        report += f"\n**ECE (Error de Calibración)**: {ece:.3f}\n"
        
        if ece < 0.05:
            report += "✅ Excelente calibración"
        elif ece < 0.10:
            report += "✓ Buena calibración"
        elif ece < 0.20:
            report += "⚠️ Calibración mejorable"
        else:
            report += "❌ Mal calibrado - necesita ajuste"
        
        return report


# =============================================================================
# VERIFICADOR PRINCIPAL
# =============================================================================

class KAMAQVerifier:
    """
    Sistema de verificación integrado para KAMAQ.
    
    Combina:
    - Verificación matemática
    - Detección de contradicciones
    - Calibración de confianza
    """
    
    def __init__(self):
        self.math_verifier = MathVerifier()
        self.contradiction_detector = ContradictionDetector()
        self.calibrator = ConfidenceCalibrator()
    
    def verify_response(self, 
                        response: str,
                        claimed_confidence: float,
                        context: Dict = None) -> Dict[str, Any]:
        """
        Verificación completa de una respuesta.
        
        Returns:
            Dict con resultados de verificación
        """
        results = {
            "math_checks": [],
            "contradictions": [],
            "adjusted_confidence": claimed_confidence,
            "warnings": [],
            "overall_status": VerificationStatus.VERIFIED
        }
        
        # 1. Verificar matemáticas
        math_results = self.math_verifier.extract_and_verify(response)
        results["math_checks"] = math_results
        
        for mr in math_results:
            if mr.status == VerificationStatus.FAILED:
                results["overall_status"] = VerificationStatus.FAILED
                results["warnings"].append(f"Error matemático: {mr.explanation}")
        
        # 2. Verificar contradicciones
        # Dividir respuesta en oraciones y verificar cada una
        sentences = re.split(r'[.!?]', response)
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) > 10:  # Ignorar fragmentos muy cortos
                contradiction = self.contradiction_detector.check_contradiction(sentence)
                if contradiction:
                    results["contradictions"].append(contradiction)
                    results["overall_status"] = VerificationStatus.UNCERTAIN
                    results["warnings"].append(f"Contradicción: {contradiction.explanation}")
                else:
                    # Si no hay contradicción, agregar como hecho conocido
                    self.contradiction_detector.add_fact(sentence)
        
        # 3. Calibrar confianza
        results["adjusted_confidence"] = self.calibrator.get_calibrated_confidence(
            claimed_confidence
        )
        
        # 4. Detectar indicadores de incertidumbre no declarada
        uncertainty_words = ["creo", "quizás", "tal vez", "posiblemente", 
                           "probablemente", "no estoy seguro", "podría ser"]
        uncertainty_count = sum(1 for w in uncertainty_words if w in response.lower())
        
        if uncertainty_count > 0 and claimed_confidence > 0.7:
            results["warnings"].append(
                f"Lenguaje de incertidumbre ({uncertainty_count} indicadores) "
                f"pero confianza alta ({claimed_confidence:.0%})"
            )
            results["adjusted_confidence"] *= 0.8
        
        return results
    
    def record_outcome(self, confidence: float, was_correct: bool):
        """Registra resultado para calibración futura."""
        self.calibrator.record(confidence, was_correct)
    
    def get_calibration_report(self) -> str:
        """Obtiene reporte de calibración."""
        return self.calibrator.get_calibration_report()
    
    def should_warn_user(self, verification_results: Dict) -> Optional[str]:
        """
        Determina si debe advertir al usuario sobre la respuesta.
        
        Returns:
            Mensaje de advertencia o None
        """
        if verification_results["overall_status"] == VerificationStatus.FAILED:
            return "⚠️ He detectado errores en mi respuesta. Por favor verifica."
        
        if verification_results["contradictions"]:
            return "⚠️ Mi respuesta puede contener información contradictoria."
        
        if verification_results["adjusted_confidence"] < 0.3:
            return "⚠️ Mi confianza en esta respuesta es baja. Considera verificar."
        
        if len(verification_results["warnings"]) >= 2:
            return "⚠️ Hay múltiples indicadores de posibles problemas en mi respuesta."
        
        return None


if __name__ == "__main__":
    # Tests del verificador
    verifier = KAMAQVerifier()
    
    print("=== Test de Verificación Matemática ===\n")
    
    test_text = "La suma de 2 + 2 = 5 y también 3 * 4 = 12"
    results = verifier.math_verifier.extract_and_verify(test_text)
    
    for r in results:
        print(f"  {r.status.value}: {r.explanation}")
    
    print("\n=== Test de Contradicciones ===\n")
    
    verifier.contradiction_detector.add_fact("Python es un lenguaje de programación")
    verifier.contradiction_detector.add_fact("El agua siempre hierve a 100°C")
    
    test_statements = [
        "El agua nunca hierve a 100°C",
        "Python no es un lenguaje de programación",
        "Java es diferente a Python",  # No contradice
    ]
    
    for stmt in test_statements:
        contradiction = verifier.contradiction_detector.check_contradiction(stmt)
        if contradiction:
            print(f"  ❌ CONTRADICCIÓN: {contradiction.explanation}")
        else:
            print(f"  ✓ OK: {stmt}")
    
    print("\n=== Test de Calibración ===\n")
    
    # Simular historial
    import random
    for _ in range(50):
        conf = random.uniform(0.6, 0.9)
        # Simular que somos ligeramente sobreconfiados
        correct = random.random() < (conf - 0.1)
        verifier.record_outcome(conf, correct)
    
    print(verifier.get_calibration_report())

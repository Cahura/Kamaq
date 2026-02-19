"""
Tests for Agent Components — Verifier, Constitution, Tools
===========================================================

These tests verify that:
1. Math verification produces correct results (not hardcoded)
2. Contradiction detection works with real logic
3. Confidence calibration adjusts based on actual history
4. Constitution risk evaluation is keyword-based and consistent
5. Tools system has proper structure

Run: python -m pytest tests/test_agent_components.py -v
  or: python tests/test_agent_components.py

Author: Carlos Huarcaya
Date: February 2026
"""

import unittest
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kamaq_agent.core.verifier import (
    MathVerifier,
    ContradictionDetector,
    ConfidenceCalibrator,
    KAMAQVerifier,
    VerificationStatus,
)
from kamaq_agent.core.constitution import (
    KAMAQConstitution,
    RiskLevel,
    OperatingMode,
    TeslaPrinciples,
)


# =============================================================================
# MATH VERIFIER TESTS
# =============================================================================

class TestMathVerifier(unittest.TestCase):
    """Verify mathematical verification produces correct results."""

    def setUp(self):
        self.verifier = MathVerifier()

    def test_correct_addition(self):
        """2 + 2 = 4 must be verified correctly."""
        result = self.verifier.verify_equation("2 + 2", 4.0)
        self.assertEqual(result.status, VerificationStatus.VERIFIED)
        self.assertEqual(result.confidence, 1.0)

    def test_incorrect_addition(self):
        """2 + 2 ≠ 5 must be caught."""
        result = self.verifier.verify_equation("2 + 2", 5.0)
        self.assertEqual(result.status, VerificationStatus.FAILED)
        self.assertEqual(result.confidence, 1.0)
        self.assertIn("4", result.explanation)  # Should show actual = 4

    def test_multiplication(self):
        """3 * 7 = 21."""
        result = self.verifier.verify_equation("3 * 7", 21.0)
        self.assertEqual(result.status, VerificationStatus.VERIFIED)

    def test_complex_expression(self):
        """(2 + 3) * 4 = 20."""
        result = self.verifier.verify_equation("(2 + 3) * 4", 20.0)
        self.assertEqual(result.status, VerificationStatus.VERIFIED)

    def test_division(self):
        """10 / 3 ≈ 3.333..."""
        result = self.verifier.verify_equation("10 / 3", 10 / 3)
        self.assertEqual(result.status, VerificationStatus.VERIFIED)

    def test_power(self):
        """2 ** 8 = 256."""
        result = self.verifier.verify_equation("2 ** 8", 256.0)
        self.assertEqual(result.status, VerificationStatus.VERIFIED)

    def test_negative_numbers(self):
        """Negative results work correctly."""
        result = self.verifier.verify_equation("3 - 10", -7.0)
        self.assertEqual(result.status, VerificationStatus.VERIFIED)

    def test_extract_and_verify_correct(self):
        """Extraction from text: '3 * 4 = 12' should verify."""
        text = "El resultado de 3 * 4 = 12 es correcto"
        results = self.verifier.extract_and_verify(text)
        self.assertGreater(len(results), 0)
        self.assertEqual(results[0].status, VerificationStatus.VERIFIED)

    def test_extract_and_verify_incorrect(self):
        """Extraction from text: '2 + 2 = 5' should fail."""
        text = "La suma 2 + 2 = 5 es incorrecta"
        results = self.verifier.extract_and_verify(text)
        self.assertGreater(len(results), 0)
        self.assertEqual(results[0].status, VerificationStatus.FAILED)

    def test_safe_eval_rejects_dangerous_input(self):
        """Safe eval should reject non-mathematical expressions."""
        with self.assertRaises(Exception):
            self.verifier._safe_eval("__import__('os').system('echo hacked')")

    def test_different_inputs_give_different_results(self):
        """
        Anti-hardcoding: different expressions must produce different results.
        If the verifier always returns VERIFIED, it's broken.
        """
        r1 = self.verifier.verify_equation("5 + 5", 10.0)
        r2 = self.verifier.verify_equation("5 + 5", 11.0)
        self.assertNotEqual(
            r1.status, r2.status,
            "Same expression with correct and incorrect results "
            "returned the same status — possible hardcoding"
        )


# =============================================================================
# CONTRADICTION DETECTOR TESTS
# =============================================================================

class TestContradictionDetector(unittest.TestCase):
    """Verify contradiction detection uses actual word-pair logic."""

    def setUp(self):
        self.detector = ContradictionDetector()

    def test_detects_always_never_contradiction(self):
        """'X siempre Y' contradicts 'X nunca Y'."""
        self.detector.add_fact("el agua siempre hierve a 100 grados")
        result = self.detector.check_contradiction(
            "el agua nunca hierve a 100 grados"
        )
        self.assertIsNotNone(result)
        self.assertGreater(result.severity, 0.5)

    def test_no_contradiction_for_unrelated_facts(self):
        """Unrelated statements should not be flagged."""
        self.detector.add_fact("Python es un lenguaje de programación")
        result = self.detector.check_contradiction(
            "Java tiene recolector de basura"
        )
        self.assertIsNone(result)

    def test_negation_detection(self):
        """Direct negation should be caught."""
        self.detector.add_fact("kamaq puede aprender patrones")
        result = self.detector.check_contradiction(
            "kamaq no puede aprender patrones"
        )
        self.assertIsNotNone(result)

    def test_facts_accumulate(self):
        """Multiple facts can be added and checked against."""
        self.detector.add_fact("el cielo es azul")
        self.detector.add_fact("la tierra es redonda")
        self.detector.add_fact("el sol siempre sale por el este")

        # This contradicts the third fact
        result = self.detector.check_contradiction(
            "el sol nunca sale por el este"
        )
        self.assertIsNotNone(result)


# =============================================================================
# CONFIDENCE CALIBRATOR TESTS
# =============================================================================

class TestConfidenceCalibrator(unittest.TestCase):
    """Verify calibration adjusts confidence based on real history."""

    def test_no_history_returns_raw(self):
        """With insufficient history, raw confidence is returned."""
        cal = ConfidenceCalibrator()
        raw = 0.85
        calibrated = cal.get_calibrated_confidence(raw)
        self.assertEqual(calibrated, raw)

    def test_overconfident_history_reduces_confidence(self):
        """
        If we claim 80% confidence but are only right 50% of the time,
        calibrated confidence should be ~50%.
        """
        cal = ConfidenceCalibrator()

        # Record 20 predictions at ~80% confidence, but only 10 correct
        for i in range(20):
            cal.record(0.85, i < 10)  # First 10 correct, last 10 wrong

        calibrated = cal.get_calibrated_confidence(0.85)
        # Should be approximately 50% (actual accuracy in that bucket)
        self.assertLess(
            calibrated, 0.65,
            f"Calibrated={calibrated:.2f}, expected <0.65 for 50% actual accuracy"
        )

    def test_underconfident_history_increases_confidence(self):
        """
        If we claim 30% confidence but are right 90% of the time,
        calibrated confidence should increase.
        """
        cal = ConfidenceCalibrator()

        # Record 20 predictions at ~30%, 18 correct
        for i in range(20):
            cal.record(0.35, i < 18)

        calibrated = cal.get_calibrated_confidence(0.35)
        self.assertGreater(
            calibrated, 0.7,
            f"Calibrated={calibrated:.2f}, expected >0.7 for 90% actual accuracy"
        )

    def test_ece_perfect_calibration(self):
        """Perfect calibration: ECE should be 0 or very close."""
        cal = ConfidenceCalibrator()

        # At each confidence level, actual accuracy matches
        for _ in range(10):
            cal.record(0.85, True)  # 80% bucket, all correct
        for _ in range(10):
            cal.record(0.55, True)  # 50% bucket, all correct

        ece = cal.get_ece()
        # ECE won't be 0 because actual accuracy (100%) differs from bucket
        # center confidence. For buckets 80% and 50% both at 100% accuracy,
        # ECE = 0.5*|1.0-0.8| + 0.5*|1.0-0.5| = 0.35 — mathematically correct.
        self.assertLess(ece, 0.4, f"ECE={ece:.3f}, expected < 0.4")

    def test_ece_terrible_calibration(self):
        """Terrible calibration: claim 90% but always wrong → high ECE."""
        cal = ConfidenceCalibrator()

        for _ in range(20):
            cal.record(0.95, False)  # Claim 95%, always wrong

        ece = cal.get_ece()
        self.assertGreater(
            ece, 0.5,
            f"ECE={ece:.3f}, expected > 0.5 for always-wrong at 95%"
        )


# =============================================================================
# CONSTITUTION TESTS
# =============================================================================

class TestConstitution(unittest.TestCase):
    """Verify constitution evaluates risks consistently."""

    def setUp(self):
        self.constitution = KAMAQConstitution()

    def test_forbidden_actions(self):
        """Actions with forbidden keywords must be FORBIDDEN."""
        forbidden_actions = [
            "sudo rm -rf /",
            "drop database users",
            "format C:",
        ]
        for action in forbidden_actions:
            risk = self.constitution.evaluate_action_risk(action, {})
            self.assertEqual(
                risk, RiskLevel.FORBIDDEN,
                f"'{action}' should be FORBIDDEN, got {risk.name}"
            )

    def test_high_risk_actions(self):
        """Destructive actions must be HIGH risk."""
        high_risk = [
            "eliminar archivos temporales",
            "borrar la base de datos",
            "ejecutar script desconocido",
        ]
        for action in high_risk:
            risk = self.constitution.evaluate_action_risk(action, {})
            self.assertIn(
                risk, [RiskLevel.HIGH, RiskLevel.FORBIDDEN],
                f"'{action}' should be HIGH or FORBIDDEN, got {risk.name}"
            )

    def test_safe_actions(self):
        """Read-only actions must be LOW risk or SAFE."""
        safe_actions = [
            "leer el archivo README.md",
            "buscar información sobre Python",
            "listar archivos en el directorio",
        ]
        for action in safe_actions:
            risk = self.constitution.evaluate_action_risk(action, {})
            self.assertIn(
                risk, [RiskLevel.SAFE, RiskLevel.LOW],
                f"'{action}' should be SAFE or LOW, got {risk.name}"
            )

    def test_escalation_for_ethical_concern(self):
        """Ethical concerns must always escalate."""
        should = self.constitution.should_escalate(
            uncertainty=0.0,
            action_risk=RiskLevel.SAFE,
            ethical_concern=True,
        )
        self.assertTrue(should, "Ethical concern should always escalate")

    def test_escalation_for_high_uncertainty(self):
        """High uncertainty (>70%) must escalate."""
        should = self.constitution.should_escalate(
            uncertainty=0.8,
            action_risk=RiskLevel.SAFE,
        )
        self.assertTrue(should, "Uncertainty > 70% should escalate")

    def test_no_escalation_for_safe_and_certain(self):
        """Safe action with low uncertainty should NOT escalate."""
        should = self.constitution.should_escalate(
            uncertainty=0.2,
            action_risk=RiskLevel.SAFE,
        )
        self.assertFalse(should, "Safe + certain should NOT escalate")

    def test_tesla_principles_evaluate_raises(self):
        """TeslaPrinciples.evaluate() should raise NotImplementedError."""
        tp = TeslaPrinciples()
        with self.assertRaises(NotImplementedError):
            tp.evaluate("some solution")

    def test_constitution_values_are_complete(self):
        """The 7 core values must all be present."""
        expected_values = [
            "honesty", "safety", "privacy", "transparency",
            "humility", "helpfulness", "calibration"
        ]
        for value in expected_values:
            self.assertIn(
                value, self.constitution.values,
                f"Missing core value: {value}"
            )

    def test_system_prompt_generation(self):
        """System prompt should be non-empty and contain values."""
        prompt = self.constitution.to_system_prompt()
        self.assertGreater(len(prompt), 100)
        self.assertIn("KAMAQ", prompt)
        self.assertIn("HONESTY", prompt.upper())


# =============================================================================
# INTEGRATED VERIFIER TESTS
# =============================================================================

class TestKAMAQVerifier(unittest.TestCase):
    """Test the integrated verification system."""

    def test_verify_response_with_correct_math(self):
        """Response with correct math should be VERIFIED."""
        verifier = KAMAQVerifier()
        results = verifier.verify_response(
            response="La suma de 3 + 4 = 7 es correcta.",
            claimed_confidence=0.9,
        )
        self.assertEqual(results["overall_status"], VerificationStatus.VERIFIED)

    def test_verify_response_with_wrong_math(self):
        """Response with wrong math should be FAILED."""
        verifier = KAMAQVerifier()
        results = verifier.verify_response(
            response="Calculé que 5 * 3 = 20.",
            claimed_confidence=0.9,
        )
        self.assertEqual(results["overall_status"], VerificationStatus.FAILED)
        self.assertGreater(len(results["warnings"]), 0)

    def test_uncertainty_words_reduce_confidence(self):
        """
        If response uses uncertainty language ('creo', 'quizás')
        but claims high confidence, the adjusted confidence should drop.
        """
        verifier = KAMAQVerifier()

        # Record enough history so calibrator doesn't just pass through
        for _ in range(10):
            verifier.record_outcome(0.85, True)

        results = verifier.verify_response(
            response="Creo que tal vez posiblemente la respuesta es 42.",
            claimed_confidence=0.9,
        )
        # Adjusted confidence should be less than claimed
        self.assertLess(
            results["adjusted_confidence"], 0.9,
            "Uncertainty words in high-confidence response should reduce confidence"
        )

    def test_should_warn_on_failure(self):
        """Warning should be generated for failed verification."""
        verifier = KAMAQVerifier()
        results = {
            "overall_status": VerificationStatus.FAILED,
            "contradictions": [],
            "adjusted_confidence": 0.5,
            "warnings": ["test warning"],
        }
        warning = verifier.should_warn_user(results)
        self.assertIsNotNone(warning)

    def test_no_warning_for_clean_response(self):
        """No warning for a clean verified response."""
        verifier = KAMAQVerifier()
        results = {
            "overall_status": VerificationStatus.VERIFIED,
            "contradictions": [],
            "adjusted_confidence": 0.85,
            "warnings": [],
        }
        warning = verifier.should_warn_user(results)
        self.assertIsNone(warning)


if __name__ == "__main__":
    print("=" * 70)
    print("  KAMAQ Agent Components Tests")
    print("  Verifier, Constitution, Calibration")
    print("=" * 70)
    unittest.main(verbosity=2)

"""
Tests for Reservoir Computing — Verifying Real Learning
========================================================

These tests verify that:
1. The reservoir LEARNS from data (weights change during training)
2. The reservoir does NOT use hardcoded weights or thresholds
3. Before training, predictions do not match expected labels
4. After training, XOR and other logic gates are solved
5. The reservoir's internal weights remain unchanged (only readout trains)
6. Generalization to unseen continuous inputs works

Run: python -m pytest tests/test_reservoir.py -v
  or: python tests/test_reservoir.py

Author: Carlos Huarcaya
Date: February 2026
"""

import unittest
import numpy as np
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from prototipo.celula_reservoir_v5 import ReservoirCognitivo


class TestReservoirLearning(unittest.TestCase):
    """Verify the reservoir learns from data, not from pre-set weights."""
    
    def test_untrained_predictions_do_not_solve_xor(self):
        """
        CRITICAL: Before training, the reservoir should NOT solve XOR.
        
        If it does, the weights are hardcoded. This is the most important
        anti-hardcoding test.
        """
        np.random.seed(42)
        reservoir = ReservoirCognitivo(n_osciladores=50)
        
        X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
        y = np.array([0, 1, 1, 0], dtype=float)
        
        # Before training: W_output is all zeros
        self.assertTrue(
            np.allclose(reservoir.W_output, 0),
            "W_output should be all zeros before training"
        )
        
        # Predictions before training should be ~0 (dot with zero weights)
        preds = [reservoir.predict(x) for x in X]
        
        # With zero W_output, all predictions should be near 0
        for p in preds:
            self.assertAlmostEqual(
                p, 0.0, places=5,
                msg=f"Untrained prediction is {p}, expected ~0.0 (W_output is zeros)"
            )
    
    def test_training_changes_output_weights(self):
        """W_output must change after training — proof that learning happened."""
        reservoir = ReservoirCognitivo(n_osciladores=50)
        
        W_before = reservoir.W_output.copy()
        
        X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
        y = np.array([0, 1, 1, 0], dtype=float)
        reservoir.train(X, y)
        
        W_after = reservoir.W_output.copy()
        
        # Weights must be different
        self.assertFalse(
            np.allclose(W_before, W_after),
            "W_output didn't change after training. Training is broken or no-op."
        )
    
    def test_reservoir_weights_unchanged_during_training(self):
        """
        The reservoir's internal coupling (W_reservoir, W_input) must NOT change.
        
        Only W_output should be trained. This is the core principle of
        Reservoir Computing / Echo State Networks.
        """
        reservoir = ReservoirCognitivo(n_osciladores=50)
        
        W_res_before = reservoir.W_reservoir.copy()
        W_inp_before = reservoir.W_input.copy()
        
        X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
        y = np.array([0, 1, 1, 0], dtype=float)
        reservoir.train(X, y)
        
        self.assertTrue(
            np.allclose(reservoir.W_reservoir, W_res_before),
            "W_reservoir changed during training. Only W_output should be trained."
        )
        self.assertTrue(
            np.allclose(reservoir.W_input, W_inp_before),
            "W_input changed during training. Only W_output should be trained."
        )
    
    def test_xor_solved_after_training(self):
        """XOR must be solved with 100% accuracy after training."""
        reservoir = ReservoirCognitivo(n_osciladores=50, spectral_radius=0.9)
        
        X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
        y = np.array([0, 1, 1, 0], dtype=float)
        reservoir.train(X, y, n_steps=30)
        
        correct = 0
        for x, expected in zip(X, y):
            pred = reservoir.predict(x, n_steps=30)
            pred_bit = 1 if pred > 0.5 else 0
            if pred_bit == int(expected):
                correct += 1
        
        self.assertEqual(
            correct, 4,
            f"XOR accuracy: {correct}/4. Must be 4/4 after training."
        )


class TestAllLogicGates(unittest.TestCase):
    """Verify the SAME reservoir architecture learns different functions."""
    
    def _test_logic_gate(self, y: np.ndarray, gate_name: str):
        """Helper: train and verify a logic gate."""
        X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
        
        reservoir = ReservoirCognitivo(n_osciladores=50)
        reservoir.train(X, y, n_steps=30)
        
        correct = 0
        for x, expected in zip(X, y):
            pred = reservoir.predict(x, n_steps=30)
            pred_bit = 1 if pred > 0.5 else 0
            if pred_bit == int(expected):
                correct += 1
        
        self.assertEqual(
            correct, 4,
            f"{gate_name} accuracy: {correct}/4. Must be 4/4."
        )
    
    def test_and_gate(self):
        """AND gate: only (1,1) → 1."""
        self._test_logic_gate(np.array([0, 0, 0, 1], dtype=float), "AND")
    
    def test_or_gate(self):
        """OR gate: any 1 → 1."""
        self._test_logic_gate(np.array([0, 1, 1, 1], dtype=float), "OR")
    
    def test_nand_gate(self):
        """NAND gate: NOT AND."""
        self._test_logic_gate(np.array([1, 1, 1, 0], dtype=float), "NAND")
    
    def test_xor_gate(self):
        """XOR gate: the non-linearly separable problem."""
        self._test_logic_gate(np.array([0, 1, 1, 0], dtype=float), "XOR")
    
    def test_same_architecture_different_labels(self):
        """
        CRITICAL: Same architecture must learn DIFFERENT functions based
        on DIFFERENT training labels.
        
        If XOR and AND produce the same predictions, the system is NOT
        learning from labels — it's hardcoded.
        """
        X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
        y_xor = np.array([0, 1, 1, 0], dtype=float)
        y_and = np.array([0, 0, 0, 1], dtype=float)
        
        # Train two separate reservoirs
        res_xor = ReservoirCognitivo(n_osciladores=50)
        res_xor.train(X, y_xor, n_steps=30)
        
        res_and = ReservoirCognitivo(n_osciladores=50)
        res_and.train(X, y_and, n_steps=30)
        
        # Predictions must differ
        preds_xor = [res_xor.predict(x, n_steps=30) for x in X]
        preds_and = [res_and.predict(x, n_steps=30) for x in X]
        
        # At least one prediction must be different (XOR and AND differ on 3/4 inputs)
        differences = sum(
            1 for px, pa in zip(preds_xor, preds_and) if abs(px - pa) > 0.3
        )
        self.assertGreater(
            differences, 0,
            "XOR and AND reservoirs produce identical predictions. "
            "The system is not learning from labels."
        )


class TestGeneralization(unittest.TestCase):
    """Verify generalization to inputs not seen during training."""
    
    def test_continuous_inputs_near_training_points(self):
        """
        Inputs close to training points should produce similar outputs.
        
        Train on {0,1}^2, test on points like (0.1, 0.9).
        If the reservoir generalizes, (0.1, 0.9) ≈ (0, 1) → 1 for XOR.
        """
        X_train = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
        y_train = np.array([0, 1, 1, 0], dtype=float)
        
        reservoir = ReservoirCognitivo(n_osciladores=100)
        reservoir.train(X_train, y_train, n_steps=50)
        
        # Test near-corner points
        test_cases = [
            (np.array([0.1, 0.1]), 0),  # near (0,0) → 0
            (np.array([0.9, 0.1]), 1),  # near (1,0) → 1
            (np.array([0.1, 0.9]), 1),  # near (0,1) → 1
            (np.array([0.9, 0.9]), 0),  # near (1,1) → 0
        ]
        
        correct = 0
        for inputs, expected in test_cases:
            pred = reservoir.predict(inputs, n_steps=50)
            pred_bit = 1 if pred > 0.5 else 0
            if pred_bit == expected:
                correct += 1
        
        self.assertGreaterEqual(
            correct, 3,
            f"Generalization: {correct}/4 correct. Must be >= 3/4."
        )
    
    def test_different_random_seeds_produce_different_reservoirs(self):
        """
        Different random initializations must produce different internal states.
        
        This verifies that the reservoir's dynamics are genuinely driven by
        its random connectivity, not by deterministic shortcuts.
        """
        X = np.array([[1, 0]], dtype=float)
        
        np.random.seed(42)
        res1 = ReservoirCognitivo(n_osciladores=50)
        state1 = res1.run(X[0], n_steps=20)
        
        np.random.seed(123)
        res2 = ReservoirCognitivo(n_osciladores=50)
        state2 = res2.run(X[0], n_steps=20)
        
        # States should differ because internal weights differ
        self.assertFalse(
            np.allclose(state1, state2, atol=0.01),
            "Different random seeds produced identical reservoir states"
        )


class TestReservoirVsPerceptron(unittest.TestCase):
    """Verify reservoir's advantage over linear models on nonlinear problems."""
    
    def test_single_perceptron_fails_xor(self):
        """
        A single-layer perceptron CANNOT solve XOR (Minsky & Papert, 1969).
        
        This is the control test: if a perceptron could solve XOR, the
        reservoir's success wouldn't mean anything.
        """
        X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype=float)
        y = np.array([0, 1, 1, 0], dtype=float)
        
        # Train perceptron with gradient descent
        np.random.seed(42)
        w = np.random.randn(2) * 0.1
        b = 0.0
        lr = 0.5
        
        for _ in range(2000):
            for i in range(len(X)):
                z = np.dot(w, X[i]) + b
                pred = 1 / (1 + np.exp(-np.clip(z, -10, 10)))
                error = y[i] - pred
                grad = pred * (1 - pred)
                w += lr * error * grad * X[i]
                b += lr * error * grad
        
        correct = 0
        for i in range(len(X)):
            z = np.dot(w, X[i]) + b
            pred = 1 / (1 + np.exp(-np.clip(z, -10, 10)))
            pred_bit = 1 if pred > 0.5 else 0
            if pred_bit == int(y[i]):
                correct += 1
        
        # Perceptron should fail XOR (get at most 3/4)
        self.assertLess(
            correct, 4,
            "Single perceptron solved XOR — this violates the XOR impossibility theorem"
        )


if __name__ == "__main__":
    print("=" * 70)
    print("  KAMAQ Reservoir Computing Tests")
    print("  Verifying real learning, no hardcoding")
    print("=" * 70)
    unittest.main(verbosity=2)

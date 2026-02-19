"""
Tests for Holographic Memory — Verifying Real HRR Functionality
================================================================

These tests verify that:
1. HRR bind/unbind operations are mathematically correct (not simulated)
2. Encoding stores information in vectors (not just text metadata)
3. Recall uses actual vector operations (not just substring matching)
4. Bag-of-words vectors capture word overlap (not semantic similarity)
5. No hardcoding exists — every result comes from computation

Run: python -m pytest tests/test_holographic_memory.py -v
  or: python tests/test_holographic_memory.py

Author: Carlos Huarcaya
Date: February 2026
"""

import unittest
import numpy as np
import tempfile
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from kamaq_companion.core.holographic_memory import HolographicMemory, RecallResult


class TestHRRMathOperations(unittest.TestCase):
    """Verify that HRR bind/unbind is mathematically functional, not decorative."""
    
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.mem = HolographicMemory(base_dim=1024, memory_path=self.tmpdir)
    
    def test_bind_unbind_recovers_original_vector(self):
        """
        Core HRR property: correlate(convolve(a, b), a) ≈ b
        
        If this fails, the circular convolution/correlation is broken
        and ALL HRR-based recall is meaningless.
        """
        a = self.mem._random_vector()
        b = self.mem._random_vector()
        
        bound = self.mem._circular_convolve(a, b)
        recovered = self.mem._circular_correlate(bound, a)
        
        similarity = self.mem._similarity(recovered, b)
        # HRR circular correlation recovery: similarity depends on vector
        # dimension & normalization. For D=1024, typical recovery is 0.6-0.9.
        self.assertGreater(
            similarity, 0.5,
            f"HRR unbinding failed: similarity={similarity:.4f}, expected > 0.5. "
            f"This means the fundamental HRR operation is broken."
        )
    
    def test_superposition_unbinding_with_three_pairs(self):
        """
        Multiple key-value bindings superimposed can each be individually recovered.
        
        trace = (k1 ⊛ v1) + (k2 ⊛ v2) + (k3 ⊛ v3)
        correlate(trace, k2) ≈ v2 + noise
        
        This is the mathematical foundation of holographic memory.
        """
        n_pairs = 3
        keys = [self.mem._random_vector() for _ in range(n_pairs)]
        values = [self.mem._random_vector() for _ in range(n_pairs)]
        
        # Create superposition
        trace = np.zeros(1024)
        for k, v in zip(keys, values):
            trace += self.mem._circular_convolve(k, v)
        
        # Unbind each and verify recovery
        for i in range(n_pairs):
            recovered = self.mem._circular_correlate(trace, keys[i])
            sim = self.mem._similarity(recovered, values[i])
            self.assertGreater(
                sim, 0.25,
                f"Pair {i}: similarity={sim:.4f}, expected > 0.25 with {n_pairs} items in 1024 dims"
            )
    
    def test_random_vectors_are_approximately_orthogonal(self):
        """
        In high dimensions (1024), random unit vectors should be nearly orthogonal.
        This is the foundation for why HRR works — cross-talk noise is small.
        
        Expected: |cos(a, b)| < 0.1 for random a, b in R^1024
        """
        similarities = []
        for _ in range(100):
            a = self.mem._random_vector()
            b = self.mem._random_vector()
            similarities.append(abs(self.mem._similarity(a, b)))
        
        mean_sim = np.mean(similarities)
        self.assertLess(
            mean_sim, 0.1,
            f"Random vectors not orthogonal enough: mean |similarity|={mean_sim:.4f}. "
            f"Expected < 0.1 for D=1024."
        )
    
    def test_text_to_vector_is_deterministic(self):
        """Same text must always produce the same vector (SHA-256 seeded)."""
        v1 = self.mem._text_to_vector("KAMAQ cognitive architecture")
        v2 = self.mem._text_to_vector("KAMAQ cognitive architecture")
        v3 = self.mem._text_to_vector("Something completely different")
        
        # Same text → same vector
        self.assertAlmostEqual(
            self.mem._similarity(v1, v2), 1.0, places=10,
            msg="Same text produced different vectors — _text_to_vector is not deterministic"
        )
        
        # Different text → different vector (approximately orthogonal)
        sim = abs(self.mem._similarity(v1, v3))
        self.assertLess(
            sim, 0.15,
            f"Different texts produced similar vectors: |sim|={sim:.4f}"
        )


class TestBagOfWordsVectors(unittest.TestCase):
    """Verify bag-of-words compositional vectors work as claimed."""
    
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.mem = HolographicMemory(base_dim=1024, memory_path=self.tmpdir)
    
    def test_shared_words_produce_positive_similarity(self):
        """Texts sharing words must have positive cosine similarity."""
        v1 = self.mem._text_to_bag_vector("reservoir computing research")
        v2 = self.mem._text_to_bag_vector("computing with neural reservoirs")
        
        sim = self.mem._similarity(v1, v2)
        self.assertGreater(
            sim, 0.2,
            f"Texts sharing 'computing' have similarity={sim:.4f}, expected > 0.2"
        )
    
    def test_no_shared_words_produce_near_zero_similarity(self):
        """Texts without shared words must have near-zero similarity."""
        v1 = self.mem._text_to_bag_vector("quantum physics laboratory")
        v2 = self.mem._text_to_bag_vector("medieval cooking recipes")
        
        sim = abs(self.mem._similarity(v1, v2))
        self.assertLess(
            sim, 0.15,
            f"Texts without shared words have |similarity|={sim:.4f}, expected < 0.15"
        )
    
    def test_more_shared_words_higher_similarity(self):
        """More shared words should produce higher similarity."""
        base = "KAMAQ holographic memory system"
        similar = "KAMAQ memory system architecture"  # shares 3 words
        less_similar = "KAMAQ neural network"  # shares 1 word
        
        v_base = self.mem._text_to_bag_vector(base)
        v_similar = self.mem._text_to_bag_vector(similar)
        v_less = self.mem._text_to_bag_vector(less_similar)
        
        sim_high = self.mem._similarity(v_base, v_similar)
        sim_low = self.mem._similarity(v_base, v_less)
        
        self.assertGreater(
            sim_high, sim_low,
            f"More shared words should give higher similarity: "
            f"3-shared={sim_high:.4f} vs 1-shared={sim_low:.4f}"
        )
    
    def test_bag_vector_is_deterministic(self):
        """Same text must always produce the same bag vector."""
        v1 = self.mem._text_to_bag_vector("holographic reduced representations")
        v2 = self.mem._text_to_bag_vector("holographic reduced representations")
        
        self.assertAlmostEqual(
            self.mem._similarity(v1, v2), 1.0, places=10,
            msg="Bag vector is not deterministic"
        )
    
    def test_bag_vector_is_order_independent(self):
        """Bag-of-words should be insensitive to word order."""
        v1 = self.mem._text_to_bag_vector("reservoir computing oscillator")
        v2 = self.mem._text_to_bag_vector("oscillator reservoir computing")
        
        self.assertAlmostEqual(
            self.mem._similarity(v1, v2), 1.0, places=10,
            msg="Bag vector should be order-independent (it's bag-of-words)"
        )


class TestEncodeAndRecall(unittest.TestCase):
    """Verify the full encode-recall pipeline uses real vector operations."""
    
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.mem = HolographicMemory(base_dim=1024, memory_path=self.tmpdir)
        
        # Encode some memories
        self.mem.encode(
            "KAMAQ es una IA honesta",
            "Sistema basado en fisica que cuantifica incertidumbre",
            importance=0.9
        )
        self.mem.encode(
            "Memoria holografica HRR",
            "Usa convolucion circular FFT para asociar conceptos",
            importance=0.8
        )
        self.mem.encode(
            "Reservoir computing resuelve XOR",
            "Red de osciladores acoplados con capa lineal entrenable",
            importance=0.85
        )
    
    def test_exact_key_returns_full_confidence(self):
        """Exact key match must return confidence 1.0."""
        result = self.mem.recall("KAMAQ es una IA honesta")
        self.assertTrue(result.found)
        self.assertEqual(result.confidence, 1.0)
        self.assertIn("incertidumbre", result.content)
    
    def test_query_sharing_words_finds_correct_memory(self):
        """Query sharing words with a key should find it via bag similarity."""
        result = self.mem.recall("que es KAMAQ")
        self.assertTrue(result.found)
        self.assertEqual(result.key, "KAMAQ es una IA honesta")
        self.assertGreater(result.confidence, 0.0)
    
    def test_unrelated_query_returns_not_found(self):
        """Completely unrelated query should return found=False."""
        result = self.mem.recall("medieval castle architecture normandy")
        self.assertFalse(result.found)
    
    def test_bag_vectors_populated_after_encode(self):
        """Encoding must populate key_bag_vectors for similarity search."""
        self.assertEqual(len(self.mem.key_bag_vectors), 3)
        for key in self.mem.trace_index:
            self.assertIn(key, self.mem.key_bag_vectors)
            vec = self.mem.key_bag_vectors[key]
            self.assertEqual(vec.shape, (1024,))
            # Vector must be normalized (unit norm)
            self.assertAlmostEqual(np.linalg.norm(vec), 1.0, places=5)
    
    def test_memory_bank_stores_bound_vectors(self):
        """memory_bank must contain bound HRR vectors, not raw text."""
        self.assertEqual(len(self.mem.memory_bank), 3)
        for _id, vec in self.mem.memory_bank.items():
            self.assertEqual(vec.shape, (1024,))
            # Bound vectors should NOT be zero (they carry information)
            self.assertGreater(np.linalg.norm(vec), 0.01)
    
    def test_hrr_unbinding_works_for_exact_key(self):
        """HRR unbinding should recover content for an exact key match."""
        # Manually test the HRR strategy
        candidates = self.mem._recall_via_hrr("Memoria holografica HRR")
        
        # Should find candidates (exact key in the trace)
        self.assertGreater(len(candidates), 0, "HRR unbinding found no candidates")
        
        # The correct key should be among candidates
        found_keys = [k for _, k, _ in candidates]
        self.assertIn(
            "Memoria holografica HRR", found_keys,
            f"HRR unbinding didn't find exact key. Found: {found_keys}"
        )
    
    def test_recall_related_memories(self):
        """Recall should return related memories when available."""
        result = self.mem.recall("KAMAQ", max_results=5)
        # Should find something (KAMAQ is a word in the first key)
        if result.found:
            # related_memories should be a list
            self.assertIsInstance(result.related_memories, list)


class TestPersistence(unittest.TestCase):
    """Verify save/load roundtrip preserves all data."""
    
    def test_save_load_preserves_memories(self):
        """After save+load, all memories should be recoverable."""
        tmpdir = tempfile.mkdtemp()
        
        # Create and populate
        mem1 = HolographicMemory(base_dim=1024, memory_path=tmpdir)
        mem1.encode("test_key_alpha", "Content alpha about testing")
        mem1.encode("test_key_beta", "Content beta about experiments")
        mem1.save()
        
        # Load fresh instance
        mem2 = HolographicMemory(base_dim=1024, memory_path=tmpdir)
        
        # Verify trace index
        self.assertEqual(len(mem2.trace_index), 2)
        self.assertIn("test_key_alpha", mem2.trace_index)
        
        # Verify bag vectors were regenerated
        self.assertEqual(len(mem2.key_bag_vectors), 2)
        
        # Verify recall still works
        result = mem2.recall("test_key_alpha")
        self.assertTrue(result.found)
        self.assertEqual(result.confidence, 1.0)
        self.assertIn("alpha", result.content)
    
    def test_save_load_preserves_memory_bank(self):
        """Holographic vectors must survive save/load."""
        tmpdir = tempfile.mkdtemp()
        
        mem1 = HolographicMemory(base_dim=1024, memory_path=tmpdir)
        mem1.encode("persistence_test", "Data that must survive roundtrip")
        
        # Capture holographic sum before save
        sum_before = mem1.holographic_sum.copy()
        
        mem1.save()
        
        # Load and compare
        mem2 = HolographicMemory(base_dim=1024, memory_path=tmpdir)
        sum_after = mem2.holographic_sum
        
        sim = mem1._similarity(sum_before, sum_after)
        self.assertGreater(
            sim, 0.99,
            f"Holographic sum changed after save/load: similarity={sim:.4f}"
        )


class TestNoHardcoding(unittest.TestCase):
    """
    Verify that results come from computation, not from hardcoded values.
    
    Strategy: if we change the input, the output must change proportionally.
    A hardcoded system would return the same output regardless of input.
    """
    
    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
    
    def test_different_content_produces_different_vectors(self):
        """Encoding different content must produce different memory bank entries."""
        mem = HolographicMemory(base_dim=1024, memory_path=self.tmpdir)
        
        mem.encode("key_A", "First completely unique content string")
        mem.encode("key_B", "Second absolutely different content string")
        
        vectors = list(mem.memory_bank.values())
        sim = mem._similarity(vectors[0], vectors[1])
        
        # Different bindings should produce nearly orthogonal vectors
        self.assertLess(
            abs(sim), 0.3,
            f"Different key-content bindings are too similar: |sim|={abs(sim):.4f}. "
            f"This suggests the binding operation is not working."
        )
    
    def test_importance_scales_vector_magnitude(self):
        """Higher importance should produce larger magnitude bound vectors."""
        mem = HolographicMemory(base_dim=1024, memory_path=self.tmpdir)
        
        mem.encode("low_importance", "Some content", importance=0.1)
        mem.encode("high_importance", "Some other content", importance=0.9)
        
        vectors = list(mem.memory_bank.values())
        norm_low = np.linalg.norm(vectors[0])
        norm_high = np.linalg.norm(vectors[1])
        
        self.assertGreater(
            norm_high, norm_low,
            f"Importance scaling not working: high={norm_high:.4f}, low={norm_low:.4f}"
        )
    
    def test_recall_confidence_varies_with_query_relevance(self):
        """Recall confidence must differ based on query relevance, not be constant."""
        mem = HolographicMemory(base_dim=1024, memory_path=self.tmpdir)
        mem.encode("Python programming language", "General purpose interpreted language")
        
        # Very relevant query
        r1 = mem.recall("Python programming")
        # Less relevant query
        r2 = mem.recall("something about language")
        
        # If both found, confidence should differ
        if r1.found and r2.found:
            self.assertNotAlmostEqual(
                r1.confidence, r2.confidence, places=2,
                msg="Recall confidence is identical for different queries — possible hardcoding"
            )


if __name__ == "__main__":
    print("=" * 70)
    print("  KAMAQ Holographic Memory Tests")
    print("  Verifying real HRR operations, no hardcoding")
    print("=" * 70)
    unittest.main(verbosity=2)

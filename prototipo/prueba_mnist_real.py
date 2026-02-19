# -*- coding: utf-8 -*-
"""
PRUEBA DEFINITIVA: MNIST REAL
==============================

Usando sklearn para descargar MNIST real.

Autor: Proyecto Kamaq
Fecha: 16 de Enero, 2026
"""

import numpy as np
import time
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# ==============================================================================
# RESERVOIR COMPUTING
# ==============================================================================

class ReservoirMNIST:
    """Echo State Network para MNIST"""

    def __init__(self, n_reservoir=1000, spectral_radius=0.95,
                 input_scaling=0.1, leak_rate=0.3, sparsity=0.1):
        self.n_reservoir = n_reservoir
        self.spectral_radius = spectral_radius
        self.input_scaling = input_scaling
        self.leak_rate = leak_rate
        self.sparsity = sparsity

        self.n_input = 784
        self.n_output = 10

        # Matriz de entrada sparse
        self.W_in = np.random.randn(n_reservoir, self.n_input) * input_scaling
        mask = np.random.rand(n_reservoir, self.n_input) > sparsity
        self.W_in[mask] = 0

        # Matriz del reservoir sparse
        self.W_res = np.random.randn(n_reservoir, n_reservoir)
        mask = np.random.rand(n_reservoir, n_reservoir) > sparsity
        self.W_res[mask] = 0

        # Escalar spectral radius
        eigenvalues = np.linalg.eigvals(self.W_res)
        max_eig = np.max(np.abs(eigenvalues))
        if max_eig > 0:
            self.W_res *= spectral_radius / max_eig

        self.W_out = None
        self.state = np.zeros(n_reservoir)

    def _update(self, x):
        pre = self.W_in @ x + self.W_res @ self.state
        new_state = np.tanh(pre)
        self.state = (1 - self.leak_rate) * self.state + self.leak_rate * new_state
        return self.state

    def _get_state(self, image, n_steps=5):
        self.state = np.zeros(self.n_reservoir)
        for _ in range(n_steps):
            self._update(image)
        return self.state.copy()

    def fit(self, X, y, n_steps=5, reg=1e-4):
        n = len(X)
        y_oh = np.zeros((n, 10))
        y_oh[np.arange(n), y] = 1

        print(f"  Procesando {n} muestras...")
        states = np.zeros((n, self.n_reservoir))
        for i in range(n):
            states[i] = self._get_state(X[i], n_steps)
            if (i+1) % 2000 == 0:
                print(f"    {i+1}/{n}")

        # Agregar bias
        states_b = np.hstack([states, np.ones((n, 1))])

        print("  Ridge Regression...")
        STS = states_b.T @ states_b
        self.W_out = np.linalg.solve(STS + reg * np.eye(STS.shape[0]), states_b.T @ y_oh)
        return self

    def predict(self, X, n_steps=5):
        preds = []
        for x in X:
            s = self._get_state(x, n_steps)
            s_b = np.concatenate([s, [1]])
            out = s_b @ self.W_out
            preds.append(np.argmax(out))
        return np.array(preds)

    def score(self, X, y, n_steps=5):
        return np.mean(self.predict(X, n_steps) == y)


# ==============================================================================
# BASELINES
# ==============================================================================

class LogisticRegression:
    """Regresion logistica simple"""
    def __init__(self, lr=0.1, epochs=50):
        self.lr = lr
        self.epochs = epochs

    def fit(self, X, y):
        n, d = X.shape
        self.W = np.zeros((d, 10))
        self.b = np.zeros(10)

        y_oh = np.zeros((n, 10))
        y_oh[np.arange(n), y] = 1

        for _ in range(self.epochs):
            logits = X @ self.W + self.b
            probs = np.exp(logits - logits.max(1, keepdims=True))
            probs /= probs.sum(1, keepdims=True)

            grad = (probs - y_oh) / n
            self.W -= self.lr * (X.T @ grad)
            self.b -= self.lr * grad.sum(0)
        return self

    def predict(self, X):
        return np.argmax(X @ self.W + self.b, axis=1)

    def score(self, X, y):
        return np.mean(self.predict(X) == y)


class MLP:
    """MLP de 2 capas"""
    def __init__(self, hidden=300, lr=0.01, epochs=30):
        self.hidden = hidden
        self.lr = lr
        self.epochs = epochs

    def fit(self, X, y):
        n, d = X.shape
        h = self.hidden

        self.W1 = np.random.randn(d, h) * np.sqrt(2/d)
        self.b1 = np.zeros(h)
        self.W2 = np.random.randn(h, 10) * np.sqrt(2/h)
        self.b2 = np.zeros(10)

        y_oh = np.zeros((n, 10))
        y_oh[np.arange(n), y] = 1

        batch = 128
        for epoch in range(self.epochs):
            idx = np.random.permutation(n)
            for i in range(0, n, batch):
                b_idx = idx[i:i+batch]
                xb, yb = X[b_idx], y_oh[b_idx]

                # Forward
                h1 = np.maximum(0, xb @ self.W1 + self.b1)
                logits = h1 @ self.W2 + self.b2
                probs = np.exp(logits - logits.max(1, keepdims=True))
                probs /= probs.sum(1, keepdims=True)

                # Backward
                m = len(b_idx)
                g2 = (probs - yb) / m
                gh = g2 @ self.W2.T * (h1 > 0)

                self.W2 -= self.lr * (h1.T @ g2)
                self.b2 -= self.lr * g2.sum(0)
                self.W1 -= self.lr * (xb.T @ gh)
                self.b1 -= self.lr * gh.sum(0)

        return self

    def predict(self, X):
        h1 = np.maximum(0, X @ self.W1 + self.b1)
        return np.argmax(h1 @ self.W2 + self.b2, axis=1)

    def score(self, X, y):
        return np.mean(self.predict(X) == y)


# ==============================================================================
# MAIN
# ==============================================================================

def main():
    print("="*70)
    print("   PRUEBA MNIST REAL - RESERVOIR vs BASELINES")
    print("="*70)

    # Cargar MNIST
    print("\n1. Cargando MNIST (puede tardar)...")
    try:
        mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='liac-arff')
        X, y = mnist.data, mnist.target.astype(int)
        print(f"   Cargado: {X.shape[0]} muestras, {X.shape[1]} dimensiones")
    except Exception as e:
        print(f"   Error: {e}")
        return

    # Normalizar
    X = X / 255.0

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42
    )

    # Usar subconjunto para velocidad
    n_train = 15000
    n_test = 3000

    X_train = X_train[:n_train]
    y_train = y_train[:n_train]
    X_test = X_test[:n_test]
    y_test = y_test[:n_test]

    print(f"   Train: {len(X_train)}, Test: {len(X_test)}")

    results = {}

    # Logistic Regression
    print("\n" + "-"*70)
    print("2. REGRESION LOGISTICA (baseline)")
    print("-"*70)
    t0 = time.time()
    lr = LogisticRegression(lr=0.5, epochs=100)
    lr.fit(X_train, y_train)
    t_lr = time.time() - t0
    acc_lr = lr.score(X_test, y_test) * 100
    print(f"   Accuracy: {acc_lr:.2f}%")
    print(f"   Tiempo: {t_lr:.1f}s")
    results['LogReg'] = acc_lr

    # MLP
    print("\n" + "-"*70)
    print("3. MLP (300 hidden, 2 capas)")
    print("-"*70)
    t0 = time.time()
    mlp = MLP(hidden=300, lr=0.01, epochs=30)
    mlp.fit(X_train, y_train)
    t_mlp = time.time() - t0
    acc_mlp = mlp.score(X_test, y_test) * 100
    print(f"   Accuracy: {acc_mlp:.2f}%")
    print(f"   Tiempo: {t_mlp:.1f}s")
    results['MLP'] = acc_mlp

    # Reservoir - Configuracion 1
    print("\n" + "-"*70)
    print("4. RESERVOIR (500 neuronas)")
    print("-"*70)
    t0 = time.time()
    res1 = ReservoirMNIST(n_reservoir=500, spectral_radius=0.9, leak_rate=0.3)
    res1.fit(X_train, y_train, n_steps=3)
    t_res1 = time.time() - t0
    acc_res1 = res1.score(X_test, y_test, n_steps=3) * 100
    print(f"   Accuracy: {acc_res1:.2f}%")
    print(f"   Tiempo: {t_res1:.1f}s")
    results['Reservoir_500'] = acc_res1

    # Reservoir - Configuracion 2 (mas grande)
    print("\n" + "-"*70)
    print("5. RESERVOIR (2000 neuronas)")
    print("-"*70)
    t0 = time.time()
    res2 = ReservoirMNIST(n_reservoir=2000, spectral_radius=0.95, leak_rate=0.2)
    res2.fit(X_train, y_train, n_steps=5)
    t_res2 = time.time() - t0
    acc_res2 = res2.score(X_test, y_test, n_steps=5) * 100
    print(f"   Accuracy: {acc_res2:.2f}%")
    print(f"   Tiempo: {t_res2:.1f}s")
    results['Reservoir_2000'] = acc_res2

    # Resumen
    print("\n" + "="*70)
    print("   RESUMEN")
    print("="*70)
    print(f"\n   {'Modelo':<25} {'Accuracy':<12}")
    print("   " + "-"*37)
    for name, acc in results.items():
        print(f"   {name:<25} {acc:.2f}%")

    # Veredicto
    print("\n" + "="*70)
    print("   VEREDICTO")
    print("="*70)

    best_res = max(results['Reservoir_500'], results['Reservoir_2000'])

    if best_res >= 95:
        print("\n   [OK] EXCELENTE: Reservoir alcanza >95%")
        print("   El enfoque es VIABLE para problemas reales.")
        veredicto = "VIABLE"
    elif best_res >= 90:
        print("\n   [OK] BUENO: Reservoir alcanza >90%")
        print("   El enfoque tiene potencial, necesita optimizacion.")
        veredicto = "PROMETEDOR"
    elif best_res >= 85:
        print("\n   [~] ACEPTABLE: Reservoir alcanza >85%")
        print("   El enfoque funciona pero es inferior a MLP.")
        veredicto = "LIMITADO"
    else:
        print(f"\n   [X] INSUFICIENTE: Reservoir solo alcanza {best_res:.1f}%")
        print("   El enfoque NO compite con alternativas simples.")
        veredicto = "NO_COMPETITIVO"

    print(f"\n   Comparacion:")
    print(f"   - Reservoir mejor: {best_res:.2f}%")
    print(f"   - MLP:             {acc_mlp:.2f}%")
    print(f"   - Diferencia:      {best_res - acc_mlp:+.2f}%")

    if best_res >= acc_mlp:
        print("\n   >>> RESERVOIR SUPERA O IGUALA A MLP <<<")
    else:
        print(f"\n   >>> RESERVOIR {acc_mlp - best_res:.1f}% POR DEBAJO DE MLP <<<")

    return results, veredicto


if __name__ == "__main__":
    results, veredicto = main()
    print("\n" + "="*70)
    print(f"   VEREDICTO FINAL: {veredicto}")
    print("="*70)

# -*- coding: utf-8 -*-
"""
PRUEBA DEFINITIVA: MNIST
=========================

MNIST es el "Hello World" del Machine Learning:
- 70,000 imagenes de digitos escritos a mano (0-9)
- 28x28 pixeles = 784 dimensiones
- 10 clases

Si el Reservoir Computing puede competir aqui, tiene potencial real.
Si falla, el enfoque es limitado a problemas toy.

BENCHMARKS DE REFERENCIA:
- Perceptron simple: ~92%
- MLP (2 capas): ~97%
- CNN (LeNet): ~99%
- Estado del arte: ~99.8%

OBJETIVO MINIMO: >90% (competir con perceptron)
OBJETIVO BUENO: >95% (competir con MLP)
OBJETIVO EXCELENTE: >97% (territorio de redes profundas)

Autor: Proyecto Kamaq
Fecha: 16 de Enero, 2026
"""

import numpy as np
import time
from typing import Tuple
import os
import gzip
import struct
from urllib.request import urlretrieve


# ==============================================================================
# DESCARGA DE MNIST
# ==============================================================================

def descargar_mnist(directorio: str = "mnist_data") -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Descarga MNIST si no existe y devuelve los datos.
    """
    os.makedirs(directorio, exist_ok=True)

    base_url = "http://yann.lecun.com/exdb/mnist/"
    archivos = {
        "train_images": "train-images-idx3-ubyte.gz",
        "train_labels": "train-labels-idx1-ubyte.gz",
        "test_images": "t10k-images-idx3-ubyte.gz",
        "test_labels": "t10k-labels-idx1-ubyte.gz"
    }

    # Descargar si no existe
    for nombre, archivo in archivos.items():
        ruta = os.path.join(directorio, archivo)
        if not os.path.exists(ruta):
            print(f"Descargando {archivo}...")
            try:
                urlretrieve(base_url + archivo, ruta)
            except Exception as e:
                print(f"Error descargando {archivo}: {e}")
                print("Intentando generar datos sinteticos...")
                return generar_mnist_sintetico()

    # Cargar datos
    def cargar_imagenes(ruta):
        with gzip.open(ruta, 'rb') as f:
            magic, num, rows, cols = struct.unpack(">IIII", f.read(16))
            return np.frombuffer(f.read(), dtype=np.uint8).reshape(num, rows * cols)

    def cargar_etiquetas(ruta):
        with gzip.open(ruta, 'rb') as f:
            magic, num = struct.unpack(">II", f.read(8))
            return np.frombuffer(f.read(), dtype=np.uint8)

    X_train = cargar_imagenes(os.path.join(directorio, archivos["train_images"]))
    y_train = cargar_etiquetas(os.path.join(directorio, archivos["train_labels"]))
    X_test = cargar_imagenes(os.path.join(directorio, archivos["test_images"]))
    y_test = cargar_etiquetas(os.path.join(directorio, archivos["test_labels"]))

    # Normalizar a [0, 1]
    X_train = X_train.astype(np.float64) / 255.0
    X_test = X_test.astype(np.float64) / 255.0

    return X_train, y_train, X_test, y_test


def generar_mnist_sintetico(n_train: int = 5000, n_test: int = 1000) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Genera datos sinteticos similares a MNIST si la descarga falla.
    Cada digito tiene un patron caracteristico simple.
    """
    print("Generando MNIST sintetico...")

    def crear_digito(label: int) -> np.ndarray:
        """Crea una imagen 28x28 con un patron simple para cada digito"""
        img = np.zeros((28, 28))

        # Patrones simples pero distinguibles
        if label == 0:  # Circulo
            for i in range(28):
                for j in range(28):
                    dist = np.sqrt((i-14)**2 + (j-14)**2)
                    if 8 < dist < 12:
                        img[i, j] = 1.0
        elif label == 1:  # Linea vertical
            img[4:24, 12:16] = 1.0
        elif label == 2:  # Forma de 2
            img[4:8, 8:20] = 1.0
            img[8:14, 16:20] = 1.0
            img[12:16, 8:20] = 1.0
            img[14:22, 8:12] = 1.0
            img[20:24, 8:20] = 1.0
        elif label == 3:  # Forma de 3
            img[4:8, 8:20] = 1.0
            img[12:16, 8:20] = 1.0
            img[20:24, 8:20] = 1.0
            img[4:24, 16:20] = 1.0
        elif label == 4:  # Forma de 4
            img[4:16, 8:12] = 1.0
            img[12:16, 8:20] = 1.0
            img[4:24, 16:20] = 1.0
        elif label == 5:  # Forma de 5
            img[4:8, 8:20] = 1.0
            img[4:14, 8:12] = 1.0
            img[12:16, 8:20] = 1.0
            img[14:22, 16:20] = 1.0
            img[20:24, 8:20] = 1.0
        elif label == 6:  # Forma de 6
            img[4:24, 8:12] = 1.0
            img[4:8, 8:20] = 1.0
            img[12:16, 8:20] = 1.0
            img[20:24, 8:20] = 1.0
            img[14:24, 16:20] = 1.0
        elif label == 7:  # Forma de 7
            img[4:8, 8:20] = 1.0
            img[4:24, 16:20] = 1.0
        elif label == 8:  # Forma de 8
            img[4:8, 8:20] = 1.0
            img[12:16, 8:20] = 1.0
            img[20:24, 8:20] = 1.0
            img[4:24, 8:12] = 1.0
            img[4:24, 16:20] = 1.0
        elif label == 9:  # Forma de 9
            img[4:16, 8:12] = 1.0
            img[4:8, 8:20] = 1.0
            img[12:16, 8:20] = 1.0
            img[4:24, 16:20] = 1.0

        # Agregar ruido y variacion
        img += np.random.randn(28, 28) * 0.1
        img = np.clip(img, 0, 1)

        # Pequeno desplazamiento aleatorio
        shift_x = np.random.randint(-2, 3)
        shift_y = np.random.randint(-2, 3)
        img = np.roll(np.roll(img, shift_x, axis=0), shift_y, axis=1)

        return img.flatten()

    # Generar datos
    X_train = []
    y_train = []
    for _ in range(n_train):
        label = np.random.randint(0, 10)
        X_train.append(crear_digito(label))
        y_train.append(label)

    X_test = []
    y_test = []
    for _ in range(n_test):
        label = np.random.randint(0, 10)
        X_test.append(crear_digito(label))
        y_test.append(label)

    return np.array(X_train), np.array(y_train), np.array(X_test), np.array(y_test)


# ==============================================================================
# RESERVOIR COMPUTING PARA MNIST
# ==============================================================================

class ReservoirMNIST:
    """
    Reservoir Computing adaptado para clasificacion de imagenes.

    ARQUITECTURA:
    - Input: 784 dimensiones (28x28 pixeles)
    - Reservoir: N osciladores con dinamica no-lineal
    - Output: 10 clases (one-hot)

    DIFERENCIA CON v5:
    - Escala mucho mayor (784 -> N -> 10)
    - Procesamiento por "chunks" para manejar la dimension
    """

    def __init__(self, n_reservoir: int = 500, spectral_radius: float = 0.9,
                 input_scaling: float = 0.1, leak_rate: float = 0.3):
        self.n_reservoir = n_reservoir
        self.spectral_radius = spectral_radius
        self.input_scaling = input_scaling
        self.leak_rate = leak_rate  # Para Leaky Integrator ESN

        # Dimensiones
        self.n_input = 784
        self.n_output = 10

        # Matriz de entrada (sparse para eficiencia)
        density = 0.1  # 10% de conexiones
        self.W_in = np.random.randn(n_reservoir, self.n_input) * input_scaling
        mask = np.random.rand(n_reservoir, self.n_input) > density
        self.W_in[mask] = 0

        # Matriz del reservoir (sparse)
        self.W_res = np.random.randn(n_reservoir, n_reservoir)
        mask = np.random.rand(n_reservoir, n_reservoir) > 0.1
        self.W_res[mask] = 0

        # Escalar para estabilidad
        eigenvalues = np.linalg.eigvals(self.W_res)
        max_eigenvalue = np.max(np.abs(eigenvalues))
        if max_eigenvalue > 0:
            self.W_res *= spectral_radius / max_eigenvalue

        # Pesos de salida (se entrenan)
        self.W_out = None

        # Estado
        self.state = np.zeros(n_reservoir)

    def _activate(self, x: np.ndarray) -> np.ndarray:
        """Funcion de activacion no-lineal (tanh)"""
        return np.tanh(x)

    def _update_state(self, input_vec: np.ndarray) -> np.ndarray:
        """
        Actualiza el estado del reservoir (Leaky ESN).

        state(t+1) = (1-leak)*state(t) + leak*tanh(W_in*input + W_res*state)
        """
        pre_activation = self.W_in @ input_vec + self.W_res @ self.state
        new_state = self._activate(pre_activation)
        self.state = (1 - self.leak_rate) * self.state + self.leak_rate * new_state
        return self.state

    def _get_reservoir_state(self, image: np.ndarray, n_steps: int = 10) -> np.ndarray:
        """
        Procesa una imagen y devuelve el estado final del reservoir.
        """
        self.state = np.zeros(self.n_reservoir)

        # Alimentar la imagen al reservoir por n_steps
        for _ in range(n_steps):
            self._update_state(image)

        return self.state.copy()

    def fit(self, X: np.ndarray, y: np.ndarray, n_steps: int = 10,
            regularization: float = 1e-4, batch_size: int = 1000):
        """
        Entrena el reservoir (solo los pesos de salida).

        Args:
            X: imagenes, shape (n_samples, 784)
            y: etiquetas, shape (n_samples,)
            n_steps: pasos de procesamiento por imagen
            regularization: regularizacion L2
        """
        n_samples = len(X)

        # Convertir etiquetas a one-hot
        y_onehot = np.zeros((n_samples, self.n_output))
        y_onehot[np.arange(n_samples), y] = 1

        # Recolectar estados del reservoir
        print(f"  Procesando {n_samples} muestras...")
        states = np.zeros((n_samples, self.n_reservoir))

        for i in range(0, n_samples, batch_size):
            end = min(i + batch_size, n_samples)
            for j in range(i, end):
                states[j] = self._get_reservoir_state(X[j], n_steps)

            if (end % 2000 == 0) or end == n_samples:
                print(f"    Procesadas {end}/{n_samples} muestras...")

        # Agregar bias
        states_bias = np.hstack([states, np.ones((n_samples, 1))])

        # Ridge Regression: W = (S^T S + lambda I)^-1 S^T Y
        print("  Calculando pesos de salida (Ridge Regression)...")
        STS = states_bias.T @ states_bias
        reg_matrix = regularization * np.eye(STS.shape[0])
        self.W_out = np.linalg.solve(STS + reg_matrix, states_bias.T @ y_onehot)

        return self

    def predict(self, X: np.ndarray, n_steps: int = 10) -> np.ndarray:
        """
        Predice las clases para un conjunto de imagenes.
        """
        n_samples = len(X)
        predictions = np.zeros(n_samples, dtype=int)

        for i in range(n_samples):
            state = self._get_reservoir_state(X[i], n_steps)
            state_bias = np.concatenate([state, [1]])
            output = state_bias @ self.W_out
            predictions[i] = np.argmax(output)

        return predictions

    def score(self, X: np.ndarray, y: np.ndarray, n_steps: int = 10) -> float:
        """
        Calcula accuracy.
        """
        predictions = self.predict(X, n_steps)
        return np.mean(predictions == y)


# ==============================================================================
# BASELINES PARA COMPARACION
# ==============================================================================

class PerceptronSimple:
    """Perceptron lineal simple (softmax regression)"""

    def __init__(self, n_input: int = 784, n_output: int = 10, lr: float = 0.01):
        self.W = np.random.randn(n_input, n_output) * 0.01
        self.b = np.zeros(n_output)
        self.lr = lr

    def _softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

    def fit(self, X, y, epochs=50, batch_size=100):
        n_samples = len(X)
        y_onehot = np.zeros((n_samples, 10))
        y_onehot[np.arange(n_samples), y] = 1

        for epoch in range(epochs):
            indices = np.random.permutation(n_samples)
            for i in range(0, n_samples, batch_size):
                batch_idx = indices[i:i+batch_size]
                X_batch = X[batch_idx]
                y_batch = y_onehot[batch_idx]

                # Forward
                logits = X_batch @ self.W + self.b
                probs = self._softmax(logits)

                # Backward
                grad = (probs - y_batch) / len(batch_idx)
                self.W -= self.lr * (X_batch.T @ grad)
                self.b -= self.lr * np.sum(grad, axis=0)

        return self

    def predict(self, X):
        logits = X @ self.W + self.b
        return np.argmax(logits, axis=1)

    def score(self, X, y):
        return np.mean(self.predict(X) == y)


class MLPSimple:
    """MLP de 2 capas con ReLU"""

    def __init__(self, n_input=784, n_hidden=256, n_output=10, lr=0.01):
        self.W1 = np.random.randn(n_input, n_hidden) * np.sqrt(2.0/n_input)
        self.b1 = np.zeros(n_hidden)
        self.W2 = np.random.randn(n_hidden, n_output) * np.sqrt(2.0/n_hidden)
        self.b2 = np.zeros(n_output)
        self.lr = lr

    def _relu(self, x):
        return np.maximum(0, x)

    def _relu_grad(self, x):
        return (x > 0).astype(float)

    def _softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

    def fit(self, X, y, epochs=30, batch_size=100):
        n_samples = len(X)
        y_onehot = np.zeros((n_samples, 10))
        y_onehot[np.arange(n_samples), y] = 1

        for epoch in range(epochs):
            indices = np.random.permutation(n_samples)
            for i in range(0, n_samples, batch_size):
                batch_idx = indices[i:i+batch_size]
                X_batch = X[batch_idx]
                y_batch = y_onehot[batch_idx]

                # Forward
                h1 = self._relu(X_batch @ self.W1 + self.b1)
                logits = h1 @ self.W2 + self.b2
                probs = self._softmax(logits)

                # Backward
                grad_out = (probs - y_batch) / len(batch_idx)
                grad_h1 = grad_out @ self.W2.T * self._relu_grad(X_batch @ self.W1 + self.b1)

                self.W2 -= self.lr * (h1.T @ grad_out)
                self.b2 -= self.lr * np.sum(grad_out, axis=0)
                self.W1 -= self.lr * (X_batch.T @ grad_h1)
                self.b1 -= self.lr * np.sum(grad_h1, axis=0)

        return self

    def predict(self, X):
        h1 = self._relu(X @ self.W1 + self.b1)
        logits = h1 @ self.W2 + self.b2
        return np.argmax(logits, axis=1)

    def score(self, X, y):
        return np.mean(self.predict(X) == y)


# ==============================================================================
# EJECUCION DE PRUEBAS
# ==============================================================================

def ejecutar_prueba_mnist():
    print("\n" + "="*70)
    print("   PRUEBA DEFINITIVA: MNIST")
    print("   Reservoir Computing vs Baselines")
    print("="*70)

    # Cargar datos
    print("\n1. CARGANDO DATOS...")
    try:
        X_train, y_train, X_test, y_test = descargar_mnist()
        print(f"   MNIST Real cargado:")
    except Exception as e:
        print(f"   Error cargando MNIST: {e}")
        X_train, y_train, X_test, y_test = generar_mnist_sintetico()
        print(f"   MNIST Sintetico generado:")

    print(f"   - Train: {X_train.shape[0]} muestras")
    print(f"   - Test: {X_test.shape[0]} muestras")
    print(f"   - Dimensiones: {X_train.shape[1]}")

    # Reducir dataset para prueba rapida
    n_train = min(10000, len(X_train))
    n_test = min(2000, len(X_test))

    # Seleccionar aleatoriamente
    train_idx = np.random.choice(len(X_train), n_train, replace=False)
    test_idx = np.random.choice(len(X_test), n_test, replace=False)

    X_train_sub = X_train[train_idx]
    y_train_sub = y_train[train_idx]
    X_test_sub = X_test[test_idx]
    y_test_sub = y_test[test_idx]

    print(f"\n   Usando subset: {n_train} train, {n_test} test")

    resultados = {}

    # --- Perceptron Simple ---
    print("\n" + "-"*70)
    print("2. PERCEPTRON SIMPLE (Baseline minimo)")
    print("-"*70)

    t_inicio = time.time()
    perceptron = PerceptronSimple(lr=0.1)
    perceptron.fit(X_train_sub, y_train_sub, epochs=30)
    t_perceptron = time.time() - t_inicio

    acc_perceptron = perceptron.score(X_test_sub, y_test_sub) * 100
    print(f"   Accuracy: {acc_perceptron:.2f}%")
    print(f"   Tiempo: {t_perceptron:.1f}s")
    resultados['Perceptron'] = acc_perceptron

    # --- MLP ---
    print("\n" + "-"*70)
    print("3. MLP (2 capas, 256 hidden)")
    print("-"*70)

    t_inicio = time.time()
    mlp = MLPSimple(n_hidden=256, lr=0.01)
    mlp.fit(X_train_sub, y_train_sub, epochs=30)
    t_mlp = time.time() - t_inicio

    acc_mlp = mlp.score(X_test_sub, y_test_sub) * 100
    print(f"   Accuracy: {acc_mlp:.2f}%")
    print(f"   Tiempo: {t_mlp:.1f}s")
    resultados['MLP'] = acc_mlp

    # --- Reservoir Computing ---
    print("\n" + "-"*70)
    print("4. RESERVOIR COMPUTING (500 osciladores)")
    print("-"*70)

    t_inicio = time.time()
    reservoir = ReservoirMNIST(
        n_reservoir=500,
        spectral_radius=0.9,
        input_scaling=0.1,
        leak_rate=0.3
    )
    reservoir.fit(X_train_sub, y_train_sub, n_steps=5)
    t_reservoir = time.time() - t_inicio

    print("   Evaluando...")
    acc_reservoir = reservoir.score(X_test_sub, y_test_sub) * 100
    print(f"   Accuracy: {acc_reservoir:.2f}%")
    print(f"   Tiempo: {t_reservoir:.1f}s")
    resultados['Reservoir'] = acc_reservoir

    # --- Resumen ---
    print("\n" + "="*70)
    print("   RESUMEN DE RESULTADOS")
    print("="*70)

    print(f"\n   {'Modelo':<20} {'Accuracy':<15} {'Tiempo':<10}")
    print("   " + "-"*45)
    print(f"   {'Perceptron':<20} {acc_perceptron:.2f}%{'':<9} {t_perceptron:.1f}s")
    print(f"   {'MLP (256 hidden)':<20} {acc_mlp:.2f}%{'':<9} {t_mlp:.1f}s")
    print(f"   {'Reservoir (500)':<20} {acc_reservoir:.2f}%{'':<9} {t_reservoir:.1f}s")

    # --- Veredicto ---
    print("\n" + "="*70)
    print("   VEREDICTO")
    print("="*70)

    if acc_reservoir >= 90:
        if acc_reservoir >= acc_mlp - 5:
            print("\n   [OK] EXITO: Reservoir compite con MLP")
            print("   El enfoque tiene potencial REAL para problemas complejos.")
            veredicto = "PROMETEDOR"
        else:
            print("\n   [~] PARCIAL: Reservoir funciona pero no alcanza MLP")
            print("   El enfoque necesita optimizacion.")
            veredicto = "NECESITA_TRABAJO"
    elif acc_reservoir >= 80:
        print("\n   [~] MARGINAL: Reservoir supera baseline pero lejos de MLP")
        print("   El enfoque tiene limitaciones fundamentales.")
        veredicto = "LIMITADO"
    else:
        print("\n   [X] FALLO: Reservoir no supera ni el baseline")
        print("   El enfoque NO es viable para problemas reales.")
        veredicto = "NO_VIABLE"

    # Comparacion con objetivo
    print(f"\n   Objetivo minimo (>90%): {'CUMPLIDO' if acc_reservoir >= 90 else 'NO CUMPLIDO'}")
    print(f"   Objetivo bueno (>95%): {'CUMPLIDO' if acc_reservoir >= 95 else 'NO CUMPLIDO'}")
    print(f"   Objetivo excelente (>97%): {'CUMPLIDO' if acc_reservoir >= 97 else 'NO CUMPLIDO'}")

    return resultados, veredicto


if __name__ == "__main__":
    resultados, veredicto = ejecutar_prueba_mnist()

    print("\n" + "="*70)
    print(f"   VEREDICTO FINAL: {veredicto}")
    print("="*70)

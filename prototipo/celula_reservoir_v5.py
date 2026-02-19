# -*- coding: utf-8 -*-
"""
CELULA COGNITIVA v5: Reservoir Computing en el Borde del Caos
==============================================================

Este es un intento HONESTO de crear un sistema que:
1. APRENDA de datos (no hardcodee umbrales)
2. Resuelva XOR y OTRAS funciones logicas
3. Use principios fisicos reales (no if-else disfrazados)

ENFOQUE: Reservoir Computing
- Un "reservoir" es una red recurrente de dinamica no-lineal
- El reservoir transforma inputs de baja dimension a alta dimension
- Solo se entrena una capa de salida lineal (eficiente)
- La no-linealidad del reservoir permite resolver XOR

Este enfoque es usado en investigacion real (Echo State Networks, Liquid State Machines)
y tiene fundamento teorico solido.

Autor: Proyecto Kamaq
Fecha: 16 de Enero, 2026
"""

import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass, field


@dataclass
class Oscilador:
    """
    Un oscilador no-lineal simple.
    Basado en la ecuacion de Stuart-Landau (bifurcacion de Hopf supercritica).
    """
    # Estado complejo (amplitud + fase)
    z: complex = complex(0.01, 0.0)

    # Parametros
    mu: float = 0.1      # Control de bifurcacion (>0 = oscila, <0 = decae)
    omega: float = 1.0   # Frecuencia natural
    gamma: float = 0.1   # Acoplamiento no-lineal

    def step(self, dt: float, input_signal: complex = 0.0) -> complex:
        """
        Evolucion temporal: dz/dt = (mu + i*omega)*z - gamma*|z|^2*z + input
        """
        dzdt = (self.mu + 1j * self.omega) * self.z - self.gamma * abs(self.z)**2 * self.z
        dzdt += input_signal
        self.z += dzdt * dt
        return self.z

    @property
    def output(self) -> float:
        """Salida real del oscilador"""
        return self.z.real


class ReservoirCognitivo:
    """
    Reservoir Computing implementado con osciladores acoplados.

    ARQUITECTURA:
    - N osciladores con frecuencias diferentes (heterogeneidad)
    - Acoplamiento aleatorio entre osciladores (conectividad)
    - Entrada distribuida a multiples osciladores
    - Salida: combinacion lineal entrenable de estados

    PRINCIPIO:
    El reservoir proyecta los inputs a un espacio de alta dimension
    donde problemas no-lineales (como XOR) se vuelven linealmente separables.
    """

    def __init__(self, n_osciladores: int = 50, spectral_radius: float = 0.9):
        self.n = n_osciladores
        self.spectral_radius = spectral_radius

        # Crear osciladores con frecuencias heterogeneas
        self.osciladores = []
        for i in range(n_osciladores):
            osc = Oscilador(
                z=complex(np.random.randn() * 0.01, np.random.randn() * 0.01),
                mu=0.1,
                omega=1.0 + 0.5 * np.random.randn(),  # Frecuencias variadas
                gamma=0.1
            )
            self.osciladores.append(osc)

        # Matriz de acoplamiento interno (sparse, random)
        self.W_reservoir = np.random.randn(n_osciladores, n_osciladores) * 0.1
        # Escalar para estabilidad (spectral radius < 1)
        eigenvalues = np.linalg.eigvals(self.W_reservoir)
        self.W_reservoir *= spectral_radius / np.max(np.abs(eigenvalues))

        # Matriz de entrada (random)
        self.W_input = np.random.randn(n_osciladores, 2) * 0.5  # 2 inputs

        # Pesos de salida (ESTO es lo que se entrena)
        self.W_output = np.zeros(n_osciladores + 1)  # +1 para bias

        # Estado del reservoir
        self.state = np.zeros(n_osciladores, dtype=complex)

    def reset(self):
        """Reiniciar estado del reservoir"""
        for osc in self.osciladores:
            osc.z = complex(np.random.randn() * 0.01, np.random.randn() * 0.01)
        self.state = np.array([osc.z for osc in self.osciladores])

    def step(self, inputs: np.ndarray, dt: float = 0.1) -> np.ndarray:
        """
        Un paso de evolucion del reservoir.

        Args:
            inputs: array de shape (2,) con los inputs
            dt: paso de tiempo

        Returns:
            Estado actual del reservoir (partes reales)
        """
        # Calcular influencia de entrada a cada oscilador
        input_signals = self.W_input @ inputs

        # Calcular acoplamiento interno
        state_real = np.array([osc.z.real for osc in self.osciladores])
        coupling = self.W_reservoir @ state_real

        # Evolucionar cada oscilador
        for i, osc in enumerate(self.osciladores):
            total_input = complex(input_signals[i] + coupling[i], 0)
            osc.step(dt, total_input)

        # Actualizar estado
        self.state = np.array([osc.z for osc in self.osciladores])

        return np.array([osc.output for osc in self.osciladores])

    def run(self, inputs: np.ndarray, n_steps: int = 50) -> np.ndarray:
        """
        Ejecutar el reservoir por n_steps y devolver estado final.
        """
        self.reset()
        for _ in range(n_steps):
            state = self.step(inputs)
        return state

    def predict(self, inputs: np.ndarray, n_steps: int = 50) -> float:
        """
        Hacer prediccion para un input.
        """
        state = self.run(inputs, n_steps)
        # Agregar bias
        state_with_bias = np.concatenate([state, [1.0]])
        # Combinacion lineal
        output = np.dot(self.W_output, state_with_bias)
        return output

    def train(self, X: np.ndarray, y: np.ndarray, n_steps: int = 50, regularization: float = 1e-6):
        """
        Entrenar los pesos de salida usando Ridge Regression.

        IMPORTANTE: Solo se entrenan los pesos de salida (W_output).
        El reservoir interno NO se modifica.
        Esto es lo que hace eficiente al Reservoir Computing.

        Args:
            X: datos de entrada, shape (n_samples, 2)
            y: etiquetas, shape (n_samples,)
            n_steps: pasos de simulacion por muestra
            regularization: regularizacion L2
        """
        n_samples = len(X)

        # Recolectar estados del reservoir para cada muestra
        states = []
        for i in range(n_samples):
            state = self.run(X[i], n_steps)
            states.append(np.concatenate([state, [1.0]]))  # Con bias

        states = np.array(states)  # Shape: (n_samples, n_osciladores + 1)

        # Ridge Regression: W = (S^T S + lambda I)^-1 S^T y
        STS = states.T @ states
        reg_matrix = regularization * np.eye(STS.shape[0])
        self.W_output = np.linalg.solve(STS + reg_matrix, states.T @ y)

        return self


def prueba_xor():
    """
    Prueba: Resolver XOR con Reservoir Computing.

    XOR es el benchmark minimo para demostrar capacidad no-lineal.
    Un perceptron simple NO puede resolverlo.
    """
    print("\n" + "="*60)
    print("PRUEBA: XOR con Reservoir Computing")
    print("="*60)

    # Datos XOR
    X = np.array([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1]
    ], dtype=float)

    y = np.array([0, 1, 1, 0], dtype=float)

    print("Datos de entrenamiento:")
    for i in range(len(X)):
        print(f"  {X[i]} -> {y[i]}")

    # Crear y entrenar reservoir
    print("\nCreando reservoir (50 osciladores)...")
    reservoir = ReservoirCognitivo(n_osciladores=50, spectral_radius=0.9)

    print("Entrenando (solo pesos de salida)...")
    reservoir.train(X, y, n_steps=30)

    # Evaluar
    print("\nEvaluacion:")
    correctos = 0
    for i in range(len(X)):
        pred = reservoir.predict(X[i], n_steps=30)
        pred_bit = 1 if pred > 0.5 else 0
        esperado = int(y[i])
        ok = pred_bit == esperado
        if ok:
            correctos += 1
        print(f"  {X[i]} -> pred={pred:.3f} (bit={pred_bit}) esperado={esperado} {'[OK]' if ok else '[X]'}")

    accuracy = correctos / len(X) * 100
    exito = accuracy == 100

    print(f"\nAccuracy: {accuracy:.0f}%")
    print(f"Resultado: {'[OK] EXITO' if exito else '[X] FALLO'}")

    return exito, accuracy


def prueba_and():
    """Prueba: AND"""
    print("\n" + "="*60)
    print("PRUEBA: AND con Reservoir Computing")
    print("="*60)

    X = np.array([[0,0], [0,1], [1,0], [1,1]], dtype=float)
    y = np.array([0, 0, 0, 1], dtype=float)

    reservoir = ReservoirCognitivo(n_osciladores=50)
    reservoir.train(X, y, n_steps=30)

    correctos = 0
    for i in range(len(X)):
        pred = reservoir.predict(X[i], n_steps=30)
        pred_bit = 1 if pred > 0.5 else 0
        if pred_bit == int(y[i]):
            correctos += 1
        print(f"  {X[i]} -> {pred:.3f} (bit={pred_bit}) esperado={int(y[i])}")

    accuracy = correctos / len(X) * 100
    print(f"Accuracy: {accuracy:.0f}%")
    return accuracy == 100, accuracy


def prueba_or():
    """Prueba: OR"""
    print("\n" + "="*60)
    print("PRUEBA: OR con Reservoir Computing")
    print("="*60)

    X = np.array([[0,0], [0,1], [1,0], [1,1]], dtype=float)
    y = np.array([0, 1, 1, 1], dtype=float)

    reservoir = ReservoirCognitivo(n_osciladores=50)
    reservoir.train(X, y, n_steps=30)

    correctos = 0
    for i in range(len(X)):
        pred = reservoir.predict(X[i], n_steps=30)
        pred_bit = 1 if pred > 0.5 else 0
        if pred_bit == int(y[i]):
            correctos += 1
        print(f"  {X[i]} -> {pred:.3f} (bit={pred_bit}) esperado={int(y[i])}")

    accuracy = correctos / len(X) * 100
    print(f"Accuracy: {accuracy:.0f}%")
    return accuracy == 100, accuracy


def prueba_nand():
    """Prueba: NAND (la base de toda computacion digital)"""
    print("\n" + "="*60)
    print("PRUEBA: NAND con Reservoir Computing")
    print("="*60)

    X = np.array([[0,0], [0,1], [1,0], [1,1]], dtype=float)
    y = np.array([1, 1, 1, 0], dtype=float)

    reservoir = ReservoirCognitivo(n_osciladores=50)
    reservoir.train(X, y, n_steps=30)

    correctos = 0
    for i in range(len(X)):
        pred = reservoir.predict(X[i], n_steps=30)
        pred_bit = 1 if pred > 0.5 else 0
        if pred_bit == int(y[i]):
            correctos += 1
        print(f"  {X[i]} -> {pred:.3f} (bit={pred_bit}) esperado={int(y[i])}")

    accuracy = correctos / len(X) * 100
    print(f"Accuracy: {accuracy:.0f}%")
    return accuracy == 100, accuracy


def prueba_generalizacion():
    """
    Prueba CRITICA: Generalizacion a inputs no vistos.

    Entrenamos con 4 puntos, probamos con inputs continuos.
    Si el sistema generaliza, deberia interpolar correctamente.
    """
    print("\n" + "="*60)
    print("PRUEBA: Generalizacion (inputs continuos)")
    print("="*60)

    # Entrenar con XOR discreto
    X_train = np.array([[0,0], [0,1], [1,0], [1,1]], dtype=float)
    y_train = np.array([0, 1, 1, 0], dtype=float)

    reservoir = ReservoirCognitivo(n_osciladores=100)  # Mas capacidad
    reservoir.train(X_train, y_train, n_steps=50)

    # Probar con inputs continuos
    print("\nPredicciones para inputs continuos:")

    test_cases = [
        ([0.1, 0.1], "~(0,0) -> esperado ~0"),
        ([0.9, 0.1], "~(1,0) -> esperado ~1"),
        ([0.1, 0.9], "~(0,1) -> esperado ~1"),
        ([0.9, 0.9], "~(1,1) -> esperado ~0"),
        ([0.5, 0.5], "ambiguo -> ?"),
        ([0.3, 0.7], "intermedio -> ?"),
    ]

    for inputs, descripcion in test_cases:
        pred = reservoir.predict(np.array(inputs), n_steps=50)
        print(f"  {inputs} -> {pred:.3f} ({descripcion})")

    # Evaluar si las esquinas cercanas dan resultados correctos
    corners = [
        ([0.1, 0.1], 0),
        ([0.9, 0.1], 1),
        ([0.1, 0.9], 1),
        ([0.9, 0.9], 0),
    ]

    correctos = 0
    for inputs, esperado in corners:
        pred = reservoir.predict(np.array(inputs), n_steps=50)
        pred_bit = 1 if pred > 0.5 else 0
        if pred_bit == esperado:
            correctos += 1

    accuracy = correctos / len(corners) * 100
    exito = accuracy >= 75  # Al menos 3/4

    print(f"\nAccuracy en esquinas cercanas: {accuracy:.0f}%")
    print(f"Resultado: {'[OK] GENERALIZA' if exito else '[X] NO GENERALIZA'}")

    return exito, accuracy


def prueba_comparacion_perceptron():
    """
    Comparacion: Reservoir vs Perceptron simple en XOR.

    El perceptron DEBE fallar (XOR no es linealmente separable).
    El reservoir DEBE funcionar.
    """
    print("\n" + "="*60)
    print("PRUEBA: Comparacion Reservoir vs Perceptron")
    print("="*60)

    X = np.array([[0,0], [0,1], [1,0], [1,1]], dtype=float)
    y = np.array([0, 1, 1, 0], dtype=float)

    # --- Perceptron simple ---
    print("\n1. Perceptron Simple (debe fallar en XOR):")

    # Entrenar perceptron: y = sigmoid(w1*x1 + w2*x2 + b)
    w = np.random.randn(2) * 0.1
    b = 0.0
    lr = 0.5

    for epoch in range(1000):
        for i in range(len(X)):
            z = np.dot(w, X[i]) + b
            pred = 1 / (1 + np.exp(-z))
            error = y[i] - pred
            grad = pred * (1 - pred)
            w += lr * error * grad * X[i]
            b += lr * error * grad

    correctos_perceptron = 0
    for i in range(len(X)):
        z = np.dot(w, X[i]) + b
        pred = 1 / (1 + np.exp(-z))
        pred_bit = 1 if pred > 0.5 else 0
        if pred_bit == int(y[i]):
            correctos_perceptron += 1
        print(f"  {X[i]} -> {pred:.3f} (bit={pred_bit}) esperado={int(y[i])}")

    acc_perceptron = correctos_perceptron / len(X) * 100
    print(f"  Accuracy Perceptron: {acc_perceptron:.0f}%")

    # --- Reservoir ---
    print("\n2. Reservoir Computing (debe funcionar en XOR):")

    reservoir = ReservoirCognitivo(n_osciladores=50)
    reservoir.train(X, y, n_steps=30)

    correctos_reservoir = 0
    for i in range(len(X)):
        pred = reservoir.predict(X[i], n_steps=30)
        pred_bit = 1 if pred > 0.5 else 0
        if pred_bit == int(y[i]):
            correctos_reservoir += 1
        print(f"  {X[i]} -> {pred:.3f} (bit={pred_bit}) esperado={int(y[i])}")

    acc_reservoir = correctos_reservoir / len(X) * 100
    print(f"  Accuracy Reservoir: {acc_reservoir:.0f}%")

    # --- Comparacion ---
    print("\n3. Comparacion:")
    print(f"  Perceptron: {acc_perceptron:.0f}% (esperado: <100% en XOR)")
    print(f"  Reservoir:  {acc_reservoir:.0f}% (esperado: 100%)")

    exito = (acc_reservoir > acc_perceptron) and (acc_reservoir == 100)
    print(f"\nResultado: {'[OK] RESERVOIR SUPERIOR' if exito else '[X] FALLO'}")

    return exito, acc_reservoir


def ejecutar_todas():
    """Ejecutar suite completa de pruebas"""

    print("\n" + "="*60)
    print("   CELULA COGNITIVA v5: RESERVOIR COMPUTING")
    print("   Implementacion Honesta con Aprendizaje Real")
    print("="*60)

    resultados = {}

    ok, acc = prueba_xor()
    resultados['XOR'] = (ok, acc)

    ok, acc = prueba_and()
    resultados['AND'] = (ok, acc)

    ok, acc = prueba_or()
    resultados['OR'] = (ok, acc)

    ok, acc = prueba_nand()
    resultados['NAND'] = (ok, acc)

    ok, acc = prueba_generalizacion()
    resultados['Generalizacion'] = (ok, acc)

    ok, acc = prueba_comparacion_perceptron()
    resultados['vs_Perceptron'] = (ok, acc)

    # Resumen
    print("\n" + "="*60)
    print("   RESUMEN DE RESULTADOS")
    print("="*60)

    for nombre, (exito, acc) in resultados.items():
        print(f"  [{'OK' if exito else 'X'}] {nombre}: {acc:.0f}%")

    total_exitos = sum(1 for ok, _ in resultados.values() if ok)
    total = len(resultados)

    print(f"\nTotal: {total_exitos}/{total} pruebas exitosas")

    if total_exitos == total:
        print("\n" + "*"*60)
        print("   VALIDACION COMPLETA")
        print("*"*60)
        print("\nEl sistema demuestra:")
        print("  - Aprendizaje REAL (no hardcodeado)")
        print("  - Resolucion de XOR y otras funciones")
        print("  - Generalizacion a inputs no vistos")
        print("  - Superioridad sobre perceptron simple")
        print("\n>>> ENFOQUE VIABLE <<<")
        return True
    elif total_exitos >= 4:
        print("\n>>> ENFOQUE PROMETEDOR <<<")
        return True
    else:
        print("\n>>> ENFOQUE INSUFICIENTE <<<")
        return False


if __name__ == "__main__":
    ejecutar_todas()

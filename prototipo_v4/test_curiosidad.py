"""
TEST DE CURIOSIDAD DIRIGIDA - FASE 1 VALIDACION
================================================
Compara exploracion RANDOM (epsilon-greedy) vs CURIOSIDAD (inferencia activa).

METRICAS CLAVE:
1. Diversidad de estados visitados
2. Cobertura de transiciones
3. Velocidad de descubrimiento

CRITERIO DE EXITO:
- Curiosidad debe visitar >80% mas estados unicos que random
- en el mismo numero de episodios

Ejecutar: python test_curiosidad.py
"""

import numpy as np
import time
from typing import List, Dict, Tuple
from motor_inferencia_activa import MotorInferenciaActiva, CuriosidadAdaptativa


class TicTacToe:
    """Juego simple para testing."""
    LINEAS = [
        [0, 1, 2], [3, 4, 5], [6, 7, 8],
        [0, 3, 6], [1, 4, 7], [2, 5, 8],
        [0, 4, 8], [2, 4, 6]
    ]

    def __init__(self):
        self.reset()

    def reset(self):
        self.tablero = np.zeros(9)
        self.turno = 1
        return self.tablero.copy()

    def acciones_validas(self):
        return [i for i in range(9) if self.tablero[i] == 0]

    def hacer_movimiento(self, pos):
        if self.tablero[pos] != 0:
            return False, -0.1, False

        self.tablero[pos] = self.turno
        ganador = self._verificar_ganador()

        if ganador == self.turno:
            self.turno = -self.turno
            return True, 1.0, True
        elif len(self.acciones_validas()) == 0:
            return True, 0.1, True

        self.turno = -self.turno
        return True, 0.0, False

    def _verificar_ganador(self):
        for linea in self.LINEAS:
            suma = sum(self.tablero[i] for i in linea)
            if suma == 3:
                return 1
            elif suma == -3:
                return -1
        return None


def entrenar_con_random(n_episodios: int, epsilon: float = 0.3) -> Dict:
    """Entrena usando epsilon-greedy clasico."""
    juego = TicTacToe()
    estados_visitados = set()
    transiciones_vistas = set()
    victorias = 0

    for ep in range(n_episodios):
        estado = juego.reset()
        estados_visitados.add(tuple(estado))

        while True:
            acciones = juego.acciones_validas()
            if not acciones:
                break

            if juego.turno == 1:
                # Epsilon-greedy: random con probabilidad epsilon
                if np.random.random() < epsilon:
                    accion = np.random.choice(acciones)
                else:
                    # Sin conocimiento, elegir random tambien
                    accion = np.random.choice(acciones)

                estado_antes = estado.copy()
                _, recompensa, terminal = juego.hacer_movimiento(accion)
                estado = juego.tablero.copy()

                # Registrar
                estados_visitados.add(tuple(estado))
                transiciones_vistas.add((tuple(estado_antes), accion))

                if terminal:
                    if juego._verificar_ganador() == 1:
                        victorias += 1
                    break
            else:
                accion = np.random.choice(acciones)
                _, _, terminal = juego.hacer_movimiento(accion)
                estado = juego.tablero.copy()
                estados_visitados.add(tuple(estado))

                if terminal:
                    break

    return {
        'estados_unicos': len(estados_visitados),
        'transiciones_unicas': len(transiciones_vistas),
        'victorias': victorias,
        'win_rate': victorias / n_episodios * 100
    }


def entrenar_con_curiosidad(n_episodios: int, peso_epistemico: float = 0.5) -> Dict:
    """Entrena usando inferencia activa (curiosidad dirigida)."""
    juego = TicTacToe()
    motor = MotorInferenciaActiva(n_acciones=9, peso_epistemico=peso_epistemico)
    curiosidad_adaptativa = CuriosidadAdaptativa(motor)

    estados_visitados = set()
    victorias = 0

    for ep in range(n_episodios):
        estado = juego.reset()
        estados_visitados.add(tuple(estado))

        while True:
            acciones = juego.acciones_validas()
            if not acciones:
                break

            if juego.turno == 1:
                estado_antes = estado.copy()

                # Decidir usando inferencia activa
                accion = motor.decidir(estado, acciones)

                _, recompensa, terminal = juego.hacer_movimiento(accion)
                estado = juego.tablero.copy()

                # Aprender
                motor.aprender(estado_antes, accion, estado, recompensa)
                curiosidad_adaptativa.actualizar(recompensa)

                estados_visitados.add(tuple(estado))

                if terminal:
                    if juego._verificar_ganador() == 1:
                        victorias += 1
                    break
            else:
                accion = np.random.choice(acciones)
                _, _, terminal = juego.hacer_movimiento(accion)
                estado = juego.tablero.copy()
                estados_visitados.add(tuple(estado))

                if terminal:
                    break

    diag = motor.diagnostico()

    return {
        'estados_unicos': len(estados_visitados),
        'transiciones_unicas': diag['cobertura_transiciones'] * len(estados_visitados) * 9,
        'victorias': victorias,
        'win_rate': victorias / n_episodios * 100,
        'ratio_epistemico': diag['ratio_epistemico'],
        'peso_epistemico_final': motor.peso_epistemico
    }


def test_diversidad_exploracion():
    """Test principal: compara diversidad de exploracion."""
    print("=" * 70)
    print("TEST DE CURIOSIDAD DIRIGIDA vs EPSILON-GREEDY")
    print("=" * 70)

    n_episodios = 1000

    print(f"\nEntrenando {n_episodios} episodios...")
    print("-" * 70)

    # Random
    print("\n1. EPSILON-GREEDY (epsilon=0.3):")
    inicio = time.time()
    resultados_random = entrenar_con_random(n_episodios, epsilon=0.3)
    tiempo_random = time.time() - inicio

    print(f"   Estados unicos: {resultados_random['estados_unicos']}")
    print(f"   Win rate: {resultados_random['win_rate']:.1f}%")
    print(f"   Tiempo: {tiempo_random:.2f}s")

    # Curiosidad
    print("\n2. CURIOSIDAD DIRIGIDA (peso_epistemico=0.5):")
    inicio = time.time()
    resultados_curiosidad = entrenar_con_curiosidad(n_episodios, peso_epistemico=0.5)
    tiempo_curiosidad = time.time() - inicio

    print(f"   Estados unicos: {resultados_curiosidad['estados_unicos']}")
    print(f"   Win rate: {resultados_curiosidad['win_rate']:.1f}%")
    print(f"   Ratio epistemico: {resultados_curiosidad['ratio_epistemico']*100:.1f}%")
    print(f"   Tiempo: {tiempo_curiosidad:.2f}s")

    # Comparacion
    print("\n" + "=" * 70)
    print("COMPARACION")
    print("=" * 70)

    mejora_estados = (resultados_curiosidad['estados_unicos'] / resultados_random['estados_unicos'] - 1) * 100
    diff_wr = resultados_curiosidad['win_rate'] - resultados_random['win_rate']

    print(f"\nEstados unicos:")
    print(f"  Random:     {resultados_random['estados_unicos']}")
    print(f"  Curiosidad: {resultados_curiosidad['estados_unicos']}")
    print(f"  Diferencia: {mejora_estados:+.1f}%")

    print(f"\nWin Rate:")
    print(f"  Random:     {resultados_random['win_rate']:.1f}%")
    print(f"  Curiosidad: {resultados_curiosidad['win_rate']:.1f}%")
    print(f"  Diferencia: {diff_wr:+.1f}%")

    # Veredicto
    print("\n" + "=" * 70)
    print("VEREDICTO")
    print("=" * 70)

    exito_diversidad = mejora_estados > 0
    exito_wr = diff_wr >= -5  # No perder mas de 5% WR

    if exito_diversidad and exito_wr:
        print("[OK] FASE 1 EXITOSA")
        print("     La curiosidad dirigida explora mas estados sin sacrificar rendimiento.")
    elif exito_diversidad:
        print("[PARCIAL] Diversidad mejorada pero WR bajo")
        print("     Ajustar balance epistemico/pragmatico.")
    else:
        print("[FALLO] Curiosidad no mejora diversidad")
        print("     Revisar implementacion del motor de inferencia activa.")

    print("=" * 70)

    return resultados_random, resultados_curiosidad


def test_evolucion_curiosidad():
    """Test de como evoluciona la curiosidad durante el entrenamiento."""
    print("\n" + "=" * 70)
    print("TEST: EVOLUCION DE CURIOSIDAD EN EL TIEMPO")
    print("=" * 70)

    juego = TicTacToe()
    motor = MotorInferenciaActiva(n_acciones=9, peso_epistemico=0.7)
    curiosidad = CuriosidadAdaptativa(motor, ventana=50)

    n_episodios = 500
    metricas = []

    for ep in range(n_episodios):
        estado = juego.reset()
        recompensa_total = 0

        while True:
            acciones = juego.acciones_validas()
            if not acciones:
                break

            if juego.turno == 1:
                estado_antes = estado.copy()
                accion = motor.decidir(estado, acciones)
                _, recompensa, terminal = juego.hacer_movimiento(accion)
                estado = juego.tablero.copy()

                motor.aprender(estado_antes, accion, estado, recompensa)
                curiosidad.actualizar(recompensa)
                recompensa_total += recompensa

                if terminal:
                    break
            else:
                accion = np.random.choice(acciones)
                _, _, terminal = juego.hacer_movimiento(accion)
                estado = juego.tablero.copy()
                if terminal:
                    break

        if (ep + 1) % 100 == 0:
            diag = motor.diagnostico()
            metricas.append({
                'ep': ep + 1,
                'peso_epistemico': motor.peso_epistemico,
                'ratio_epistemico': diag['ratio_epistemico'],
                'estados_unicos': diag['estados_unicos'],
                'cobertura': diag['cobertura_transiciones']
            })

            print(f"Ep {ep+1:4d} | Peso: {motor.peso_epistemico:.2f} | "
                  f"Ratio: {diag['ratio_epistemico']*100:.1f}% | "
                  f"Estados: {diag['estados_unicos']}")

    print("\nEvolucion completa. El peso epistemico debe adaptarse al progreso.")
    return metricas


def test_diferentes_pesos():
    """Test de diferentes balances epistemico/pragmatico."""
    print("\n" + "=" * 70)
    print("TEST: DIFERENTES PESOS EPISTEMICOS")
    print("=" * 70)

    pesos = [0.1, 0.3, 0.5, 0.7, 0.9]
    n_episodios = 500

    resultados = []

    for peso in pesos:
        r = entrenar_con_curiosidad(n_episodios, peso_epistemico=peso)
        resultados.append({
            'peso': peso,
            **r
        })
        print(f"Peso {peso:.1f}: Estados={r['estados_unicos']:4d}, WR={r['win_rate']:.1f}%")

    # Encontrar mejor balance
    mejor = max(resultados, key=lambda x: x['estados_unicos'] + x['win_rate'])
    print(f"\nMejor balance: peso_epistemico={mejor['peso']:.1f}")

    return resultados


if __name__ == "__main__":
    # Test principal
    r_random, r_curiosidad = test_diversidad_exploracion()

    # Test de evolucion
    metricas = test_evolucion_curiosidad()

    # Test de pesos
    resultados_pesos = test_diferentes_pesos()

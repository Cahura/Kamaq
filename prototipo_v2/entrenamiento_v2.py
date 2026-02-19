"""
ENTRENAMIENTO V2 - CON CAMPO COGNITIVO MEJORADO
===============================================
Usa CampoCognitivoV2 con:
- Separación de J positiva/negativa
- Eligibility traces
- Epsilon decay
- Sin curriculum agresivo

Este es un intento honesto de hacer que emerja conocimiento.
"""

import numpy as np
import time
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from campo_cognitivo_v2 import AgenteEmergentev2


@dataclass
class Experiencia:
    estado: np.ndarray
    accion: int
    recompensa: float
    estado_siguiente: np.ndarray
    terminal: bool


class TicTacToe:
    LINEAS = [
        [0, 1, 2], [3, 4, 5], [6, 7, 8],
        [0, 3, 6], [1, 4, 7], [2, 5, 8],
        [0, 4, 8], [2, 4, 6]
    ]

    def __init__(self):
        self.reset()

    def reset(self) -> np.ndarray:
        self.tablero = np.zeros(9)
        self.turno = 1
        return self.tablero.copy()

    def acciones_validas(self) -> List[int]:
        return [i for i in range(9) if self.tablero[i] == 0]

    def hacer_movimiento(self, pos: int) -> Tuple[bool, float, bool]:
        if self.tablero[pos] != 0:
            return False, -0.1, False

        self.tablero[pos] = self.turno
        ganador = self._verificar_ganador()

        if ganador == self.turno:
            self.turno = -self.turno
            return True, 1.0, True
        elif len(self.acciones_validas()) == 0:
            return True, 0.1, True  # Empate ligeramente positivo

        self.turno = -self.turno
        return True, 0.0, False

    def _verificar_ganador(self) -> Optional[int]:
        for linea in self.LINEAS:
            suma = sum(self.tablero[i] for i in linea)
            if suma == 3:
                return 1
            elif suma == -3:
                return -1
        return None

    def analizar_movimiento(self, pos: int, jugador: int) -> Dict:
        """Análisis post-hoc del movimiento."""
        tablero_antes = self.tablero.copy()
        tablero_antes[pos] = 0

        analisis = {'gano': False, 'bloqueo': False, 'amenaza': False}

        # ¿Ganó?
        for linea in self.LINEAS:
            if pos in linea:
                suma = sum(self.tablero[i] for i in linea)
                if suma == 3 * jugador:
                    analisis['gano'] = True

        # ¿Bloqueó?
        oponente = -jugador
        for linea in self.LINEAS:
            if pos in linea:
                fichas_op = sum(1 for i in linea if tablero_antes[i] == oponente)
                vacias = sum(1 for i in linea if tablero_antes[i] == 0)
                if fichas_op == 2 and vacias == 1:
                    analisis['bloqueo'] = True

        # ¿Creó amenaza?
        for linea in self.LINEAS:
            if pos in linea:
                fichas = sum(1 for i in linea if self.tablero[i] == jugador)
                vacias = sum(1 for i in linea if self.tablero[i] == 0)
                if fichas == 2 and vacias == 1:
                    analisis['amenaza'] = True

        return analisis


def entrenar_v2(n_episodios: int = 3000):
    """
    Entrenamiento con campo cognitivo V2.

    Cambios respecto a V1:
    1. Sin curriculum agresivo (solo random)
    2. Recompensas más informativas
    3. Más consolidación
    """
    print("=" * 70)
    print("ENTRENAMIENTO V2 - CAMPO COGNITIVO MEJORADO")
    print("=" * 70)

    agente = AgenteEmergentev2(dim_estado=9, n_acciones=9)
    juego = TicTacToe()

    victorias = 0
    derrotas = 0
    empates = 0

    historial_wr = []

    inicio = time.time()

    for ep in range(1, n_episodios + 1):
        estado = juego.reset()

        while True:
            acciones = juego.acciones_validas()
            if not acciones:
                break

            if juego.turno == 1:  # Agente
                estado_antes = estado.copy()
                accion = agente.decidir(estado, acciones)

                valido, recompensa_base, terminal = juego.hacer_movimiento(accion)
                estado = juego.tablero.copy()

                # Análisis para recompensa informativa
                analisis = juego.analizar_movimiento(accion, 1)

                recompensa = recompensa_base
                if analisis['gano']:
                    recompensa += 0.5
                if analisis['bloqueo']:
                    recompensa += 0.3
                if analisis['amenaza']:
                    recompensa += 0.1

                exp = Experiencia(
                    estado=estado_antes,
                    accion=accion,
                    recompensa=recompensa,
                    estado_siguiente=estado,
                    terminal=terminal
                )
                agente.aprender(exp)

                if terminal:
                    if juego._verificar_ganador() == 1:
                        victorias += 1
                    else:
                        empates += 1
                    break

            else:  # Oponente random
                accion = np.random.choice(acciones)
                _, _, terminal = juego.hacer_movimiento(accion)
                estado = juego.tablero.copy()

                if terminal:
                    if juego._verificar_ganador() == -1:
                        derrotas += 1
                        # Feedback negativo
                        if agente.experiencias:
                            exp = agente.experiencias[-1]
                            exp_derrota = Experiencia(
                                estado=exp.estado_siguiente,
                                accion=exp.accion,
                                recompensa=-0.8,
                                estado_siguiente=estado,
                                terminal=True
                            )
                            agente.aprender(exp_derrota)
                    else:
                        empates += 1
                    break

        # Consolidar
        if ep % 30 == 0:
            agente.consolidar()

        # Métricas
        if ep % 100 == 0:
            wr = victorias / ep * 100
            historial_wr.append(wr)

            # Medir conocimiento emergente
            pref_centro = medir_preferencia_centro(agente)
            tasa_bloqueo = medir_tasa_bloqueo(agente)
            tasa_victoria = medir_tasa_victoria(agente)

            print(f"Ep {ep:4d} | WR: {wr:5.1f}% | eps: {agente.epsilon:.3f} | "
                  f"Centro: {pref_centro*100:4.1f}% | Bloq: {tasa_bloqueo*100:4.1f}% | "
                  f"Vic: {tasa_victoria*100:4.1f}%")

    tiempo = time.time() - inicio

    print("=" * 70)
    print(f"Completado en {tiempo:.1f}s")
    print(f"Final: {victorias}V / {derrotas}D / {empates}E")
    print("=" * 70)

    # Evaluación final
    print("\nEVALUACIÓN FINAL (200 partidas vs random):")
    v, d, e = evaluar(agente, 200)
    print(f"  Victorias: {v}")
    print(f"  Derrotas: {d}")
    print(f"  Empates: {e}")
    print(f"  Win Rate: {v/2:.1f}%")

    print("\nEMERGENCIA DE CONOCIMIENTO:")
    pref_centro = medir_preferencia_centro(agente, 100)
    tasa_bloqueo = medir_tasa_bloqueo(agente)
    tasa_victoria = medir_tasa_victoria(agente)
    print(f"  Preferencia centro: {pref_centro*100:.1f}%")
    print(f"  Tasa bloqueo: {tasa_bloqueo*100:.1f}%")
    print(f"  Tasa victoria directa: {tasa_victoria*100:.1f}%")

    # Diagnóstico
    print("\nDIAGNÓSTICO DEL AGENTE:")
    diag = agente.diagnostico()
    for k, v in diag.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.3f}")
        else:
            print(f"  {k}: {v}")

    # Veredicto
    print("\n" + "=" * 70)
    emergencia = pref_centro > 0.2 or tasa_bloqueo > 0.4 or tasa_victoria > 0.4
    mejora = v/200 > 0.55

    if emergencia and mejora:
        print("VEREDICTO: ÉXITO - Conocimiento emergió y rendimiento mejoró")
    elif mejora:
        print("VEREDICTO: PARCIAL - Mejoró rendimiento pero sin conceptos claros")
    elif emergencia:
        print("VEREDICTO: PARCIAL - Conceptos emergieron pero rendimiento bajo")
    else:
        print("VEREDICTO: INSUFICIENTE - Requiere más ajustes")
    print("=" * 70)

    return agente, historial_wr


def evaluar(agente, n_partidas: int):
    juego = TicTacToe()
    victorias = 0
    derrotas = 0
    empates = 0

    for _ in range(n_partidas):
        estado = juego.reset()

        while True:
            acciones = juego.acciones_validas()
            if not acciones:
                break

            if juego.turno == 1:
                accion = agente.decidir(estado, acciones)
            else:
                accion = np.random.choice(acciones)

            _, _, terminal = juego.hacer_movimiento(accion)
            estado = juego.tablero.copy()

            if terminal:
                ganador = juego._verificar_ganador()
                if ganador == 1:
                    victorias += 1
                elif ganador == -1:
                    derrotas += 1
                else:
                    empates += 1
                break

    return victorias, derrotas, empates


def medir_preferencia_centro(agente, n_tests: int = 50) -> float:
    veces = 0
    for _ in range(n_tests):
        estado = np.zeros(9)
        accion = agente.decidir(estado, list(range(9)))
        if accion == 4:
            veces += 1
    return veces / n_tests


def medir_tasa_bloqueo(agente) -> float:
    situaciones = [
        (np.array([0, 0, 0, -1, -1, 0, 0, 0, 0]), 5),
        (np.array([-1, 0, 0, 0, -1, 0, 0, 0, 0]), 8),
        (np.array([-1, -1, 0, 0, 0, 0, 0, 0, 0]), 2),
        (np.array([0, -1, -1, 0, 0, 0, 0, 0, 0]), 0),
        (np.array([0, 0, -1, 0, 0, -1, 0, 0, 0]), 8),
    ]
    correctos = 0
    for estado, esperado in situaciones:
        validas = [i for i in range(9) if estado[i] == 0]
        accion = agente.decidir(estado, validas)
        if accion == esperado:
            correctos += 1
    return correctos / len(situaciones)


def medir_tasa_victoria(agente) -> float:
    situaciones = [
        (np.array([1, 1, 0, 0, 0, 0, 0, 0, 0]), 2),
        (np.array([0, 0, 0, 1, 1, 0, 0, 0, 0]), 5),
        (np.array([1, 0, 0, 0, 1, 0, 0, 0, 0]), 8),
        (np.array([1, 0, 0, 1, 0, 0, 0, 0, 0]), 6),
        (np.array([0, 1, 0, 0, 1, 0, 0, 0, 0]), 7),
    ]
    correctos = 0
    for estado, esperado in situaciones:
        validas = [i for i in range(9) if estado[i] == 0]
        accion = agente.decidir(estado, validas)
        if accion == esperado:
            correctos += 1
    return correctos / len(situaciones)


if __name__ == "__main__":
    agente, historial = entrenar_v2(3000)

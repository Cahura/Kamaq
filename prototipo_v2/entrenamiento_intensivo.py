"""
ENTRENAMIENTO INTENSIVO - 10,000 EPISODIOS
==========================================
Prueba exhaustiva para determinar si el conocimiento puede emerger
con suficiente entrenamiento.

HIPÓTESIS: Con más episodios y decay de epsilon más lento,
los conceptos estratégicos deberían emerger.
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

    def analizar(self, pos, jugador):
        tablero_antes = self.tablero.copy()
        tablero_antes[pos] = 0

        analisis = {'gano': False, 'bloqueo': False, 'amenaza': False}

        for linea in self.LINEAS:
            if pos in linea:
                suma = sum(self.tablero[i] for i in linea)
                if suma == 3 * jugador:
                    analisis['gano'] = True

        oponente = -jugador
        for linea in self.LINEAS:
            if pos in linea:
                fichas_op = sum(1 for i in linea if tablero_antes[i] == oponente)
                vacias = sum(1 for i in linea if tablero_antes[i] == 0)
                if fichas_op == 2 and vacias == 1:
                    analisis['bloqueo'] = True

        for linea in self.LINEAS:
            if pos in linea:
                fichas = sum(1 for i in linea if self.tablero[i] == jugador)
                vacias = sum(1 for i in linea if self.tablero[i] == 0)
                if fichas == 2 and vacias == 1:
                    analisis['amenaza'] = True

        return analisis


def entrenar_intensivo():
    """Entrenamiento intensivo de 10,000 episodios."""
    print("=" * 70)
    print("ENTRENAMIENTO INTENSIVO - 10,000 EPISODIOS")
    print("=" * 70)
    print("Objetivo: Determinar si los conceptos emergen con suficiente tiempo")
    print("=" * 70)

    agente = AgenteEmergentev2(dim_estado=9, n_acciones=9)

    # Configurar epsilon decay más lento
    agente.epsilon_inicial = 0.8
    agente.epsilon_final = 0.02
    agente.epsilon_decay = 0.9998  # Más lento
    agente.epsilon = agente.epsilon_inicial

    juego = TicTacToe()

    victorias = 0
    derrotas = 0
    empates = 0

    # Tracking de métricas por fase
    metricas_por_fase = []

    inicio = time.time()
    n_episodios = 10000

    for ep in range(1, n_episodios + 1):
        estado = juego.reset()

        while True:
            acciones = juego.acciones_validas()
            if not acciones:
                break

            if juego.turno == 1:
                estado_antes = estado.copy()
                accion = agente.decidir(estado, acciones)

                valido, recompensa_base, terminal = juego.hacer_movimiento(accion)
                estado = juego.tablero.copy()

                analisis = juego.analizar(accion, 1)

                recompensa = recompensa_base
                if analisis['gano']:
                    recompensa += 0.5
                if analisis['bloqueo']:
                    recompensa += 0.4
                if analisis['amenaza']:
                    recompensa += 0.15

                exp = Experiencia(estado_antes, accion, recompensa, estado, terminal)
                agente.aprender(exp)

                if terminal:
                    if juego._verificar_ganador() == 1:
                        victorias += 1
                    else:
                        empates += 1
                    break

            else:
                accion = np.random.choice(acciones)
                _, _, terminal = juego.hacer_movimiento(accion)
                estado = juego.tablero.copy()

                if terminal:
                    if juego._verificar_ganador() == -1:
                        derrotas += 1
                        if agente.experiencias:
                            exp = agente.experiencias[-1]
                            exp_derrota = Experiencia(exp.estado_siguiente, exp.accion,
                                                       -1.0, estado, True)
                            agente.aprender(exp_derrota)
                    else:
                        empates += 1
                    break

        if ep % 25 == 0:
            agente.consolidar()

        if ep % 500 == 0:
            wr_actual = victorias / ep * 100
            pref_centro = medir_centro(agente, 50)
            tasa_bloq = medir_bloqueo(agente)
            tasa_vic = medir_victoria(agente)

            metricas_por_fase.append({
                'ep': ep,
                'wr': wr_actual,
                'eps': agente.epsilon,
                'centro': pref_centro,
                'bloqueo': tasa_bloq,
                'victoria': tasa_vic,
            })

            print(f"Ep {ep:5d} | WR: {wr_actual:5.1f}% | eps: {agente.epsilon:.3f} | "
                  f"Centro: {pref_centro*100:5.1f}% | Bloq: {tasa_bloq*100:5.1f}% | "
                  f"Vic: {tasa_vic*100:5.1f}%")

    tiempo = time.time() - inicio

    print("=" * 70)
    print(f"COMPLETADO en {tiempo:.1f}s ({tiempo/60:.1f} min)")
    print(f"Total: {victorias}V / {derrotas}D / {empates}E")
    print("=" * 70)

    # Evaluación exhaustiva
    print("\nEVALUACIÓN FINAL (500 partidas):")
    v, d, e = evaluar(agente, 500)
    wr_final = v / 500 * 100
    print(f"  Win Rate: {wr_final:.1f}%")
    print(f"  V/D/E: {v}/{d}/{e}")

    print("\nANÁLISIS DE EMERGENCIA:")
    pref_centro = medir_centro(agente, 200)
    tasa_bloq = medir_bloqueo(agente)
    tasa_vic = medir_victoria(agente)

    print(f"  Preferencia centro: {pref_centro*100:.1f}%")
    print(f"  Tasa bloqueo: {tasa_bloq*100:.1f}%")
    print(f"  Tasa victoria directa: {tasa_vic*100:.1f}%")

    # Análisis de tendencias
    print("\nTENDENCIAS DE APRENDIZAJE:")
    if len(metricas_por_fase) >= 4:
        inicio_fase = metricas_por_fase[:len(metricas_por_fase)//4]
        fin_fase = metricas_por_fase[-len(metricas_por_fase)//4:]

        wr_inicio = np.mean([m['wr'] for m in inicio_fase])
        wr_fin = np.mean([m['wr'] for m in fin_fase])

        centro_inicio = np.mean([m['centro'] for m in inicio_fase])
        centro_fin = np.mean([m['centro'] for m in fin_fase])

        bloq_inicio = np.mean([m['bloqueo'] for m in inicio_fase])
        bloq_fin = np.mean([m['bloqueo'] for m in fin_fase])

        vic_inicio = np.mean([m['victoria'] for m in inicio_fase])
        vic_fin = np.mean([m['victoria'] for m in fin_fase])

        print(f"  Win Rate:     {wr_inicio:.1f}% -> {wr_fin:.1f}% ({wr_fin-wr_inicio:+.1f}%)")
        print(f"  Centro:       {centro_inicio*100:.1f}% -> {centro_fin*100:.1f}% ({(centro_fin-centro_inicio)*100:+.1f}%)")
        print(f"  Bloqueo:      {bloq_inicio*100:.1f}% -> {bloq_fin*100:.1f}% ({(bloq_fin-bloq_inicio)*100:+.1f}%)")
        print(f"  Victoria:     {vic_inicio*100:.1f}% -> {vic_fin*100:.1f}% ({(vic_fin-vic_inicio)*100:+.1f}%)")

    # Veredicto final
    print("\n" + "=" * 70)
    print("VEREDICTO FINAL")
    print("=" * 70)

    emergencia_clara = (pref_centro > 0.25 or tasa_bloq > 0.5 or tasa_vic > 0.5)
    mejora_significativa = wr_final > 60

    if emergencia_clara and mejora_significativa:
        print("ÉXITO: El conocimiento estratégico HA EMERGIDO")
        print("El sistema desarrolló conceptos sin que se los programáramos.")
    elif mejora_significativa:
        print("PARCIAL: El rendimiento mejoró pero sin conceptos claros")
        print("El aprendizaje ocurrió pero no de forma interpretable.")
    elif emergencia_clara:
        print("PARCIAL: Hay señales de conceptos pero rendimiento bajo")
        print("Los conceptos emergieron pero no se traducen en victorias.")
    else:
        print("ANÁLISIS HONESTO: El paradigma actual no logra emergencia clara")
        print("Con 10,000 episodios, los conceptos estratégicos no emergen.")
        print("")
        print("POSIBLES CAUSAS:")
        print("  1. El campo cognitivo necesita más dimensionalidad")
        print("  2. Las recompensas aún no son suficientemente informativas")
        print("  3. El aprendizaje Hebbiano requiere ajustes")
        print("  4. Se necesita una arquitectura diferente para este problema")

    print("=" * 70)

    # Guardar resultados
    return agente, metricas_por_fase


def evaluar(agente, n_partidas):
    juego = TicTacToe()
    v, d, e = 0, 0, 0

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
                    v += 1
                elif ganador == -1:
                    d += 1
                else:
                    e += 1
                break

    return v, d, e


def medir_centro(agente, n=50):
    veces = 0
    for _ in range(n):
        estado = np.zeros(9)
        accion = agente.decidir(estado, list(range(9)))
        if accion == 4:
            veces += 1
    return veces / n


def medir_bloqueo(agente):
    situaciones = [
        (np.array([0, 0, 0, -1, -1, 0, 0, 0, 0]), 5),
        (np.array([-1, 0, 0, 0, -1, 0, 0, 0, 0]), 8),
        (np.array([-1, -1, 0, 0, 0, 0, 0, 0, 0]), 2),
        (np.array([0, -1, -1, 0, 0, 0, 0, 0, 0]), 0),
        (np.array([0, 0, -1, 0, 0, -1, 0, 0, 0]), 8),
        (np.array([0, 0, 0, 0, -1, 0, 0, -1, 0]), 2),
        (np.array([-1, 0, 0, -1, 0, 0, 0, 0, 0]), 6),
        (np.array([0, -1, 0, 0, -1, 0, 0, 0, 0]), 7),
    ]
    correctos = 0
    for estado, esperado in situaciones:
        validas = [i for i in range(9) if estado[i] == 0]
        accion = agente.decidir(estado, validas)
        if accion == esperado:
            correctos += 1
    return correctos / len(situaciones)


def medir_victoria(agente):
    situaciones = [
        (np.array([1, 1, 0, 0, 0, 0, 0, 0, 0]), 2),
        (np.array([0, 0, 0, 1, 1, 0, 0, 0, 0]), 5),
        (np.array([1, 0, 0, 0, 1, 0, 0, 0, 0]), 8),
        (np.array([1, 0, 0, 1, 0, 0, 0, 0, 0]), 6),
        (np.array([0, 1, 0, 0, 1, 0, 0, 0, 0]), 7),
        (np.array([0, 0, 1, 0, 0, 1, 0, 0, 0]), 8),
        (np.array([1, 0, 0, 0, 0, 0, 1, 0, 0]), 3),
        (np.array([0, 0, 0, 0, 1, 0, 1, 0, 0]), 2),
    ]
    correctos = 0
    for estado, esperado in situaciones:
        validas = [i for i in range(9) if estado[i] == 0]
        accion = agente.decidir(estado, validas)
        if accion == esperado:
            correctos += 1
    return correctos / len(situaciones)


if __name__ == "__main__":
    agente, historial = entrenar_intensivo()

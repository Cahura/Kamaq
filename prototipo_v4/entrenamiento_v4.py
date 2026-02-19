"""
ENTRENAMIENTO V4 - EXPERIMENTO COMPLETO
========================================
Entrena el agente KAMAQ V4 y compara con V3.

METRICAS DE EXITO (del plan):
- Win rate vs random: >70%
- Bloqueo sin seeds: >50%
- Diversidad de exploracion: >80%

COMPARACION:
- V3 mejor resultado: 60.8% WR, 0% bloqueo emergente
- V4 objetivo: superar en ambas metricas

Ejecutar: python entrenamiento_v4.py
"""

import numpy as np
import time
from typing import List, Dict, Tuple
from dataclasses import dataclass

from campo_cognitivo_v4 import AgenteKAMAQv4


class TicTacToe:
    """Juego de Tic-Tac-Toe."""
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

    def hay_amenaza(self) -> List[int]:
        """Retorna posiciones que bloquean amenazas."""
        bloqueos = []
        for linea in self.LINEAS:
            valores = [self.tablero[i] for i in linea]
            if valores.count(-1) == 2 and valores.count(0) == 1:
                for i in linea:
                    if self.tablero[i] == 0:
                        bloqueos.append(i)
        return bloqueos

    def hay_victoria(self) -> List[int]:
        """Retorna posiciones que dan victoria."""
        victorias = []
        for linea in self.LINEAS:
            valores = [self.tablero[i] for i in linea]
            if valores.count(1) == 2 and valores.count(0) == 1:
                for i in linea:
                    if self.tablero[i] == 0:
                        victorias.append(i)
        return victorias


def calcular_recompensa_shaped(juego: TicTacToe, accion: int,
                                terminal: bool, ganador: int) -> Tuple[float, bool]:
    """Calcula recompensa con shaping y determina exito."""
    recompensa = 0.0
    exito = False

    if terminal:
        if ganador == 1:
            recompensa = 1.0
            exito = True
        elif ganador == -1:
            recompensa = -1.0
        else:
            recompensa = 0.1

    # Bonus por bloquear amenaza
    tablero_antes = juego.tablero.copy()
    tablero_antes[accion] = 0  # Revertir para ver estado anterior

    # Verificar si bloqueo amenaza
    for linea in juego.LINEAS:
        if accion in linea:
            valores_antes = [tablero_antes[i] for i in linea]
            if valores_antes.count(-1) == 2 and valores_antes.count(0) == 1:
                recompensa += 0.4
                exito = True
                break

    # Bonus por crear amenaza
    for linea in juego.LINEAS:
        if accion in linea:
            valores = [juego.tablero[i] for i in linea]
            if valores.count(1) == 2 and valores.count(0) == 1:
                recompensa += 0.2
                exito = True

    # Bonus por centro en turno 1
    if accion == 4 and np.sum(np.abs(tablero_antes)) == 0:
        recompensa += 0.1
        exito = True

    return recompensa, exito


def entrenar_v4(n_episodios: int = 5000) -> Tuple[AgenteKAMAQv4, List[Dict]]:
    """Entrena el agente V4."""
    print("=" * 70)
    print("ENTRENAMIENTO KAMAQ V4")
    print("=" * 70)
    print(f"Episodios: {n_episodios}")
    print("Componentes: Inferencia Activa + Poblacion Estrategias + Hopfield")
    print("=" * 70)

    agente = AgenteKAMAQv4()
    juego = TicTacToe()
    metricas = []

    inicio = time.time()

    for ep in range(1, n_episodios + 1):
        estado = juego.reset()

        while True:
            acciones = juego.acciones_validas()
            if not acciones:
                break

            if juego.turno == 1:
                estado_antes = estado.copy()
                accion = agente.decidir(estado, acciones)

                _, _, terminal = juego.hacer_movimiento(accion)
                estado = juego.tablero.copy()
                ganador = juego._verificar_ganador()

                recompensa, exito = calcular_recompensa_shaped(juego, accion, terminal, ganador)

                agente.aprender(
                    estado_antes, accion, estado, recompensa, terminal,
                    {'exito': exito}
                )

                if terminal:
                    if ganador == 1:
                        agente.fin_episodio('victoria')
                    elif ganador == -1:
                        agente.fin_episodio('derrota')
                    else:
                        agente.fin_episodio('empate')
                    break
            else:
                accion = np.random.choice(acciones)
                _, _, terminal = juego.hacer_movimiento(accion)
                estado = juego.tablero.copy()

                if terminal:
                    ganador = juego._verificar_ganador()
                    if ganador == -1:
                        agente.fin_episodio('derrota')
                        # Penalizar ultima accion del agente
                        if agente.campo.experiencias:
                            ultima = agente.campo.experiencias[-1]
                            agente.aprender(
                                ultima['estado'], ultima['accion'],
                                estado, -1.0, True, {'exito': False}
                            )
                    else:
                        agente.fin_episodio('empate')
                    break

        # Metricas cada 500 episodios
        if ep % 500 == 0:
            diag = agente.diagnostico()

            # Medir bloqueo
            tasa_bloqueo = medir_tasa_bloqueo(agente)
            tasa_victoria = medir_tasa_victoria(agente)
            pref_centro = medir_preferencia_centro(agente)

            metricas.append({
                'ep': ep,
                'wr': diag['win_rate'],
                'bloqueo': tasa_bloqueo,
                'victoria': tasa_victoria,
                'centro': pref_centro,
                'estrategias': diag['n_estrategias'],
                'fitness': diag['fitness_promedio'],
                'fuente_estrategia': diag['fuente_estrategia'],
                'fuente_curiosidad': diag['fuente_curiosidad']
            })

            print(f"Ep {ep:5d} | WR: {diag['win_rate']:5.1f}% | "
                  f"Bloq: {tasa_bloqueo*100:5.1f}% | Vic: {tasa_victoria*100:5.1f}% | "
                  f"Estr: {diag['n_estrategias']:3d} | Fit: {diag['fitness_promedio']:.2f}")

    tiempo_total = time.time() - inicio

    print("\n" + "=" * 70)
    print(f"COMPLETADO en {tiempo_total:.1f}s ({tiempo_total/60:.1f} min)")
    print("=" * 70)

    return agente, metricas


def medir_tasa_bloqueo(agente: AgenteKAMAQv4, n_tests: int = 8) -> float:
    """Mide tasa de bloqueo correcto."""
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


def medir_tasa_victoria(agente: AgenteKAMAQv4, n_tests: int = 8) -> float:
    """Mide tasa de completar lineas ganadoras."""
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


def medir_preferencia_centro(agente: AgenteKAMAQv4, n_tests: int = 50) -> float:
    """Mide preferencia por centro en tablero vacio."""
    centro = 0
    for _ in range(n_tests):
        estado = np.zeros(9)
        accion = agente.decidir(estado, list(range(9)))
        if accion == 4:
            centro += 1
    return centro / n_tests


def evaluar_final(agente: AgenteKAMAQv4, n_partidas: int = 500) -> Dict:
    """Evaluacion final contra random."""
    juego = TicTacToe()
    victorias, derrotas, empates = 0, 0, 0

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

    return {
        'victorias': victorias,
        'derrotas': derrotas,
        'empates': empates,
        'win_rate': victorias / n_partidas * 100
    }


def experimento_completo():
    """Ejecuta el experimento completo y compara con V3."""
    print("\n" + "=" * 70)
    print("EXPERIMENTO COMPLETO: KAMAQ V4 vs V3")
    print("=" * 70)

    # Resultados V3 (de experimentos anteriores)
    resultados_v3 = {
        'win_rate': 60.8,
        'bloqueo': 0.0,
        'victoria': 25.0,
        'centro': 12.0
    }

    # Entrenar V4
    agente, metricas = entrenar_v4(n_episodios=5000)

    # Evaluacion final
    print("\n" + "=" * 70)
    print("EVALUACION FINAL (500 partidas)")
    print("=" * 70)

    eval_final = evaluar_final(agente, 500)
    tasa_bloqueo = medir_tasa_bloqueo(agente)
    tasa_victoria = medir_tasa_victoria(agente)
    pref_centro = medir_preferencia_centro(agente, 100)

    print(f"Win Rate: {eval_final['win_rate']:.1f}%")
    print(f"V/D/E: {eval_final['victorias']}/{eval_final['derrotas']}/{eval_final['empates']}")
    print(f"Tasa Bloqueo: {tasa_bloqueo*100:.1f}%")
    print(f"Tasa Victoria: {tasa_victoria*100:.1f}%")
    print(f"Preferencia Centro: {pref_centro*100:.1f}%")

    # Comparacion con V3
    print("\n" + "=" * 70)
    print("COMPARACION V4 vs V3")
    print("=" * 70)

    print(f"\n{'Metrica':<25} {'V3':>12} {'V4':>12} {'Diff':>12}")
    print("-" * 61)

    v4_wr = eval_final['win_rate']
    v4_bloqueo = tasa_bloqueo * 100
    v4_victoria = tasa_victoria * 100
    v4_centro = pref_centro * 100

    print(f"{'Win Rate':<25} {resultados_v3['win_rate']:>11.1f}% {v4_wr:>11.1f}% {v4_wr - resultados_v3['win_rate']:>+11.1f}%")
    print(f"{'Tasa Bloqueo':<25} {resultados_v3['bloqueo']:>11.1f}% {v4_bloqueo:>11.1f}% {v4_bloqueo - resultados_v3['bloqueo']:>+11.1f}%")
    print(f"{'Tasa Victoria':<25} {resultados_v3['victoria']:>11.1f}% {v4_victoria:>11.1f}% {v4_victoria - resultados_v3['victoria']:>+11.1f}%")
    print(f"{'Preferencia Centro':<25} {resultados_v3['centro']:>11.1f}% {v4_centro:>11.1f}% {v4_centro - resultados_v3['centro']:>+11.1f}%")

    # Veredicto
    print("\n" + "=" * 70)
    print("VEREDICTO")
    print("=" * 70)

    exito_wr = v4_wr > resultados_v3['win_rate']
    exito_bloqueo = v4_bloqueo > 20  # Mucho mejor que V3
    exito_victoria = v4_victoria > resultados_v3['victoria']
    meta_wr = v4_wr >= 70
    meta_bloqueo = v4_bloqueo >= 50

    print(f"\nObjetivos del plan:")
    print(f"  Win Rate >= 70%: {'[OK]' if meta_wr else '[NO]'} ({v4_wr:.1f}%)")
    print(f"  Bloqueo >= 50%:  {'[OK]' if meta_bloqueo else '[NO]'} ({v4_bloqueo:.1f}%)")

    print(f"\nMejora sobre V3:")
    print(f"  Win Rate:        {'[OK]' if exito_wr else '[NO]'} ({v4_wr - resultados_v3['win_rate']:+.1f}%)")
    print(f"  Bloqueo:         {'[OK]' if exito_bloqueo else '[NO]'} ({v4_bloqueo - resultados_v3['bloqueo']:+.1f}%)")
    print(f"  Victoria:        {'[OK]' if exito_victoria else '[NO]'} ({v4_victoria - resultados_v3['victoria']:+.1f}%)")

    if meta_wr and meta_bloqueo:
        print("\n[EXITO COMPLETO] V4 cumple todos los objetivos del plan")
    elif exito_wr and exito_bloqueo:
        print("\n[EXITO PARCIAL] V4 mejora significativamente sobre V3")
        print("  pero no alcanza las metas ambiciosas del plan")
    elif exito_wr or exito_bloqueo:
        print("\n[MEJORA MODERADA] V4 mejora en algunas metricas")
    else:
        print("\n[SIN MEJORA] V4 no mejora sobre V3")
        print("  Revisar arquitectura o parametros")

    print("=" * 70)

    # Diagnostico final
    print("\n" + "=" * 70)
    print("DIAGNOSTICO FINAL DEL AGENTE")
    print("=" * 70)

    diag = agente.diagnostico()
    print(f"\nComponentes:")
    print(f"  Estrategias en poblacion: {diag['n_estrategias']}")
    print(f"  Fitness promedio: {diag['fitness_promedio']:.3f}")
    print(f"  Generaciones: {diag['generacion']}")

    print(f"\nFuentes de decision:")
    print(f"  Estrategia: {diag['fuente_estrategia']:.1f}%")
    print(f"  Curiosidad: {diag['fuente_curiosidad']:.1f}%")

    print(f"\nInferencia Activa:")
    print(f"  Estados unicos: {diag['estados_unicos']}")
    print(f"  Cobertura transiciones: {diag['cobertura_transiciones']*100:.1f}%")

    return agente, metricas, eval_final


if __name__ == "__main__":
    agente, metricas, eval_final = experimento_completo()

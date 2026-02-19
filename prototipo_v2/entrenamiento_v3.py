"""
ENTRENAMIENTO V3 - ARQUITECTURA CORREGIDA
==========================================
Entrena el agente V3 con:
1. Features estructuradas
2. TD(lambda) real
3. Seeds opcionales para bootstrap
4. Diagnosticos detallados

EXPERIMENTOS:
1. Sin seeds (emergencia pura) - 10,000 episodios
2. Con seeds (bootstrap + generalizacion) - 10,000 episodios
3. Comparacion honesta

Ejecutar: python entrenamiento_v3.py
"""

import numpy as np
import time
from typing import List, Dict, Tuple
from dataclasses import dataclass
from campo_cognitivo_v3 import AgenteEmergentev3


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

    def analizar_movimiento(self, pos, jugador) -> Dict[str, bool]:
        """Analiza un movimiento para dar recompensas shaped."""
        tablero_antes = self.tablero.copy()
        tablero_antes[pos] = 0

        resultado = {
            'gano': False,
            'bloqueo': False,
            'amenaza_creada': False,
            'centro': pos == 4,
            'esquina': pos in [0, 2, 6, 8],
        }

        # Verificar si gano
        for linea in self.LINEAS:
            if pos in linea:
                suma = sum(self.tablero[i] for i in linea)
                if suma == 3 * jugador:
                    resultado['gano'] = True

        # Verificar si bloqueo
        oponente = -jugador
        for linea in self.LINEAS:
            if pos in linea:
                fichas_op = sum(1 for i in linea if tablero_antes[i] == oponente)
                vacias = sum(1 for i in linea if tablero_antes[i] == 0)
                if fichas_op == 2 and vacias == 1:
                    resultado['bloqueo'] = True

        # Verificar si creo amenaza
        for linea in self.LINEAS:
            if pos in linea:
                fichas = sum(1 for i in linea if self.tablero[i] == jugador)
                vacias = sum(1 for i in linea if self.tablero[i] == 0)
                if fichas == 2 and vacias == 1:
                    resultado['amenaza_creada'] = True

        return resultado


def calcular_recompensa_shaped(analisis: Dict[str, bool], turno: int) -> float:
    """Calcula recompensa con shaping estructurado."""
    recompensa = 0.0

    if analisis['gano']:
        recompensa += 0.5  # Bonus adicional por ganar
    if analisis['bloqueo']:
        recompensa += 0.4  # Bloquear es casi tan bueno como ganar
    if analisis['amenaza_creada']:
        recompensa += 0.2  # Crear amenazas es bueno

    # Centro en primer movimiento
    if analisis['centro'] and turno == 1:
        recompensa += 0.1

    return recompensa


def evaluar_agente(agente: AgenteEmergentev3, n_partidas: int = 200) -> Tuple[int, int, int, float]:
    """Evalua el agente contra random."""
    juego = TicTacToe()
    victorias, derrotas, empates = 0, 0, 0

    # Guardar epsilon y forzar greedy
    epsilon_original = agente.epsilon
    agente.epsilon = 0.0

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

    agente.epsilon = epsilon_original
    win_rate = victorias / n_partidas * 100
    return victorias, derrotas, empates, win_rate


def medir_preferencia_centro(agente: AgenteEmergentev3, n_tests: int = 100) -> float:
    """Mide con que frecuencia elige centro en tablero vacio."""
    epsilon_original = agente.epsilon
    agente.epsilon = 0.0

    centro_elegido = 0
    for _ in range(n_tests):
        tablero = np.zeros(9)
        accion = agente.decidir(tablero, list(range(9)))
        if accion == 4:
            centro_elegido += 1

    agente.epsilon = epsilon_original
    return centro_elegido / n_tests


def medir_tasa_bloqueo(agente: AgenteEmergentev3) -> float:
    """Mide tasa de bloqueo en situaciones de amenaza."""
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

    epsilon_original = agente.epsilon
    agente.epsilon = 0.0

    correctos = 0
    for estado, esperado in situaciones:
        validas = [i for i in range(9) if estado[i] == 0]
        accion = agente.decidir(estado, validas)
        if accion == esperado:
            correctos += 1

    agente.epsilon = epsilon_original
    return correctos / len(situaciones)


def medir_tasa_victoria(agente: AgenteEmergentev3) -> float:
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

    epsilon_original = agente.epsilon
    agente.epsilon = 0.0

    correctos = 0
    for estado, esperado in situaciones:
        validas = [i for i in range(9) if estado[i] == 0]
        accion = agente.decidir(estado, validas)
        if accion == esperado:
            correctos += 1

    agente.epsilon = epsilon_original
    return correctos / len(situaciones)


def entrenar(agente: AgenteEmergentev3, n_episodios: int = 10000,
             nombre: str = "Agente") -> List[Dict]:
    """Entrena el agente y retorna metricas por fase."""
    print(f"\n{'='*70}")
    print(f"ENTRENANDO: {nombre}")
    print(f"Episodios: {n_episodios}")
    print(f"{'='*70}")

    juego = TicTacToe()
    metricas = []

    victorias = 0
    derrotas = 0
    empates = 0

    inicio = time.time()

    for ep in range(1, n_episodios + 1):
        estado = juego.reset()
        historial_episodio = []

        while True:
            acciones = juego.acciones_validas()
            if not acciones:
                break

            if juego.turno == 1:
                estado_antes = estado.copy()
                accion = agente.decidir(estado, acciones)

                valido, recompensa_base, terminal = juego.hacer_movimiento(accion)
                estado = juego.tablero.copy()

                # Analizar y calcular reward shaping
                analisis = juego.analizar_movimiento(accion, 1)
                recompensa_shaped = calcular_recompensa_shaped(analisis, juego.turno)
                recompensa_total = recompensa_base + recompensa_shaped

                # Aprender
                agente.aprender(estado_antes, accion, recompensa_total, estado, terminal)

                historial_episodio.append({
                    'estado': estado_antes,
                    'accion': accion,
                    'recompensa': recompensa_total,
                    'analisis': analisis
                })

                if terminal:
                    if juego._verificar_ganador() == 1:
                        victorias += 1
                    else:
                        empates += 1
                    break

            else:
                # Oponente random
                accion = np.random.choice(acciones)
                _, _, terminal = juego.hacer_movimiento(accion)
                estado = juego.tablero.copy()

                if terminal:
                    if juego._verificar_ganador() == -1:
                        derrotas += 1
                        # Penalizar ultima accion del agente
                        if historial_episodio:
                            ultimo = historial_episodio[-1]
                            agente.aprender(ultimo['estado'], ultimo['accion'],
                                          -1.0, estado, True)
                    else:
                        empates += 1
                    break

        # Consolidar periodicamente
        if ep % 25 == 0:
            agente.consolidar(n_replay=40)

        # Metricas cada 500 episodios
        if ep % 500 == 0:
            wr = victorias / ep * 100
            pref_centro = medir_preferencia_centro(agente, 50)
            tasa_bloqueo = medir_tasa_bloqueo(agente)
            tasa_victoria = medir_tasa_victoria(agente)

            metricas.append({
                'ep': ep,
                'wr': wr,
                'eps': agente.epsilon,
                'centro': pref_centro,
                'bloqueo': tasa_bloqueo,
                'victoria': tasa_victoria,
            })

            diag = agente.diagnostico()

            print(f"Ep {ep:5d} | WR: {wr:5.1f}% | eps: {agente.epsilon:.3f} | "
                  f"Centro: {pref_centro*100:5.1f}% | Bloq: {tasa_bloqueo*100:5.1f}% | "
                  f"Vic: {tasa_victoria*100:5.1f}% | J_pa: {diag['J_percepcion_accion']:.4f}")

    tiempo_total = time.time() - inicio

    print(f"\n{'='*70}")
    print(f"COMPLETADO: {nombre}")
    print(f"Tiempo: {tiempo_total:.1f}s ({tiempo_total/60:.1f} min)")
    print(f"Total: {victorias}V / {derrotas}D / {empates}E")
    print(f"{'='*70}")

    return metricas


def experimento_completo():
    """Ejecuta experimento comparativo: sin seeds vs con seeds."""
    print("="*70)
    print("EXPERIMENTO V3: EMERGENCIA VS BOOTSTRAP")
    print("="*70)
    print("Comparando dos configuraciones:")
    print("  1. SIN SEEDS: Emergencia pura desde cero")
    print("  2. CON SEEDS: Bootstrap con 3 ejemplos")
    print("="*70)

    n_episodios = 10000

    # Experimento 1: Sin seeds
    print("\n" + "="*70)
    print("EXPERIMENTO 1: EMERGENCIA PURA (SIN SEEDS)")
    print("="*70)

    agente_puro = AgenteEmergentev3()
    metricas_puro = entrenar(agente_puro, n_episodios, "Emergencia Pura")

    # Evaluacion final
    print("\nEVALUACION FINAL (SIN SEEDS):")
    v, d, e, wr = evaluar_agente(agente_puro, 500)
    print(f"  Win Rate: {wr:.1f}%")
    print(f"  V/D/E: {v}/{d}/{e}")

    centro_puro = medir_preferencia_centro(agente_puro, 200)
    bloqueo_puro = medir_tasa_bloqueo(agente_puro)
    victoria_puro = medir_tasa_victoria(agente_puro)

    print(f"  Preferencia centro: {centro_puro*100:.1f}%")
    print(f"  Tasa bloqueo: {bloqueo_puro*100:.1f}%")
    print(f"  Tasa victoria directa: {victoria_puro*100:.1f}%")

    # Experimento 2: Con seeds
    print("\n" + "="*70)
    print("EXPERIMENTO 2: BOOTSTRAP (CON SEEDS)")
    print("="*70)

    agente_bootstrap = AgenteEmergentev3()
    agente_bootstrap.seed_estrategia_basica()  # Plantar 3 seeds
    print("Seeds plantados: centro_inicial, completar_linea, bloquear_amenaza")

    metricas_bootstrap = entrenar(agente_bootstrap, n_episodios, "Bootstrap")

    # Evaluacion final
    print("\nEVALUACION FINAL (CON SEEDS):")
    v, d, e, wr = evaluar_agente(agente_bootstrap, 500)
    print(f"  Win Rate: {wr:.1f}%")
    print(f"  V/D/E: {v}/{d}/{e}")

    centro_boot = medir_preferencia_centro(agente_bootstrap, 200)
    bloqueo_boot = medir_tasa_bloqueo(agente_bootstrap)
    victoria_boot = medir_tasa_victoria(agente_bootstrap)

    print(f"  Preferencia centro: {centro_boot*100:.1f}%")
    print(f"  Tasa bloqueo: {bloqueo_boot*100:.1f}%")
    print(f"  Tasa victoria directa: {victoria_boot*100:.1f}%")

    # Comparacion
    print("\n" + "="*70)
    print("COMPARACION FINAL")
    print("="*70)
    print(f"{'Metrica':<25} {'Sin Seeds':>12} {'Con Seeds':>12} {'Diff':>10}")
    print("-"*70)

    wr_puro = evaluar_agente(agente_puro, 500)[3]
    wr_boot = evaluar_agente(agente_bootstrap, 500)[3]

    print(f"{'Win Rate':<25} {wr_puro:>11.1f}% {wr_boot:>11.1f}% {wr_boot-wr_puro:>+9.1f}%")
    print(f"{'Preferencia Centro':<25} {centro_puro*100:>11.1f}% {centro_boot*100:>11.1f}% {(centro_boot-centro_puro)*100:>+9.1f}%")
    print(f"{'Tasa Bloqueo':<25} {bloqueo_puro*100:>11.1f}% {bloqueo_boot*100:>11.1f}% {(bloqueo_boot-bloqueo_puro)*100:>+9.1f}%")
    print(f"{'Tasa Victoria Directa':<25} {victoria_puro*100:>11.1f}% {victoria_boot*100:>11.1f}% {(victoria_boot-victoria_puro)*100:>+9.1f}%")

    # Analisis de tendencias
    print("\n" + "="*70)
    print("ANALISIS DE TENDENCIAS")
    print("="*70)

    if len(metricas_puro) >= 4:
        inicio = metricas_puro[:len(metricas_puro)//4]
        fin = metricas_puro[-len(metricas_puro)//4:]

        print("\nSIN SEEDS (emergencia pura):")
        print(f"  WR:      {np.mean([m['wr'] for m in inicio]):.1f}% -> {np.mean([m['wr'] for m in fin]):.1f}%")
        print(f"  Centro:  {np.mean([m['centro'] for m in inicio])*100:.1f}% -> {np.mean([m['centro'] for m in fin])*100:.1f}%")
        print(f"  Bloqueo: {np.mean([m['bloqueo'] for m in inicio])*100:.1f}% -> {np.mean([m['bloqueo'] for m in fin])*100:.1f}%")
        print(f"  Victoria:{np.mean([m['victoria'] for m in inicio])*100:.1f}% -> {np.mean([m['victoria'] for m in fin])*100:.1f}%")

    if len(metricas_bootstrap) >= 4:
        inicio = metricas_bootstrap[:len(metricas_bootstrap)//4]
        fin = metricas_bootstrap[-len(metricas_bootstrap)//4:]

        print("\nCON SEEDS (bootstrap):")
        print(f"  WR:      {np.mean([m['wr'] for m in inicio]):.1f}% -> {np.mean([m['wr'] for m in fin]):.1f}%")
        print(f"  Centro:  {np.mean([m['centro'] for m in inicio])*100:.1f}% -> {np.mean([m['centro'] for m in fin])*100:.1f}%")
        print(f"  Bloqueo: {np.mean([m['bloqueo'] for m in inicio])*100:.1f}% -> {np.mean([m['bloqueo'] for m in fin])*100:.1f}%")
        print(f"  Victoria:{np.mean([m['victoria'] for m in inicio])*100:.1f}% -> {np.mean([m['victoria'] for m in fin])*100:.1f}%")

    # Veredicto
    print("\n" + "="*70)
    print("VEREDICTO FINAL")
    print("="*70)

    # Criterios de exito
    emergencia_pura = (centro_puro > 0.2 or bloqueo_puro > 0.4 or victoria_puro > 0.4)
    emergencia_boot = (centro_boot > 0.3 or bloqueo_boot > 0.5 or victoria_boot > 0.5)
    generalizacion = bloqueo_boot > bloqueo_puro + 0.1  # Mejoro significativamente con seeds

    if emergencia_pura:
        print("EMERGENCIA PURA: EXITO")
        print("  Conceptos estrategicos emergieron sin ningun seed.")
    else:
        print("EMERGENCIA PURA: NO LOGRADA")
        print("  Con 10,000 episodios y arquitectura V3, no emerge conocimiento claro.")

    if emergencia_boot:
        print("\nBOOTSTRAP: EXITO")
        print("  Con seeds iniciales, el sistema desarrollo conocimiento.")
    else:
        print("\nBOOTSTRAP: NO LOGRADO")
        print("  Incluso con seeds, el rendimiento es bajo.")

    if generalizacion:
        print("\nGENERALIZACION: EXITO")
        print("  El sistema generalizo desde los 3 seeds a casos nuevos.")
    else:
        print("\nGENERALIZACION: NO LOGRADA")
        print("  Los seeds no se generalizaron a situaciones nuevas.")

    print("\n" + "="*70)
    print("CONCLUSION HONESTA")
    print("="*70)

    if emergencia_pura or (emergencia_boot and generalizacion):
        print("El paradigma KAMAQ V3 FUNCIONA.")
        print("La arquitectura corregida permite emergencia de conocimiento.")
    else:
        print("ANALISIS HONESTO: La arquitectura V3 aun requiere ajustes.")
        print("\nPOSIBLES CAUSAS:")
        print("  1. Representacion de features aun insuficiente")
        print("  2. TD(lambda) necesita mas tiempo para propagar credito")
        print("  3. El espacio de 128 dimensiones es insuficiente")
        print("  4. Tic-Tac-Toe puede no ser el dominio ideal para probar emergencia")

    print("="*70)

    return agente_puro, agente_bootstrap, metricas_puro, metricas_bootstrap


if __name__ == "__main__":
    agente_puro, agente_boot, m_puro, m_boot = experimento_completo()

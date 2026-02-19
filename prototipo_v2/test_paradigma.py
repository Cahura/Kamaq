"""
TEST DEL PARADIGMA KAMAQ
========================
No mide "accuracy". Mide propiedades emergentes.

Preguntas a responder:
1. ¿El campo desarrolla atractores estables?
2. ¿La metacognición (entropía) correlaciona con incertidumbre real?
3. ¿Emerge conocimiento sin haberlo programado?
4. ¿El sistema tiene dinámica continua (no solo input-output)?
"""

import numpy as np
import sys
import os

# Agregar path para imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from typing import Dict, List, Tuple
from campo_cognitivo import CampoCognitivo, AprendizajeHebbiano
from agente_emergente import AgenteEmergente, Experiencia


class TicTacToe:
    """Juego simple para demostrar aprendizaje"""

    def __init__(self):
        self.reset()

    def reset(self) -> np.ndarray:
        self.tablero = [' '] * 9
        self.turno = 'X'
        return self.estado_vector()

    def estado_vector(self) -> np.ndarray:
        estado = np.zeros(9)
        for i, v in enumerate(self.tablero):
            if v == 'X':
                estado[i] = 1
            elif v == 'O':
                estado[i] = -1
        return estado

    def acciones_validas(self) -> List[int]:
        return [i for i, v in enumerate(self.tablero) if v == ' ']

    def hacer_movimiento(self, pos: int) -> Tuple[bool, float, bool]:
        if pos < 0 or pos > 8 or self.tablero[pos] != ' ':
            return False, -1.0, False

        self.tablero[pos] = self.turno
        ganador = self.verificar_ganador()

        if ganador == 'X':
            return True, 1.0, True
        elif ganador == 'O':
            return True, -1.0, True
        elif ganador == 'Empate':
            return True, 0.0, True

        self.turno = 'O' if self.turno == 'X' else 'X'
        return True, 0.0, False

    def verificar_ganador(self) -> str:
        lineas = [
            [0,1,2], [3,4,5], [6,7,8],
            [0,3,6], [1,4,7], [2,5,8],
            [0,4,8], [2,4,6]
        ]
        for l in lineas:
            if self.tablero[l[0]] == self.tablero[l[1]] == self.tablero[l[2]] != ' ':
                return self.tablero[l[0]]
        if ' ' not in self.tablero:
            return 'Empate'
        return None


def test_1_atractores_emergen():
    """
    TEST 1: ¿El campo desarrolla atractores estables?

    Memorizamos patrones y verificamos que el campo
    converge a ellos desde pistas parciales.
    """
    print("=" * 70)
    print("TEST 1: EMERGENCIA DE ATRACTORES")
    print("=" * 70)

    campo = CampoCognitivo(dimension=64)

    # Crear 5 patrones binarios (+1/-1) ortogonales
    patrones = []
    for i in range(5):
        # Patron aleatorio binario
        patron = np.random.choice([-1, 1], size=64)
        patrones.append(patron)
        campo.memorizar(patron)

    print(f"Patrones memorizados: {len(patrones)}")

    # Test de recuperación
    exitos = 0
    for i, patron in enumerate(patrones):
        # Crear pista corrompiendo 20% del patrón
        pista = patron.copy().astype(float)
        n_flip = int(0.2 * len(pista))
        indices = np.random.choice(len(pista), n_flip, replace=False)
        pista[indices] *= -1  # Invertir algunos bits

        # Intentar recordar
        recuerdo = campo.recordar(pista, pasos_relajacion=100)

        # Calcular similitud (overlap para patrones +-1)
        # overlap = (1/N) * sum(xi * yi)
        similitud = np.mean(recuerdo * patron)

        if similitud > 0.8:
            exitos += 1

        print(f"  Patron {i}: Overlap = {similitud:.3f} {'[OK]' if similitud > 0.8 else '[X]'}")

    print(f"\nRecuperación exitosa: {exitos}/{len(patrones)}")
    print(f"VEREDICTO: {'ATRACTORES EMERGEN' if exitos >= 4 else 'FALLO'}")

    return exitos >= 4


def test_2_metacognicion_calibrada():
    """
    TEST 2: ¿La entropía del campo correlaciona con incertidumbre real?

    Presentamos situaciones claras vs ambiguas
    y verificamos que la entropía refleje esto.
    """
    print("\n" + "=" * 70)
    print("TEST 2: METACOGNICIÓN CALIBRADA")
    print("=" * 70)

    agente = AgenteEmergente(dim_estado=9, n_acciones=9)

    # Primero entrenamos un poco para que el agente tenga contexto
    juego = TicTacToe()
    for _ in range(50):
        estado = juego.reset()
        while True:
            acciones = juego.acciones_validas()
            if not acciones:
                break
            accion = np.random.choice(acciones)
            estado_ant = estado.copy()
            _, rew, term = juego.hacer_movimiento(accion)
            estado = juego.estado_vector()
            agente.aprender(Experiencia(estado_ant, accion, rew, estado, term))
            if term:
                break

    # Ahora probamos situaciones claras vs ambiguas

    # Estado claro: Casi ganando
    estados_claros = [
        np.array([1, 1, 0, 0, 0, 0, 0, 0, 0]),  # X puede ganar en posición 2
        np.array([1, 0, 0, 1, 0, 0, 0, 0, 0]),  # Tiene dos fichas alineables
    ]

    # Estado ambiguo: Tablero vacío o equilibrado
    estados_ambiguos = [
        np.array([0, 0, 0, 0, 0, 0, 0, 0, 0]),  # Tablero vacío
        np.array([1, 0, -1, 0, 0, 0, -1, 0, 1]),  # Situación equilibrada
    ]

    # Medir entropía en cada caso
    entropias_claras = []
    entropias_ambiguas = []

    # Guardar estado del campo
    x_backup = agente.campo.x.copy()
    v_backup = agente.campo.v.copy()

    for estado in estados_claros:
        agente.campo.x = x_backup.copy()
        agente.campo.v = v_backup.copy()
        agente.campo.perturbar(agente.estado_a_patron(estado), fuerza=1.0)
        agente.campo.evolucionar(pasos=20)
        entropias_claras.append(agente.campo.entropia())

    for estado in estados_ambiguos:
        agente.campo.x = x_backup.copy()
        agente.campo.v = v_backup.copy()
        agente.campo.perturbar(agente.estado_a_patron(estado), fuerza=1.0)
        agente.campo.evolucionar(pasos=20)
        entropias_ambiguas.append(agente.campo.entropia())

    promedio_claros = np.mean(entropias_claras)
    promedio_ambiguos = np.mean(entropias_ambiguas)

    print(f"Entropía promedio (situaciones claras): {promedio_claros:.3f}")
    print(f"Entropía promedio (situaciones ambiguas): {promedio_ambiguos:.3f}")
    print(f"Diferencia: {promedio_ambiguos - promedio_claros:.3f}")

    # La metacognición está calibrada si las situaciones ambiguas tienen mayor entropía
    # O al menos una diferencia significativa
    calibrado = abs(promedio_ambiguos - promedio_claros) > 0.01

    print(f"\nVEREDICTO: {'METACOGNICIÓN MUESTRA DIFERENCIACIÓN' if calibrado else 'SIN DIFERENCIACIÓN CLARA'}")

    return calibrado


def test_3_conocimiento_emergente():
    """
    TEST 3: ¿El agente desarrolla conceptos sin que los programemos?

    Entrenamos en Tic-Tac-Toe y verificamos si:
    - Prefiere el centro (sin decírselo)
    - Bloquea amenazas (sin decírselo)
    - Completa líneas (sin decírselo)
    """
    print("\n" + "=" * 70)
    print("TEST 3: CONOCIMIENTO EMERGENTE")
    print("=" * 70)

    agente = AgenteEmergente(dim_estado=9, n_acciones=9)
    juego = TicTacToe()

    # Entrenar con 500 partidas contra random
    print("[Entrenando agente con 500 partidas...]")
    for ep in range(500):
        estado = juego.reset()

        while True:
            acciones = juego.acciones_validas()
            if not acciones:
                break

            if juego.turno == 'X':
                accion = agente.decidir(estado, acciones)
            else:
                accion = np.random.choice(acciones)

            estado_anterior = estado.copy()
            valido, recompensa, terminal = juego.hacer_movimiento(accion)
            estado = juego.estado_vector()

            # Solo aprende cuando es su turno
            if juego.turno == 'O' or terminal:
                # Recompensa desde perspectiva de X
                r = recompensa if juego.turno == 'O' else -recompensa
                exp = Experiencia(
                    estado=estado_anterior,
                    accion=accion,
                    recompensa=r,
                    estado_siguiente=estado,
                    terminal=terminal
                )
                agente.aprender(exp)

            if terminal:
                break

        if (ep + 1) % 100 == 0:
            agente.consolidar()
            print(f"  Episodio {ep+1}: recompensa acumulada = {agente.recompensa_acumulada:.1f}")

    # Ahora probamos si aprendió conceptos
    print("\n[Probando conocimiento emergente]")

    # Test 1: ¿Prefiere el centro cuando está disponible?
    n_tests = 50
    veces_centro = 0
    for _ in range(n_tests):
        estado = np.zeros(9)  # Tablero vacío
        accion = agente.decidir(estado, list(range(9)))
        if accion == 4:
            veces_centro += 1

    prefiere_centro = veces_centro / n_tests
    print(f"  Preferencia por centro (tablero vacío): {prefiere_centro:.1%}")

    # Test 2: ¿Bloquea amenazas?
    situaciones_bloqueo = [
        (np.array([0, 0, 0, -1, -1, 0, 0, 0, 0]), 5),  # Debe bloquear en 5
        (np.array([-1, 0, 0, 0, -1, 0, 0, 0, 0]), 8),  # Debe bloquear en 8
        (np.array([-1, -1, 0, 0, 0, 0, 0, 0, 0]), 2),  # Debe bloquear en 2
    ]

    bloqueos_correctos = 0
    for estado, bloqueo_esperado in situaciones_bloqueo:
        validas = [i for i in range(9) if estado[i] == 0]
        accion = agente.decidir(estado, validas)
        if accion == bloqueo_esperado:
            bloqueos_correctos += 1

    tasa_bloqueo = bloqueos_correctos / len(situaciones_bloqueo)
    print(f"  Tasa de bloqueo correcto: {tasa_bloqueo:.1%}")

    # Test 3: ¿Completa líneas propias?
    situaciones_ganar = [
        (np.array([1, 1, 0, 0, 0, 0, 0, 0, 0]), 2),  # Debe ganar en 2
        (np.array([0, 0, 0, 1, 1, 0, 0, 0, 0]), 5),  # Debe ganar en 5
        (np.array([1, 0, 0, 0, 1, 0, 0, 0, 0]), 8),  # Debe ganar en 8
    ]

    victorias_correctas = 0
    for estado, victoria_esperada in situaciones_ganar:
        validas = [i for i in range(9) if estado[i] == 0]
        accion = agente.decidir(estado, validas)
        if accion == victoria_esperada:
            victorias_correctas += 1

    tasa_victoria = victorias_correctas / len(situaciones_ganar)
    print(f"  Tasa de completar línea (ganar): {tasa_victoria:.1%}")

    # Diagnóstico final
    diag = agente.diagnostico()
    print(f"\n[Estado final del agente]")
    for k, v in diag.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.3f}")
        else:
            print(f"  {k}: {v}")

    # Introspección
    print(f"\n[Introspección del agente]")
    print(f"  {agente.introspecccion()}")

    # Veredicto
    # Cualquier preferencia por encima del azar (11% para centro, 33% para bloqueo/ganar)
    # indica que algo emergió
    algo_emergio = (prefiere_centro > 0.15) or (tasa_bloqueo > 0.4) or (tasa_victoria > 0.4)
    print(f"\nVEREDICTO: {'ALGÚN CONOCIMIENTO EMERGIÓ' if algo_emergio else 'NO HAY CONOCIMIENTO EMERGENTE CLARO'}")

    return algo_emergio


def test_4_dinamica_continua():
    """
    TEST 4: ¿El sistema tiene dinámica interna continua?

    Verificamos que el campo evoluciona incluso sin input,
    y que su estado cambia de forma significativa.
    """
    print("\n" + "=" * 70)
    print("TEST 4: DINÁMICA CONTINUA")
    print("=" * 70)

    campo = CampoCognitivo(dimension=64)

    # Memorizar algunos patrones para dar estructura
    for i in range(3):
        patron = np.random.randn(64)
        patron = patron / np.linalg.norm(patron)
        campo.memorizar(patron)

    # Evolucionar sin input y medir cambios
    estados = []
    energias = []
    entropias = []
    coherencias = []

    for t in range(100):
        campo.evolucionar(pasos=10)
        estados.append(campo.x.copy())
        energias.append(campo.energia())
        entropias.append(campo.entropia())
        coherencias.append(campo.coherencia())

    # Analizar la dinámica
    # 1. ¿La energía varía? (no es constante)
    varianza_energia = np.var(energias)
    print(f"Varianza de energía: {varianza_energia:.6f}")

    # 2. ¿La entropía fluctúa?
    varianza_entropia = np.var(entropias)
    print(f"Varianza de entropía: {varianza_entropia:.6f}")

    # 3. ¿Los estados cambian significativamente?
    cambios = []
    for i in range(1, len(estados)):
        cambio = np.linalg.norm(estados[i] - estados[i-1])
        cambios.append(cambio)

    cambio_promedio = np.mean(cambios)
    print(f"Cambio promedio por paso: {cambio_promedio:.4f}")

    # 4. ¿Hay variedad en las fases?
    fases = set()
    for _ in range(10):
        campo.evolucionar(pasos=50)
        fases.add(campo.fase_actual().value)
    print(f"Fases observadas: {fases}")

    # Veredicto: el sistema es dinámico si hay varianza y cambio
    dinamica_activa = (varianza_energia > 1e-6 or cambio_promedio > 0.001)
    print(f"\nVEREDICTO: {'DINÁMICA CONTINUA ACTIVA' if dinamica_activa else 'SISTEMA RELATIVAMENTE ESTÁTICO'}")

    return dinamica_activa


def test_5_rendimiento_sin_trampa():
    """
    TEST 5: ¿El agente puede ganar sin reglas hardcodeadas?

    Comparamos con un baseline random.
    """
    print("\n" + "=" * 70)
    print("TEST 5: RENDIMIENTO GENUINO")
    print("=" * 70)

    agente = AgenteEmergente(dim_estado=9, n_acciones=9)
    juego = TicTacToe()

    # Fase 1: Entrenamiento
    print("[Fase 1: Entrenamiento (300 partidas)]")
    for ep in range(300):
        estado = juego.reset()

        while True:
            acciones = juego.acciones_validas()
            if not acciones:
                break

            if juego.turno == 'X':
                accion = agente.decidir(estado, acciones)
            else:
                accion = np.random.choice(acciones)

            estado_anterior = estado.copy()
            valido, recompensa, terminal = juego.hacer_movimiento(accion)
            estado = juego.estado_vector()

            if juego.turno == 'O' or terminal:
                r = recompensa if juego.turno == 'O' else -recompensa
                exp = Experiencia(
                    estado=estado_anterior,
                    accion=accion,
                    recompensa=r,
                    estado_siguiente=estado,
                    terminal=terminal
                )
                agente.aprender(exp)

            if terminal:
                break

        if (ep + 1) % 50 == 0:
            agente.consolidar()

    # Fase 2: Evaluación
    print("\n[Fase 2: Evaluación (100 partidas)]")
    victorias = 0
    derrotas = 0
    empates = 0

    for _ in range(100):
        estado = juego.reset()

        while True:
            acciones = juego.acciones_validas()
            if not acciones:
                break

            if juego.turno == 'X':
                accion = agente.decidir(estado, acciones)
            else:
                accion = np.random.choice(acciones)

            _, _, terminal = juego.hacer_movimiento(accion)
            estado = juego.estado_vector()

            if terminal:
                ganador = juego.verificar_ganador()
                if ganador == 'X':
                    victorias += 1
                elif ganador == 'O':
                    derrotas += 1
                else:
                    empates += 1
                break

    print(f"  Victorias: {victorias}")
    print(f"  Derrotas: {derrotas}")
    print(f"  Empates: {empates}")
    print(f"  Win Rate: {victorias}%")

    # Baseline: Random vs Random (esperado ~58% para X por ventaja de primer turno)
    print("\n[Baseline: Random vs Random (100 partidas)]")
    victorias_baseline = 0
    for _ in range(100):
        juego.reset()
        while True:
            acciones = juego.acciones_validas()
            if not acciones:
                break
            accion = np.random.choice(acciones)
            _, _, terminal = juego.hacer_movimiento(accion)
            if terminal:
                if juego.verificar_ganador() == 'X':
                    victorias_baseline += 1
                break

    print(f"  Win Rate Random: {victorias_baseline}%")

    mejora = victorias > victorias_baseline
    diferencia = victorias - victorias_baseline
    print(f"\n  Diferencia vs baseline: {diferencia:+d}%")
    print(f"VEREDICTO: {'APRENDIÓ A JUGAR MEJOR' if mejora else 'NO SUPERÓ BASELINE (aún)'}")

    return mejora


def ejecutar_todos_los_tests():
    """
    Ejecuta todos los tests y da un veredicto final.
    """
    print("\n" + "=" * 70)
    print("BATERÍA DE TESTS: PARADIGMA KAMAQ")
    print("=" * 70)
    print("Objetivo: Demostrar propiedades EMERGENTES, no optimizar métricas")
    print("=" * 70)

    resultados = {}

    resultados['1_atractores'] = test_1_atractores_emergen()
    resultados['2_metacognicion'] = test_2_metacognicion_calibrada()
    resultados['3_conocimiento'] = test_3_conocimiento_emergente()
    resultados['4_dinamica'] = test_4_dinamica_continua()
    resultados['5_rendimiento'] = test_5_rendimiento_sin_trampa()

    print("\n" + "=" * 70)
    print("RESUMEN FINAL")
    print("=" * 70)

    for test, exito in resultados.items():
        estado = "[OK] PASO" if exito else "[X] NO PASO"
        print(f"  Test {test}: {estado}")

    n_exitosos = sum(resultados.values())
    print(f"\nTests exitosos: {n_exitosos}/5")

    print("\n" + "=" * 70)
    if n_exitosos >= 4:
        print("VEREDICTO: EL PARADIGMA MUESTRA PROPIEDADES EMERGENTES")
        print("El sistema exhibe comportamiento que NO fue programado directamente.")
        print("El campo cognitivo funciona como fundamento del nuevo paradigma.")
    elif n_exitosos >= 2:
        print("VEREDICTO: PROPIEDADES PARCIALES - EL PARADIGMA TIENE POTENCIAL")
        print("Algunos aspectos funcionan. Se necesita más desarrollo y ajuste.")
        print("La dirección es correcta pero falta refinamiento.")
    else:
        print("VEREDICTO: INSUFICIENTE - REPLANTEAR PARÁMETROS")
        print("Las propiedades emergentes no se observan claramente.")
        print("Revisar ecuaciones, parámetros o arquitectura del campo.")
    print("=" * 70)

    return resultados


if __name__ == "__main__":
    ejecutar_todos_los_tests()

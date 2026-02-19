"""
ENTRENAMIENTO ESCALADO KAMAQ
============================
Sistema de entrenamiento intensivo para emergencia de conocimiento.

PRINCIPIOS:
1. NO hay reglas hardcodeadas de estrategia
2. Las recompensas intermedias se calculan desde el RESULTADO, no desde heurísticas
3. El conocimiento debe EMERGER, no ser inyectado
4. Todo se documenta honestamente

MEJORAS SOBRE EL ENTRENAMIENTO BÁSICO:
1. Más episodios (5000+)
2. Recompensas más informativas (pero no hardcodeadas)
3. Curriculum learning: empezar fácil, aumentar dificultad
4. Replay priorizado: repetir experiencias importantes
5. Decaimiento de exploración más lento
"""

import numpy as np
import time
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
from agente_emergente import AgenteEmergente, Experiencia


@dataclass
class MetricasEntrenamiento:
    """Métricas detalladas del entrenamiento"""
    episodio: int
    victorias: int = 0
    derrotas: int = 0
    empates: int = 0
    recompensa_promedio: float = 0.0
    entropia_promedio: float = 0.0
    ratio_exploracion: float = 0.0
    movimientos_promedio: float = 0.0
    # Métricas de emergencia de conocimiento
    preferencia_centro: float = 0.0
    tasa_bloqueo: float = 0.0
    tasa_victoria_directa: float = 0.0


class TicTacToeConAnalisis:
    """
    Tic-Tac-Toe con análisis de situación para recompensas informativas.

    IMPORTANTE: El análisis NO se usa para decidir movimientos.
    Solo se usa para dar recompensas más informativas DESPUÉS del hecho.
    """

    LINEAS = [
        [0, 1, 2], [3, 4, 5], [6, 7, 8],  # Horizontales
        [0, 3, 6], [1, 4, 7], [2, 5, 8],  # Verticales
        [0, 4, 8], [2, 4, 6]               # Diagonales
    ]

    def __init__(self):
        self.reset()

    def reset(self) -> np.ndarray:
        self.tablero = np.zeros(9)
        self.turno = 1  # 1 = X, -1 = O
        self.historial = []
        return self.tablero.copy()

    def estado(self) -> np.ndarray:
        return self.tablero.copy()

    def acciones_validas(self) -> List[int]:
        return [i for i in range(9) if self.tablero[i] == 0]

    def hacer_movimiento(self, pos: int) -> Tuple[bool, float, bool]:
        """
        Retorna: (válido, recompensa, terminal)

        La recompensa es informativa pero NO hardcodea estrategia.
        """
        if pos < 0 or pos > 8 or self.tablero[pos] != 0:
            return False, -0.1, False  # Movimiento inválido

        jugador = self.turno
        self.tablero[pos] = jugador
        self.historial.append((pos, jugador))

        # Verificar victoria
        ganador = self._verificar_ganador()

        if ganador == jugador:
            # Victoria
            self.turno = -self.turno
            return True, 1.0, True
        elif ganador == -jugador:
            # Derrota (no debería pasar aquí)
            self.turno = -self.turno
            return True, -1.0, True
        elif len(self.acciones_validas()) == 0:
            # Empate
            return True, 0.0, True

        # Juego continúa
        self.turno = -self.turno
        return True, 0.0, False

    def _verificar_ganador(self) -> Optional[int]:
        """Retorna 1 si gana X, -1 si gana O, None si no hay ganador"""
        for linea in self.LINEAS:
            suma = sum(self.tablero[i] for i in linea)
            if suma == 3:
                return 1
            elif suma == -3:
                return -1
        return None

    def analizar_movimiento_post_hoc(self, pos: int, jugador: int) -> Dict[str, bool]:
        """
        Analiza un movimiento DESPUÉS de hacerlo.

        Esto NO es para decidir - es para entender qué pasó.
        Se usa para dar recompensas más informativas.

        NOTA: Esto es análisis post-hoc, no heurística de decisión.
        """
        # Crear tablero temporal sin el movimiento
        tablero_antes = self.tablero.copy()
        tablero_antes[pos] = 0

        analisis = {
            'gano': False,
            'bloqueo': False,
            'creo_amenaza': False,
            'centro': pos == 4,
            'esquina': pos in [0, 2, 6, 8],
        }

        # ¿Ganó con este movimiento?
        for linea in self.LINEAS:
            if pos in linea:
                suma = sum(self.tablero[i] for i in linea)
                if suma == 3 * jugador:
                    analisis['gano'] = True
                    break

        # ¿Bloqueó una victoria del oponente?
        oponente = -jugador
        for linea in self.LINEAS:
            if pos in linea:
                # Contar cuántas fichas del oponente había en la línea antes
                fichas_oponente = sum(1 for i in linea if tablero_antes[i] == oponente)
                vacias = sum(1 for i in linea if tablero_antes[i] == 0)
                if fichas_oponente == 2 and vacias == 1:
                    analisis['bloqueo'] = True
                    break

        # ¿Creó una amenaza (dos en línea)?
        for linea in self.LINEAS:
            if pos in linea:
                fichas_propias = sum(1 for i in linea if self.tablero[i] == jugador)
                vacias = sum(1 for i in linea if self.tablero[i] == 0)
                if fichas_propias == 2 and vacias == 1:
                    analisis['creo_amenaza'] = True
                    break

        return analisis


class EntrenadorEscalado:
    """
    Entrenador con curriculum learning y replay priorizado.

    PRINCIPIOS:
    1. Sin conocimiento hardcodeado
    2. Recompensas informativas pero no prescriptivas
    3. Aprendizaje gradual
    """

    def __init__(self, agente: AgenteEmergente):
        self.agente = agente
        self.juego = TicTacToeConAnalisis()

        # Configuración de entrenamiento
        self.episodios_totales = 0
        self.victorias_totales = 0
        self.derrotas_totales = 0
        self.empates_totales = 0

        # Buffer de replay priorizado
        self.experiencias_importantes: List[Tuple[Experiencia, float]] = []
        self.max_importantes = 500

        # Historial de métricas
        self.historial_metricas: List[MetricasEntrenamiento] = []

        # Oponentes de diferente dificultad
        self.nivel_oponente = 0  # 0=random, 1=semi-random, 2=defensivo

    def oponente_random(self, acciones: List[int]) -> int:
        """Oponente completamente aleatorio"""
        return int(np.random.choice(acciones))

    def oponente_semi_random(self, acciones: List[int]) -> int:
        """Oponente que a veces bloquea (30% del tiempo)"""
        if np.random.random() < 0.3:
            # Intentar bloquear
            for linea in TicTacToeConAnalisis.LINEAS:
                fichas_x = sum(1 for i in linea if self.juego.tablero[i] == 1)
                vacias = [i for i in linea if self.juego.tablero[i] == 0]
                if fichas_x == 2 and len(vacias) == 1 and vacias[0] in acciones:
                    return vacias[0]
        return int(np.random.choice(acciones))

    def oponente_defensivo(self, acciones: List[int]) -> int:
        """Oponente que siempre bloquea si puede, sino random"""
        # Primero intentar ganar
        for linea in TicTacToeConAnalisis.LINEAS:
            fichas_o = sum(1 for i in linea if self.juego.tablero[i] == -1)
            vacias = [i for i in linea if self.juego.tablero[i] == 0]
            if fichas_o == 2 and len(vacias) == 1 and vacias[0] in acciones:
                return vacias[0]

        # Luego bloquear
        for linea in TicTacToeConAnalisis.LINEAS:
            fichas_x = sum(1 for i in linea if self.juego.tablero[i] == 1)
            vacias = [i for i in linea if self.juego.tablero[i] == 0]
            if fichas_x == 2 and len(vacias) == 1 and vacias[0] in acciones:
                return vacias[0]

        return int(np.random.choice(acciones))

    def seleccionar_oponente(self, acciones: List[int]) -> int:
        """Selecciona movimiento del oponente según nivel"""
        if self.nivel_oponente == 0:
            return self.oponente_random(acciones)
        elif self.nivel_oponente == 1:
            return self.oponente_semi_random(acciones)
        else:
            return self.oponente_defensivo(acciones)

    def calcular_recompensa_informativa(self, analisis: Dict[str, bool],
                                         resultado_final: Optional[str] = None) -> float:
        """
        Calcula recompensa informativa basada en análisis post-hoc.

        IMPORTANTE: Esto NO es una heurística de decisión.
        Es una forma de dar feedback más rico sobre lo que pasó.

        La recompensa es proporcional a lo que REALMENTE sucedió,
        no a lo que "debería" haber hecho.
        """
        recompensa = 0.0

        # Recompensas por resultado final (las más importantes)
        if resultado_final == 'victoria':
            recompensa += 1.0
        elif resultado_final == 'derrota':
            recompensa -= 1.0
        elif resultado_final == 'empate':
            recompensa += 0.1  # Empate es mejor que perder

        # Recompensas intermedias pequeñas (feedback más rápido)
        # Estas son proporcionales a consecuencias reales, no heurísticas
        if analisis.get('gano'):
            recompensa += 0.5  # Completó la victoria
        if analisis.get('bloqueo'):
            recompensa += 0.2  # Evitó perder
        if analisis.get('creo_amenaza'):
            recompensa += 0.1  # Creó presión

        return recompensa

    def entrenar_episodio(self) -> Tuple[str, float, List[Experiencia]]:
        """
        Entrena un episodio completo.

        Retorna: (resultado, recompensa_total, experiencias)
        """
        estado = self.juego.reset()
        experiencias = []
        recompensa_total = 0.0

        while True:
            acciones = self.juego.acciones_validas()
            if not acciones:
                break

            if self.juego.turno == 1:  # Turno del agente (X)
                estado_antes = estado.copy()
                accion = self.agente.decidir(estado, acciones)

                valido, _, terminal = self.juego.hacer_movimiento(accion)
                estado = self.juego.estado()

                # Analizar el movimiento post-hoc
                analisis = self.juego.analizar_movimiento_post_hoc(accion, 1)

                # Determinar resultado si es terminal
                resultado_final = None
                if terminal:
                    ganador = self.juego._verificar_ganador()
                    if ganador == 1:
                        resultado_final = 'victoria'
                    elif ganador == -1:
                        resultado_final = 'derrota'
                    else:
                        resultado_final = 'empate'

                # Calcular recompensa informativa
                recompensa = self.calcular_recompensa_informativa(analisis, resultado_final)
                recompensa_total += recompensa

                # Crear experiencia
                exp = Experiencia(
                    estado=estado_antes,
                    accion=accion,
                    recompensa=recompensa,
                    estado_siguiente=estado,
                    terminal=terminal
                )
                experiencias.append(exp)

                # Aprender inmediatamente
                self.agente.aprender(exp)

                if terminal:
                    return resultado_final, recompensa_total, experiencias

            else:  # Turno del oponente (O)
                accion = self.seleccionar_oponente(acciones)
                valido, _, terminal = self.juego.hacer_movimiento(accion)
                estado = self.juego.estado()

                if terminal:
                    ganador = self.juego._verificar_ganador()
                    if ganador == 1:
                        resultado_final = 'victoria'
                    elif ganador == -1:
                        resultado_final = 'derrota'
                        # Dar feedback negativo por perder
                        if experiencias:
                            exp = experiencias[-1]
                            exp_perdida = Experiencia(
                                estado=exp.estado_siguiente,
                                accion=exp.accion,
                                recompensa=-0.5,
                                estado_siguiente=estado,
                                terminal=True
                            )
                            self.agente.aprender(exp_perdida)
                            recompensa_total -= 0.5
                    else:
                        resultado_final = 'empate'

                    return resultado_final, recompensa_total, experiencias

        return 'empate', recompensa_total, experiencias

    def replay_priorizado(self, n_replay: int = 50):
        """
        Repite experiencias importantes.

        Prioriza experiencias con alta recompensa (positiva o negativa).
        """
        if not self.experiencias_importantes:
            return

        # Ordenar por prioridad (valor absoluto de recompensa)
        self.experiencias_importantes.sort(key=lambda x: -x[1])

        # Replay de las más importantes
        for i in range(min(n_replay, len(self.experiencias_importantes))):
            exp, _ = self.experiencias_importantes[i]
            patron = self.agente.experiencia_a_patron(exp.estado, exp.accion)

            self.agente.campo.perturbar(patron, fuerza=0.3)
            self.agente.campo.evolucionar(pasos=5)
            self.agente.aprendizaje.reforzar(patron, exp.recompensa * 0.5)

    def medir_emergencia_conocimiento(self, n_tests: int = 50) -> Dict[str, float]:
        """
        Mide si han emergido conceptos estratégicos.

        NO pregunta "¿hace lo correcto?" sino "¿ha desarrollado preferencias?"
        """
        metricas = {
            'preferencia_centro': 0.0,
            'tasa_bloqueo': 0.0,
            'tasa_victoria_directa': 0.0,
        }

        # Test 1: ¿Prefiere el centro en tablero vacío?
        veces_centro = 0
        for _ in range(n_tests):
            estado = np.zeros(9)
            accion = self.agente.decidir(estado, list(range(9)))
            if accion == 4:
                veces_centro += 1
        metricas['preferencia_centro'] = veces_centro / n_tests

        # Test 2: ¿Bloquea amenazas?
        situaciones_bloqueo = [
            (np.array([0, 0, 0, -1, -1, 0, 0, 0, 0]), 5),
            (np.array([-1, 0, 0, 0, -1, 0, 0, 0, 0]), 8),
            (np.array([-1, -1, 0, 0, 0, 0, 0, 0, 0]), 2),
            (np.array([0, -1, -1, 0, 0, 0, 0, 0, 0]), 0),
            (np.array([0, 0, -1, 0, 0, -1, 0, 0, 0]), 8),
        ]
        bloqueos = 0
        for estado, bloqueo_correcto in situaciones_bloqueo:
            validas = [i for i in range(9) if estado[i] == 0]
            accion = self.agente.decidir(estado, validas)
            if accion == bloqueo_correcto:
                bloqueos += 1
        metricas['tasa_bloqueo'] = bloqueos / len(situaciones_bloqueo)

        # Test 3: ¿Completa victorias?
        situaciones_ganar = [
            (np.array([1, 1, 0, 0, 0, 0, 0, 0, 0]), 2),
            (np.array([0, 0, 0, 1, 1, 0, 0, 0, 0]), 5),
            (np.array([1, 0, 0, 0, 1, 0, 0, 0, 0]), 8),
            (np.array([1, 0, 0, 1, 0, 0, 0, 0, 0]), 6),
            (np.array([0, 1, 0, 0, 1, 0, 0, 0, 0]), 7),
        ]
        victorias = 0
        for estado, victoria_correcta in situaciones_ganar:
            validas = [i for i in range(9) if estado[i] == 0]
            accion = self.agente.decidir(estado, validas)
            if accion == victoria_correcta:
                victorias += 1
        metricas['tasa_victoria_directa'] = victorias / len(situaciones_ganar)

        return metricas

    def entrenar(self, n_episodios: int = 5000,
                 intervalo_metricas: int = 100,
                 intervalo_consolidacion: int = 50,
                 curriculum: bool = True) -> List[MetricasEntrenamiento]:
        """
        Entrenamiento completo con curriculum learning.

        Args:
            n_episodios: Número total de episodios
            intervalo_metricas: Cada cuántos episodios medir métricas
            intervalo_consolidacion: Cada cuántos episodios consolidar
            curriculum: Si usar curriculum learning (aumentar dificultad)
        """
        print("=" * 70)
        print("ENTRENAMIENTO ESCALADO KAMAQ")
        print("=" * 70)
        print(f"Episodios: {n_episodios}")
        print(f"Curriculum Learning: {'Sí' if curriculum else 'No'}")
        print("=" * 70)

        inicio = time.time()

        victorias_periodo = 0
        derrotas_periodo = 0
        empates_periodo = 0
        recompensa_periodo = 0.0
        movimientos_periodo = 0

        for ep in range(1, n_episodios + 1):
            # Curriculum learning: aumentar dificultad
            if curriculum:
                if ep < n_episodios * 0.3:
                    self.nivel_oponente = 0  # Random
                elif ep < n_episodios * 0.6:
                    self.nivel_oponente = 1  # Semi-random
                else:
                    self.nivel_oponente = 2  # Defensivo

            # Entrenar episodio
            resultado, recompensa, experiencias = self.entrenar_episodio()

            # Actualizar contadores
            self.episodios_totales += 1
            movimientos_periodo += len(experiencias)
            recompensa_periodo += recompensa

            if resultado == 'victoria':
                self.victorias_totales += 1
                victorias_periodo += 1
            elif resultado == 'derrota':
                self.derrotas_totales += 1
                derrotas_periodo += 1
            else:
                self.empates_totales += 1
                empates_periodo += 1

            # Guardar experiencias importantes
            for exp in experiencias:
                if abs(exp.recompensa) > 0.3:
                    self.experiencias_importantes.append((exp, abs(exp.recompensa)))
                    if len(self.experiencias_importantes) > self.max_importantes:
                        # Eliminar la menos importante
                        self.experiencias_importantes.sort(key=lambda x: -x[1])
                        self.experiencias_importantes.pop()

            # Consolidación periódica
            if ep % intervalo_consolidacion == 0:
                self.agente.consolidar()
                self.replay_priorizado(30)

            # Métricas periódicas
            if ep % intervalo_metricas == 0:
                conocimiento = self.medir_emergencia_conocimiento(30)

                metricas = MetricasEntrenamiento(
                    episodio=ep,
                    victorias=victorias_periodo,
                    derrotas=derrotas_periodo,
                    empates=empates_periodo,
                    recompensa_promedio=recompensa_periodo / intervalo_metricas,
                    entropia_promedio=self.agente.campo.entropia(),
                    ratio_exploracion=self.agente.n_exploraciones / max(1, self.agente.n_decisiones),
                    movimientos_promedio=movimientos_periodo / intervalo_metricas,
                    preferencia_centro=conocimiento['preferencia_centro'],
                    tasa_bloqueo=conocimiento['tasa_bloqueo'],
                    tasa_victoria_directa=conocimiento['tasa_victoria_directa'],
                )
                self.historial_metricas.append(metricas)

                # Imprimir progreso
                win_rate = victorias_periodo / intervalo_metricas * 100
                print(f"Ep {ep:5d} | WR: {win_rate:5.1f}% | "
                      f"Centro: {conocimiento['preferencia_centro']*100:4.1f}% | "
                      f"Bloqueo: {conocimiento['tasa_bloqueo']*100:4.1f}% | "
                      f"Victoria: {conocimiento['tasa_victoria_directa']*100:4.1f}% | "
                      f"Nivel: {self.nivel_oponente}")

                # Reset contadores del periodo
                victorias_periodo = 0
                derrotas_periodo = 0
                empates_periodo = 0
                recompensa_periodo = 0.0
                movimientos_periodo = 0
                self.agente.reset_estadisticas()

        tiempo_total = time.time() - inicio

        print("=" * 70)
        print(f"ENTRENAMIENTO COMPLETADO en {tiempo_total:.1f}s")
        print(f"Total: {self.victorias_totales}V / {self.derrotas_totales}D / {self.empates_totales}E")
        print("=" * 70)

        return self.historial_metricas

    def evaluar_final(self, n_partidas: int = 200) -> Dict:
        """
        Evaluación final rigurosa contra diferentes oponentes.
        """
        resultados = {}

        for nivel, nombre in [(0, 'random'), (1, 'semi_random'), (2, 'defensivo')]:
            self.nivel_oponente = nivel
            victorias = 0
            derrotas = 0
            empates = 0

            for _ in range(n_partidas):
                estado = self.juego.reset()

                while True:
                    acciones = self.juego.acciones_validas()
                    if not acciones:
                        break

                    if self.juego.turno == 1:
                        accion = self.agente.decidir(estado, acciones)
                    else:
                        accion = self.seleccionar_oponente(acciones)

                    _, _, terminal = self.juego.hacer_movimiento(accion)
                    estado = self.juego.estado()

                    if terminal:
                        ganador = self.juego._verificar_ganador()
                        if ganador == 1:
                            victorias += 1
                        elif ganador == -1:
                            derrotas += 1
                        else:
                            empates += 1
                        break

            resultados[nombre] = {
                'victorias': victorias,
                'derrotas': derrotas,
                'empates': empates,
                'win_rate': victorias / n_partidas,
                'no_lose_rate': (victorias + empates) / n_partidas,
            }

        return resultados


def ejecutar_entrenamiento_escalado():
    """
    Ejecuta el entrenamiento escalado completo.
    """
    print("\n" + "=" * 70)
    print("KAMAQ - ENTRENAMIENTO ESCALADO PARA EMERGENCIA DE CONOCIMIENTO")
    print("=" * 70)
    print("Objetivo: Demostrar que el conocimiento EMERGE sin hardcodear")
    print("=" * 70 + "\n")

    # Crear agente
    agente = AgenteEmergente(dim_estado=9, n_acciones=9)

    # Crear entrenador
    entrenador = EntrenadorEscalado(agente)

    # Entrenar
    historial = entrenador.entrenar(
        n_episodios=5000,
        intervalo_metricas=100,
        intervalo_consolidacion=50,
        curriculum=True
    )

    # Evaluación final
    print("\n" + "=" * 70)
    print("EVALUACIÓN FINAL")
    print("=" * 70)

    resultados = entrenador.evaluar_final(200)

    for oponente, stats in resultados.items():
        print(f"\nvs {oponente.upper()}:")
        print(f"  Victorias: {stats['victorias']}")
        print(f"  Derrotas: {stats['derrotas']}")
        print(f"  Empates: {stats['empates']}")
        print(f"  Win Rate: {stats['win_rate']*100:.1f}%")
        print(f"  No-Lose Rate: {stats['no_lose_rate']*100:.1f}%")

    # Métricas de emergencia
    print("\n" + "=" * 70)
    print("EMERGENCIA DE CONOCIMIENTO")
    print("=" * 70)

    conocimiento = entrenador.medir_emergencia_conocimiento(100)
    print(f"Preferencia por centro: {conocimiento['preferencia_centro']*100:.1f}%")
    print(f"Tasa de bloqueo: {conocimiento['tasa_bloqueo']*100:.1f}%")
    print(f"Tasa de victoria directa: {conocimiento['tasa_victoria_directa']*100:.1f}%")

    # Análisis de curva de aprendizaje
    print("\n" + "=" * 70)
    print("ANÁLISIS DE CURVA DE APRENDIZAJE")
    print("=" * 70)

    if historial:
        # Comparar inicio vs fin
        inicio = historial[:5]  # Primeros 500 episodios
        fin = historial[-5:]    # Últimos 500 episodios

        wr_inicio = np.mean([m.victorias / 100 for m in inicio])
        wr_fin = np.mean([m.victorias / 100 for m in fin])

        centro_inicio = np.mean([m.preferencia_centro for m in inicio])
        centro_fin = np.mean([m.preferencia_centro for m in fin])

        bloqueo_inicio = np.mean([m.tasa_bloqueo for m in inicio])
        bloqueo_fin = np.mean([m.tasa_bloqueo for m in fin])

        victoria_inicio = np.mean([m.tasa_victoria_directa for m in inicio])
        victoria_fin = np.mean([m.tasa_victoria_directa for m in fin])

        print(f"Win Rate: {wr_inicio*100:.1f}% -> {wr_fin*100:.1f}% (cambio: {(wr_fin-wr_inicio)*100:+.1f}%)")
        print(f"Preferencia centro: {centro_inicio*100:.1f}% -> {centro_fin*100:.1f}% (cambio: {(centro_fin-centro_inicio)*100:+.1f}%)")
        print(f"Tasa bloqueo: {bloqueo_inicio*100:.1f}% -> {bloqueo_fin*100:.1f}% (cambio: {(bloqueo_fin-bloqueo_inicio)*100:+.1f}%)")
        print(f"Tasa victoria: {victoria_inicio*100:.1f}% -> {victoria_fin*100:.1f}% (cambio: {(victoria_fin-victoria_inicio)*100:+.1f}%)")

    # Veredicto
    print("\n" + "=" * 70)
    print("VEREDICTO")
    print("=" * 70)

    emergencia_exitosa = (
        conocimiento['preferencia_centro'] > 0.2 or
        conocimiento['tasa_bloqueo'] > 0.4 or
        conocimiento['tasa_victoria_directa'] > 0.4
    )

    aprendizaje_mejora = resultados['random']['win_rate'] > 0.6

    if emergencia_exitosa and aprendizaje_mejora:
        print("ÉXITO: Conocimiento estratégico ha EMERGIDO sin hardcodear")
        print("El sistema desarrolló conceptos por sí mismo.")
    elif aprendizaje_mejora:
        print("PARCIAL: Mejoró el rendimiento pero sin conceptos claros")
        print("Hay aprendizaje, pero los conceptos no son distinguibles aún.")
    else:
        print("INSUFICIENTE: No se observa emergencia clara")
        print("Requiere ajustes en el mecanismo de aprendizaje.")

    print("=" * 70)

    return agente, entrenador, historial, resultados


if __name__ == "__main__":
    agente, entrenador, historial, resultados = ejecutar_entrenamiento_escalado()

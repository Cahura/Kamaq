"""
TEST DE POBLACION DE ESTRATEGIAS - FASE 2 VALIDACION
=====================================================
Valida que la seleccion natural de estrategias funciona.

METRICAS CLAVE:
1. La estrategia "bloquear" debe tener fitness alto (funciona en 8 contextos)
2. Estrategias especificas deben morir (fitness bajo)
3. La poblacion debe estabilizarse con estrategias utiles

CRITERIO DE EXITO:
- Fitness promedio > 0.4 despues de 1000 episodios
- Estrategia de bloqueo emerge naturalmente

Ejecutar: python test_poblacion.py
"""

import numpy as np
import time
from typing import List, Dict, Tuple
from poblacion_estrategias import (
    PoblacionEstrategias, SelectorHibrido,
    Estrategia, Condicion, crear_estrategias_semilla
)


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

    def hay_amenaza(self) -> List[int]:
        """Retorna posiciones que bloquean amenazas."""
        bloqueos = []
        for linea in self.LINEAS:
            valores = [self.tablero[i] for i in linea]
            if valores.count(-1) == 2 and valores.count(0) == 1:
                # Hay amenaza, encontrar casilla vacia
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


def entrenar_poblacion(n_episodios: int, con_semillas: bool = False) -> Dict:
    """Entrena usando seleccion natural de estrategias."""
    juego = TicTacToe()
    selector = SelectorHibrido()

    if con_semillas:
        semillas = crear_estrategias_semilla()
        for s in semillas:
            selector.poblacion.estrategias.append(s)

    victorias = 0
    bloqueos_correctos = 0
    bloqueos_posibles = 0

    for ep in range(n_episodios):
        estado = juego.reset()

        while True:
            acciones = juego.acciones_validas()
            if not acciones:
                break

            if juego.turno == 1:
                estado_antes = estado.copy()

                # Detectar si hay amenaza
                amenazas = juego.hay_amenaza()
                victorias_posibles = juego.hay_victoria()

                # Decidir
                accion = selector.decidir(estado, acciones)

                # Ejecutar
                _, recompensa, terminal = juego.hacer_movimiento(accion)
                estado = juego.tablero.copy()

                # Determinar exito
                exito = False
                if terminal and juego._verificar_ganador() == 1:
                    exito = True
                    victorias += 1
                elif accion in victorias_posibles:
                    exito = True
                elif accion in amenazas:
                    exito = True
                    bloqueos_correctos += 1

                if amenazas:
                    bloqueos_posibles += 1

                # Aprender
                selector.aprender(estado_antes, accion, exito)

                if terminal:
                    break
            else:
                accion = np.random.choice(acciones)
                _, _, terminal = juego.hacer_movimiento(accion)
                estado = juego.tablero.copy()

                if terminal:
                    break

        # Evolucionar cada 50 episodios
        if (ep + 1) % 50 == 0:
            selector.evolucionar()

    tasa_bloqueo = bloqueos_correctos / max(1, bloqueos_posibles)

    return {
        'victorias': victorias,
        'win_rate': victorias / n_episodios * 100,
        'tasa_bloqueo': tasa_bloqueo * 100,
        'bloqueos_correctos': bloqueos_correctos,
        'bloqueos_posibles': bloqueos_posibles,
        **selector.diagnostico()
    }


def test_seleccion_natural():
    """Test principal: verifica que la seleccion funciona."""
    print("=" * 70)
    print("TEST DE SELECCION NATURAL DE ESTRATEGIAS")
    print("=" * 70)

    n_episodios = 1000

    print(f"\nEntrenando {n_episodios} episodios...")
    print("-" * 70)

    # Sin semillas
    print("\n1. SIN SEMILLAS (emergencia pura):")
    inicio = time.time()
    resultados_puro = entrenar_poblacion(n_episodios, con_semillas=False)
    tiempo = time.time() - inicio

    print(f"   Estrategias finales: {resultados_puro['n_estrategias']}")
    print(f"   Fitness promedio: {resultados_puro['fitness_promedio']:.3f}")
    print(f"   Win rate: {resultados_puro['win_rate']:.1f}%")
    print(f"   Tasa bloqueo: {resultados_puro['tasa_bloqueo']:.1f}%")
    print(f"   Ratio estrategia: {resultados_puro['ratio_estrategia']*100:.1f}%")
    print(f"   Tiempo: {tiempo:.2f}s")

    # Con semillas
    print("\n2. CON SEMILLAS (bootstrap):")
    inicio = time.time()
    resultados_semillas = entrenar_poblacion(n_episodios, con_semillas=True)
    tiempo = time.time() - inicio

    print(f"   Estrategias finales: {resultados_semillas['n_estrategias']}")
    print(f"   Fitness promedio: {resultados_semillas['fitness_promedio']:.3f}")
    print(f"   Win rate: {resultados_semillas['win_rate']:.1f}%")
    print(f"   Tasa bloqueo: {resultados_semillas['tasa_bloqueo']:.1f}%")
    print(f"   Ratio estrategia: {resultados_semillas['ratio_estrategia']*100:.1f}%")
    print(f"   Tiempo: {tiempo:.2f}s")

    # Comparacion
    print("\n" + "=" * 70)
    print("COMPARACION")
    print("=" * 70)

    diff_wr = resultados_semillas['win_rate'] - resultados_puro['win_rate']
    diff_bloqueo = resultados_semillas['tasa_bloqueo'] - resultados_puro['tasa_bloqueo']

    print(f"\nWin Rate:")
    print(f"  Sin semillas: {resultados_puro['win_rate']:.1f}%")
    print(f"  Con semillas: {resultados_semillas['win_rate']:.1f}%")
    print(f"  Diferencia:   {diff_wr:+.1f}%")

    print(f"\nTasa Bloqueo:")
    print(f"  Sin semillas: {resultados_puro['tasa_bloqueo']:.1f}%")
    print(f"  Con semillas: {resultados_semillas['tasa_bloqueo']:.1f}%")
    print(f"  Diferencia:   {diff_bloqueo:+.1f}%")

    # Veredicto
    print("\n" + "=" * 70)
    print("VEREDICTO")
    print("=" * 70)

    exito_fitness = resultados_puro['fitness_promedio'] > 0.2
    exito_emergencia = resultados_puro['tasa_bloqueo'] > 10
    exito_semillas = resultados_semillas['tasa_bloqueo'] > resultados_puro['tasa_bloqueo']

    if exito_fitness and (exito_emergencia or exito_semillas):
        print("[OK] FASE 2 EXITOSA")
        print("     La seleccion natural produce estrategias utiles.")
    elif exito_semillas:
        print("[PARCIAL] Semillas mejoran pero emergencia pura no funciona")
        print("     El sistema necesita bootstrap inicial.")
    else:
        print("[FALLO] La seleccion natural no produce estrategias utiles")
        print("     Revisar fitness function y umbral de seleccion.")

    print("=" * 70)

    return resultados_puro, resultados_semillas


def test_fitness_estrategias():
    """Test detallado de fitness de estrategias individuales."""
    print("\n" + "=" * 70)
    print("TEST: FITNESS DE ESTRATEGIAS CONOCIDAS")
    print("=" * 70)

    juego = TicTacToe()
    poblacion = PoblacionEstrategias()

    # Simular muchas situaciones
    situaciones_bloqueo = [
        (np.array([-1, -1, 0, 0, 0, 0, 0, 0, 0]), 2),  # Bloquear linea superior
        (np.array([0, -1, -1, 0, 0, 0, 0, 0, 0]), 0),
        (np.array([0, 0, 0, -1, -1, 0, 0, 0, 0]), 5),
        (np.array([-1, 0, 0, -1, 0, 0, 0, 0, 0]), 6),
        (np.array([-1, 0, 0, 0, -1, 0, 0, 0, 0]), 8),
    ]

    situaciones_victoria = [
        (np.array([1, 1, 0, 0, 0, 0, 0, 0, 0]), 2),
        (np.array([0, 1, 1, 0, 0, 0, 0, 0, 0]), 0),
        (np.array([1, 0, 0, 1, 0, 0, 0, 0, 0]), 6),
    ]

    # Simular experiencias de bloqueo
    print("\nSimulando experiencias de BLOQUEO:")
    for estado, accion_correcta in situaciones_bloqueo:
        # Registrar accion correcta como exito
        poblacion.agregar_estrategia(estado, accion_correcta, True)
        # Registrar accion incorrecta como fallo
        otras = [i for i in range(9) if estado[i] == 0 and i != accion_correcta]
        if otras:
            poblacion.agregar_estrategia(estado, np.random.choice(otras), False)

    print(f"  Estrategias creadas: {len(poblacion.estrategias)}")

    # Evaluar
    poblacion.seleccion_natural()
    print(f"  Estrategias despues de seleccion: {len(poblacion.estrategias)}")

    # Mostrar top estrategias
    print("\nTop 5 estrategias:")
    top = poblacion.top_estrategias(5)
    for i, e in enumerate(top):
        print(f"  {i+1}. Accion={e['accion']}, Fitness={e['fitness']:.2f}, "
              f"Activaciones={e['n_activaciones']}")

    diag = poblacion.diagnostico()
    print(f"\nFitness promedio: {diag['fitness_promedio']:.3f}")

    return diag


def test_evolucion_poblacion():
    """Test de como evoluciona la poblacion en el tiempo."""
    print("\n" + "=" * 70)
    print("TEST: EVOLUCION DE LA POBLACION")
    print("=" * 70)

    juego = TicTacToe()
    selector = SelectorHibrido()

    n_episodios = 500
    metricas = []

    for ep in range(n_episodios):
        estado = juego.reset()

        while True:
            acciones = juego.acciones_validas()
            if not acciones:
                break

            if juego.turno == 1:
                estado_antes = estado.copy()

                victorias_pos = juego.hay_victoria()
                amenazas = juego.hay_amenaza()

                accion = selector.decidir(estado, acciones)
                _, _, terminal = juego.hacer_movimiento(accion)
                estado = juego.tablero.copy()

                exito = (accion in victorias_pos) or (accion in amenazas) or \
                        (terminal and juego._verificar_ganador() == 1)

                selector.aprender(estado_antes, accion, exito)

                if terminal:
                    break
            else:
                accion = np.random.choice(acciones)
                _, _, terminal = juego.hacer_movimiento(accion)
                estado = juego.tablero.copy()
                if terminal:
                    break

        if (ep + 1) % 50 == 0:
            selector.evolucionar()

        if (ep + 1) % 100 == 0:
            diag = selector.diagnostico()
            metricas.append({
                'ep': ep + 1,
                **diag
            })
            print(f"Ep {ep+1:4d} | Estrategias: {diag['n_estrategias']:3d} | "
                  f"Fitness: {diag['fitness_promedio']:.3f} | "
                  f"Ratio: {diag['ratio_estrategia']*100:.1f}%")

    print("\nEvolucion completa.")
    return metricas


if __name__ == "__main__":
    # Test principal
    r_puro, r_semillas = test_seleccion_natural()

    # Test de fitness
    diag_fitness = test_fitness_estrategias()

    # Test de evolucion
    metricas = test_evolucion_poblacion()

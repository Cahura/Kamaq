"""
EVALUACION KAMAQ-ARC V2
========================
Usa el motor de transformaciones basado en primitivas.

ESTRATEGIAS:
1. Transformaciones geometricas (rotar, reflejar, escalar)
2. Cambios de color
3. Deteccion de patrones (tile, extraer)
"""

import numpy as np
import time
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass

from arc_loader import ARCDataset, ARCTask, visualize_grid
from motor_transformaciones import MotorTransformacionesARC


@dataclass
class ResultadoTarea:
    """Resultado de una tarea individual."""
    task_id: str
    correcto: bool
    accuracy_pixeles: float
    transformacion: Optional[str]
    candidatas: List[str]
    tiempo_ms: float
    error_msg: Optional[str] = None


class EvaluadorARCv2:
    """Evaluador usando motor de transformaciones."""

    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.motor = MotorTransformacionesARC()
        self.resultados: List[ResultadoTarea] = []

    def evaluar_tarea(self, task: ARCTask) -> ResultadoTarea:
        """Evalua una tarea."""
        inicio = time.time()

        try:
            # Preparar ejemplos
            train_examples = [
                (ex.input_grid, ex.output_grid)
                for ex in task.train_examples
            ]

            test_ex = task.test_examples[0]
            test_input = test_ex.input_grid
            test_output = test_ex.output_grid

            if test_output is None:
                return ResultadoTarea(
                    task_id=task.task_id,
                    correcto=False,
                    accuracy_pixeles=0.0,
                    transformacion=None,
                    candidatas=[],
                    tiempo_ms=0,
                    error_msg="No ground truth"
                )

            # Evaluar
            resultado = self.motor.evaluar_tarea(
                train_examples=train_examples,
                test_input=test_input,
                test_output=test_output
            )

            tiempo_ms = (time.time() - inicio) * 1000

            return ResultadoTarea(
                task_id=task.task_id,
                correcto=resultado['correcto'],
                accuracy_pixeles=resultado['accuracy_pixeles'],
                transformacion=resultado['transformacion'],
                candidatas=resultado['candidatas'],
                tiempo_ms=tiempo_ms
            )

        except Exception as e:
            return ResultadoTarea(
                task_id=task.task_id,
                correcto=False,
                accuracy_pixeles=0.0,
                transformacion=None,
                candidatas=[],
                tiempo_ms=(time.time() - inicio) * 1000,
                error_msg=str(e)
            )

    def evaluar_dataset(self, tasks: Dict[str, ARCTask],
                        max_tasks: Optional[int] = None) -> Dict:
        """Evalua el dataset."""
        task_list = list(tasks.values())
        if max_tasks:
            task_list = task_list[:max_tasks]

        self.resultados = []
        inicio = time.time()

        for i, task in enumerate(task_list):
            if self.verbose and (i % 20 == 0 or i < 5):
                print(f"  [{i+1}/{len(task_list)}] {task.task_id}")

            res = self.evaluar_tarea(task)
            self.resultados.append(res)

        tiempo_total = time.time() - inicio

        # Calcular metricas
        n_correctas = sum(1 for r in self.resultados if r.correcto)
        accuracy_pixeles = np.mean([r.accuracy_pixeles for r in self.resultados])

        # Contar transformaciones detectadas
        trans_counts = {}
        for r in self.resultados:
            if r.transformacion:
                trans_counts[r.transformacion] = trans_counts.get(r.transformacion, 0) + 1

        return {
            'n_tareas': len(task_list),
            'n_correctas': n_correctas,
            'accuracy_exacta': n_correctas / len(task_list),
            'accuracy_pixeles_promedio': accuracy_pixeles,
            'tiempo_total_s': tiempo_total,
            'transformaciones': trans_counts,
        }

    def mostrar_ejemplos(self, n_correctas: int = 5, n_incorrectas: int = 5):
        """Muestra ejemplos de predicciones."""
        correctas = [r for r in self.resultados if r.correcto]
        incorrectas = [r for r in self.resultados if not r.correcto]

        print(f"\n--- CORRECTAS ({len(correctas)} total) ---")
        for r in correctas[:n_correctas]:
            print(f"  {r.task_id}: {r.transformacion}")

        print(f"\n--- INCORRECTAS (muestra de {len(incorrectas)}) ---")
        for r in incorrectas[:n_incorrectas]:
            trans = r.transformacion or "ninguna"
            print(f"  {r.task_id}: acc={r.accuracy_pixeles*100:.1f}%, trans={trans}")
            if r.candidatas:
                print(f"    candidatas: {r.candidatas}")


def generar_reporte_v2(resultados: Dict, output_path: Optional[Path] = None) -> str:
    """Genera reporte de evaluacion."""
    lines = []
    lines.append("=" * 70)
    lines.append("EVALUACION KAMAQ-ARC V2 - MOTOR DE TRANSFORMACIONES")
    lines.append("=" * 70)
    lines.append(f"\nFecha: {time.strftime('%Y-%m-%d %H:%M:%S')}")

    lines.append("\n" + "-" * 70)
    lines.append("RESULTADOS")
    lines.append("-" * 70)
    lines.append(f"  Tareas evaluadas: {resultados['n_tareas']}")
    lines.append(f"  Tareas correctas: {resultados['n_correctas']}")
    lines.append(f"  Accuracy exacta:  {resultados['accuracy_exacta']*100:.2f}%")
    lines.append(f"  Accuracy pixeles: {resultados['accuracy_pixeles_promedio']*100:.2f}%")
    lines.append(f"  Tiempo total:     {resultados['tiempo_total_s']:.2f}s")

    lines.append("\n" + "-" * 70)
    lines.append("TRANSFORMACIONES DETECTADAS")
    lines.append("-" * 70)
    for trans, count in sorted(resultados['transformaciones'].items(),
                               key=lambda x: -x[1]):
        lines.append(f"  {trans}: {count}")

    lines.append("\n" + "-" * 70)
    lines.append("COMPARACION CON BASELINES")
    lines.append("-" * 70)
    acc = resultados['accuracy_exacta'] * 100
    lines.append(f"  KAMAQ-ARC v2: {acc:.2f}%")
    lines.append(f"  Random:       <1%")
    lines.append(f"  GPT-4:        ~13%")
    lines.append(f"  Humanos:      ~85%")

    lines.append("\n" + "-" * 70)
    lines.append("INTERPRETACION")
    lines.append("-" * 70)
    if acc < 1:
        lines.append("  [FALLO] No detecta transformaciones basicas")
    elif acc < 5:
        lines.append("  [MINIMO] Detecta algunas transformaciones simples")
    elif acc < 10:
        lines.append("  [PARCIAL] Detecta transformaciones geometricas")
    elif acc < 15:
        lines.append("  [COMPARABLE] Similar a GPT-4 vanilla")
    elif acc < 25:
        lines.append("  [PROMETEDOR] Supera LLMs sin fine-tuning")
    else:
        lines.append("  [EXITO] Rendimiento significativo")

    lines.append("\n" + "=" * 70)

    texto = "\n".join(lines)

    if output_path:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(texto)

    return texto


if __name__ == "__main__":
    print("=" * 70)
    print("EVALUACION KAMAQ-ARC V2")
    print("Motor de Transformaciones Primitivas")
    print("=" * 70)

    # Cargar dataset
    data_dir = Path(__file__).parent.parent / "arc_dataset" / "data"

    print("\nCargando dataset...")
    dataset = ARCDataset(data_dir)
    dataset.load_all()

    if not dataset.training_tasks:
        print("ERROR: No se encontraron tareas")
        exit(1)

    # Evaluacion
    evaluador = EvaluadorARCv2(verbose=True)

    # Fase 1: Rapida
    print("\n" + "-" * 70)
    print("[Fase 1] Evaluacion rapida (50 tareas)")
    print("-" * 70)

    resultados_50 = evaluador.evaluar_dataset(dataset.training_tasks, max_tasks=50)

    print(f"\nResultados (50 tareas):")
    print(f"  Correctas: {resultados_50['n_correctas']}/{resultados_50['n_tareas']}")
    print(f"  Accuracy exacta: {resultados_50['accuracy_exacta']*100:.2f}%")
    print(f"  Accuracy pixeles: {resultados_50['accuracy_pixeles_promedio']*100:.2f}%")

    evaluador.mostrar_ejemplos(n_correctas=10, n_incorrectas=5)

    # Fase 2: Completa (si hay exitos)
    if resultados_50['n_correctas'] > 0:
        print("\n" + "-" * 70)
        print("[Fase 2] Evaluacion completa (400 tareas)")
        print("-" * 70)

        resultados_completo = evaluador.evaluar_dataset(dataset.training_tasks)

        reporte = generar_reporte_v2(
            resultados_completo,
            Path(__file__).parent / "RESULTADOS_V2.md"
        )
        print(reporte)
    else:
        # Generar reporte parcial
        reporte = generar_reporte_v2(
            resultados_50,
            Path(__file__).parent / "RESULTADOS_V2.md"
        )
        print(reporte)

    print("\nReporte guardado en: arc_challenge/RESULTADOS_V2.md")

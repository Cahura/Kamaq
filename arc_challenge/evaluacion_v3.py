"""
EVALUACION KAMAQ-ARC V3
========================
Usa el motor avanzado con:
- Mas transformaciones
- Composicion de transformaciones
- Analisis de patrones

OBJETIVO: Superar 5% (mejor que V2), acercarse a 10%
"""

import numpy as np
import time
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass

from arc_loader import ARCDataset, ARCTask
from motor_avanzado import MotorAvanzadoARC


@dataclass
class ResultadoTarea:
    task_id: str
    correcto: bool
    accuracy_pixeles: float
    transformacion: Optional[List[str]]
    tiempo_ms: float
    error: Optional[str] = None


class EvaluadorARCv3:
    """Evaluador con motor avanzado."""

    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.motor = MotorAvanzadoARC()
        self.resultados: List[ResultadoTarea] = []

    def evaluar_tarea(self, task: ARCTask) -> ResultadoTarea:
        inicio = time.time()

        try:
            train = [(ex.input_grid, ex.output_grid) for ex in task.train_examples]
            test_ex = task.test_examples[0]

            if test_ex.output_grid is None:
                return ResultadoTarea(
                    task_id=task.task_id,
                    correcto=False,
                    accuracy_pixeles=0.0,
                    transformacion=None,
                    tiempo_ms=0,
                    error="No ground truth"
                )

            res = self.motor.evaluar_tarea(train, test_ex.input_grid, test_ex.output_grid)

            return ResultadoTarea(
                task_id=task.task_id,
                correcto=res['correcto'],
                accuracy_pixeles=res['accuracy_pixeles'],
                transformacion=res['transformacion'],
                tiempo_ms=(time.time() - inicio) * 1000
            )

        except Exception as e:
            return ResultadoTarea(
                task_id=task.task_id,
                correcto=False,
                accuracy_pixeles=0.0,
                transformacion=None,
                tiempo_ms=(time.time() - inicio) * 1000,
                error=str(e)
            )

    def evaluar_dataset(self, tasks: Dict[str, ARCTask],
                        max_tasks: Optional[int] = None) -> Dict:
        task_list = list(tasks.values())
        if max_tasks:
            task_list = task_list[:max_tasks]

        self.resultados = []
        inicio = time.time()

        for i, task in enumerate(task_list):
            if self.verbose and (i % 50 == 0 or i < 5):
                print(f"  [{i+1}/{len(task_list)}] {task.task_id}")

            self.resultados.append(self.evaluar_tarea(task))

        # Calcular metricas
        n_correctas = sum(1 for r in self.resultados if r.correcto)
        acc_pixeles = np.mean([r.accuracy_pixeles for r in self.resultados])

        # Contar transformaciones
        trans_counts = {}
        for r in self.resultados:
            if r.transformacion:
                key = '->'.join(r.transformacion)
                trans_counts[key] = trans_counts.get(key, 0) + 1

        return {
            'n_tareas': len(task_list),
            'n_correctas': n_correctas,
            'accuracy_exacta': n_correctas / len(task_list),
            'accuracy_pixeles': acc_pixeles,
            'tiempo_s': time.time() - inicio,
            'transformaciones': trans_counts,
        }


def generar_reporte(resultados: Dict, output_path: Optional[Path] = None) -> str:
    lines = []
    lines.append("=" * 70)
    lines.append("EVALUACION KAMAQ-ARC V3 - MOTOR AVANZADO")
    lines.append("=" * 70)
    lines.append(f"\nFecha: {time.strftime('%Y-%m-%d %H:%M:%S')}")

    lines.append("\n" + "-" * 70)
    lines.append("RESULTADOS PRINCIPALES")
    lines.append("-" * 70)
    lines.append(f"  Tareas evaluadas:    {resultados['n_tareas']}")
    lines.append(f"  Tareas correctas:    {resultados['n_correctas']}")
    lines.append(f"  ACCURACY EXACTA:     {resultados['accuracy_exacta']*100:.2f}%")
    lines.append(f"  Accuracy pixeles:    {resultados['accuracy_pixeles']*100:.2f}%")
    lines.append(f"  Tiempo total:        {resultados['tiempo_s']:.2f}s")

    lines.append("\n" + "-" * 70)
    lines.append("TRANSFORMACIONES EXITOSAS (top 20)")
    lines.append("-" * 70)
    sorted_trans = sorted(resultados['transformaciones'].items(), key=lambda x: -x[1])
    for trans, count in sorted_trans[:20]:
        lines.append(f"  {trans}: {count}")

    lines.append("\n" + "-" * 70)
    lines.append("COMPARACION CON BENCHMARKS")
    lines.append("-" * 70)
    acc = resultados['accuracy_exacta'] * 100
    lines.append(f"  KAMAQ-ARC V3:   {acc:.2f}%")
    lines.append(f"  Random:         <1%")
    lines.append(f"  KAMAQ V2:       3%")
    lines.append(f"  GPT-4 vanilla:  ~13%")
    lines.append(f"  Humanos:        ~85%")

    # Analisis
    lines.append("\n" + "-" * 70)
    lines.append("ANALISIS")
    lines.append("-" * 70)

    if acc < 3:
        lines.append("  [REGRESION] Peor que V2")
    elif acc < 5:
        lines.append("  [ESTABLE] Similar a V2")
    elif acc < 10:
        lines.append("  [MEJORA] Progreso significativo")
    elif acc < 15:
        lines.append("  [EXITO] Comparable a GPT-4")
    else:
        lines.append("  [EXCEPCIONAL] Supera GPT-4")

    # Desglose por tipo de transformacion
    n_simples = sum(1 for r in sorted_trans if '->' not in r[0])
    n_compuestas = sum(1 for r in sorted_trans if '->' in r[0])
    lines.append(f"\n  Transformaciones simples:    {n_simples}")
    lines.append(f"  Transformaciones compuestas: {n_compuestas}")

    lines.append("\n" + "=" * 70)

    texto = "\n".join(lines)
    if output_path:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(texto)

    return texto


if __name__ == "__main__":
    print("=" * 70)
    print("EVALUACION KAMAQ-ARC V3")
    print("Motor Avanzado con Composicion")
    print("=" * 70)

    # Cargar dataset
    data_dir = Path(__file__).parent.parent / "arc_dataset" / "data"

    print("\nCargando dataset...")
    dataset = ARCDataset(data_dir)
    dataset.load_all()

    if not dataset.training_tasks:
        print("ERROR: No hay tareas")
        exit(1)

    evaluador = EvaluadorARCv3(verbose=True)

    # Evaluacion completa
    print("\n" + "-" * 70)
    print("Evaluando 400 tareas de entrenamiento...")
    print("-" * 70)

    resultados = evaluador.evaluar_dataset(dataset.training_tasks)

    # Mostrar ejemplos correctos
    correctas = [r for r in evaluador.resultados if r.correcto]
    print(f"\n--- TAREAS RESUELTAS ({len(correctas)}) ---")
    for r in correctas[:20]:
        trans = '->'.join(r.transformacion) if r.transformacion else 'N/A'
        print(f"  {r.task_id}: {trans}")

    # Generar reporte
    reporte = generar_reporte(
        resultados,
        Path(__file__).parent / "RESULTADOS_V3.md"
    )
    print(reporte)

    print("\nReporte guardado en: arc_challenge/RESULTADOS_V3.md")

    # Evaluar tambien en evaluation set si hay exito
    if resultados['n_correctas'] > 10:
        print("\n" + "=" * 70)
        print("Evaluando en EVALUATION set (400 tareas)...")
        print("=" * 70)

        resultados_eval = evaluador.evaluar_dataset(dataset.evaluation_tasks)

        print(f"\nResultados Evaluation Set:")
        print(f"  Correctas: {resultados_eval['n_correctas']}/{resultados_eval['n_tareas']}")
        print(f"  Accuracy:  {resultados_eval['accuracy_exacta']*100:.2f}%")

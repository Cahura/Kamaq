"""
EVALUACION KAMAQ-ARC
====================
Script para evaluar el sistema KAMAQ en tareas ARC reales.

METRICAS:
- Accuracy exacta: prediccion == ground truth (todo el grid)
- Accuracy por pixel: % de pixeles correctos
- Accuracy por forma: tamaño de output correcto

BASELINES:
- Random: <1%
- GPT-4 vanilla: ~13%
- Humanos: ~85%

OBJETIVO HONESTO: Superar 10% seria prometedor
"""

import numpy as np
import time
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
import json

# Importar nuestros modulos
from arc_loader import ARCDataset, ARCTask, ARCExample, visualize_grid
from campo_arc import MotorARC, EncoderARC, DetectorObjetos, analizar_tarea


@dataclass
class ResultadoTarea:
    """Resultado de evaluar una tarea."""
    task_id: str
    correcto: bool  # Prediccion exacta
    accuracy_pixeles: float  # % pixeles correctos
    forma_correcta: bool  # Tamaño de output correcto
    tiempo_ms: float
    prediccion: np.ndarray
    ground_truth: np.ndarray
    error_msg: Optional[str] = None


@dataclass
class ResultadosGlobales:
    """Resultados agregados de evaluacion."""
    n_tareas: int
    n_correctas: int
    accuracy_exacta: float
    accuracy_pixeles_promedio: float
    n_forma_correcta: int
    tiempo_total_s: float
    tiempo_promedio_ms: float

    def to_dict(self) -> Dict:
        return {
            'n_tareas': self.n_tareas,
            'n_correctas': self.n_correctas,
            'accuracy_exacta': f"{self.accuracy_exacta*100:.2f}%",
            'accuracy_pixeles_promedio': f"{self.accuracy_pixeles_promedio*100:.2f}%",
            'n_forma_correcta': self.n_forma_correcta,
            'tiempo_total_s': f"{self.tiempo_total_s:.2f}s",
            'tiempo_promedio_ms': f"{self.tiempo_promedio_ms:.2f}ms",
        }


class EvaluadorARC:
    """Evaluador del sistema KAMAQ en ARC."""

    def __init__(self, verbose: bool = True):
        self.verbose = verbose
        self.motor = MotorARC()
        self.resultados: List[ResultadoTarea] = []

    def evaluar_tarea(self, task: ARCTask) -> ResultadoTarea:
        """Evalua una tarea individual."""
        inicio = time.time()

        try:
            # Preparar ejemplos de entrenamiento
            train_examples = [
                (ex.input_grid, ex.output_grid)
                for ex in task.train_examples
            ]

            # Obtener test
            test_ex = task.test_examples[0]  # Usamos primer test
            test_input = test_ex.input_grid
            test_output = test_ex.output_grid

            if test_output is None:
                return ResultadoTarea(
                    task_id=task.task_id,
                    correcto=False,
                    accuracy_pixeles=0.0,
                    forma_correcta=False,
                    tiempo_ms=0,
                    prediccion=np.zeros((1,1)),
                    ground_truth=np.zeros((1,1)),
                    error_msg="No ground truth disponible"
                )

            # Evaluar con nuestro motor
            resultado = self.motor.evaluar_en_tarea(
                train_examples=train_examples,
                test_input=test_input,
                test_output=test_output
            )

            tiempo_ms = (time.time() - inicio) * 1000

            prediccion = resultado['prediccion']
            forma_correcta = prediccion.shape == test_output.shape

            return ResultadoTarea(
                task_id=task.task_id,
                correcto=resultado['correcto'],
                accuracy_pixeles=resultado['accuracy_pixeles'],
                forma_correcta=forma_correcta,
                tiempo_ms=tiempo_ms,
                prediccion=prediccion,
                ground_truth=test_output
            )

        except Exception as e:
            tiempo_ms = (time.time() - inicio) * 1000
            return ResultadoTarea(
                task_id=task.task_id,
                correcto=False,
                accuracy_pixeles=0.0,
                forma_correcta=False,
                tiempo_ms=tiempo_ms,
                prediccion=np.zeros((1,1)),
                ground_truth=np.zeros((1,1)) if test_ex.output_grid is None else test_ex.output_grid,
                error_msg=str(e)
            )

    def evaluar_dataset(self, tasks: Dict[str, ARCTask],
                        max_tasks: Optional[int] = None) -> ResultadosGlobales:
        """Evalua multiples tareas."""
        task_list = list(tasks.values())
        if max_tasks:
            task_list = task_list[:max_tasks]

        self.resultados = []
        n_correctas = 0
        total_accuracy_pixeles = 0
        n_forma_correcta = 0

        inicio_total = time.time()

        for i, task in enumerate(task_list):
            if self.verbose and (i % 10 == 0 or i < 5):
                print(f"  Evaluando tarea {i+1}/{len(task_list)}: {task.task_id}")

            resultado = self.evaluar_tarea(task)
            self.resultados.append(resultado)

            if resultado.correcto:
                n_correctas += 1
            total_accuracy_pixeles += resultado.accuracy_pixeles
            if resultado.forma_correcta:
                n_forma_correcta += 1

        tiempo_total = time.time() - inicio_total
        n_tareas = len(task_list)

        return ResultadosGlobales(
            n_tareas=n_tareas,
            n_correctas=n_correctas,
            accuracy_exacta=n_correctas / n_tareas if n_tareas > 0 else 0,
            accuracy_pixeles_promedio=total_accuracy_pixeles / n_tareas if n_tareas > 0 else 0,
            n_forma_correcta=n_forma_correcta,
            tiempo_total_s=tiempo_total,
            tiempo_promedio_ms=(tiempo_total * 1000) / n_tareas if n_tareas > 0 else 0
        )

    def mostrar_ejemplos(self, n: int = 5):
        """Muestra ejemplos de predicciones."""
        print("\n" + "="*70)
        print("EJEMPLOS DE PREDICCIONES")
        print("="*70)

        # Mostrar algunas correctas y algunas incorrectas
        correctas = [r for r in self.resultados if r.correcto]
        incorrectas = [r for r in self.resultados if not r.correcto]

        print(f"\n--- PREDICCIONES CORRECTAS ({len(correctas)} total) ---")
        for r in correctas[:min(n, len(correctas))]:
            print(f"\nTarea: {r.task_id}")
            print(f"Accuracy pixeles: {r.accuracy_pixeles*100:.1f}%")
            print(f"Tiempo: {r.tiempo_ms:.1f}ms")

        print(f"\n--- PREDICCIONES INCORRECTAS ({len(incorrectas)} total) ---")
        for r in incorrectas[:min(n, len(incorrectas))]:
            print(f"\nTarea: {r.task_id}")
            print(f"Accuracy pixeles: {r.accuracy_pixeles*100:.1f}%")
            print(f"Forma correcta: {r.forma_correcta}")
            if r.error_msg:
                print(f"Error: {r.error_msg}")

            # Mostrar comparacion visual
            if r.prediccion.size < 100 and r.ground_truth.size < 100:
                print("\nPrediccion:")
                print(visualize_grid(r.prediccion))
                print("\nGround Truth:")
                print(visualize_grid(r.ground_truth))


def baseline_random(task: ARCTask) -> np.ndarray:
    """Baseline: prediccion aleatoria."""
    test_input = task.test_examples[0].input_grid
    test_output = task.test_examples[0].output_grid

    # Usar misma forma que output (si disponible) o input
    shape = test_output.shape if test_output is not None else test_input.shape

    # Generar grid aleatorio
    return np.random.randint(0, 10, size=shape, dtype=np.int8)


def evaluar_baseline_random(tasks: Dict[str, ARCTask],
                           max_tasks: Optional[int] = None) -> float:
    """Evalua el baseline random."""
    task_list = list(tasks.values())
    if max_tasks:
        task_list = task_list[:max_tasks]

    n_correctas = 0

    for task in task_list:
        test_output = task.test_examples[0].output_grid
        if test_output is None:
            continue

        prediccion = baseline_random(task)
        if prediccion.shape == test_output.shape:
            if np.array_equal(prediccion, test_output):
                n_correctas += 1

    return n_correctas / len(task_list) if task_list else 0


def generar_reporte(resultados: ResultadosGlobales,
                    baseline_acc: float,
                    output_path: Optional[Path] = None) -> str:
    """Genera reporte de evaluacion."""

    reporte = []
    reporte.append("="*70)
    reporte.append("REPORTE DE EVALUACION KAMAQ-ARC")
    reporte.append("="*70)
    reporte.append(f"\nFecha: {time.strftime('%Y-%m-%d %H:%M:%S')}")

    reporte.append("\n" + "-"*70)
    reporte.append("RESULTADOS PRINCIPALES")
    reporte.append("-"*70)

    for k, v in resultados.to_dict().items():
        reporte.append(f"  {k}: {v}")

    reporte.append("\n" + "-"*70)
    reporte.append("COMPARACION CON BASELINES")
    reporte.append("-"*70)

    reporte.append(f"  Random baseline: {baseline_acc*100:.4f}%")
    reporte.append(f"  KAMAQ-ARC:       {resultados.accuracy_exacta*100:.2f}%")
    reporte.append(f"  GPT-4 (ref):     ~13%")
    reporte.append(f"  Humanos (ref):   ~85%")

    # Interpretacion
    reporte.append("\n" + "-"*70)
    reporte.append("INTERPRETACION HONESTA")
    reporte.append("-"*70)

    if resultados.accuracy_exacta < 0.01:
        reporte.append("  [FALLO] Rendimiento a nivel de random")
        reporte.append("  El sistema no aprende nada util de los ejemplos")
    elif resultados.accuracy_exacta < 0.10:
        reporte.append("  [INSUFICIENTE] Por debajo de GPT-4")
        reporte.append("  Necesita mejoras fundamentales")
    elif resultados.accuracy_exacta < 0.20:
        reporte.append("  [PROMETEDOR] Comparable a LLMs vanilla")
        reporte.append("  El paradigma tiene potencial")
    elif resultados.accuracy_exacta < 0.40:
        reporte.append("  [EXITO] Supera LLMs sin fine-tuning")
        reporte.append("  Demostrado valor del enfoque")
    else:
        reporte.append("  [HISTORICO] Rendimiento excepcional")
        reporte.append("  Cerca de nivel humano")

    reporte.append("\n" + "="*70)

    texto = "\n".join(reporte)

    if output_path:
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(texto)

    return texto


if __name__ == "__main__":
    print("="*70)
    print("EVALUACION KAMAQ-ARC")
    print("="*70)

    # Cargar dataset
    data_dir = Path(__file__).parent.parent / "arc_dataset" / "data"

    print("\nCargando dataset ARC...")
    dataset = ARCDataset(data_dir)
    dataset.load_all()

    if len(dataset.training_tasks) == 0:
        print("ERROR: No se encontraron tareas. Verifica la ruta del dataset.")
        print(f"Buscando en: {data_dir}")
        exit(1)

    # Estadisticas del dataset
    stats = dataset.statistics()
    print("\nEstadisticas del dataset:")
    for k, v in stats.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.2f}")
        else:
            print(f"  {k}: {v}")

    # Evaluar baseline random
    print("\n" + "-"*70)
    print("Evaluando baseline random...")
    baseline_acc = evaluar_baseline_random(dataset.training_tasks, max_tasks=100)
    print(f"Baseline random accuracy: {baseline_acc*100:.4f}%")

    # Evaluar KAMAQ-ARC
    print("\n" + "-"*70)
    print("Evaluando KAMAQ-ARC...")
    print("-"*70)

    evaluador = EvaluadorARC(verbose=True)

    # Empezar con pocas tareas para ver si funciona
    print("\n[Fase 1] Evaluacion rapida (20 tareas)...")
    resultados_rapido = evaluador.evaluar_dataset(
        dataset.training_tasks,
        max_tasks=20
    )

    print("\nResultados rapidos:")
    for k, v in resultados_rapido.to_dict().items():
        print(f"  {k}: {v}")

    # Mostrar ejemplos
    evaluador.mostrar_ejemplos(n=3)

    # Si hay al menos una correcta, evaluar mas
    if resultados_rapido.n_correctas > 0 or resultados_rapido.accuracy_pixeles_promedio > 0.3:
        print("\n[Fase 2] Evaluacion extendida (100 tareas)...")
        resultados_extendido = evaluador.evaluar_dataset(
            dataset.training_tasks,
            max_tasks=100
        )

        # Generar reporte final
        reporte = generar_reporte(
            resultados_extendido,
            baseline_acc,
            output_path=Path(__file__).parent / "RESULTADOS_EVALUACION.md"
        )
        print(reporte)
    else:
        # Generar reporte con resultados rapidos
        reporte = generar_reporte(
            resultados_rapido,
            baseline_acc,
            output_path=Path(__file__).parent / "RESULTADOS_EVALUACION.md"
        )
        print(reporte)

    print("\nReporte guardado en: arc_challenge/RESULTADOS_EVALUACION.md")

"""
ARC DATASET LOADER
==================
Carga y visualiza tareas del ARC Challenge.

Estructura de una tarea ARC:
{
    "train": [
        {"input": [[...]], "output": [[...]]},
        {"input": [[...]], "output": [[...]]},
        {"input": [[...]], "output": [[...]]}
    ],
    "test": [
        {"input": [[...]], "output": [[...]]}  # output solo en training set
    ]
}

Colores ARC (0-9):
0: Negro (fondo)
1: Azul
2: Rojo
3: Verde
4: Amarillo
5: Gris
6: Fucsia
7: Naranja
8: Cyan
9: Marron
"""

import json
import os
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from pathlib import Path


# Colores ARC para visualizacion
ARC_COLORS = {
    0: '\033[40m  \033[0m',  # Negro
    1: '\033[44m  \033[0m',  # Azul
    2: '\033[41m  \033[0m',  # Rojo
    3: '\033[42m  \033[0m',  # Verde
    4: '\033[43m  \033[0m',  # Amarillo
    5: '\033[47m  \033[0m',  # Gris/Blanco
    6: '\033[45m  \033[0m',  # Magenta
    7: '\033[48;5;208m  \033[0m',  # Naranja
    8: '\033[46m  \033[0m',  # Cyan
    9: '\033[48;5;94m  \033[0m',  # Marron
}

# Version ASCII sin colores ANSI
ARC_SYMBOLS = {
    0: '. ',
    1: '1 ',
    2: '2 ',
    3: '3 ',
    4: '4 ',
    5: '5 ',
    6: '6 ',
    7: '7 ',
    8: '8 ',
    9: '9 ',
}


@dataclass
class ARCExample:
    """Un ejemplo de entrenamiento o test."""
    input_grid: np.ndarray
    output_grid: Optional[np.ndarray]

    @property
    def input_shape(self) -> Tuple[int, int]:
        return self.input_grid.shape

    @property
    def output_shape(self) -> Optional[Tuple[int, int]]:
        return self.output_grid.shape if self.output_grid is not None else None

    @property
    def colors_used(self) -> set:
        colors = set(self.input_grid.flatten())
        if self.output_grid is not None:
            colors.update(self.output_grid.flatten())
        return colors


@dataclass
class ARCTask:
    """Una tarea completa de ARC."""
    task_id: str
    train_examples: List[ARCExample]
    test_examples: List[ARCExample]

    @property
    def n_train(self) -> int:
        return len(self.train_examples)

    @property
    def n_test(self) -> int:
        return len(self.test_examples)

    def get_all_colors(self) -> set:
        colors = set()
        for ex in self.train_examples + self.test_examples:
            colors.update(ex.colors_used)
        return colors

    def get_grid_sizes(self) -> Dict:
        """Retorna estadisticas de tamanos de grids."""
        sizes = {
            'train_inputs': [ex.input_shape for ex in self.train_examples],
            'train_outputs': [ex.output_shape for ex in self.train_examples],
            'test_inputs': [ex.input_shape for ex in self.test_examples],
        }
        return sizes


class ARCDataset:
    """Cargador del dataset ARC."""

    def __init__(self, data_dir: str):
        self.data_dir = Path(data_dir)
        self.training_dir = self.data_dir / "training"
        self.evaluation_dir = self.data_dir / "evaluation"

        self.training_tasks: Dict[str, ARCTask] = {}
        self.evaluation_tasks: Dict[str, ARCTask] = {}

    def load_all(self):
        """Carga todas las tareas."""
        print("Cargando tareas de entrenamiento...")
        self._load_directory(self.training_dir, self.training_tasks)
        print(f"  {len(self.training_tasks)} tareas cargadas")

        print("Cargando tareas de evaluacion...")
        self._load_directory(self.evaluation_dir, self.evaluation_tasks)
        print(f"  {len(self.evaluation_tasks)} tareas cargadas")

    def _load_directory(self, directory: Path, target: Dict):
        """Carga todas las tareas de un directorio."""
        if not directory.exists():
            print(f"  Directorio no encontrado: {directory}")
            return

        for file_path in directory.glob("*.json"):
            task_id = file_path.stem
            task = self._load_task(file_path, task_id)
            if task:
                target[task_id] = task

    def _load_task(self, file_path: Path, task_id: str) -> Optional[ARCTask]:
        """Carga una tarea individual."""
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)

            train_examples = []
            for ex in data.get('train', []):
                input_grid = np.array(ex['input'], dtype=np.int8)
                output_grid = np.array(ex['output'], dtype=np.int8)
                train_examples.append(ARCExample(input_grid, output_grid))

            test_examples = []
            for ex in data.get('test', []):
                input_grid = np.array(ex['input'], dtype=np.int8)
                output_grid = np.array(ex.get('output'), dtype=np.int8) if 'output' in ex else None
                test_examples.append(ARCExample(input_grid, output_grid))

            return ARCTask(task_id, train_examples, test_examples)

        except Exception as e:
            print(f"  Error cargando {file_path}: {e}")
            return None

    def get_task(self, task_id: str) -> Optional[ARCTask]:
        """Obtiene una tarea por ID."""
        if task_id in self.training_tasks:
            return self.training_tasks[task_id]
        if task_id in self.evaluation_tasks:
            return self.evaluation_tasks[task_id]
        return None

    def get_random_task(self, from_training: bool = True) -> ARCTask:
        """Obtiene una tarea aleatoria."""
        tasks = self.training_tasks if from_training else self.evaluation_tasks
        task_id = np.random.choice(list(tasks.keys()))
        return tasks[task_id]

    def statistics(self) -> Dict:
        """Retorna estadisticas del dataset."""
        all_tasks = list(self.training_tasks.values()) + list(self.evaluation_tasks.values())

        grid_sizes = []
        n_colors = []
        n_train_examples = []

        for task in all_tasks:
            for ex in task.train_examples:
                grid_sizes.append(ex.input_shape[0] * ex.input_shape[1])
            n_colors.append(len(task.get_all_colors()))
            n_train_examples.append(task.n_train)

        return {
            'n_training_tasks': len(self.training_tasks),
            'n_evaluation_tasks': len(self.evaluation_tasks),
            'avg_grid_size': np.mean(grid_sizes) if grid_sizes else 0,
            'max_grid_size': np.max(grid_sizes) if grid_sizes else 0,
            'avg_colors': np.mean(n_colors) if n_colors else 0,
            'avg_train_examples': np.mean(n_train_examples) if n_train_examples else 0,
        }


def visualize_grid(grid: np.ndarray, use_color: bool = False) -> str:
    """Genera representacion visual de un grid."""
    lines = []
    symbols = ARC_COLORS if use_color else ARC_SYMBOLS

    for row in grid:
        line = ''
        for val in row:
            line += symbols.get(int(val), '? ')
        lines.append(line)

    return '\n'.join(lines)


def visualize_example(example: ARCExample, use_color: bool = False) -> str:
    """Visualiza un ejemplo input -> output."""
    result = []
    result.append("INPUT:")
    result.append(visualize_grid(example.input_grid, use_color))

    if example.output_grid is not None:
        result.append("\nOUTPUT:")
        result.append(visualize_grid(example.output_grid, use_color))

    return '\n'.join(result)


def visualize_task(task: ARCTask, use_color: bool = False) -> str:
    """Visualiza una tarea completa."""
    result = []
    result.append(f"{'='*60}")
    result.append(f"TAREA: {task.task_id}")
    result.append(f"Ejemplos de entrenamiento: {task.n_train}")
    result.append(f"Tests: {task.n_test}")
    result.append(f"Colores usados: {task.get_all_colors()}")
    result.append(f"{'='*60}")

    for i, ex in enumerate(task.train_examples):
        result.append(f"\n--- Ejemplo {i+1} ---")
        result.append(visualize_example(ex, use_color))

    for i, ex in enumerate(task.test_examples):
        result.append(f"\n--- Test {i+1} ---")
        result.append("INPUT:")
        result.append(visualize_grid(ex.input_grid, use_color))
        if ex.output_grid is not None:
            result.append("\nOUTPUT (ground truth):")
            result.append(visualize_grid(ex.output_grid, use_color))
        else:
            result.append("\nOUTPUT: ???")

    return '\n'.join(result)


def analizar_transformacion(example: ARCExample) -> Dict:
    """Analiza que tipo de transformacion ocurrio."""
    inp = example.input_grid
    out = example.output_grid

    if out is None:
        return {'type': 'unknown'}

    analysis = {
        'input_shape': inp.shape,
        'output_shape': out.shape,
        'same_shape': inp.shape == out.shape,
        'input_colors': set(inp.flatten()),
        'output_colors': set(out.flatten()),
    }

    # Detectar transformaciones comunes
    if analysis['same_shape']:
        # Comparar pixel a pixel
        diff = inp != out
        analysis['n_changed_pixels'] = np.sum(diff)
        analysis['change_ratio'] = np.sum(diff) / inp.size

        # Detectar si es solo cambio de color
        if analysis['input_colors'] != analysis['output_colors']:
            analysis['color_change'] = True

    # Detectar cambio de tamano
    if inp.shape != out.shape:
        analysis['shape_change'] = True
        if out.shape[0] > inp.shape[0] or out.shape[1] > inp.shape[1]:
            analysis['expansion'] = True
        else:
            analysis['contraction'] = True

    return analysis


if __name__ == "__main__":
    # Demo
    import sys

    # Ruta al dataset
    data_dir = Path(__file__).parent.parent / "arc_dataset" / "data"

    print("Cargando ARC Dataset...")
    dataset = ARCDataset(data_dir)
    dataset.load_all()

    # Estadisticas
    stats = dataset.statistics()
    print("\nEstadisticas del dataset:")
    for k, v in stats.items():
        print(f"  {k}: {v:.2f}" if isinstance(v, float) else f"  {k}: {v}")

    # Mostrar una tarea aleatoria
    print("\n" + "="*60)
    print("TAREA ALEATORIA:")
    print("="*60)

    task = dataset.get_random_task()
    print(visualize_task(task, use_color=False))

    # Analizar transformaciones
    print("\n" + "="*60)
    print("ANALISIS DE TRANSFORMACIONES:")
    print("="*60)

    for i, ex in enumerate(task.train_examples):
        print(f"\nEjemplo {i+1}:")
        analysis = analizar_transformacion(ex)
        for k, v in analysis.items():
            print(f"  {k}: {v}")

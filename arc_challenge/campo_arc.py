"""
CAMPO COGNITIVO PARA ARC
========================
Adapta la fisica de KAMAQ para resolver tareas ARC.

PRINCIPIO CENTRAL:
Las transformaciones ARC son ATRACTORES en un espacio de transformaciones.
El campo cognitivo converge al atractor correcto a partir de 3 ejemplos.

COMPONENTES:
1. Encoder: Grid ARC -> Campo Cognitivo
2. Espacio de Transformaciones: Donde viven las "reglas"
3. Motor de Inferencia: Encuentra la transformacion correcta
4. Decoder: Campo -> Grid ARC

DIFERENCIA CON V4:
- No hay acciones discretas, hay transformaciones continuas
- Los "objetos" en el grid son sub-atractores
- La sincronizacion de Kuramoto detecta patrones de color
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass
from collections import defaultdict
import scipy.ndimage as ndimage


@dataclass
class Objeto:
    """Un objeto detectado en el grid."""
    pixeles: List[Tuple[int, int]]  # Lista de (fila, columna)
    color: int
    bbox: Tuple[int, int, int, int]  # (min_row, min_col, max_row, max_col)
    mask: np.ndarray  # Mascara binaria del objeto

    @property
    def size(self) -> int:
        return len(self.pixeles)

    @property
    def centroid(self) -> Tuple[float, float]:
        rows = [p[0] for p in self.pixeles]
        cols = [p[1] for p in self.pixeles]
        return (np.mean(rows), np.mean(cols))

    @property
    def shape_signature(self) -> tuple:
        """Firma de forma normalizada (para comparar objetos)."""
        min_r, min_c, max_r, max_c = self.bbox
        h = max_r - min_r + 1
        w = max_c - min_c + 1
        # Normalizar a forma relativa
        rel_pixels = tuple(sorted([(p[0] - min_r, p[1] - min_c) for p in self.pixeles]))
        return (h, w, rel_pixels)


class DetectorObjetos:
    """
    Detecta objetos en grids ARC usando componentes conexos.

    Un "objeto" es un grupo de pixeles del mismo color conectados.
    El fondo (color 0) generalmente no se considera objeto.
    """

    def __init__(self, incluir_fondo: bool = False):
        self.incluir_fondo = incluir_fondo

    def detectar(self, grid: np.ndarray) -> List[Objeto]:
        """Detecta todos los objetos en el grid."""
        objetos = []
        colores_vistos = set()

        for color in range(10):
            if color == 0 and not self.incluir_fondo:
                continue

            mask = (grid == color)
            if not np.any(mask):
                continue

            # Encontrar componentes conexos
            labeled, n_components = ndimage.label(mask)

            for component_id in range(1, n_components + 1):
                component_mask = (labeled == component_id)
                pixeles = list(zip(*np.where(component_mask)))

                if pixeles:
                    rows = [p[0] for p in pixeles]
                    cols = [p[1] for p in pixeles]
                    bbox = (min(rows), min(cols), max(rows), max(cols))

                    obj = Objeto(
                        pixeles=pixeles,
                        color=color,
                        bbox=bbox,
                        mask=component_mask
                    )
                    objetos.append(obj)

        return objetos


class EncoderARC:
    """
    Codifica grids ARC en representacion para el campo cognitivo.

    Estrategia:
    1. Codificar cada celda como vector
    2. Codificar relaciones espaciales
    3. Codificar objetos detectados
    """

    def __init__(self, dim_celda: int = 16, max_grid_size: int = 30):
        self.dim_celda = dim_celda
        self.max_grid_size = max_grid_size
        self.detector = DetectorObjetos()

        # Dimension total del campo
        self.dim_campo = max_grid_size * max_grid_size * dim_celda

    def encode_grid(self, grid: np.ndarray) -> np.ndarray:
        """Codifica un grid completo."""
        h, w = grid.shape

        # Inicializar campo
        campo = np.zeros(self.dim_campo, dtype=np.float64)

        # Codificar cada celda
        for i in range(h):
            for j in range(w):
                color = int(grid[i, j])  # Asegurar que es int nativo
                color = min(max(color, 0), 9)  # Clamp a rango valido 0-9
                idx_base = (i * self.max_grid_size + j) * self.dim_celda

                if idx_base + self.dim_celda <= self.dim_campo:
                    # One-hot del color
                    campo[idx_base + color] = 1.0

                    # Posicion normalizada
                    campo[idx_base + 10] = i / max(h - 1, 1)  # Fila normalizada
                    campo[idx_base + 11] = j / max(w - 1, 1)  # Columna normalizada

                    # Es borde?
                    campo[idx_base + 12] = 1.0 if (i == 0 or i == h-1 or j == 0 or j == w-1) else 0.0

                    # Es centro?
                    centro_i, centro_j = h // 2, w // 2
                    dist_centro = np.sqrt((i - centro_i)**2 + (j - centro_j)**2)
                    campo[idx_base + 13] = 1.0 / (1.0 + dist_centro)

        # Normalizar
        norma = np.linalg.norm(campo)
        if norma > 1e-10:
            campo = campo / norma

        return campo

    def encode_objetos(self, grid: np.ndarray) -> Dict:
        """Codifica los objetos detectados."""
        objetos = self.detector.detectar(grid)

        encoding = {
            'n_objetos': len(objetos),
            'colores': [obj.color for obj in objetos],
            'tamanos': [obj.size for obj in objetos],
            'centroides': [obj.centroid for obj in objetos],
            'formas': [obj.shape_signature for obj in objetos],
        }

        return encoding

    def encode_diferencia(self, grid_in: np.ndarray, grid_out: np.ndarray) -> np.ndarray:
        """Codifica la diferencia entre input y output."""
        campo_in = self.encode_grid(grid_in)
        campo_out = self.encode_grid(grid_out)

        # La diferencia es la "transformacion"
        delta = campo_out - campo_in

        return delta


class DecoderARC:
    """
    Decodifica campo cognitivo a grid ARC.
    """

    def __init__(self, dim_celda: int = 16, max_grid_size: int = 30):
        self.dim_celda = dim_celda
        self.max_grid_size = max_grid_size

    def decode_grid(self, campo: np.ndarray, target_shape: Tuple[int, int]) -> np.ndarray:
        """Decodifica campo a grid."""
        h, w = target_shape
        grid = np.zeros((h, w), dtype=np.int32)  # Usar int32 para evitar overflow

        for i in range(h):
            for j in range(w):
                idx_base = (i * self.max_grid_size + j) * self.dim_celda

                if idx_base + 10 <= len(campo):
                    # Leer one-hot de color
                    color_probs = campo[idx_base:idx_base + 10]
                    color = int(np.argmax(color_probs))
                    grid[i, j] = min(max(color, 0), 9)  # Clamp a 0-9

        return grid.astype(np.int8)


class CampoTransformaciones:
    """
    Campo donde viven las transformaciones ARC.

    Cada transformacion es un atractor.
    Aprender = formar un nuevo atractor a partir de ejemplos.
    """

    def __init__(self, dim: int = 256):
        self.dim = dim

        # Matriz de acoplamiento (Hopfield)
        self.J = np.zeros((dim, dim))

        # Estado actual del campo
        self.estado = np.random.randn(dim) * 0.1

        # Transformaciones aprendidas
        self.transformaciones: List[np.ndarray] = []

        # Parametros de dinamica
        self.temperatura = 0.01
        self.gamma = 0.5

    def memorizar_transformacion(self, deltas: List[np.ndarray]):
        """
        Memoriza una transformacion a partir de los deltas de ejemplos.

        Los deltas de los ejemplos deben converger a un patron comun.
        """
        if not deltas:
            return

        # Promediar deltas y normalizar
        delta_promedio = np.mean(deltas, axis=0)

        # Ajustar dimension si es necesario
        if len(delta_promedio) != self.dim:
            # Proyectar a la dimension del campo
            if len(delta_promedio) > self.dim:
                delta_promedio = delta_promedio[:self.dim]
            else:
                delta_padded = np.zeros(self.dim)
                delta_padded[:len(delta_promedio)] = delta_promedio
                delta_promedio = delta_padded

        # Normalizar
        norma = np.linalg.norm(delta_promedio)
        if norma > 1e-10:
            delta_promedio = delta_promedio / norma

        # Regla de Hebb: memorizar como atractor
        delta_J = np.outer(delta_promedio, delta_promedio)
        np.fill_diagonal(delta_J, 0)
        self.J += delta_J

        # Limitar magnitud
        self.J = np.clip(self.J, -1.0, 1.0)

        self.transformaciones.append(delta_promedio.copy())

    def recuperar_transformacion(self, pista: np.ndarray, pasos: int = 100) -> np.ndarray:
        """
        Recupera la transformacion mas cercana a la pista.

        Usa dinamica de Hopfield para converger al atractor.
        """
        # Ajustar dimension
        if len(pista) != self.dim:
            if len(pista) > self.dim:
                pista = pista[:self.dim]
            else:
                pista_padded = np.zeros(self.dim)
                pista_padded[:len(pista)] = pista
                pista = pista_padded

        # Inicializar estado con la pista
        self.estado = pista.copy()

        # Evolucionar (Hopfield asincrono)
        for _ in range(pasos):
            orden = np.random.permutation(self.dim)
            cambios = 0

            for i in orden:
                h_i = np.dot(self.J[i], self.estado)
                nuevo = np.tanh(h_i / self.temperatura)

                if abs(nuevo - self.estado[i]) > 0.01:
                    self.estado[i] = nuevo
                    cambios += 1

            if cambios == 0:
                break

        return self.estado.copy()


class MotorARC:
    """
    Motor principal para resolver tareas ARC.

    Proceso:
    1. Codificar ejemplos de entrenamiento
    2. Extraer transformacion comun
    3. Aplicar transformacion al test
    4. Decodificar resultado
    """

    def __init__(self):
        self.encoder = EncoderARC()
        self.decoder = DecoderARC()
        self.campo = CampoTransformaciones()

    def aprender_de_ejemplos(self, ejemplos: List[Tuple[np.ndarray, np.ndarray]]):
        """
        Aprende la transformacion de los ejemplos.

        ejemplos: Lista de (input_grid, output_grid)
        """
        deltas = []

        for inp, out in ejemplos:
            delta = self.encoder.encode_diferencia(inp, out)
            deltas.append(delta)

        # Memorizar la transformacion
        self.campo.memorizar_transformacion(deltas)

    def predecir(self, test_input: np.ndarray, output_shape: Optional[Tuple[int, int]] = None) -> np.ndarray:
        """
        Predice el output para un input de test.
        """
        # Codificar input
        campo_in = self.encoder.encode_grid(test_input)

        # Recuperar transformacion
        if self.campo.transformaciones:
            transformacion = self.campo.transformaciones[-1]
        else:
            transformacion = np.zeros(self.campo.dim)

        # Aplicar transformacion
        campo_out = campo_in.copy()

        # Ajustar dimensiones
        min_len = min(len(campo_out), len(transformacion))
        campo_out[:min_len] += transformacion[:min_len]

        # Determinar forma del output
        if output_shape is None:
            output_shape = test_input.shape

        # Decodificar
        output_grid = self.decoder.decode_grid(campo_out, output_shape)

        return output_grid

    def evaluar_en_tarea(self, train_examples: List[Tuple[np.ndarray, np.ndarray]],
                        test_input: np.ndarray,
                        test_output: Optional[np.ndarray] = None) -> Dict:
        """
        Evalua en una tarea completa.
        """
        # Resetear campo
        self.campo = CampoTransformaciones()

        # Aprender de ejemplos
        self.aprender_de_ejemplos(train_examples)

        # Predecir
        output_shape = test_output.shape if test_output is not None else None
        prediccion = self.predecir(test_input, output_shape)

        # Evaluar si tenemos ground truth
        resultado = {
            'prediccion': prediccion,
            'correcto': False,
            'accuracy_pixeles': 0.0,
        }

        if test_output is not None:
            # Asegurar misma forma
            if prediccion.shape == test_output.shape:
                correcto = np.array_equal(prediccion, test_output)
                accuracy = np.mean(prediccion == test_output)
                resultado['correcto'] = correcto
                resultado['accuracy_pixeles'] = accuracy
            else:
                resultado['shape_mismatch'] = True

        return resultado


# Funciones de utilidad

def analizar_tarea(train_examples: List[Tuple[np.ndarray, np.ndarray]]) -> Dict:
    """Analiza una tarea para entender que tipo de transformacion es."""
    analisis = {
        'n_ejemplos': len(train_examples),
        'cambios_forma': [],
        'cambios_color': [],
        'patrones_detectados': [],
    }

    detector = DetectorObjetos()

    for inp, out in train_examples:
        # Analizar cambio de forma
        if inp.shape != out.shape:
            analisis['cambios_forma'].append({
                'input': inp.shape,
                'output': out.shape,
                'ratio': (out.shape[0] / inp.shape[0], out.shape[1] / inp.shape[1])
            })

        # Analizar objetos
        obj_in = detector.detectar(inp)
        obj_out = detector.detectar(out)

        analisis['n_objetos_in'] = len(obj_in)
        analisis['n_objetos_out'] = len(obj_out)

    return analisis


if __name__ == "__main__":
    # Test basico
    print("Test del Campo ARC")
    print("=" * 50)

    # Crear grid de prueba
    grid_in = np.array([
        [0, 0, 0, 0, 0],
        [0, 1, 1, 0, 0],
        [0, 1, 1, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
    ], dtype=np.int8)

    grid_out = np.array([
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 1, 1, 0],
        [0, 0, 1, 1, 0],
    ], dtype=np.int8)

    print("Grid Input:")
    print(grid_in)
    print("\nGrid Output:")
    print(grid_out)

    # Detectar objetos
    detector = DetectorObjetos()
    objetos = detector.detectar(grid_in)
    print(f"\nObjetos detectados: {len(objetos)}")
    for obj in objetos:
        print(f"  Color {obj.color}: {obj.size} pixeles, centroide {obj.centroid}")

    # Probar encoder
    encoder = EncoderARC()
    campo = encoder.encode_grid(grid_in)
    print(f"\nCampo codificado: {campo.shape}, norma={np.linalg.norm(campo):.4f}")

    # Probar motor
    motor = MotorARC()
    motor.aprender_de_ejemplos([(grid_in, grid_out)])

    prediccion = motor.predecir(grid_in, grid_out.shape)
    print(f"\nPrediccion:")
    print(prediccion)

    accuracy = np.mean(prediccion == grid_out)
    print(f"\nAccuracy: {accuracy*100:.1f}%")

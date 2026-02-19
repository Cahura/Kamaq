"""
MOTOR DE TRANSFORMACIONES ARC
==============================
Implementa transformaciones primitivas y su composicion.

ESTRATEGIA:
1. Detectar objetos en input/output
2. Identificar transformacion por comparacion
3. Aplicar transformacion al test

TRANSFORMACIONES PRIMITIVAS:
- Identidad
- Mover (translate)
- Rotar (90, 180, 270)
- Reflejar (horizontal, vertical)
- Escalar (2x, 3x)
- Cambiar color
- Rellenar
- Copiar/duplicar
- Extraer
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Callable
from dataclasses import dataclass
from collections import Counter
import scipy.ndimage as ndimage


@dataclass
class Objeto:
    """Un objeto detectado en el grid."""
    pixeles: List[Tuple[int, int]]
    color: int
    bbox: Tuple[int, int, int, int]  # (min_r, min_c, max_r, max_c)

    @property
    def size(self) -> int:
        return len(self.pixeles)

    @property
    def width(self) -> int:
        return self.bbox[3] - self.bbox[1] + 1

    @property
    def height(self) -> int:
        return self.bbox[2] - self.bbox[0] + 1

    @property
    def centroid(self) -> Tuple[float, float]:
        rows = [p[0] for p in self.pixeles]
        cols = [p[1] for p in self.pixeles]
        return (np.mean(rows), np.mean(cols))

    def get_mask(self, shape: Tuple[int, int]) -> np.ndarray:
        """Retorna mascara binaria."""
        mask = np.zeros(shape, dtype=bool)
        for r, c in self.pixeles:
            if 0 <= r < shape[0] and 0 <= c < shape[1]:
                mask[r, c] = True
        return mask

    def get_normalized_shape(self) -> np.ndarray:
        """Retorna la forma del objeto normalizada (topleft en 0,0)."""
        min_r, min_c, max_r, max_c = self.bbox
        h = max_r - min_r + 1
        w = max_c - min_c + 1
        shape = np.zeros((h, w), dtype=np.int8)
        for r, c in self.pixeles:
            shape[r - min_r, c - min_c] = self.color
        return shape


class DetectorObjetos:
    """Detecta objetos en grids ARC."""

    def detectar(self, grid: np.ndarray, incluir_fondo: bool = False) -> List[Objeto]:
        """Detecta objetos por componentes conexos."""
        objetos = []

        for color in range(10):
            if color == 0 and not incluir_fondo:
                continue

            mask = (grid == color)
            if not np.any(mask):
                continue

            labeled, n_components = ndimage.label(mask)

            for comp_id in range(1, n_components + 1):
                comp_mask = (labeled == comp_id)
                pixeles = list(zip(*np.where(comp_mask)))

                if pixeles:
                    rows = [p[0] for p in pixeles]
                    cols = [p[1] for p in pixeles]
                    bbox = (min(rows), min(cols), max(rows), max(cols))

                    obj = Objeto(
                        pixeles=pixeles,
                        color=color,
                        bbox=bbox
                    )
                    objetos.append(obj)

        return objetos


# =============================================================================
# TRANSFORMACIONES PRIMITIVAS
# =============================================================================

def transformacion_identidad(grid: np.ndarray) -> np.ndarray:
    """No hace nada."""
    return grid.copy()


def transformacion_rotar_90(grid: np.ndarray) -> np.ndarray:
    """Rota 90 grados en sentido horario."""
    return np.rot90(grid, k=-1)


def transformacion_rotar_180(grid: np.ndarray) -> np.ndarray:
    """Rota 180 grados."""
    return np.rot90(grid, k=2)


def transformacion_rotar_270(grid: np.ndarray) -> np.ndarray:
    """Rota 270 grados (90 antihorario)."""
    return np.rot90(grid, k=1)


def transformacion_reflejar_h(grid: np.ndarray) -> np.ndarray:
    """Refleja horizontalmente."""
    return np.fliplr(grid)


def transformacion_reflejar_v(grid: np.ndarray) -> np.ndarray:
    """Refleja verticalmente."""
    return np.flipud(grid)


def transformacion_transponer(grid: np.ndarray) -> np.ndarray:
    """Transpone el grid."""
    return grid.T


def hacer_cambio_color(color_orig: int, color_nuevo: int) -> Callable:
    """Factory para cambio de color."""
    def transformacion(grid: np.ndarray) -> np.ndarray:
        result = grid.copy()
        result[grid == color_orig] = color_nuevo
        return result
    return transformacion


def transformacion_escalar_2x(grid: np.ndarray) -> np.ndarray:
    """Escala 2x."""
    return np.repeat(np.repeat(grid, 2, axis=0), 2, axis=1)


def transformacion_escalar_3x(grid: np.ndarray) -> np.ndarray:
    """Escala 3x."""
    return np.repeat(np.repeat(grid, 3, axis=0), 3, axis=1)


def transformacion_tile_2x2(grid: np.ndarray) -> np.ndarray:
    """Crea un tile 2x2 del grid."""
    return np.tile(grid, (2, 2))


def transformacion_tile_3x3(grid: np.ndarray) -> np.ndarray:
    """Crea un tile 3x3 del grid."""
    return np.tile(grid, (3, 3))


def transformacion_rellenar_fondo(grid: np.ndarray) -> np.ndarray:
    """Rellena el fondo (0) con el color mas comun no-fondo."""
    colores = grid[grid != 0].flatten()
    if len(colores) == 0:
        return grid.copy()
    color_comun = Counter(colores).most_common(1)[0][0]
    result = grid.copy()
    result[grid == 0] = color_comun
    return result


def transformacion_invertir_colores(grid: np.ndarray) -> np.ndarray:
    """Invierte colores (no-fondo)."""
    result = grid.copy()
    colores_usados = set(grid.flatten()) - {0}
    if len(colores_usados) == 2:
        c1, c2 = list(colores_usados)
        result[grid == c1] = c2
        result[grid == c2] = c1
    return result


def transformacion_extraer_patron(grid: np.ndarray) -> np.ndarray:
    """Extrae el patron no-fondo a su bbox."""
    mask = grid != 0
    if not np.any(mask):
        return grid.copy()

    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    return grid[rmin:rmax+1, cmin:cmax+1]


# =============================================================================
# BIBLIOTECA DE TRANSFORMACIONES
# =============================================================================

TRANSFORMACIONES_BASICAS = {
    'identidad': transformacion_identidad,
    'rotar_90': transformacion_rotar_90,
    'rotar_180': transformacion_rotar_180,
    'rotar_270': transformacion_rotar_270,
    'reflejar_h': transformacion_reflejar_h,
    'reflejar_v': transformacion_reflejar_v,
    'transponer': transformacion_transponer,
    'escalar_2x': transformacion_escalar_2x,
    'escalar_3x': transformacion_escalar_3x,
    'tile_2x2': transformacion_tile_2x2,
    'tile_3x3': transformacion_tile_3x3,
    'extraer_patron': transformacion_extraer_patron,
    'rellenar_fondo': transformacion_rellenar_fondo,
    'invertir_colores': transformacion_invertir_colores,
}


class IdentificadorTransformacion:
    """Identifica que transformacion se aplico comparando input/output."""

    def __init__(self):
        self.detector = DetectorObjetos()

    def identificar(self, inp: np.ndarray, out: np.ndarray) -> List[str]:
        """
        Identifica las transformaciones que convierten inp en out.
        Retorna lista de nombres de transformaciones candidatas.
        """
        candidatas = []

        # Probar cada transformacion basica
        for nombre, func in TRANSFORMACIONES_BASICAS.items():
            try:
                resultado = func(inp)
                if resultado.shape == out.shape:
                    if np.array_equal(resultado, out):
                        candidatas.append(nombre)
            except Exception:
                pass

        # Probar cambios de color
        colores_inp = set(inp.flatten())
        colores_out = set(out.flatten())

        if inp.shape == out.shape:
            for c_in in colores_inp:
                for c_out in colores_out:
                    if c_in != c_out:
                        func = hacer_cambio_color(c_in, c_out)
                        resultado = func(inp)
                        if np.array_equal(resultado, out):
                            candidatas.append(f'color_{c_in}_a_{c_out}')

        # Si no hay candidatas, analizar objetos
        if not candidatas:
            candidatas = self._analizar_por_objetos(inp, out)

        return candidatas if candidatas else ['desconocida']

    def _analizar_por_objetos(self, inp: np.ndarray, out: np.ndarray) -> List[str]:
        """Analiza transformacion a nivel de objetos."""
        obj_inp = self.detector.detectar(inp)
        obj_out = self.detector.detectar(out)

        candidatas = []

        # Detectar si hay mismo numero de objetos
        if len(obj_inp) == len(obj_out):
            candidatas.append('conserva_objetos')

            # Verificar si los objetos se movieron
            for oi, oo in zip(obj_inp, obj_out):
                if oi.color == oo.color and oi.size == oo.size:
                    dr = oo.centroid[0] - oi.centroid[0]
                    dc = oo.centroid[1] - oi.centroid[1]
                    if abs(dr) > 0.5 or abs(dc) > 0.5:
                        candidatas.append(f'mover_{int(dr)}_{int(dc)}')

        # Detectar si hay objetos nuevos
        if len(obj_out) > len(obj_inp):
            candidatas.append('crea_objetos')

        # Detectar si se eliminaron objetos
        if len(obj_out) < len(obj_inp):
            candidatas.append('elimina_objetos')

        return candidatas


class MotorTransformacionesARC:
    """
    Motor principal para resolver tareas ARC.

    ESTRATEGIA:
    1. Para cada par (input, output) de entrenamiento:
       - Identificar transformacion
    2. Encontrar transformacion consistente en TODOS los ejemplos
    3. Aplicar al test
    """

    def __init__(self):
        self.identificador = IdentificadorTransformacion()
        self.transformacion_aprendida: Optional[str] = None
        self.transformaciones_candidatas: List[str] = []

    def aprender(self, ejemplos: List[Tuple[np.ndarray, np.ndarray]]):
        """Aprende la transformacion de los ejemplos."""
        if not ejemplos:
            return

        # Identificar transformaciones para cada ejemplo
        todas_transformaciones = []
        for inp, out in ejemplos:
            trans = self.identificador.identificar(inp, out)
            todas_transformaciones.append(set(trans))

        # Encontrar interseccion (transformaciones consistentes)
        if todas_transformaciones:
            consistentes = todas_transformaciones[0]
            for t in todas_transformaciones[1:]:
                consistentes = consistentes.intersection(t)

            self.transformaciones_candidatas = list(consistentes)

            # Elegir la mas probable (priorizando transformaciones simples)
            if consistentes:
                for pref in ['identidad', 'rotar_90', 'rotar_180', 'rotar_270',
                            'reflejar_h', 'reflejar_v', 'extraer_patron']:
                    if pref in consistentes:
                        self.transformacion_aprendida = pref
                        return

                # Si no, tomar la primera
                self.transformacion_aprendida = list(consistentes)[0]

    def predecir(self, test_input: np.ndarray,
                 output_shape: Optional[Tuple[int, int]] = None) -> np.ndarray:
        """Predice el output para un test input."""

        if self.transformacion_aprendida is None:
            # Sin transformacion aprendida, devolver input
            return test_input.copy()

        nombre = self.transformacion_aprendida

        # Aplicar transformacion basica si existe
        if nombre in TRANSFORMACIONES_BASICAS:
            resultado = TRANSFORMACIONES_BASICAS[nombre](test_input)

            # Ajustar forma si es necesario
            if output_shape and resultado.shape != output_shape:
                resultado = self._ajustar_forma(resultado, output_shape)

            return resultado

        # Manejar cambios de color
        if nombre.startswith('color_'):
            parts = nombre.split('_')
            if len(parts) >= 4:
                c_in = int(parts[1])
                c_out = int(parts[3])
                func = hacer_cambio_color(c_in, c_out)
                return func(test_input)

        # Por defecto, devolver input
        return test_input.copy()

    def _ajustar_forma(self, grid: np.ndarray,
                       target_shape: Tuple[int, int]) -> np.ndarray:
        """Ajusta el grid a la forma objetivo."""
        h, w = target_shape
        result = np.zeros((h, w), dtype=grid.dtype)

        # Copiar lo que quepa
        copy_h = min(h, grid.shape[0])
        copy_w = min(w, grid.shape[1])
        result[:copy_h, :copy_w] = grid[:copy_h, :copy_w]

        return result

    def evaluar_tarea(self, train_examples: List[Tuple[np.ndarray, np.ndarray]],
                      test_input: np.ndarray,
                      test_output: Optional[np.ndarray] = None) -> Dict:
        """Evalua en una tarea completa."""
        # Reset
        self.transformacion_aprendida = None
        self.transformaciones_candidatas = []

        # Aprender
        self.aprender(train_examples)

        # Predecir
        output_shape = test_output.shape if test_output is not None else None
        prediccion = self.predecir(test_input, output_shape)

        resultado = {
            'prediccion': prediccion,
            'transformacion': self.transformacion_aprendida,
            'candidatas': self.transformaciones_candidatas,
            'correcto': False,
            'accuracy_pixeles': 0.0,
        }

        if test_output is not None:
            if prediccion.shape == test_output.shape:
                resultado['correcto'] = np.array_equal(prediccion, test_output)
                resultado['accuracy_pixeles'] = np.mean(prediccion == test_output)
            else:
                resultado['shape_mismatch'] = True

        return resultado


if __name__ == "__main__":
    print("Test del Motor de Transformaciones")
    print("="*50)

    # Test: rotacion 90
    inp = np.array([
        [1, 1, 0],
        [1, 0, 0],
        [0, 0, 0]
    ], dtype=np.int8)

    out_rot = transformacion_rotar_90(inp)
    print("\nRotacion 90:")
    print("Input:")
    print(inp)
    print("Output:")
    print(out_rot)

    # Test: identificacion
    identificador = IdentificadorTransformacion()
    trans = identificador.identificar(inp, out_rot)
    print(f"\nTransformacion identificada: {trans}")

    # Test: motor completo
    print("\n" + "="*50)
    print("Test Motor Completo")

    motor = MotorTransformacionesARC()

    # Ejemplos de entrenamiento (rotacion 90)
    ejemplos = [
        (inp, out_rot),
        (np.array([[2, 2], [0, 2]], dtype=np.int8),
         np.array([[0, 2], [2, 2]], dtype=np.int8)),
    ]

    motor.aprender(ejemplos)
    print(f"Transformacion aprendida: {motor.transformacion_aprendida}")
    print(f"Candidatas: {motor.transformaciones_candidatas}")

    # Predecir
    test_in = np.array([
        [3, 3, 3],
        [3, 0, 0],
        [3, 0, 0]
    ], dtype=np.int8)

    pred = motor.predecir(test_in)
    print("\nTest input:")
    print(test_in)
    print("Prediccion:")
    print(pred)

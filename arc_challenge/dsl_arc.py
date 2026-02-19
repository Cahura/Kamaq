"""
DSL PARA TRANSFORMACIONES ARC
==============================
Un lenguaje especifico para describir transformaciones ARC.

ESTRATEGIA:
1. Detectar objetos en input y output
2. Mapear objetos (cual se convirtio en cual)
3. Inferir operaciones sobre cada objeto
4. Generalizar a regla

OPERACIONES DEL DSL:
- COPY(obj) -> obj
- MOVE(obj, dr, dc) -> obj en nueva posicion
- COLOR(obj, nuevo_color) -> obj con color cambiado
- SCALE(obj, factor) -> obj escalado
- ROTATE(obj, angulo) -> obj rotado
- TILE(patron, n, m) -> patron repetido
- FILTER(propiedad) -> solo objetos con propiedad
- SORT(criterio) -> ordenar objetos
"""

import numpy as np
from typing import List, Dict, Tuple, Optional, Set, Any
from dataclasses import dataclass, field
from collections import defaultdict
import scipy.ndimage as ndimage


@dataclass
class Objeto:
    """Objeto detectado en grid ARC."""
    id: int
    pixeles: List[Tuple[int, int]]
    color: int
    bbox: Tuple[int, int, int, int]

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

    @property
    def top_left(self) -> Tuple[int, int]:
        return (self.bbox[0], self.bbox[1])

    def get_shape(self) -> np.ndarray:
        """Retorna la forma normalizada del objeto."""
        min_r, min_c, max_r, max_c = self.bbox
        h = max_r - min_r + 1
        w = max_c - min_c + 1
        shape = np.zeros((h, w), dtype=np.int8)
        for r, c in self.pixeles:
            shape[r - min_r, c - min_c] = self.color
        return shape

    def get_signature(self) -> tuple:
        """Firma unica del objeto (para matching)."""
        shape = self.get_shape()
        # Normalizar rotaciones para comparacion
        return (self.size, self.width, self.height, tuple(shape.flatten()))


@dataclass
class Operacion:
    """Una operacion del DSL."""
    tipo: str
    objeto_id: Optional[int] = None
    parametros: Dict[str, Any] = field(default_factory=dict)

    def __str__(self):
        if self.objeto_id is not None:
            return f"{self.tipo}(obj_{self.objeto_id}, {self.parametros})"
        return f"{self.tipo}({self.parametros})"


class DetectorObjetosARC:
    """Detecta objetos con IDs unicos."""

    def detectar(self, grid: np.ndarray, incluir_fondo: bool = False) -> List[Objeto]:
        objetos = []
        obj_id = 0

        for color in range(10):
            if color == 0 and not incluir_fondo:
                continue

            mask = (grid == color)
            if not np.any(mask):
                continue

            labeled, n = ndimage.label(mask)

            for comp in range(1, n + 1):
                comp_mask = (labeled == comp)
                pixeles = list(zip(*np.where(comp_mask)))

                if pixeles:
                    rows = [p[0] for p in pixeles]
                    cols = [p[1] for p in pixeles]
                    bbox = (min(rows), min(cols), max(rows), max(cols))

                    objetos.append(Objeto(
                        id=obj_id,
                        pixeles=pixeles,
                        color=color,
                        bbox=bbox
                    ))
                    obj_id += 1

        return objetos


class MapperObjetos:
    """Mapea objetos entre input y output."""

    def mapear(self, obj_in: List[Objeto], obj_out: List[Objeto]) -> Dict[int, int]:
        """
        Mapea objetos de input a output.
        Retorna dict {id_input: id_output}
        """
        mapping = {}

        # Estrategia 1: Por forma identica
        for oi in obj_in:
            sig_i = oi.get_signature()
            for oo in obj_out:
                if oo.id in mapping.values():
                    continue
                sig_o = oo.get_signature()
                if sig_i == sig_o or (oi.size == oo.size and oi.color == oo.color):
                    mapping[oi.id] = oo.id
                    break

        # Estrategia 2: Por forma similar (mismo tamano)
        for oi in obj_in:
            if oi.id in mapping:
                continue
            for oo in obj_out:
                if oo.id in mapping.values():
                    continue
                if oi.size == oo.size:
                    mapping[oi.id] = oo.id
                    break

        # Estrategia 3: Por posicion similar
        for oi in obj_in:
            if oi.id in mapping:
                continue
            mejor_dist = float('inf')
            mejor_oo = None
            for oo in obj_out:
                if oo.id in mapping.values():
                    continue
                dist = np.sqrt(
                    (oi.centroid[0] - oo.centroid[0])**2 +
                    (oi.centroid[1] - oo.centroid[1])**2
                )
                if dist < mejor_dist:
                    mejor_dist = dist
                    mejor_oo = oo

            if mejor_oo and mejor_dist < 10:
                mapping[oi.id] = mejor_oo.id

        return mapping


class InferenciaDSL:
    """Infiere programa DSL de ejemplos."""

    def __init__(self):
        self.detector = DetectorObjetosARC()
        self.mapper = MapperObjetos()

    def inferir(self, ejemplos: List[Tuple[np.ndarray, np.ndarray]]) -> List[Operacion]:
        """Infiere programa DSL de los ejemplos."""
        programas_por_ejemplo = []

        for inp, out in ejemplos:
            ops = self._inferir_un_ejemplo(inp, out)
            programas_por_ejemplo.append(ops)

        # Encontrar operaciones consistentes
        return self._unificar_programas(programas_por_ejemplo)

    def _inferir_un_ejemplo(self, inp: np.ndarray, out: np.ndarray) -> List[Operacion]:
        """Infiere operaciones para un ejemplo."""
        ops = []

        obj_in = self.detector.detectar(inp)
        obj_out = self.detector.detectar(out)

        # Caso: mismo numero de objetos
        if len(obj_in) == len(obj_out):
            mapping = self.mapper.mapear(obj_in, obj_out)

            for oi in obj_in:
                if oi.id not in mapping:
                    continue

                oo_id = mapping[oi.id]
                oo = next((o for o in obj_out if o.id == oo_id), None)
                if not oo:
                    continue

                # Detectar tipo de transformacion
                if oi.color != oo.color:
                    ops.append(Operacion(
                        tipo='COLOR',
                        objeto_id=oi.id,
                        parametros={'nuevo_color': oo.color}
                    ))

                # Detectar movimiento
                dr = oo.centroid[0] - oi.centroid[0]
                dc = oo.centroid[1] - oi.centroid[1]
                if abs(dr) > 0.5 or abs(dc) > 0.5:
                    ops.append(Operacion(
                        tipo='MOVE',
                        objeto_id=oi.id,
                        parametros={'dr': int(round(dr)), 'dc': int(round(dc))}
                    ))

        # Caso: output tiene mas objetos (replicacion)
        elif len(obj_out) > len(obj_in):
            ratio = len(obj_out) / max(len(obj_in), 1)
            if ratio == int(ratio):
                ops.append(Operacion(
                    tipo='REPLICATE',
                    parametros={'factor': int(ratio)}
                ))

        # Caso: output tiene menos objetos (filtrado)
        elif len(obj_out) < len(obj_in):
            # Detectar criterio de filtrado
            colores_out = set(o.color for o in obj_out)
            if len(colores_out) == 1:
                ops.append(Operacion(
                    tipo='FILTER_COLOR',
                    parametros={'color': list(colores_out)[0]}
                ))

            # Filtrar por tamano
            if obj_out:
                max_size_out = max(o.size for o in obj_out)
                min_size_out = min(o.size for o in obj_out)

                if all(o.size == max_size_out for o in obj_out):
                    ops.append(Operacion(
                        tipo='FILTER_MAX_SIZE'
                    ))
                elif all(o.size == min_size_out for o in obj_out):
                    ops.append(Operacion(
                        tipo='FILTER_MIN_SIZE'
                    ))

        # Detectar cambio de forma global
        if inp.shape != out.shape:
            ratio_h = out.shape[0] / inp.shape[0]
            ratio_w = out.shape[1] / inp.shape[1]

            if ratio_h == ratio_w and ratio_h in [2, 3]:
                ops.append(Operacion(
                    tipo='SCALE_GRID',
                    parametros={'factor': int(ratio_h)}
                ))
            elif ratio_h > 1 or ratio_w > 1:
                ops.append(Operacion(
                    tipo='TILE_GRID',
                    parametros={'n': int(ratio_h), 'm': int(ratio_w)}
                ))

        return ops

    def _unificar_programas(self, programas: List[List[Operacion]]) -> List[Operacion]:
        """Encuentra operaciones consistentes en todos los ejemplos."""
        if not programas:
            return []

        if len(programas) == 1:
            return programas[0]

        # Encontrar operaciones por tipo que aparecen en todos
        tipos_comunes = set(op.tipo for op in programas[0])
        for prog in programas[1:]:
            tipos_comunes &= set(op.tipo for op in prog)

        # Tomar la primera operacion de cada tipo comun
        resultado = []
        for tipo in tipos_comunes:
            for prog in programas:
                for op in prog:
                    if op.tipo == tipo:
                        resultado.append(op)
                        break
                break

        return resultado


class EjecutorDSL:
    """Ejecuta programas DSL."""

    def __init__(self):
        self.detector = DetectorObjetosARC()

    def ejecutar(self, grid: np.ndarray, programa: List[Operacion]) -> np.ndarray:
        """Ejecuta un programa DSL sobre un grid."""
        resultado = grid.copy()

        for op in programa:
            resultado = self._ejecutar_operacion(resultado, op)

        return resultado

    def _ejecutar_operacion(self, grid: np.ndarray, op: Operacion) -> np.ndarray:
        """Ejecuta una operacion individual."""

        if op.tipo == 'COLOR':
            # Cambiar color de un objeto
            objetos = self.detector.detectar(grid)
            resultado = grid.copy()

            if op.objeto_id is not None and op.objeto_id < len(objetos):
                obj = objetos[op.objeto_id]
                nuevo_color = op.parametros.get('nuevo_color', obj.color)
                for r, c in obj.pixeles:
                    resultado[r, c] = nuevo_color
            else:
                # Cambiar todos los objetos
                nuevo_color = op.parametros.get('nuevo_color', 1)
                for obj in objetos:
                    for r, c in obj.pixeles:
                        resultado[r, c] = nuevo_color

            return resultado

        elif op.tipo == 'MOVE':
            dr = op.parametros.get('dr', 0)
            dc = op.parametros.get('dc', 0)

            objetos = self.detector.detectar(grid)
            resultado = np.zeros_like(grid)

            for obj in objetos:
                for r, c in obj.pixeles:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < grid.shape[0] and 0 <= nc < grid.shape[1]:
                        resultado[nr, nc] = obj.color

            return resultado

        elif op.tipo == 'SCALE_GRID':
            factor = op.parametros.get('factor', 2)
            return np.repeat(np.repeat(grid, factor, axis=0), factor, axis=1)

        elif op.tipo == 'TILE_GRID':
            n = op.parametros.get('n', 2)
            m = op.parametros.get('m', 2)
            return np.tile(grid, (n, m))

        elif op.tipo == 'FILTER_COLOR':
            color = op.parametros.get('color', 1)
            resultado = np.zeros_like(grid)
            resultado[grid == color] = color
            return resultado

        elif op.tipo == 'FILTER_MAX_SIZE':
            objetos = self.detector.detectar(grid)
            if not objetos:
                return grid.copy()

            max_size = max(o.size for o in objetos)
            resultado = np.zeros_like(grid)

            for obj in objetos:
                if obj.size == max_size:
                    for r, c in obj.pixeles:
                        resultado[r, c] = obj.color

            return resultado

        elif op.tipo == 'FILTER_MIN_SIZE':
            objetos = self.detector.detectar(grid)
            if not objetos:
                return grid.copy()

            min_size = min(o.size for o in objetos)
            resultado = np.zeros_like(grid)

            for obj in objetos:
                if obj.size == min_size:
                    for r, c in obj.pixeles:
                        resultado[r, c] = obj.color

            return resultado

        elif op.tipo == 'REPLICATE':
            # No implementado completamente
            return grid.copy()

        return grid.copy()


class MotorDSL:
    """Motor principal basado en DSL."""

    def __init__(self):
        self.inferencia = InferenciaDSL()
        self.ejecutor = EjecutorDSL()
        self.programa: List[Operacion] = []

    def aprender(self, ejemplos: List[Tuple[np.ndarray, np.ndarray]]):
        """Aprende programa DSL de ejemplos."""
        self.programa = self.inferencia.inferir(ejemplos)

    def predecir(self, test_input: np.ndarray,
                 output_shape: Optional[Tuple[int, int]] = None) -> np.ndarray:
        """Predice usando el programa aprendido."""
        if not self.programa:
            return test_input.copy()

        resultado = self.ejecutor.ejecutar(test_input, self.programa)

        if output_shape and resultado.shape != output_shape:
            # Ajustar forma
            h, w = output_shape
            nuevo = np.zeros((h, w), dtype=resultado.dtype)
            copy_h = min(h, resultado.shape[0])
            copy_w = min(w, resultado.shape[1])
            nuevo[:copy_h, :copy_w] = resultado[:copy_h, :copy_w]
            resultado = nuevo

        return resultado

    def evaluar_tarea(self, train_examples: List[Tuple[np.ndarray, np.ndarray]],
                      test_input: np.ndarray,
                      test_output: Optional[np.ndarray] = None) -> Dict:
        """Evalua una tarea."""
        self.programa = []
        self.aprender(train_examples)

        output_shape = test_output.shape if test_output is not None else None
        prediccion = self.predecir(test_input, output_shape)

        resultado = {
            'prediccion': prediccion,
            'programa': [str(op) for op in self.programa],
            'correcto': False,
            'accuracy_pixeles': 0.0,
        }

        if test_output is not None:
            if prediccion.shape == test_output.shape:
                resultado['correcto'] = np.array_equal(prediccion, test_output)
                resultado['accuracy_pixeles'] = np.mean(prediccion == test_output)

        return resultado


if __name__ == "__main__":
    print("Test DSL ARC")
    print("=" * 50)

    # Test: movimiento de objetos
    inp = np.array([
        [0, 0, 0, 0, 0],
        [0, 1, 1, 0, 0],
        [0, 1, 1, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
    ], dtype=np.int8)

    out = np.array([
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 1, 1, 0],
        [0, 0, 1, 1, 0],
    ], dtype=np.int8)

    print("Input:")
    print(inp)
    print("\nOutput:")
    print(out)

    motor = MotorDSL()
    motor.aprender([(inp, out)])

    print(f"\nPrograma inferido:")
    for op in motor.programa:
        print(f"  {op}")

    pred = motor.predecir(inp, out.shape)
    print("\nPrediccion:")
    print(pred)
    print(f"\nCorrecto: {np.array_equal(pred, out)}")

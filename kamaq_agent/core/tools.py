"""
KAMAQ Tools System: Sistema de Herramientas
============================================
Herramientas que el agente puede usar para interactuar
con el mundo real.

Herramientas implementadas:
- FileReader: Leer archivos
- FileWriter: Escribir archivos (con confirmación)
- ShellExecutor: Ejecutar comandos (con sandboxing)
- Calculator: Cálculos matemáticos seguros
- PythonExecutor: Ejecutar código Python
- WebSearch: Búsqueda web (placeholder)

Autor: KAMAQ Team
Fecha: Enero 2026
"""

from dataclasses import dataclass, field
from typing import List, Dict, Optional, Any, Callable
from abc import ABC, abstractmethod
from enum import Enum
from pathlib import Path
import subprocess
import json
import ast
import operator
import re


class ToolCategory(Enum):
    """Categorías de herramientas."""
    FILE = "archivos"
    CODE = "codigo"
    SEARCH = "busqueda"
    CALCULATE = "calculo"
    SYSTEM = "sistema"


class ToolRisk(Enum):
    """Nivel de riesgo de la herramienta."""
    SAFE = "seguro"           # Solo lectura
    MODERATE = "moderado"      # Escribe pero reversible
    DANGEROUS = "peligroso"    # Puede causar daño


@dataclass
class ToolResult:
    """Resultado de ejecución de herramienta."""
    success: bool
    output: Any
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolSpec:
    """Especificación de herramienta para el LLM."""
    name: str
    description: str
    parameters: Dict[str, Dict]  # {param_name: {type, description, required}}
    category: ToolCategory
    risk: ToolRisk
    examples: List[str] = field(default_factory=list)
    
    def to_function_spec(self) -> Dict:
        """Convierte a formato de function calling."""
        properties = {}
        required = []
        
        for param_name, param_info in self.parameters.items():
            properties[param_name] = {
                "type": param_info.get("type", "string"),
                "description": param_info.get("description", "")
            }
            if param_info.get("required", False):
                required.append(param_name)
        
        return {
            "name": self.name,
            "description": self.description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required
            }
        }


class Tool(ABC):
    """Clase base para herramientas."""
    
    @property
    @abstractmethod
    def spec(self) -> ToolSpec:
        """Especificación de la herramienta."""
        pass
    
    @abstractmethod
    def execute(self, **kwargs) -> ToolResult:
        """Ejecuta la herramienta."""
        pass
    
    def validate_params(self, **kwargs) -> Optional[str]:
        """Valida parámetros. Retorna mensaje de error o None."""
        for param_name, param_info in self.spec.parameters.items():
            if param_info.get("required", False) and param_name not in kwargs:
                return f"Parámetro requerido faltante: {param_name}"
        return None


# =============================================================================
# HERRAMIENTAS DE ARCHIVOS
# =============================================================================

class FileReaderTool(Tool):
    """Lee contenido de archivos."""
    
    def __init__(self, allowed_extensions: List[str] = None,
                 max_size_kb: int = 500):
        self.allowed_extensions = allowed_extensions or [
            ".txt", ".md", ".py", ".json", ".yaml", ".yml",
            ".js", ".ts", ".html", ".css", ".csv"
        ]
        self.max_size_kb = max_size_kb
    
    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="read_file",
            description="Lee el contenido de un archivo de texto",
            parameters={
                "path": {
                    "type": "string",
                    "description": "Ruta al archivo a leer",
                    "required": True
                },
                "start_line": {
                    "type": "integer",
                    "description": "Línea inicial (opcional)",
                    "required": False
                },
                "end_line": {
                    "type": "integer",
                    "description": "Línea final (opcional)",
                    "required": False
                }
            },
            category=ToolCategory.FILE,
            risk=ToolRisk.SAFE,
            examples=[
                'read_file(path="src/main.py")',
                'read_file(path="README.md", start_line=1, end_line=50)'
            ]
        )
    
    def execute(self, path: str, start_line: int = None, 
                end_line: int = None) -> ToolResult:
        try:
            file_path = Path(path)
            
            # Validar extensión
            if file_path.suffix.lower() not in self.allowed_extensions:
                return ToolResult(
                    success=False,
                    output=None,
                    error=f"Extensión no permitida: {file_path.suffix}"
                )
            
            # Validar existencia
            if not file_path.exists():
                return ToolResult(
                    success=False,
                    output=None,
                    error=f"Archivo no encontrado: {path}"
                )
            
            # Validar tamaño
            size_kb = file_path.stat().st_size / 1024
            if size_kb > self.max_size_kb:
                return ToolResult(
                    success=False,
                    output=None,
                    error=f"Archivo muy grande: {size_kb:.1f}KB > {self.max_size_kb}KB"
                )
            
            # Leer archivo
            content = file_path.read_text(encoding='utf-8')
            
            # Aplicar filtro de líneas si se especifica
            if start_line is not None or end_line is not None:
                lines = content.split('\n')
                start = (start_line - 1) if start_line else 0
                end = end_line if end_line else len(lines)
                content = '\n'.join(lines[start:end])
            
            return ToolResult(
                success=True,
                output=content,
                metadata={
                    "path": str(file_path.absolute()),
                    "size_kb": size_kb,
                    "lines": len(content.split('\n'))
                }
            )
            
        except Exception as e:
            return ToolResult(
                success=False,
                output=None,
                error=str(e)
            )


class FileWriterTool(Tool):
    """Escribe contenido a archivos (requiere confirmación)."""
    
    def __init__(self, 
                 allowed_directories: List[str] = None,
                 require_confirmation: bool = True):
        self.allowed_directories = allowed_directories
        self.require_confirmation = require_confirmation
        self._pending_write: Optional[Dict] = None
    
    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="write_file",
            description="Escribe contenido a un archivo. Requiere confirmación.",
            parameters={
                "path": {
                    "type": "string",
                    "description": "Ruta donde escribir el archivo",
                    "required": True
                },
                "content": {
                    "type": "string",
                    "description": "Contenido a escribir",
                    "required": True
                },
                "mode": {
                    "type": "string",
                    "description": "Modo: 'overwrite' o 'append'",
                    "required": False
                }
            },
            category=ToolCategory.FILE,
            risk=ToolRisk.MODERATE,
            examples=[
                'write_file(path="output.txt", content="Hello World")',
                'write_file(path="log.txt", content="Entry", mode="append")'
            ]
        )
    
    def execute(self, path: str, content: str, 
                mode: str = "overwrite",
                confirmed: bool = False) -> ToolResult:
        
        # Si requiere confirmación y no está confirmado
        if self.require_confirmation and not confirmed:
            self._pending_write = {
                "path": path,
                "content": content,
                "mode": mode
            }
            return ToolResult(
                success=False,
                output=None,
                error="REQUIRES_CONFIRMATION",
                metadata={
                    "pending_action": "write_file",
                    "path": path,
                    "content_preview": content[:100] + "..." if len(content) > 100 else content,
                    "mode": mode
                }
            )
        
        try:
            file_path = Path(path)
            
            # Crear directorio si no existe
            file_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Escribir
            if mode == "append":
                with open(file_path, 'a', encoding='utf-8') as f:
                    f.write(content)
            else:
                file_path.write_text(content, encoding='utf-8')
            
            return ToolResult(
                success=True,
                output=f"Archivo escrito: {path}",
                metadata={
                    "path": str(file_path.absolute()),
                    "size_bytes": len(content.encode('utf-8')),
                    "mode": mode
                }
            )
            
        except Exception as e:
            return ToolResult(
                success=False,
                output=None,
                error=str(e)
            )
    
    def confirm_pending(self) -> ToolResult:
        """Confirma y ejecuta la escritura pendiente."""
        if not self._pending_write:
            return ToolResult(
                success=False,
                output=None,
                error="No hay escritura pendiente"
            )
        
        result = self.execute(
            path=self._pending_write["path"],
            content=self._pending_write["content"],
            mode=self._pending_write["mode"],
            confirmed=True
        )
        self._pending_write = None
        return result


# =============================================================================
# HERRAMIENTA DE CÁLCULO SEGURO
# =============================================================================

class CalculatorTool(Tool):
    """Calculadora segura (sin eval)."""
    
    # Operadores permitidos
    OPERATORS = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.Pow: operator.pow,
        ast.USub: operator.neg,
        ast.Mod: operator.mod,
    }
    
    # Funciones matemáticas permitidas
    import math
    FUNCTIONS = {
        'sqrt': math.sqrt,
        'sin': math.sin,
        'cos': math.cos,
        'tan': math.tan,
        'log': math.log,
        'log10': math.log10,
        'exp': math.exp,
        'abs': abs,
        'round': round,
        'pi': math.pi,
        'e': math.e,
    }
    
    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="calculate",
            description="Evalúa expresiones matemáticas de forma segura",
            parameters={
                "expression": {
                    "type": "string",
                    "description": "Expresión matemática a evaluar",
                    "required": True
                }
            },
            category=ToolCategory.CALCULATE,
            risk=ToolRisk.SAFE,
            examples=[
                'calculate(expression="2 + 2")',
                'calculate(expression="sqrt(16) + pi")',
                'calculate(expression="sin(3.14159) ** 2 + cos(3.14159) ** 2")'
            ]
        )
    
    def execute(self, expression: str) -> ToolResult:
        try:
            result = self._safe_eval(expression)
            return ToolResult(
                success=True,
                output=result,
                metadata={"expression": expression}
            )
        except Exception as e:
            return ToolResult(
                success=False,
                output=None,
                error=f"Error en expresión: {str(e)}"
            )
    
    def _safe_eval(self, expression: str) -> float:
        """Evalúa expresión de forma segura usando AST."""
        # Reemplazar funciones por marcadores (word-boundary safe)
        for func_name in self.FUNCTIONS:
            if func_name in expression and not func_name.isdigit():
                expression = re.sub(
                    r'\b' + re.escape(func_name) + r'\b',
                    f"__{func_name}__",
                    expression
                )
        
        # Parsear AST
        node = ast.parse(expression, mode='eval')
        return self._eval_node(node.body)
    
    def _eval_node(self, node) -> float:
        """Evalúa nodo AST recursivamente."""
        # Python 3.8+ usa ast.Constant para todos los literales
        if isinstance(node, ast.Constant):
            return node.value
        # Compatibilidad con Python < 3.8 (ast.Num deprecado)
        elif hasattr(ast, 'Num') and isinstance(node, ast.Num):
            return node.n
        elif isinstance(node, ast.BinOp):
            left = self._eval_node(node.left)
            right = self._eval_node(node.right)
            op = self.OPERATORS.get(type(node.op))
            if op:
                return op(left, right)
            raise ValueError(f"Operador no soportado: {type(node.op)}")
        elif isinstance(node, ast.UnaryOp):
            operand = self._eval_node(node.operand)
            op = self.OPERATORS.get(type(node.op))
            if op:
                return op(operand)
            raise ValueError(f"Operador unario no soportado")
        elif isinstance(node, ast.Call):
            # Llamada a función
            if isinstance(node.func, ast.Name):
                func_name = node.func.id.replace("__", "")
                if func_name in self.FUNCTIONS:
                    func = self.FUNCTIONS[func_name]
                    if callable(func):
                        args = [self._eval_node(arg) for arg in node.args]
                        return func(*args)
                    return func  # Es una constante
            raise ValueError(f"Función no permitida")
        elif isinstance(node, ast.Name):
            # Variable (constante)
            name = node.id.replace("__", "")
            if name in self.FUNCTIONS:
                return self.FUNCTIONS[name]
            raise ValueError(f"Variable no definida: {name}")
        else:
            raise ValueError(f"Expresión no soportada: {type(node)}")


# =============================================================================
# HERRAMIENTA DE PYTHON SANDBOX
# =============================================================================

class PythonExecutorTool(Tool):
    """Ejecuta código Python en sandbox."""
    
    FORBIDDEN_IMPORTS = [
        'os', 'sys', 'subprocess', 'shutil', 'pathlib',
        'socket', 'urllib', 'requests', 'http',
        '__builtins__', 'eval', 'exec', 'compile',
        'open', 'file', 'input'
    ]
    
    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="python_exec",
            description="Ejecuta código Python en un sandbox seguro",
            parameters={
                "code": {
                    "type": "string",
                    "description": "Código Python a ejecutar",
                    "required": True
                }
            },
            category=ToolCategory.CODE,
            risk=ToolRisk.MODERATE,
            examples=[
                'python_exec(code="result = sum([1,2,3,4,5])")',
                'python_exec(code="import math\\nresult = math.factorial(10)")'
            ]
        )
    
    def execute(self, code: str) -> ToolResult:
        # Validar código
        validation_error = self._validate_code(code)
        if validation_error:
            return ToolResult(
                success=False,
                output=None,
                error=validation_error
            )
        
        try:
            # Crear namespace restringido
            import math
            import random
            import json as json_module
            import re as re_module
            from datetime import datetime, timedelta
            
            # Import seguro
            ALLOWED_IMPORTS = {'math', 'random', 'json', 're', 'datetime', 'collections', 'itertools', 'functools'}
            def safe_import(name, globals=None, locals=None, fromlist=(), level=0):
                if name in ALLOWED_IMPORTS:
                    return __import__(name, globals, locals, fromlist, level)
                raise ImportError(f"Import no permitido: {name}")
            
            safe_globals = {
                "__builtins__": {
                    "__import__": safe_import,
                    "print": print,
                    "len": len,
                    "range": range,
                    "str": str,
                    "int": int,
                    "float": float,
                    "list": list,
                    "dict": dict,
                    "set": set,
                    "tuple": tuple,
                    "bool": bool,
                    "sum": sum,
                    "min": min,
                    "max": max,
                    "sorted": sorted,
                    "enumerate": enumerate,
                    "zip": zip,
                    "map": map,
                    "filter": filter,
                    "abs": abs,
                    "round": round,
                    "True": True,
                    "False": False,
                    "None": None,
                },
                "math": math,
                "random": random,
                "json": json_module,
                "re": re_module,
                "datetime": datetime,
                "timedelta": timedelta,
            }
            
            local_vars = {}
            
            # Capturar stdout
            import io
            import sys
            stdout_capture = io.StringIO()
            old_stdout = sys.stdout
            sys.stdout = stdout_capture
            
            try:
                # Ejecutar con timeout implícito (no es perfecto pero ayuda)
                exec(code, safe_globals, local_vars)
            finally:
                sys.stdout = old_stdout
            
            captured_output = stdout_capture.getvalue()
            
            # Buscar variable 'result' si existe
            if 'result' in local_vars:
                result = local_vars['result']
            elif captured_output:
                result = captured_output.strip()
            elif local_vars:
                # Retornar última variable definida
                result = list(local_vars.values())[-1]
            else:
                result = "Código ejecutado exitosamente (sin resultado explícito)"
            
            return ToolResult(
                success=True,
                output=result,
                metadata={"executed_code": code[:200]}
            )
            
        except Exception as e:
            return ToolResult(
                success=False,
                output=None,
                error=f"Error de ejecución: {str(e)}"
            )
    
    def _validate_code(self, code: str) -> Optional[str]:
        """Valida que el código no use imports peligrosos."""
        code_lower = code.lower()
        
        for forbidden in self.FORBIDDEN_IMPORTS:
            patterns = [
                f"import {forbidden}",
                f"from {forbidden}",
                f"__{forbidden}__",
            ]
            for pattern in patterns:
                if pattern.lower() in code_lower:
                    return f"Import prohibido: {forbidden}"
        
        # Detectar llamadas peligrosas
        dangerous_patterns = [
            r"eval\s*\(",
            r"exec\s*\(",
            r"compile\s*\(",
            r"__import__\s*\(",
            r"open\s*\(",
            r"\.read\s*\(",
            r"\.write\s*\(",
        ]
        
        for pattern in dangerous_patterns:
            if re.search(pattern, code):
                return f"Patrón peligroso detectado: {pattern}"
        
        return None


# =============================================================================
# HERRAMIENTA DE SHELL (con restricciones)
# =============================================================================

class ShellExecutorTool(Tool):
    """Ejecuta comandos shell con restricciones."""
    
    ALLOWED_COMMANDS = [
        "ls", "dir", "pwd", "cd", "echo", "cat", "head", "tail",
        "grep", "find", "wc", "date", "whoami",
        "python", "pip", "node", "npm",
        "git"  # Con cuidado
    ]
    
    FORBIDDEN_PATTERNS = [
        "rm -rf", "del /", "format", "mkfs",
        "sudo", "su ", "> /dev/", "| sh", "| bash",
        "curl", "wget", "nc ", "netcat",
    ]
    
    def __init__(self, require_confirmation: bool = True):
        self.require_confirmation = require_confirmation
        self._pending_command: Optional[str] = None
    
    @property
    def spec(self) -> ToolSpec:
        return ToolSpec(
            name="shell",
            description="Ejecuta comandos shell seguros",
            parameters={
                "command": {
                    "type": "string",
                    "description": "Comando a ejecutar",
                    "required": True
                },
                "cwd": {
                    "type": "string",
                    "description": "Directorio de trabajo",
                    "required": False
                }
            },
            category=ToolCategory.SYSTEM,
            risk=ToolRisk.DANGEROUS,
            examples=[
                'shell(command="ls -la")',
                'shell(command="python --version")'
            ]
        )
    
    def execute(self, command: str, cwd: str = None,
                confirmed: bool = False) -> ToolResult:
        
        # Validar comando
        validation_error = self._validate_command(command)
        if validation_error:
            return ToolResult(
                success=False,
                output=None,
                error=validation_error
            )
        
        # Requerir confirmación
        if self.require_confirmation and not confirmed:
            self._pending_command = command
            return ToolResult(
                success=False,
                output=None,
                error="REQUIRES_CONFIRMATION",
                metadata={
                    "pending_action": "shell",
                    "command": command
                }
            )
        
        try:
            import shlex
            cmd_parts = shlex.split(command)
            result = subprocess.run(
                cmd_parts,
                shell=False,
                capture_output=True,
                text=True,
                timeout=30,
                cwd=cwd
            )
            
            return ToolResult(
                success=result.returncode == 0,
                output=result.stdout or result.stderr,
                error=result.stderr if result.returncode != 0 else None,
                metadata={
                    "command": command,
                    "return_code": result.returncode
                }
            )
            
        except subprocess.TimeoutExpired:
            return ToolResult(
                success=False,
                output=None,
                error="Timeout: comando tomó más de 30 segundos"
            )
        except Exception as e:
            return ToolResult(
                success=False,
                output=None,
                error=str(e)
            )
    
    def _validate_command(self, command: str) -> Optional[str]:
        """Valida que el comando sea seguro."""
        command_lower = command.lower()
        
        # Verificar patrones prohibidos
        for pattern in self.FORBIDDEN_PATTERNS:
            if pattern.lower() in command_lower:
                return f"Comando prohibido: contiene '{pattern}'"
        
        # Verificar que el comando base sea permitido
        base_command = command.split()[0] if command.split() else ""
        
        # Permitir rutas absolutas a python
        if "python" in base_command.lower():
            return None
        
        if base_command not in self.ALLOWED_COMMANDS:
            return f"Comando no permitido: {base_command}. Permitidos: {self.ALLOWED_COMMANDS}"
        
        return None


# =============================================================================
# REGISTRO DE HERRAMIENTAS
# =============================================================================

class ToolRegistry:
    """Registro centralizado de herramientas."""
    
    def __init__(self):
        self.tools: Dict[str, Tool] = {}
    
    def register(self, tool: Tool):
        """Registra una herramienta."""
        self.tools[tool.spec.name] = tool
    
    def get(self, name: str) -> Optional[Tool]:
        """Obtiene herramienta por nombre."""
        return self.tools.get(name)
    
    def list_tools(self) -> List[ToolSpec]:
        """Lista todas las herramientas."""
        return [tool.spec for tool in self.tools.values()]
    
    def get_safe_tools(self) -> List[ToolSpec]:
        """Lista solo herramientas seguras."""
        return [t.spec for t in self.tools.values() 
                if t.spec.risk == ToolRisk.SAFE]
    
    def to_function_specs(self) -> List[Dict]:
        """Convierte a formato de function calling."""
        return [tool.spec.to_function_spec() for tool in self.tools.values()]
    
    def execute(self, tool_name: str, **kwargs) -> ToolResult:
        """Ejecuta una herramienta por nombre."""
        tool = self.get(tool_name)
        if not tool:
            return ToolResult(
                success=False,
                output=None,
                error=f"Herramienta no encontrada: {tool_name}"
            )
        
        return tool.execute(**kwargs)


def create_default_registry() -> ToolRegistry:
    """Crea registro con herramientas por defecto."""
    registry = ToolRegistry()
    
    registry.register(FileReaderTool())
    registry.register(FileWriterTool())
    registry.register(CalculatorTool())
    registry.register(PythonExecutorTool())
    registry.register(ShellExecutorTool())
    
    return registry


if __name__ == "__main__":
    # Test de herramientas
    registry = create_default_registry()
    
    print("=== Herramientas Registradas ===")
    for spec in registry.list_tools():
        print(f"\n{spec.name} ({spec.category.value}) - Riesgo: {spec.risk.value}")
        print(f"  {spec.description}")
    
    print("\n\n=== Tests ===")
    
    # Test calculadora
    calc_result = registry.execute("calculate", expression="sqrt(16) + 3**2")
    print(f"\nCalculadora: sqrt(16) + 3**2 = {calc_result.output}")
    
    # Test Python
    py_result = registry.execute("python_exec", 
                                  code="result = [x**2 for x in range(5)]")
    print(f"\nPython: [x**2 for x in range(5)] = {py_result.output}")
    
    # Test lectura (simulado)
    print("\n\nFunction specs para LLM:")
    print(json.dumps(registry.to_function_specs(), indent=2))

# KAMAQ Super-Agent: Guía de Implementación

## 🎯 ¿Qué es KAMAQ Super-Agent?

KAMAQ Super-Agent es un sistema de IA local que **envuelve** un LLM (como Mistral, Qwen o LLaMA) 
con capas adicionales de:

1. **Constitución de Valores** - Ética y principios auditables
2. **Razonamiento Estructurado** - Pensamiento paso a paso
3. **Memoria Holográfica** - Nunca olvida
4. **Herramientas** - Interactúa con el mundo real
5. **Verificación** - Honestidad y calibración

Es como JARVIS de Iron Man, pero con la visión de Tesla sobre elegancia y eficiencia.

---

## 🚀 Inicio Rápido

### 1. Instalar Ollama

```bash
# Windows: Descargar de https://ollama.ai

# Verificar instalación
ollama --version
```

### 2. Descargar Modelo Recomendado

```bash
# Qwen 2.5 7B (recomendado para español + herramientas)
ollama pull qwen2.5:7b

# Alternativas
ollama pull mistral:7b        # Más rápido
ollama pull llama3:8b         # Mejor en inglés
ollama pull deepseek-coder:6.7b  # Mejor para código
```

### 3. Instalar Dependencias Python

```bash
pip install numpy scipy pyyaml
```

### 4. Ejecutar el Agente

```bash
cd kamaq
python -m kamaq_agent.agent
```

---

## 📦 Estructura del Proyecto

```
kamaq_agent/
├── __init__.py
├── agent.py                 # Agente principal
├── config/
│   └── constitution.yaml    # Valores y principios
└── core/
    ├── __init__.py
    ├── constitution.py      # Sistema de valores
    ├── reasoning.py         # Razonamiento estructurado
    ├── tools.py             # Herramientas (archivos, cálculo, código)
    └── verifier.py          # Verificación y calibración
```

---

## 🔧 Componentes

### 1. Constitución (`constitution.py`)

Define los valores fundamentales del agente:

```python
from kamaq_agent import KAMAQConstitution

constitution = KAMAQConstitution()

# Evaluar riesgo de una acción
risk = constitution.evaluate_action_risk("borrar archivos", {})
print(risk)  # RiskLevel.HIGH

# Obtener prompt de sistema
system_prompt = constitution.to_system_prompt()
```

**Valores incluidos:**
- Honestidad
- Seguridad
- Privacidad
- Transparencia
- Humildad
- Calibración

**Principios Tesla:**
- Elegancia
- Eficiencia
- Escalabilidad
- Robustez
- Impacto

### 2. Razonamiento (`reasoning.py`)

Pipeline de pensamiento estructurado:

```python
from kamaq_agent import ReasoningPipeline

pipeline = ReasoningPipeline(llm_caller=my_llm_function)

# Razonamiento completo
trace = pipeline.reason("¿Cómo optimizar mi código?")

print(trace.final_answer)
print(f"Confianza: {trace.overall_confidence:.0%}")
```

**Pasos del pipeline:**
1. COMPRENSIÓN - ¿Qué se pide?
2. HIPÓTESIS - Posibles respuestas
3. PLAN - Cómo verificar
4. EJECUCIÓN - Hacer el trabajo
5. VERIFICACIÓN - ¿Es correcto?
6. CONCLUSIÓN - Respuesta final

### 3. Herramientas (`tools.py`)

Interacción con el mundo real:

```python
from kamaq_agent import create_default_registry

tools = create_default_registry()

# Calculadora segura
result = tools.execute("calculate", expression="sqrt(16) + 3**2")
print(result.output)  # 13.0

# Leer archivo
result = tools.execute("read_file", path="README.md")
print(result.output)

# Ejecutar Python (sandbox)
result = tools.execute("python_exec", code="result = [x**2 for x in range(5)]")
print(result.output)  # [0, 1, 4, 9, 16]
```

**Herramientas disponibles:**
| Herramienta   | Descripción          |  Riesgo   |
| :------------ | :------------------- | :-------: |
| `read_file`   | Leer archivos        |  Seguro   |
| `write_file`  | Escribir archivos    | Moderado  |
| `calculate`   | Cálculos matemáticos |  Seguro   |
| `python_exec` | Ejecutar Python      | Moderado  |
| `shell`       | Comandos shell       | Peligroso |

### 4. Verificador (`verifier.py`)

Honestidad y calibración:

```python
from kamaq_agent import KAMAQVerifier

verifier = KAMAQVerifier()

# Verificar respuesta
results = verifier.verify_response(
    "La suma de 2 + 2 = 5",
    claimed_confidence=0.9
)

print(results["warnings"])  # ["Error matemático: 2 + 2 = 4, no 5"]

# Registrar resultado para calibración
verifier.record_outcome(confidence=0.8, was_correct=True)

# Reporte de calibración
print(verifier.get_calibration_report())
```

---

## 💬 Uso Interactivo

```bash
python -m kamaq_agent.agent
```

### Comandos Especiales

| Comando        | Descripción                 |
| :------------- | :-------------------------- |
| `/stats`       | Ver estadísticas del agente |
| `/calibration` | Reporte de calibración      |
| `/mode <modo>` | Cambiar modo de operación   |
| `salir`        | Terminar sesión             |

### Modos de Operación

- `asistente` - Conversación normal
- `enfocado` - Trabajo concentrado
- `creativo` - Brainstorming
- `critico` - Análisis riguroso
- `explorar` - Búsqueda activa

---

## 🔬 Integración con KAMAQ Existente

El agente integra automáticamente:

1. **Memoria Holográfica** (`kamaq_companion/core/holographic_memory.py`)
   - Si está disponible, la usa para recordar conversaciones

2. **Metacognición** (`prototipo_v2/metacognicion.py`)
   - Si está disponible, usa medición de incertidumbre avanzada

---

## 📊 Ejemplo de Sesión

```
=======================================================================
  🌟 KAMAQ SUPER-AGENT v0.1.0
=======================================================================
  Modelo LLM: ✓ qwen2.5:7b
  Memoria: ✓ Holográfica
  Metacognición: ✓ Avanzada
  Herramientas: 5 disponibles
  Modo: asistente
=======================================================================

💬 Escribe tu mensaje (o 'salir' para terminar):

Tú: Calcula la raíz cuadrada de 256 más 15 al cuadrado

KAMAQ: El resultado es 241.0

Calculé:
- √256 = 16
- 15² = 225
- 16 + 225 = 241

  [Confianza: 100% | Tiempo: 45ms]
  [Herramientas: calculate]

Tú: ¿Qué es la programación funcional?

KAMAQ: La programación funcional es un paradigma donde...

## PASO 1: COMPRENSIÓN
El usuario pregunta sobre un concepto de programación.

## PASO 2: HIPÓTESIS
Es un paradigma de programación basado en funciones puras...

[... respuesta completa ...]

  [Confianza: 85% | Tiempo: 2341ms]

Tú: /stats
{
  "name": "KAMAQ",
  "total_interactions": 2,
  "session_duration_minutes": 3,
  "memory": {
    "total_memories": 47,
    "session_count": 12
  }
}

Tú: salir

👋 ¡Hasta pronto!
```

---

## 🎯 Próximos Pasos

### Fase 1: Core (Actual)
- [x] Constitución de valores
- [x] Razonamiento estructurado
- [x] Herramientas básicas
- [x] Verificación y calibración
- [ ] Tests completos

### Fase 2: Agencia
- [ ] Gestor de tareas y subtareas
- [ ] Sondas exploratorias
- [ ] Ejecución multi-paso autónoma

### Fase 3: Inteligencia
- [ ] Grafo de conocimiento
- [ ] Aprendizaje continuo
- [ ] Modos de operación adaptativos

### Fase 4: Producción
- [ ] API REST
- [ ] Interfaz web
- [ ] Plugins de VS Code

---

## ⚠️ Limitaciones Honestas

### Lo que SÍ puede hacer:
- Razonar paso a paso
- Usar herramientas (archivos, código, cálculo)
- Recordar conversaciones
- Admitir cuando no sabe
- Pedir confirmación ante riesgo

### Lo que NO puede hacer (aún):
- Igualar a GPT-4/Claude en conocimiento general
- Procesar imágenes o audio
- Navegar la web en tiempo real
- Aprender durante la sesión (solo RAG)

### Lo que NUNCA hará:
- Mentir sobre sus capacidades
- Ejecutar código peligroso sin confirmación
- Inventar información

---

## 📜 Filosofía

> "La diferencia entre lo posible y lo imposible está en la determinación." 
> — Tommy Lasorda

> "El futuro pertenece a quienes creen en la belleza de sus sueños."
> — Nikola Tesla

KAMAQ no intenta ser el modelo más "inteligente". 
Intenta ser el más **honesto** y **útil**.

---

## 🤝 Contribuir

1. Fork el repositorio
2. Crea una rama: `git checkout -b feature/nueva-funcionalidad`
3. Commit: `git commit -m 'Agregar nueva funcionalidad'`
4. Push: `git push origin feature/nueva-funcionalidad`
5. Abre un Pull Request

---

**KAMAQ Team** - Enero 2026

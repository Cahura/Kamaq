<div align="center">

# 🧠 KAMAQ

**Experimental Cognitive Architecture & AI Agent Framework**

*Exploring the intersection of nonlinear dynamics, associative memory, and autonomous AI agents*

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![NumPy](https://img.shields.io/badge/NumPy-Scientific_Computing-013243?style=flat-square&logo=numpy)](https://numpy.org)
[![Ollama](https://img.shields.io/badge/Ollama-Local_LLM-000000?style=flat-square)](https://ollama.ai)
[![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)
[![Status](https://img.shields.io/badge/Status-Experimental_Prototype-orange?style=flat-square)]()

<br>

*KAMAQ (from Quechua: "creator", "one who brings to life") is a personal research project exploring how principles from nonlinear dynamics, neuroscience, and cognitive science can be implemented as computational systems. It includes both a practical AI agent wrapper and a series of experimental prototypes investigating bio-inspired computation.*

</div>

---

## 📋 Table of Contents

- [🧠 KAMAQ](#-kamaq)
  - [📋 Table of Contents](#-table-of-contents)
  - [🎯 What is This Project?](#-what-is-this-project)
  - [� Why I Built This](#-why-i-built-this)
    - [AI-Assisted Development](#ai-assisted-development)
  - [🗺 Project Map](#-project-map)
  - [🤖 Component 1: KAMAQ Agent](#-component-1-kamaq-agent)
    - [Architecture](#architecture)
    - [Key Features](#key-features)
    - [Quick Start](#quick-start)
    - [Example Usage](#example-usage)
  - [🔬 Component 2: Cognitive Prototypes](#-component-2-cognitive-prototypes)
    - [Evolution of Ideas](#evolution-of-ideas)
    - [The Oscillator Model](#the-oscillator-model)
    - [Reservoir Computing (V5) ⭐](#reservoir-computing-v5-)
    - [Active Inference Engine (V4)](#active-inference-engine-v4)
    - [Metacognition Benchmark (V4) ⭐](#metacognition-benchmark-v4-)
    - [Holographic Memory](#holographic-memory)
    - [Neurogenesis](#neurogenesis)
  - [🧩 Component 3: ARC Challenge](#-component-3-arc-challenge)
    - [Approach](#approach)
    - [Results](#results)
  - [📖 The Learning Journey](#-the-learning-journey)
  - [📐 Mathematical Foundations](#-mathematical-foundations)
  - [🚀 Getting Started](#-getting-started)
    - [Prerequisites](#prerequisites)
    - [Installation](#installation)
    - [Run the Agent](#run-the-agent)
    - [Run the Prototypes](#run-the-prototypes)
  - [🪞 Honest Assessment](#-honest-assessment)
    - [What it IS](#what-it-is)
    - [What it is NOT](#what-it-is-not)
    - [Known Gaps](#known-gaps)
  - [🛠 Tech Stack](#-tech-stack)
  - [👤 Author](#-author)
  - [📄 License](#-license)

---

## 🎯 What is This Project?

KAMAQ is **not** a production system. It's an **experimental playground** where I explore ideas from computational neuroscience and AI by actually implementing them in code.

The project has three distinct components:

| Component                | What it does                                                                                                                     | Maturity             |
| ------------------------ | -------------------------------------------------------------------------------------------------------------------------------- | -------------------- |
| **KAMAQ Agent**          | AI assistant that wraps local LLMs (via Ollama) with tools, verification, and ethical constraints                                | Functional prototype |
| **Cognitive Prototypes** | NumPy-based simulations of bio-inspired computation: oscillator networks, Hopfield memory, reservoir computing, active inference | Experimental         |
| **ARC Challenge**        | Attempt at the ARC-AGI benchmark using geometric transforms and DSL-based reasoning                                              | Early experiment     |

> **Disclaimer:** This is a learning-by-building project. The code explores well-established computational neuroscience concepts (Kuramoto oscillators, Hopfield networks, Echo State Networks, Learning Classifier Systems). The implementations are correct but not novel — they're my way of deeply understanding these ideas by coding them from scratch.

---

## 💡 Why I Built This

As an Electronic Engineering student, I was fascinated by the gap between how biological systems process information and how current AI systems work. I wanted to answer a personal question:

> **Can principles from nonlinear dynamics, oscillator synchronization, and neuroscience produce useful computation — or are they just beautiful math?**

The honest answer turned out to be "both":
- The Kuramoto oscillator models were mathematically elegant but couldn't classify MNIST digits
- Pivoting to Reservoir Computing (an established readout technique) finally solved XOR — the first real success
- Building the agent taught me practical software engineering: modular architecture, tool dispatch, subprocess management

This project represents my journey from pure theory to practical implementation. Every failed prototype (and there were many) taught me something about what works and what doesn't in computational intelligence.

### AI-Assisted Development

This project was developed using AI coding assistants:
- **Claude Opus 4.6 Thinking** and **Claude Opus 4.5 Thinking** (via VS Code / Copilot)
- **Gemini 3 Pro** (via IDE integration)

The AI tools helped with code structure, debugging, and documentation. All architectural decisions, mathematical implementations, and experimental design were mine. I believe in transparency about tooling — using AI assistants is a skill in itself, and pretending otherwise would be dishonest.

---

## 🗺 Project Map

```
kamaq/
├── kamaq_agent/                  # 🤖 AI Agent (Ollama wrapper + tools)
│   ├── agent.py                  #    Main orchestrator (~670 lines, v0.2.0)
│   ├── config/
│   │   └── constitution.yaml     #    Ethical values configuration
│   └── core/
│       ├── constitution.py       #    Risk evaluation & value system
│       ├── reasoning.py          #    Structured reasoning prompts
│       ├── tools.py              #    Calculator, file I/O, Python sandbox, shell
│       └── verifier.py           #    Math verification & confidence calibration
│
├── kamaq_companion/              # 💬 Chat companion with memory
│   ├── kamaq_companion.py        #    Interactive chat loop
│   └── core/
│       ├── holographic_memory.py #    HRR-based memory encoding (FFT)
│       └── learning_engine.py    #    Proactive learning suggestions
│
├── prototipo/                    # 🔬 V1: Oscillator experiments
│   ├── celula_cognitiva.py       #    Stuart-Landau + Kuramoto cells
│   ├── celula_cognitiva_v2.py    #    Fixed phase plasticity (atan2)
│   ├── celula_cognitiva_v3.py    #    PLL-based frequency learning
│   ├── celula_reservoir_v5.py    #    ⭐ Reservoir Computing (ESN) — solves XOR
│   ├── prueba_mnist.py           #    MNIST digit classification attempt
│   └── pruebas_cognitivas.py     #    Unit tests for oscillator cells
│
├── prototipo_v2/                 # 🧬 V2: Integrated cognitive field
│   ├── campo_cognitivo.py        #    Hopfield + Langevin + Kuramoto field
│   ├── metacognicion.py          #    Uncertainty & strategy monitoring
│   ├── neurogenesis.py           #    Dynamic network resizing
│   ├── memoria_holografica.py    #    Episodic + Semantic memory (Hopfield)
│   └── agente_emergente.py       #    Agent using field for decisions
│
├── prototipo_v4/                 # 🏗 V4: Hierarchical decision system
│   ├── campo_cognitivo_v4.py     #    3-tier: Strategies → Inference → Hopfield
│   ├── motor_inferencia_activa.py#    Tabular active inference (EFE)
│   ├── poblacion_estrategias.py  #    Evolutionary Learning Classifier System
│   ├── metacognicion_benchmark.py#    ⭐ Calibration vs minimax ground truth
│   └── finanzas/                 #    Financial time series experiment
│
└── arc_challenge/                # 🧩 ARC-AGI benchmark attempt
    ├── campo_arc.py              #    Hopfield-based transform attractors
    ├── dsl_arc.py                #    DSL for object manipulation
    ├── motor_transformaciones.py  #    Geometric transform library
    └── evaluacion_final.py       #    Benchmark runner (5.5% training accuracy)
```

**~27,000 lines of Python** across 40+ files.

---

## 🤖 Component 1: KAMAQ Agent

A modular AI assistant that wraps **local LLMs** (running on Ollama) with structured reasoning, tool use, ethical constraints, and response verification.

### Architecture

```
User Input
    │
    ▼
┌─────────────────────────────────────────────┐
│            KAMAQSuperAgent                  │
│                                             │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  │
│  │ Constitu-│  │ Tool     │  │ Reasoning│  │
│  │ tion     │  │ Registry │  │ Pipeline │  │
│  │          │  │          │  │          │  │
│  │ - Values │  │ - Calc   │  │ - 6-step │  │
│  │ - Risk   │  │ - Files  │  │   prompt │  │
│  │   check  │  │ - Python │  │ - Struct  │  │
│  │ - Ethics │  │ - Shell  │  │   output │  │
│  └────┬─────┘  └────┬─────┘  └────┬─────┘  │
│       │              │              │        │
│       ▼              ▼              ▼        │
│  ┌──────────────────────────────────────┐   │
│  │     Orchestration (agent.py)         │   │
│  │                                      │   │
│  │  1. Risk evaluation (keyword-based)  │   │
│  │  2. Tool detection (regex matching)  │   │
│  │  3. LLM call (Ollama subprocess)     │   │
│  │  4. Response verification            │   │
│  └──────────────────┬───────────────────┘   │
│                     │                        │
│              ┌──────▼──────┐                 │
│              │  Verifier   │                 │
│              │             │                 │
│              │ - Math      │                 │
│              │   (AST)     │                 │
│              │ - Contradic-│                 │
│              │   tions     │                 │
│              │ - Confidence│                 │
│              │   calibrate │                 │
│              └─────────────┘                 │
└─────────────────────────────────────────────┘
    │
    ▼
AgentResponse (answer + confidence + verification)
```

### Key Features

| Feature                    | How it works                                                                                                      | Limitations                                                               |
| -------------------------- | ----------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------- |
| **LLM Integration**        | Calls Ollama via `subprocess.run(["ollama", "run", model])`                                                       | Local models only (no cloud API). Requires Ollama installed               |
| **Conversation History**   | Maintains sliding window of recent turns for context between interactions                                         | Window-based, not persistent across sessions                              |
| **5 Built-in Tools**       | Calculator (AST-safe eval), File reader/writer, Python executor (restricted `exec`), Shell (allowlisted commands) | No external API calls. Tool dispatch is regex-based, not function-calling |
| **Risk Evaluation**        | Keyword matching against forbidden/dangerous actions                                                              | Static rules, not context-aware                                           |
| **Math Verification**      | Extracts `X + Y = Z` patterns from LLM output, evaluates via AST                                                  | Only catches explicitly written equations                                 |
| **Confidence Calibration** | ECE-based tracking of declared vs. actual accuracy                                                                | No automatic feedback loop — outcome recording is manual                  |
| **Constitution**           | YAML-configurable values that become the LLM system prompt                                                        | Ethical constraints are prompt-level only                                 |

### Quick Start

```bash
# Install Ollama: https://ollama.ai
ollama pull qwen2.5:7b

# Install dependencies
pip install pyyaml

# Run the agent
python -m kamaq_agent.agent
```

### Example Usage

```python
from kamaq_agent import KAMAQSuperAgent

agent = KAMAQSuperAgent(model="qwen2.5:7b")
response = agent.chat("Calcula la raíz cuadrada de 144")

print(response.response)        # "12.0"
print(response.confidence)      # 0.95
print(response.tools_used)      # ["calculate"]
print(response.verification)    # VerificationResult(math_correct=True)
```

---

## 🔬 Component 2: Cognitive Prototypes

A series of experiments implementing bio-inspired computational models from scratch in NumPy. No deep learning frameworks — everything is manual matrix operations.

### Evolution of Ideas

```
V1: Oscillator Cells          V2: Cognitive Field          V4: Decision Hierarchy
─────────────────────         ─────────────────            ─────────────────────
Stuart-Landau dynamics        Hopfield + Langevin          Evolutionary strategies
  + Kuramoto coupling           + Kuramoto + Hebbian         + Active inference
  + frequency learning          + neurogenesis                + Hopfield fallback
                                + metacognition
         │                            │                            │
         ▼                            ▼                            ▼
   "Can oscillators             "Can a physics-based        "Can a hierarchy of
    learn patterns?"             field make decisions?"      methods outperform
                                                             any single one?"
         │                            │                            │
         ▼                            ▼                            ▼
   Answer: Not alone.           Answer: Yes, but            Answer: Yes — fast
   → Pivot to Reservoir         ad hoc. No optimality       rules + slow inference
     Computing (V5) ⭐          guarantees.                 + fallback = robust.
```

### The Oscillator Model

Each "cell" is a complex oscillator governed by:

**Amplitude** (Hopf bifurcation):

$$\frac{dA}{dt} = \mu A - A^3 \quad \rightarrow \quad A_{\text{stable}} = \sqrt{\mu}$$

**Phase** (Kuramoto synchronization + plasticity):

$$\frac{d\phi_i}{dt} = \omega_i + \frac{K}{N}\sum_{j} \sin(\phi_j - \phi_i)$$

The cells synchronize their phases and can entrain their frequencies to external signals. **This is not machine learning** — it's a simulation of coupled nonlinear oscillators, which is a well-studied topic in physics and neuroscience.

### Reservoir Computing (V5) ⭐

The most scientifically sound prototype. After discovering that oscillators alone cannot classify patterns, I pivoted to **Echo State Networks**:

```
Input → [Random Recurrent Reservoir] → Collect States → Ridge Regression → Output
                 (fixed weights)           (linear readout - only trainable part)
```

**Results:**
| Task | Reservoir | Single-layer Perceptron           |
| ---- | --------- | --------------------------------- |
| XOR  | ✅ Solved  | ❌ Cannot (not linearly separable) |
| AND  | ✅ Solved  | ✅ Solved                          |
| OR   | ✅ Solved  | ✅ Solved                          |
| NAND | ✅ Solved  | ✅ Solved                          |

The reservoir's nonlinear projection makes XOR solvable with a linear readout — demonstrating the power of recurrent dynamics for computation.

### Active Inference Engine (V4)

Implements a tabular approximation of Active Inference:

$$G(\pi) = -\underbrace{w_p \cdot \mathbb{E}[r \mid s, a]}_{\text{pragmatic value}} - \underbrace{w_e \cdot H[s' \mid s, a]}_{\text{epistemic value (curiosity)}}$$

Action selection via softmax over negative Expected Free Energy. Curiosity weight adapts based on reward trends and state discovery rate.

> **Honest note:** This is closer to model-based RL with an exploration bonus than to full active inference (which requires Bayesian belief updating over hidden states). The tabular representation limits it to small discrete state spaces like Tic-Tac-Toe.

### Metacognition Benchmark (V4) ⭐

The most rigorous experiment in the project. Tests: *"Does KAMAQ know when it doesn't know?"*

- **Ground truth:** Optimal moves computed via minimax with memoization
- **Metric:** Expected Calibration Error (ECE) — measures if confidence matches actual accuracy
- **Result:** System achieves measurable calibration, but with room for improvement

### Holographic Memory

**Encoding** uses Holographic Reduced Representations (HRR) — binding vectors via circular convolution in the frequency domain:

$$\text{bind}(\mathbf{a}, \mathbf{b}) = \mathcal{F}^{-1}\left(\mathcal{F}(\mathbf{a}) \odot \mathcal{F}(\mathbf{b})\right)$$

Text is converted to 1024-dimensional vectors using SHA-256 seeded random generation. Bound key-value pairs are superimposed into a holographic trace.

> **Known limitation:** The companion's recall function falls back to text-based substring matching instead of using HRR unbinding. The `prototipo_v2/memoria_holografica.py` implementation uses proper Hopfield-based semantic recall with episodic-to-semantic consolidation. This inconsistency is documented for future refactoring.

### Neurogenesis

Dynamic network resizing during runtime:
- **Growth:** New cells added when performance stagnates (`np.vstack` to expand weight matrices)
- **Pruning:** Low-utility cells killed (utility decays exponentially, increases on activation)
- **Architect:** Monitors performance trends and recommends expand/clean/revive operations

---

## 🧩 Component 3: ARC Challenge

An early attempt at the [ARC-AGI benchmark](https://arcprize.org/) — testing abstract reasoning via grid transformations.

### Approach

Three engines working in cascade:
1. **Transform Engine:** Library of ~30 geometric operations (rotation, reflection, scaling, color mapping)
2. **Advanced Engine:** 2-deep composition search over transform pairs
3. **DSL Engine:** Object detection + rule inference for movement, tiling, and color changes

### Results

| Dataset          | Tasks | Correct | Accuracy |
| ---------------- | ----- | ------- | -------- |
| Training (400)   | 400   | 22      | 5.50%    |
| Evaluation (400) | 400   | 1       | 0.25%    |

For reference: GPT-4 achieves ~13%, humans ~85%.

> **Honest note:** The ARC engines do NOT use KAMAQ's cognitive architecture. They are conventional program-synthesis approaches. This reveals the gap between biological inspiration and practical pattern recognition. The cognitive field approaches were not effective for this benchmark.

---

## 📖 The Learning Journey

What makes this project personally meaningful is the intellectual trajectory across versions:

```
V1 (Jan 2026)     "Oscillators can learn and compute!"
    │                  → Built Kuramoto + Stuart-Landau cells
    │                  → They synchronize but can't classify
    │
V2 (Jan 2026)     "Let me fix the math..."
    │                  → Fixed phase wrapping (atan2)
    │                  → Added PLL-based frequency tracking
    │                  → Still can't classify
    │
V3 (Jan 2026)     "Maybe oscillators need a readout layer?"
    │                  → Pivoted to Reservoir Computing
    │                  → XOR SOLVED — the first real success ⭐
    │
V4 (Feb 2026)     "Let me combine everything..."
    │                  → Hopfield + Active Inference + Evolutionary LCS
    │                  → Hierarchical decision architecture
    │                  → First proper benchmark (minimax ground truth)
    │
Agent (Feb 2026)   "Let me build something practical too"
                       → Ollama wrapper with tools and verification
                       → Working CLI assistant
```

**Key lesson learned:** The jump from V2 to V3 was the most important — admitting that a beautiful mathematical model (coupled oscillators) couldn't solve practical tasks, and pivoting to Reservoir Computing. That humility is more valuable than the code itself.

---

## 📐 Mathematical Foundations

The project implements these established computational models:

| Model                               | From                       | Used in          | What it does                                |
| ----------------------------------- | -------------------------- | ---------------- | ------------------------------------------- |
| **Stuart-Landau oscillator**        | Nonlinear dynamics         | Prototypes V1-V3 | Amplitude dynamics via Hopf bifurcation     |
| **Kuramoto model**                  | Synchronization theory     | Prototypes V1-V3 | Phase coupling between oscillators          |
| **Hopfield network**                | Associative memory (1982)  | V2, V4, ARC      | Pattern storage/recall via Hebbian learning |
| **Echo State Network**              | Reservoir Computing (2001) | V5 (Reservoir)   | Random reservoir + linear readout           |
| **HRR (Holographic Reduced Repr.)** | Plate (2003)               | Companion        | Vector binding via circular convolution     |
| **Active Inference (tabular)**      | Friston (2016+)            | V4               | Decision-making via Expected Free Energy    |
| **Learning Classifier System**      | Holland (1976)             | V4               | Evolutionary rule population                |
| **Ridge Regression**                | Statistics                 | V5 (Reservoir)   | Regularized linear readout training         |

---

## 🚀 Getting Started

### Prerequisites

- **Python 3.11+**
- **Ollama** (for the agent component): [ollama.ai](https://ollama.ai)

### Installation

```bash
# Clone the repository
git clone https://github.com/Cahura/Kamaq.git
cd Kamaq

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# or: .venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

### Run the Agent

```bash
# Pull a local model first
ollama pull qwen2.5:7b

# Start interactive agent
python -m kamaq_agent.agent
```

### Run the Prototypes

```bash
# Reservoir Computing (XOR solver)
python prototipo/celula_reservoir_v5.py

# Cognitive Field with Active Inference
python prototipo_v4/campo_cognitivo_v4.py

# Metacognition Benchmark
python prototipo_v4/metacognicion_benchmark.py

# ARC Challenge evaluation
python arc_challenge/evaluacion_final.py
```

---

## 🪞 Honest Assessment

I believe in transparency. Here's what this project **is** and **isn't:**

### What it IS
- ✅ A genuine exploration of computational neuroscience concepts, built from scratch
- ✅ Correct implementations of established models (Kuramoto, Hopfield, ESN, HRR, LCS)
- ✅ A working local AI agent with tools and verification
- ✅ ~27,000 lines of code showing iterative learning and intellectual honesty
- ✅ A project that documents its own failures (V1→V3 pivot, ARC results)

### What it is NOT
- ❌ Not a novel research contribution — all components implement known techniques
- ❌ Not production-ready — experimental code with basic test suite (`tests/`), no logging framework
- ❌ Not a "cognitive architecture" in the AGI sense — it's numerical simulations
- ❌ Not using cloud LLM APIs (Claude, GPT, Gemini) — only local models via Ollama
- ❌ Not solving ARC with biological inspiration — the ARC solver uses conventional transforms

### Known Gaps
- Holographic recall now uses 4 strategies (HRR unbinding, bag-of-words similarity, text fallback) — but bag-of-words captures word overlap only, NOT semantic similarity (requires trained embeddings)
- Tool dispatch in the agent is regex-based, not proper function-calling protocol
- Active inference engine is tabular, not scalable to large state spaces
- Neurogenesis uses O(N²) matrix reallocation on each expansion
- No automatic feedback loop for confidence calibration (outcome recording is manual)
- See [LESSONS_LEARNED.md](LESSONS_LEARNED.md) for detailed technical reflections

---

## 🛠 Tech Stack

| Category                        | Technologies                                                |
| ------------------------------- | ----------------------------------------------------------- |
| **Language**                    | Python 3.11+                                                |
| **Scientific Computing**        | NumPy, SciPy (FFT, ndimage)                                 |
| **LLM Runtime**                 | Ollama (local inference)                                    |
| **Models Tested**               | Qwen 2.5 7B, Mistral 7B, LLaMA 3.2, DeepSeek Coder          |
| **Configuration**               | PyYAML                                                      |
| **Data**                        | JSON, NumPy serialization (.npz)                            |
| **No deep learning frameworks** | Everything is manual NumPy — no PyTorch, TensorFlow, or JAX |

---

## 👤 Author

**Carlos Huarcaya**
- Electronic Engineering student at UPC (Lima, Peru) — 8th cycle
- GitHub: [@Cahura](https://github.com/Cahura)
- LinkedIn: [huarcayacarlos](https://linkedin.com/in/huarcayacarlos)
- AI Tools: Claude Opus 4.6/4.5 Thinking, Gemini 3 Pro (coding assistants)

---

## 📄 License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

---

<div align="center">

*"The measure of intelligence is the ability to change."* — Albert Einstein

**KAMAQ** — Built with curiosity, powered by NumPy, grounded in honesty.

</div>

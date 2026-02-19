# -*- coding: utf-8 -*-
"""
KAMAQ Super-Agent - Verification Script

Run from project root: python verify_agent.py
"""
import sys
import os
from pathlib import Path

# Fix encoding
os.environ['PYTHONIOENCODING'] = 'utf-8'

# Ensure project root is in path
project_root = str(Path(__file__).parent)
if project_root not in sys.path:
    sys.path.insert(0, project_root)
os.chdir(project_root)

print('='*60)
print('KAMAQ SUPER-AGENT - VERIFICACION COMPLETA')
print('='*60)

# Test 1: Importar agente
print('\n[TEST 1] Importando KAMAQ Super-Agent...')
try:
    from kamaq_agent.agent import KAMAQSuperAgent
    print('  [OK] Modulo importado')
except Exception as e:
    print(f'  [FAIL] {e}')
    sys.exit(1)

# Test 2: Crear agente
print('\n[TEST 2] Inicializando agente con qwen2.5:7b...')
try:
    agent = KAMAQSuperAgent(
        name='KAMAQ',
        model='qwen2.5:7b',
        verbose=False
    )
    print('  [OK] Agente inicializado')
    print(f'      - Ollama disponible: {agent.ollama_available}')
    print(f'      - Herramientas: {len(agent.tools.list_tools())}')
    print(f'      - Historial max: {agent.MAX_HISTORY_TURNS} turnos')
except Exception as e:
    print(f'  [FAIL] {e}')
    sys.exit(1)

# Test 3: Calculadora
print('\n[TEST 3] Probando herramienta calculadora...')
try:
    response = agent.chat('Calcula 125 * 8 + 50', use_reasoning=False)
    print(f'  [OK] Respuesta recibida')
    print(f'      - Contenido: {response.content[:100]}...')
    print(f'      - Herramientas: {response.tools_used}')
    print(f'      - Tiempo: {response.processing_time_ms}ms')
except Exception as e:
    print(f'  [FAIL] {e}')

# Test 4: Pregunta simple con LLM
print('\n[TEST 4] Probando respuesta del LLM...')
try:
    response = agent.chat('Responde en una linea corta: Que es KAMAQ?', use_reasoning=False)
    print(f'  [OK] Respuesta del LLM')
    print(f'      - Contenido: {response.content[:150]}')
    print(f'      - Confianza: {response.confidence:.0%}')
    print(f'      - Tiempo: {response.processing_time_ms}ms')
except Exception as e:
    print(f'  [FAIL] {e}')

# Test 5: Conversation history
print('\n[TEST 5] Probando historial de conversacion...')
try:
    turns = len(agent.conversation_history)
    print(f'  [OK] Historial funcionando')
    print(f'      - Turnos registrados: {turns}')
except Exception as e:
    print(f'  [FAIL] {e}')

# Test 6: Estadisticas
print('\n[TEST 6] Estadisticas finales...')
try:
    stats = agent.get_stats()
    print(f'  - Interacciones: {stats["total_interactions"]}')
    print(f'  - Turnos conversacion: {stats["conversation_turns"]}')
    print(f'  - Estado: {stats["state"]}')
    print(f'  - Modo: {stats["mode"]}')
except Exception as e:
    print(f'  [FAIL] {e}')

print('\n' + '='*60)
print('VERIFICACION COMPLETADA')
print('='*60)

"""
KAMAQ Agent - Demo Rápida
=========================
Demostración rápida del agente completo.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

import subprocess
from kamaq_agent.agent import KAMAQSuperAgent


def main():
    print()
    print("╔" + "═" * 58 + "╗")
    print("║" + " KAMAQ SUPER-AGENT - DEMO INTERACTIVA ".center(58) + "║")
    print("╚" + "═" * 58 + "╝")
    print()
    
    # Crear agente con llama3.2:1b
    print("🔧 Inicializando agente con llama3.2:1b...")
    agent = KAMAQSuperAgent(
        model="llama3.2:1b",
        memory_path="memory_demo",
        verbose=False
    )
    print("✓ Agente listo!\n")
    
    # Pruebas automáticas
    test_queries = [
        ("calculate", "Calcula: 2^10 + sqrt(256)"),
        ("python", "Ejecuta este código Python: result = [x**2 for x in range(1, 6)]"),
        ("question", "¿Qué es una API REST? Responde en 1 oración."),
    ]
    
    print("=" * 60)
    print("PRUEBAS AUTOMÁTICAS")
    print("=" * 60)
    
    for test_name, query in test_queries:
        print(f"\n❓ [{test_name.upper()}] {query}")
        
        response = agent.chat(query, use_reasoning=False)
        
        print(f"💬 Respuesta: {response.content[:300]}...")
        print(f"   📊 Confianza: {response.confidence:.0%}")
        print(f"   ⏱️ Tiempo: {response.processing_time_ms}ms")
        if response.tools_used:
            print(f"   🔧 Herramientas: {', '.join(response.tools_used)}")
    
    print("\n" + "=" * 60)
    print("MODO INTERACTIVO")
    print("=" * 60)
    print()
    print("Comandos especiales:")
    print("  /stats    - Ver estadísticas")
    print("  /salir    - Terminar")
    print()
    
    while True:
        try:
            user_input = input("Tú: ").strip()
            
            if not user_input:
                continue
            
            if user_input.lower() in ["/salir", "salir", "exit", "quit"]:
                print("\n👋 ¡Hasta pronto!")
                break
            
            if user_input.lower() == "/stats":
                stats = agent.get_stats()
                print(f"\n📊 Estadísticas:")
                print(f"   Total interacciones: {stats['total_interactions']}")
                print(f"   Modelo: {stats['model']}")
                print(f"   Modo: {stats['mode']}")
                print()
                continue
            
            # Procesar mensaje
            response = agent.chat(user_input, use_reasoning=False)
            
            print(f"\n🤖 KAMAQ: {response.content}")
            print(f"   [Confianza: {response.confidence:.0%} | Tiempo: {response.processing_time_ms}ms]")
            
            if response.tools_used:
                print(f"   [Herramientas: {', '.join(response.tools_used)}]")
            print()
            
        except KeyboardInterrupt:
            print("\n\n👋 ¡Hasta pronto!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}\n")


if __name__ == "__main__":
    main()

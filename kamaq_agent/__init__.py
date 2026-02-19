"""
KAMAQ Agent Package
===================
Agente de IA local con razonamiento estructurado, herramientas
y verificación, usando Ollama como backend de LLM.
"""

__version__ = "0.2.0"
__author__ = "Carlos Huarcaya"

# Lazy imports — avoid side effects (Ollama subprocess, prints) on package import
def __getattr__(name):
    """Import heavy modules only when accessed."""
    _lazy_map = {
        "KAMAQSuperAgent": "kamaq_agent.agent",
        "AgentResponse": "kamaq_agent.agent",
        "AgentState": "kamaq_agent.agent",
        "ConversationTurn": "kamaq_agent.agent",
        "KAMAQConstitution": "kamaq_agent.core.constitution",
        "OperatingMode": "kamaq_agent.core.constitution",
        "RiskLevel": "kamaq_agent.core.constitution",
        "ReasoningPipeline": "kamaq_agent.core.reasoning",
        "QuickReasoner": "kamaq_agent.core.reasoning",
        "ToolRegistry": "kamaq_agent.core.tools",
        "create_default_registry": "kamaq_agent.core.tools",
        "KAMAQVerifier": "kamaq_agent.core.verifier",
    }
    if name in _lazy_map:
        import importlib
        module = importlib.import_module(_lazy_map[name])
        return getattr(module, name)
    raise AttributeError(f"module 'kamaq_agent' has no attribute {name!r}")

__all__ = [
    "KAMAQSuperAgent",
    "AgentResponse",
    "AgentState",
    "ConversationTurn",
    "KAMAQConstitution",
    "OperatingMode",
    "RiskLevel",
    "ReasoningPipeline",
    "QuickReasoner",
    "ToolRegistry",
    "create_default_registry",
    "KAMAQVerifier",
]

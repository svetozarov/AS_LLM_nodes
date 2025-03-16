from .AS_LLM_nodes import AS_GeminiCaptioning, AS_MultimodalGemini, AS_ComfyGPT

NODE_CLASS_MAPPINGS = {
    "AS_GeminiCaptioning": AS_GeminiCaptioning,
    "AS_MultimodalGemini": AS_MultimodalGemini,
    "AS_ComfyGPT": AS_ComfyGPT
}

def load_plugin():
    """
    Called by ComfyUI to load and register these custom nodes.
    """
    try:
        from modules import nodes
    except ImportError:
        print("Module 'modules.nodes' not found. Make sure ComfyUI is installed.")
        return

    nodes.register_node(AS_GeminiCaptioning)
    nodes.register_node(AS_MultimodalGemini)
    nodes.register_node(AS_ComfyGPT)

    print("AS_LLM nodes plugin successfully registered.")

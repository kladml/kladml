
from .pipeline import ComponentRegistry

def register_all_components():
    """Import and register standard components."""
    from .parsers.j1939 import J1939Parser
    from .components.resampling import TimeResampler
    from .components.cleaning import J1939Cleaner, TripSegmenter
    from .components.splitting import ChronologicalSplitter
    
    ComponentRegistry.register("J1939Parser", J1939Parser)
    # Also register aliases
    ComponentRegistry.register("J1939", J1939Parser)
    
    ComponentRegistry.register("TimeResampler", TimeResampler)
    ComponentRegistry.register("Resampler", TimeResampler) # Friendly alias
    
    ComponentRegistry.register("J1939Cleaner", J1939Cleaner)
    ComponentRegistry.register("Cleaner", J1939Cleaner) # Maybe ambiguous? Keep explicit
    
    ComponentRegistry.register("TripSegmenter", TripSegmenter)
    ComponentRegistry.register("Segmenter", TripSegmenter)
    
    ComponentRegistry.register("ChronologicalSplitter", ChronologicalSplitter)
    ComponentRegistry.register("Splitter", ChronologicalSplitter)

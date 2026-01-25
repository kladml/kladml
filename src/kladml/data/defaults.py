
from .pipeline import ComponentRegistry

def register_all_components():
    """Import and register standard components."""
    from .parsers.j1939 import J1939Parser
    from .components.resampling import TimeResampler
    from .components.cleaning import J1939Cleaner, TripSegmenter
    from .components.splitting import ChronologicalSplitter
    from .components.io import ParquetReader, ParquetWriter
    
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

    # Decorators handle main names, but we add aliases or explicit checks here
    # Since decorators run on import, just importing above is enough for main names!
    # We add aliases here manually if needed.
    
    ComponentRegistry.register("J1939", J1939Parser)
    ComponentRegistry.register("Resampler", TimeResampler)
    ComponentRegistry.register("Cleaner", J1939Cleaner)
    ComponentRegistry.register("Segmenter", TripSegmenter)
    ComponentRegistry.register("WriteParquet", ParquetWriter)
    ComponentRegistry.register("ReadParquet", ParquetReader)

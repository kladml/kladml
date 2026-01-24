from kladml.interfaces.exporter import ExporterInterface

class ExporterRegistry:
    """
    Registry for model exporters.
    """
    _exporters: dict[str, type[ExporterInterface]] = {}
    
    @classmethod
    def register(cls, name: str):
        def decorator(exporter_cls):
            cls._exporters[name] = exporter_cls
            return exporter_cls
        return decorator
        
    @classmethod
    def get(cls, name: str) -> type[ExporterInterface]:
        if name not in cls._exporters:
            available = ", ".join(cls._exporters.keys())
            raise ValueError(f"Exporter '{name}' not found. Available: {available}")
        return cls._exporters[name]
    
    @classmethod
    def list(cls) -> dict[str, str]:
        return {name: cls.__doc__ for name, cls in cls._exporters.items()}

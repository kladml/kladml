
import polars as pl
from pathlib import Path
from typing import Any
from ..pipeline import PipelineComponent, register_component

@register_component("ParquetReader")
class ParquetReader(PipelineComponent):
    """
    Reads a Parquet file into a DataFrame.
    """
    def __init__(self, path: str = None):
        super().__init__(config={"path": path})
        self.path = path

    def fit(self, data: Any) -> 'ParquetReader':
        return self

    def transform(self, data: Any) -> pl.DataFrame:
        """
        Ignores input data (usually), reads from configured path.
        If data is provided and path is None, treats data as path?
        """
        target_path = self.path or data
        if not target_path:
            raise ValueError("ParquetReader requires a path in config or input data")
            
        return pl.read_parquet(target_path)

@register_component("ParquetWriter")
class ParquetWriter(PipelineComponent):
    """
    Writes a DataFrame to a Parquet file.
    Returns the Path (str) for downstream components or logging.
    """
    def __init__(self, path: str):
        super().__init__(config={"path": path})
        self.path = path

    def fit(self, data: Any) -> 'ParquetWriter':
        return self

    def transform(self, df: pl.DataFrame) -> str:
        if not isinstance(df, pl.DataFrame):
            # Try to handle lazy frame
            if isinstance(df, pl.LazyFrame):
                df = df.collect()
            else:
                raise ValueError(f"ParquetWriter expects DataFrame, got {type(df)}")
                
        # Handle Output Path
        # Ensure parent dir exists
        p = Path(self.path)
        p.parent.mkdir(parents=True, exist_ok=True)
        
        df.write_parquet(self.path)
        return str(self.path)

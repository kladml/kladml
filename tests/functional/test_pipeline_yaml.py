
import pytest
import yaml
import polars as pl
from pathlib import Path
from kladml.data.pipeline import DataPipeline
from kladml.data.defaults import register_all_components

@pytest.fixture
def pipeline_yaml(tmp_path):
    config = {
        "pipeline": [
            {"ParquetReader": {}}, # Will use input path
            {"TimeResampler": {"rate": 1.0}},
            {"ParquetWriter": {"path": str(tmp_path / "output.parquet")}}
        ]
    }
    p = tmp_path / "pipe.yaml"
    with open(p, "w") as f:
        yaml.dump(config, f)
    return str(p)

@pytest.fixture
def input_parquet(tmp_path):
    from datetime import datetime
    df = pl.DataFrame({
        "timestamp": pl.datetime_range(
            start=datetime(2024, 1, 1), 
            end=datetime(2024, 1, 1, 0, 10), # 10 minutes
            interval="100ms", 
            eager=True, 
            time_unit="us"
        ),
        "value": [float(i) for i in range(6001)] # 10min * 60s * 10Hz = 6000 points
    }).head(101) # Keep small
    
    p = tmp_path / "input.parquet"
    df.write_parquet(p)
    return str(p)

def test_declarative_pipeline_run(pipeline_yaml, input_parquet, tmp_path):
    register_all_components()
    
    # Load
    pipe = DataPipeline.from_yaml(pipeline_yaml)
    assert len(pipe.steps) == 3
    
    # Run
    # Input is path to parquet
    # Step 1: ParquetReader(path=None) -> uses input -> returns DF
    # Step 2: TimeResampler(rate=1.0) -> requires DF -> returns DF
    # Step 3: ParquetWriter(path=...) -> requires DF -> returns Path
    
    res = pipe.transform(input_parquet)
    
    assert res == str(tmp_path / "output.parquet")
    assert Path(res).exists()
    
    # Check result
    df_out = pl.read_parquet(res)
    # Original 100ms, Resampled 1.0s -> 10x reduction roughly
    assert len(df_out) < 101
    assert len(df_out) > 0


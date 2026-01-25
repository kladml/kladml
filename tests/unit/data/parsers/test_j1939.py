
import pytest
import polars as pl
from pathlib import Path
from kladml.data.parsers.j1939 import process_single_file, parse_can_id

def test_parse_can_id():
    assert parse_can_id("0x123") == 0x123
    assert parse_can_id("123") == 0x123 # if hex string
    assert parse_can_id(123) == 123
    assert parse_can_id("F004") == 0xF004

def test_process_single_file_eec1(tmp_path):
    # Mock CSV content
    # timestamp;id;dlc;d0;d1;d2;d3;d4;d5;d6;d7
    # PGN EEC1 = F004. ID example: 0CF00400 (Source=00, Priority=3, PGN=F004)
    # RPM bytes: d3, d4. Torque: d2.
    
    # RPM = 1000 -> 8000 * 8 = 64000? No. (b5*256+b4)*0.125 = 1000 -> raw = 8000.
    # 8000 = 0x1F40. b5=1F(31), b4=40(64).
    
    # Torque = 50% -> (b3-125) = 50 -> b3=175.
    
    csv_data = """header_noise_to_skip
timestamp;id;dlc;data
2024-01-01 10:00:00.000;0CF00400;8;0;0;175;64;31;0;0;0
2024-01-01 10:00:00.100;0CF00400;8;0;0;175;64;31;0;0;0
"""
    f = tmp_path / "test_log.csv"
    f.write_bytes(csv_data.encode('utf-8')) # Use bytes for J1939Parser reading
    
    df = process_single_file(str(f))
    
    assert df is not None
    assert not df.is_empty()
    assert len(df) == 2
    
    # Check RPM
    # (31 * 256 + 64) * 0.125 = (7936 + 64) * 0.125 = 8000 * 0.125 = 1000.0
    assert df["rpm"][0] == 1000.0
    
    # Check Torque
    # 175 - 125 = 50.0
    assert df["torque"][0] == 50.0


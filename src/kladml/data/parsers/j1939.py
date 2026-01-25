
from pathlib import Path
import polars as pl
import numpy as np
import io
from loguru import logger
import gc
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Optional, Any
from ..pipeline import PipelineComponent
from ..components.resampling import TimeResampler

# J1939 PGNs
PGN_EEC1 = 0xF004  # RPM, Torque
PGN_CCVS = 0xFEF1  # Wheel Speed
PGN_EEC2 = 0xF003  # Accelerator Pedal

def parse_can_id(can_id_val):
    """Parses hex string CAN ID to integer."""
    try:
        if isinstance(can_id_val, str):
            return int(can_id_val, 16)
        if isinstance(can_id_val, (int, float)):
            return int(can_id_val)
        return 0
    except:
        return 0

def process_single_file(file_path: str) -> Optional[pl.DataFrame]:
    """
    Worker function to process a single CSV file.
    Designed for parallelism (no shared state).
    """
    try:
        with open(file_path, 'rb') as f:
            content = f.read()
            
        header_offset = content.find(b'timestamp;id;dlc;data')
        if header_offset == -1:
            return None
            
        csv_content = io.BytesIO(content[header_offset:])
        
        # Robust Parsing
        # Polars CSV reader is strict. Since lines might vary, used to use 'python' engine in pandas.
        # Here we try pl.read_csv with truncate_ragged_lines or ignore_errors if needed.
        # But usually J1939 logs are fairly structured if valid.
        
        # We manually specify schema to avoid type inference issues
        # 11 columns: timestamp, id, dlc, d0..d7
        
        col_names = ['timestamp', 'id', 'dlc'] + [f'd{i}' for i in range(8)]
        schema = {c: pl.Utf8 for c in col_names} # Read all as string first for safety
        
        try:
            df = pl.read_csv(
                csv_content, 
                separator=';', 
                has_header=False, 
                skip_rows=1,
                new_columns=col_names,
                schema_overrides=schema,
                ignore_errors=True # Drop bad lines
            )
        except Exception as e:
            print(f"J1939 Read CSV Error: {e}")
            return None
        
        if df.is_empty():
            return None

        # Clean ID
        df = df.with_columns(
            pl.col("id").str.strip_chars()
        )
        
        # Parse PGN
        # Map hex string to int
        # Optimization: use python map if cleaner
        # df = df.with_columns(pl.col("id").map_elements(parse_can_id, return_dtype=pl.Int64).alias("can_id_int"))
        
        # Or trying native polars hex parsing if strict hex?
        # Usually can logs are hex '0x...' or '...'
        
        df = df.with_columns(
             pl.col("id").map_elements(parse_can_id, return_dtype=pl.Int64).alias("can_id_int")
        )
        
        df = df.with_columns(
            ((pl.col("can_id_int") // 256) & 0xFFFF).alias("pgn")
        )

        # Filter relevant PGNs
        df = df.filter(pl.col("pgn").is_in([PGN_EEC1, PGN_CCVS, PGN_EEC2]))
        
        if df.is_empty():
            return None
            
        # Parse Timestamp
        # Format: '2023-11-20 14:00:00.123'
        df = df.with_columns(
            pl.col("timestamp").str.strptime(pl.Datetime, format="%Y-%m-%d %H:%M:%S%.f", strict=False).alias("timestamp")
        ).drop_nulls(subset=["timestamp"])
        
        if df.is_empty():
            return None
            
        # Cast Data Bytes to Int
        data_cols = [f'd{i}' for i in range(8)]
        df = df.with_columns([
            pl.col(c).cast(pl.Int32, strict=False).fill_null(0) for c in data_cols
        ])
        
        # --- DECODE ---
        # Initialize result columns
        # We can do this efficiently via conditional expressions
        
        # EEC1: RPM (d4*256 + d3)*0.125, Torque (d2 - 125)
        # Note: d0=byte1, ... d3=byte4, d4=byte5.
        # User logic: b5(d4), b4(d3).
        
        # We create columns for each signal, filled with null, then coalesce?
        # Or better: Create separate frames and concat? Or use `when().then()`
        
        # RPM/Torque
        expr_rpm = (
            pl.when(pl.col("pgn") == PGN_EEC1)
            .then(((pl.col("d4") * 256) + pl.col("d3")) * 0.125)
            .otherwise(None)
            .alias("rpm")
        )
        
        expr_torque = (
            pl.when(pl.col("pgn") == PGN_EEC1)
            .then(pl.col("d2") - 125.0)
            .otherwise(None)
            .alias("torque")
        )
        
        # Speed (CCVS)
        # b3(d2), b2(d1) -> (b3*256 + b2)/256
        expr_speed = (
             pl.when(pl.col("pgn") == PGN_CCVS)
             .then(((pl.col("d2") * 256) + pl.col("d1")) / 256.0)
             .otherwise(None)
             .alias("speed_kmh")
        )
        
        # Pedal (EEC2)
        # b2(d1) * 0.4
        expr_pedal = (
             pl.when(pl.col("pgn") == PGN_EEC2)
             .then(pl.col("d1") * 0.4)
             .otherwise(None)
             .alias("accel_pedal")
        )
        
        df = df.with_columns([expr_rpm, expr_torque, expr_speed, expr_pedal])
        
        # Select and Sort
        result = df.select(["timestamp", "rpm", "torque", "speed_kmh", "accel_pedal"]).sort("timestamp")
        
        return result
        
    except Exception as e:
        print(f"J1939 Worker Error: {e}")
        import traceback
        traceback.print_exc()
        return None

class J1939Parser(PipelineComponent):
    """
    Parses and Resamples SAE J1939 Data using Polars.
    """
    def __init__(self, output_file: str, resample_rate: float = 0.5, num_workers: int = 4, max_files: Optional[int] = None):
        super().__init__(config={"resample_rate": resample_rate})
        self.output_file = output_file
        self.resample_rate = resample_rate
        # self.resample_rate_pd not needed for Polars TimeResampler (it uses float rate)
        self.num_workers = num_workers
        self.max_files = max_files

    def fit(self, data: Any) -> 'J1939Parser':
        """Nothing to learn."""
        return self

    def transform(self, input_root: str) -> str:
        """
        Processes all files in input_root and saves to self.output_file.
        Returns path to output file.
        """
        self.temp_dir = Path(input_root) / "temp_chunks"
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"Scanning {input_root}...")
        all_files = list(Path(input_root).rglob("*.csv"))
        all_files = [str(f) for f in all_files]
        all_files.sort()
        
        if self.max_files:
            all_files = all_files[:self.max_files]
            logger.info(f"Limiting to first {self.max_files} files.")
            
        logger.info(f"Processing {len(all_files)} files with {self.num_workers} workers...")
        
        chunk_size = 50
        overlap = 2
        
        # Process Batches
        # Note: Polars itself is threaded. Processing multiple files in ProcessPool might compete for CPU.
        # But since we do file I/O and parsing, Multiprocessing is usually fine.
        
        file_batches = []
        for i in range(0, len(all_files), chunk_size):
            end = min(len(all_files), i + chunk_size + overlap)
            file_batches.append(all_files[i:end])
            
        temp_chunk_paths = []
        
        for b_idx, batch_files in enumerate(file_batches):
            logger.info(f"Processing Batch {b_idx+1}/{len(file_batches)} ({len(batch_files)} files)...")
            
            raw_dfs = []
            
            with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
                futures = {executor.submit(process_single_file, f): f for f in batch_files}
                for fut in as_completed(futures):
                    res = fut.result()
                    if res is not None and not res.is_empty():
                        raw_dfs.append(res)
            
            if not raw_dfs:
                continue
                
            # Merge Batch
            try:
                batch_raw = pl.concat(raw_dfs).sort("timestamp").unique(subset=["timestamp"], keep="last")
            except Exception as e:
                logger.error(f"Concat failed: {e}")
                continue
                
            if batch_raw.is_empty():
                continue
                
            # Use Modular Resampler (Polars Version)
            resampler = TimeResampler(rate=self.resample_rate)
            resampled_chunk = resampler.transform(batch_raw)
            
            if resampled_chunk.is_empty():
                continue
            
            save_path = self.temp_dir / f"resampled_batch_{b_idx}.parquet"
            resampled_chunk.write_parquet(save_path)
            temp_chunk_paths.append(str(save_path))
            
            del raw_dfs, batch_raw, resampled_chunk, resampler
            gc.collect()
            
        if not temp_chunk_paths:
            logger.error("No data processed!")
            return ""

        logger.info(f"Generated {len(temp_chunk_paths)} resampled chunks. Merging final dataset...")
        
        # Merge all chunks
        # pl.scan_parquet works great for this
        
        full_lf = pl.scan_parquet(temp_chunk_paths)
        full_df = (
            full_lf
            .sort("timestamp")
            .unique(subset=["timestamp"], keep="last")
            .collect()
        )
        
        logger.info(f"Final Dataset: {len(full_df)} rows. Saving to {self.output_file}...")
        full_df.write_parquet(self.output_file)
        
        # Save Metadata Passport
        from ..pipeline import DatasetMetadata
        meta = DatasetMetadata(
            source_type="j1939",
            format="parquet",
            freq="0.5s",
            is_equispaced=True,
            columns=full_df.columns,
            num_samples=len(full_df)
        )
        meta_path = str(self.output_file).replace(".parquet", "_metadata.json")
        meta.save(meta_path)
        logger.info(f"Saved Metadata Passport to {meta_path}")
        
        # Cleanup
        logger.info("Cleaning temps...")
        for p in temp_chunk_paths:
            Path(p).unlink(missing_ok=True)
        try:
            for f in self.temp_dir.glob("*"):
                f.unlink()
            self.temp_dir.rmdir()
        except:
            pass
        
        logger.info("Done.")
        return self.output_file

# Alias for backward compatibility if needed, but we prefer J1939Parser
CanBusOrchestrator = J1939Parser

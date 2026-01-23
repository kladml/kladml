
import os
import glob
import pandas as pd
import numpy as np
import io
import logging
import gc
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Optional, Any
from ..pipeline import PipelineComponent, DatasetMetadata
from ..components.resampling import TimeResampler

# Setup logging
logger = logging.getLogger(__name__)

# J1939 PGNs
PGN_EEC1 = 0xF004  # RPM, Torque
PGN_CCVS = 0xFEF1  # Wheel Speed
PGN_EEC2 = 0xF003  # Accelerator Pedal

def parse_can_id(can_id_str):
    """Parses hex string CAN ID to integer."""
    try:
        if isinstance(can_id_str, str):
            return int(can_id_str, 16)
        if isinstance(can_id_str, (int, float)):
            return int(can_id_str)
        return 0
    except:
        return 0

def process_single_file(file_path: str) -> Optional[pd.DataFrame]:
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
        
        # Robust Parsing: Use 'python' engine to handle variable columns (short frames)
        # Using 11 columns (3 header + 8 data bytes) to handle variable J1939 message lengths
        col_names = ['timestamp', 'id', 'dlc'] + [f'd{i}' for i in range(8)]
        
        df = pd.read_csv(
            csv_content, 
            sep=';', 
            header=None, 
            skiprows=1, # Skip the file header row manually
            names=col_names,
            engine='python',
            encoding='latin1'
        )
        
        if df.empty or 'id' not in df.columns:
            return None

        # PGN Parsing
        if df['id'].dtype == object:
            df['id'] = df['id'].str.strip()
            
        # Parse PGN safely
        int_ids = df['id'].apply(parse_can_id).astype(np.int64)
        pgns = (int_ids.values >> 8) & 0xFFFF
        df['pgn'] = pgns
        
        # Filter relevant PGNs
        mask_relevant = df['pgn'].isin([PGN_EEC1, PGN_CCVS, PGN_EEC2])
        relevant_df = df[mask_relevant].copy()
        
        if relevant_df.empty:
            return None
            
        # Timestamp parsing (only for relevant rows to save time)
        relevant_df['idx'] = pd.to_datetime(relevant_df['timestamp'], format='%Y-%m-%d %H:%M:%S.%f', errors='coerce')
        relevant_df = relevant_df.dropna(subset=['idx'])
        
        if relevant_df.empty:
            return None

        # Decode Data Bytes
        data_cols = [f'd{i}' for i in range(8)]
        # Ensure numeric
        for c in data_cols:
            relevant_df[c] = pd.to_numeric(relevant_df[c], errors='coerce')
            
        # Access by 0-indexed integer columns for easy math (d0 -> index 0)
        # We just use the named columns d0..d7
        
        relevant_df['rpm'] = np.nan
        relevant_df['torque'] = np.nan
        relevant_df['speed_kmh'] = np.nan
        relevant_df['accel_pedal'] = np.nan
        
        # --- DECODER ---
        
        # EEC1 (RPM/Torque) - PGN 0xF004
        mask_rpm = relevant_df['pgn'] == PGN_EEC1
        if mask_rpm.any():
            # RPM: Bytes 4,5 (d3, d4)
            b4 = relevant_df.loc[mask_rpm, 'd3']
            b5 = relevant_df.loc[mask_rpm, 'd4']
            # Torque: Byte 3 (d2)
            b3 = relevant_df.loc[mask_rpm, 'd2']
            
            relevant_df.loc[mask_rpm, 'rpm'] = ((b5 * 256) + b4) * 0.125
            relevant_df.loc[mask_rpm, 'torque'] = b3 - 125.0

        # CCVS (Speed) - PGN 0xFEF1
        mask_speed = relevant_df['pgn'] == PGN_CCVS
        if mask_speed.any():
            # Speed: Bytes 2,3 (d1, d2)
            b2 = relevant_df.loc[mask_speed, 'd1']
            b3 = relevant_df.loc[mask_speed, 'd2']
            relevant_df.loc[mask_speed, 'speed_kmh'] = ((b3 * 256) + b2) / 256.0

        # EEC2 (Pedal) - PGN 0xF003
        mask_pedal = relevant_df['pgn'] == PGN_EEC2
        if mask_pedal.any():
            # Pedal: Byte 2 (d1)
            b2 = relevant_df.loc[mask_pedal, 'd1']
            relevant_df.loc[mask_pedal, 'accel_pedal'] = b2 * 0.4

        # Select Features
        result = relevant_df[['idx', 'rpm', 'torque', 'speed_kmh', 'accel_pedal']].copy()
        result.set_index('idx', inplace=True)
        result.sort_index(inplace=True)
        
        return result
        
    except Exception as e:
        # e.g. truncated file
        return None

class J1939Parser(PipelineComponent):
    """
    Parses and Resamples SAE J1939 Data.
    Inherits from PipelineComponent for modular use.
    """
    def __init__(self, output_file: str, resample_rate: float = 0.5, num_workers: int = 4, max_files: Optional[int] = None):
        super().__init__(config={"resample_rate": resample_rate})
        self.output_file = output_file
        self.resample_rate = resample_rate
        self.resample_rate_pd = f"{int(resample_rate*1000)}ms"
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
        self.temp_dir = os.path.join(input_root, "temp_chunks")
        os.makedirs(self.temp_dir, exist_ok=True)
        
        logger.info(f"Scanning {input_root}...")
        all_files = glob.glob(os.path.join(input_root, "**", "*.csv"), recursive=True)
        all_files.sort()
        
        if self.max_files:
            all_files = all_files[:self.max_files]
            logger.info(f"Limiting to first {self.max_files} files.")
            
        logger.info(f"Processing {len(all_files)} files with {self.num_workers} workers...")
        
        chunk_size = 50
        overlap = 2
        file_batches = []
        for i in range(0, len(all_files), chunk_size):
            end = min(len(all_files), i + chunk_size + overlap)
            batch = all_files[i : end]
            if batch:
                file_batches.append(batch)
        
        temp_chunk_paths = []
        
        for b_idx, batch_files in enumerate(file_batches):
            logger.info(f"Processing Batch {b_idx+1}/{len(file_batches)} ({len(batch_files)} files)...")
            
            raw_dfs = []
            
            with ProcessPoolExecutor(max_workers=self.num_workers) as executor:
                futures = {executor.submit(process_single_file, f): f for f in batch_files}
                for fut in as_completed(futures):
                    res = fut.result()
                    if res is not None and not res.empty:
                        raw_dfs.append(res)
            
            if not raw_dfs:
                continue
                
            batch_raw = pd.concat(raw_dfs).sort_index()
            batch_raw = batch_raw[~batch_raw.index.duplicated(keep='last')]
            
            if batch_raw.empty:
                continue
                
            if batch_raw.empty:
                continue
                
            # Use Modular Resampler
            resampler = TimeResampler(rate=self.resample_rate)
            resampled_chunk = resampler.transform(batch_raw)
            
            if resampled_chunk.empty:
                continue
            
            save_path = os.path.join(self.temp_dir, f"resampled_batch_{b_idx}.parquet")
            resampled_chunk.to_parquet(save_path)
            temp_chunk_paths.append(save_path)
            
            del raw_dfs, batch_raw, resampled_chunk, resampler
            gc.collect()
                    
        if not temp_chunk_paths:
            logger.error("No data processed!")
            return ""

        logger.info(f"Generated {len(temp_chunk_paths)} resampled chunks. Merging final dataset...")
        
        full_df = pd.read_parquet(temp_chunk_paths)
        full_df = full_df.sort_index()
        full_df = full_df[~full_df.index.duplicated(keep='last')]
        
        full_df.index.name = 'timestamp'
        full_df.reset_index(inplace=True)
        
        
        logger.info(f"Final Dataset: {len(full_df)} rows. Saving to {self.output_file}...")
        full_df.to_parquet(self.output_file)
        
        # Save Metadata Passport
        from ..pipeline import DatasetMetadata
        meta = DatasetMetadata(
            source_type="j1939",
            format="parquet",
            freq="0.5s",
            is_equispaced=True,
            columns=list(full_df.columns),
            num_samples=len(full_df)
        )
        meta_path = str(self.output_file).replace(".parquet", "_metadata.json")
        meta.save(meta_path)
        logger.info(f"Saved Metadata Passport to {meta_path}")
        
        logger.info("Cleaning temps...")
        for p in temp_chunk_paths:
            try:
                os.remove(p)
            except:
                pass
        try:
            os.rmdir(self.temp_dir)
        except:
            pass
        
        logger.info("Done.")
        return self.output_file

# Alias for backward compatibility if needed, but we prefer J1939Parser
CanBusOrchestrator = J1939Parser

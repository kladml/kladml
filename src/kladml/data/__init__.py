"""
HDF5-based lazy loading datasets for KladML.

Provides memory-efficient data loading by reading directly from disk
instead of loading entire datasets into RAM.
"""

from .hdf5_dataset import HDF5GluformerDataset

__all__ = ["HDF5GluformerDataset"]

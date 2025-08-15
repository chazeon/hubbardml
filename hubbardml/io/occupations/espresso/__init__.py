"""ESPRESSO file readers for HubbardML.

This package provides readers for different ESPRESSO output formats:
- SCF text output (.out files)
- XML output (.xml files) 
- Future: Binary, HDF5, etc.
"""

from .scf import read_occupations_scf, OccupationData, Occupation
from .xml import read_occupations_xml
from .auto import read_occupations

__all__ = [
    # Main API
    "read_occupations",           # Auto-detect format
    
    # Format-specific readers  
    "read_occupations_scf",       # SCF text output
    "read_occupations_xml",       # XML output
    
    # Data structures
    "OccupationData", "Occupation"
]
"""Occupation matrix I/O operations.

This module provides functions to read/write Hubbard occupation matrices
from various sources and formats.
"""

from .espresso import (
    read_occupations_scf,
    read_occupations_xml, 
    read_occupations,
    OccupationData,
    Occupation
)

__all__ = [
    # ESPRESSO readers (auto-detect and specific)
    "read_occupations", "read_occupations_scf", "read_occupations_xml",
    
    # Data structures
    "OccupationData", "Occupation"
]
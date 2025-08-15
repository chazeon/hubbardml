"""I/O operations for HubbardML data.

Organized by data type:
- occupations: Reading/writing Hubbard occupation matrices
- predictions: Formatting model predictions for output  
- datasets: Converting between dataset formats
- formats: Core data structures
"""

# Core data structures
from .formats import Site, Edge, UData, VData

# Occupation matrices I/O
from .occupations import (
    read_occupations, 
    read_occupations_scf, 
    read_occupations_xml,
    OccupationData,
    Occupation
)

# Prediction formatting
from .predictions import render_predictions, render_template

# Dataset operations - placeholder for future implementations
# from .datasets import convert_arrow_to_hdf5, load_hdf5

__all__ = [
    # Data structures
    "Site", "Edge", "UData", "VData", "OccupationData", "Occupation",
    
    # Occupation matrices
    "read_occupations", "read_occupations_scf", "read_occupations_xml",
    
    # Output formatting
    "render_predictions", "render_template"
]
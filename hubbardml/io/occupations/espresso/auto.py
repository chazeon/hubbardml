"""Auto-detection for Quantum ESPRESSO file formats."""

from pathlib import Path
from .scf import read_occupations_scf, OccupationData
from .xml import read_occupations_xml


def read_occupations(file: Path | str) -> OccupationData:
    """Auto-detect format and read Hubbard occupation matrices.
    
    Args:
        file: Path to QE output file
        
    Returns:
        OccupationData with parsed occupation matrices
        
    Note:
        Currently only supports SCF text output. Will auto-detect
        XML, binary formats in future versions.
    """
    file_path = Path(file)
    
    if file_path.suffix == '.xml':
        return read_occupations_xml(file_path)
    else:
        # Default to SCF text format
        return read_occupations_scf(file_path)
"""SCF text output parser for Quantum ESPRESSO.

This module contains the ADHD-friendly parser for QE SCF text output files.
"""

from typing import Dict, NamedTuple
from pathlib import Path
from dataclasses import dataclass
import numpy as np


class Occupation(NamedTuple):
    """Data from one SPIN section. Only .occupations matters for HubbardML."""
    eigenvalues: np.ndarray
    eigenvectors: np.ndarray  # Not used elsewhere  
    occupations: np.ndarray   # ← This is what we actually need!

    @property
    def trace(self) -> float:
        """Total occupation number (sum of diagonal elements)."""
        return float(np.trace(self.occupations))


class ParseResult(NamedTuple):
    """Result of parsing with clear naming."""
    data: Dict[int, Occupation] | Occupation | None
    lines_processed: int


@dataclass 
class OccupationData:
    """Container for occupation matrices from QE calculations."""
    atoms: Dict[int, Dict[int, Occupation]]
    
    def __len__(self) -> int:
        """Number of atoms."""
        return len(self.atoms)
    
    def get_atom_occupations(self, atom_id: int) -> Dict[int, Occupation]:
        """Get all spin occupations for an atom."""
        return self.atoms[atom_id]
    
    def get_occupation(self, atom_id: int, spin: int) -> Occupation:
        """Get specific atom-spin occupation."""
        return self.atoms[atom_id][spin]


def read_occupations_scf(file: Path | str) -> OccupationData:
    """Read Hubbard occupation matrices from QE SCF text output.
    
    Args:
        file: Path to SCF output file (typically 'scf.out')
        
    Returns:
        OccupationData with parsed occupation matrices
        
    Example:
        >>> data = read_occupations_scf("scf.out")
        >>> len(data)  # Number of atoms
        4
        >>> occ = data.get_occupation(atom_id=1, spin=1)
        >>> occ.occupations.shape
        (5, 5)
    """
    file_path = Path(file)
    if not file_path.exists():
        raise FileNotFoundError(f"SCF file not found: {file}")

    # Read all lines at once (simpler than streaming)
    lines = file_path.read_text().strip().split('\n')
    
    # Step 1: Find the final SCF calculation
    scf_completion_line = _find_final_scf_completion(lines)
    
    # Step 2: Find the final Hubbard occupation block 
    occupation_block_start = _find_hubbard_occupation_block(lines, scf_completion_line)
    
    # Step 3: Parse all atoms in the occupation block
    atoms = _parse_all_atoms(lines, occupation_block_start)
    
    if not atoms:
        raise ValueError("No occupation data found in SCF file")
    
    return OccupationData(atoms=atoms)


def _find_final_scf_completion(lines: list[str]) -> int:
    """Find the line number of the final 'End of self-consistent calculation'."""
    scf_end_lines = []
    for line_num, line in enumerate(lines):
        if "End of self-consistent calculation" in line:
            scf_end_lines.append(line_num)
    
    if not scf_end_lines:
        raise ValueError("No SCF completion found - is this a valid QE SCF output?")
    
    return scf_end_lines[-1]  # Return the last one


def _find_hubbard_occupation_block(lines: list[str], start_after: int) -> int:
    """Find the Hubbard occupation block after the given line number."""
    hubbard_header = "=================== HUBBARD OCCUPATIONS ==================="
    
    for line_num in range(start_after, len(lines)):
        if hubbard_header in lines[line_num]:
            return line_num
    
    raise ValueError("No Hubbard occupation block found after SCF completion")


def _parse_all_atoms(lines: list[str], hubbard_block_start: int) -> Dict[int, Dict[int, Occupation]]:
    """Parse all atoms starting from the occupation block."""
    all_atoms = {}
    line_cursor = hubbard_block_start + 1  # Skip the === HUBBARD OCCUPATIONS === header
    
    while line_cursor < len(lines):
        current_line = lines[line_cursor].strip()
        
        if not current_line:  # Empty line = end of Hubbard block
            break
            
        if _is_atom_header(current_line):
            atom_num = _extract_atom_number(current_line)
            if atom_num is not None:
                atom_result = _parse_one_atom(lines, line_cursor)
                if atom_result.data:
                    all_atoms[atom_num] = atom_result.data
                line_cursor += atom_result.lines_processed
            else:
                line_cursor += 1
        else:
            line_cursor += 1
    
    return all_atoms


def _is_atom_header(line: str) -> bool:
    """Check if this line is an atom header like '--- ATOM 1 ---'."""
    return "ATOM" in line and "---" in line


def _extract_atom_number(line: str) -> int | None:
    """Extract atom number from header like '--- ATOM 42 ---'."""
    parts = line.split()
    try:
        atom_index = parts.index("ATOM")
        if atom_index + 1 < len(parts):
            return int(parts[atom_index + 1])
    except (ValueError, IndexError):
        pass
    return None


def _parse_one_atom(lines: list[str], atom_header_line: int) -> ParseResult:
    """Parse one atom's data (both SPIN 1 and SPIN 2 sections).
    
    Args:
        lines: All file lines
        atom_header_line: Line where the '--- ATOM X ---' header is
        
    Returns:
        ParseResult with {1: Occupation, 2: Occupation} and lines processed
    """
    atom_spins = {}
    line_cursor = atom_header_line + 1  # Skip the '--- ATOM X ---' line
    
    while line_cursor < len(lines):
        current_line = lines[line_cursor].strip()
        
        if not current_line or "Number of occupied Hubbard levels" in current_line:
            # End of this atom's data
            break
            
        if _is_atom_header(current_line):
            # Hit the next atom's header
            break
            
        if current_line.startswith("SPIN"):
            spin_num = _extract_spin_number(current_line)
            if spin_num is not None:
                spin_result = _parse_one_spin(lines, line_cursor + 1)  # +1 to skip SPIN line
                if spin_result.data is not None:
                    atom_spins[spin_num] = spin_result.data
                line_cursor += 1 + spin_result.lines_processed  # +1 for SPIN line
            else:
                line_cursor += 1
        else:
            line_cursor += 1
    
    total_lines_processed = line_cursor - atom_header_line
    return ParseResult(atom_spins, total_lines_processed)


def _extract_spin_number(line: str) -> int | None:
    """Extract spin number from line like 'SPIN  1' or 'SPIN  2'."""
    parts = line.split()
    try:
        if parts[0] == "SPIN" and len(parts) >= 2:
            return int(parts[1])
    except (IndexError, ValueError):
        pass
    return None


def _parse_one_spin(lines: list[str], after_spin_header: int) -> ParseResult:
    """Parse one SPIN section (eigenvalues + eigenvectors + occupation matrix).
    
    Args:
        lines: All file lines
        after_spin_header: Line number after the 'SPIN 1' line
        
    Returns:
        ParseResult with Occupation data and lines processed
    """
    # The three pieces we need to extract
    eigenvals = None
    eigenvecs = None 
    occupation_matrix = None
    
    line_cursor = after_spin_header
    
    while line_cursor < len(lines):
        current_line = lines[line_cursor].strip()
        
        # Stop when we hit end of this SPIN section
        if not current_line or current_line.startswith("SPIN") or _is_atom_header(current_line):
            break
            
        # Parse the three sections we care about
        if current_line == "eigenvalues:":
            eigenvals, lines_read = _read_eigenvalues(lines, line_cursor + 1)
            line_cursor += 1 + lines_read
            
        elif current_line == "eigenvectors (columns):":
            if eigenvals is not None:  # Need eigenvals to know matrix size
                matrix_size = len(eigenvals)
                eigenvecs, lines_read = _read_matrix(lines, line_cursor + 1, matrix_size)
                line_cursor += 1 + lines_read
            else:
                line_cursor += 1
                
        elif current_line == "occupation matrix ns (before diag.):":
            if eigenvals is not None:  # Need eigenvals to know matrix size
                matrix_size = len(eigenvals)
                occupation_matrix, lines_read = _read_matrix(lines, line_cursor + 1, matrix_size)
                line_cursor += 1 + lines_read
            else:
                line_cursor += 1
        else:
            line_cursor += 1
    
    # Build result if we got the essential data
    if eigenvals is not None and occupation_matrix is not None:
        if eigenvecs is None:
            # Create dummy eigenvectors (they're not used by HubbardML anyway)
            eigenvecs = np.eye(len(eigenvals))
        
        occupation_data = Occupation(eigenvals, eigenvecs, occupation_matrix)
        total_lines_processed = line_cursor - after_spin_header
        return ParseResult(occupation_data, total_lines_processed)
    
    # Failed to parse
    total_lines_processed = line_cursor - after_spin_header  
    return ParseResult(None, total_lines_processed)


def _read_eigenvalues(lines: list[str], eigenval_data_line: int) -> tuple[np.ndarray | None, int]:
    """Read eigenvalues from one line like '  0.845  0.849  0.967  1.036  1.037'."""
    if eigenval_data_line < len(lines):
        try:
            eigenvals = np.fromstring(lines[eigenval_data_line], sep=' ')
            return eigenvals, 1  # Read 1 line
        except ValueError:
            return None, 0
    return None, 0


def _read_matrix(lines: list[str], matrix_start_line: int, size: int) -> tuple[np.ndarray | None, int]:
    """Read a square matrix from 'size' consecutive lines.
    
    Args:
        lines: All file lines
        matrix_start_line: First line of matrix data
        size: Matrix dimensions (3x3 or 5x5)
        
    Returns:
        (matrix, lines_read) or (None, 0) if failed
    """
    if matrix_start_line + size > len(lines):
        return None, 0
    
    # Collect the matrix lines
    matrix_text_lines = []
    for row in range(size):
        matrix_text_lines.append(lines[matrix_start_line + row])
    
    # Convert to numpy matrix
    try:
        all_numbers = np.fromstring('\n'.join(matrix_text_lines), sep=' ')
        matrix = all_numbers.reshape(size, size)
        return matrix, size  # Read 'size' lines
    except ValueError:
        return None, 0


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 2:
        print("Usage: python scf.py <scf_output_file>")
        sys.exit(1)
    from pprint import pprint
    pprint(read_occupations_scf(sys.argv[1]))
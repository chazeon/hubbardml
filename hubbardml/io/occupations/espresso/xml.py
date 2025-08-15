"""XML output parser for Quantum ESPRESSO.

ADHD-friendly implementation for reading pwscf.xml files.
Each function has ONE clear purpose, with descriptive names and good error messages.
"""

import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List
import numpy as np

from .scf import OccupationData, Occupation


def read_occupations_xml(file: Path | str) -> OccupationData:
    """Read Hubbard occupation matrices from QE XML output.
    
    Args:
        file: Path to XML output file (typically 'pwscf.xml')
        
    Returns:
        OccupationData with parsed occupation matrices
        
    Example:
        >>> data = read_occupations_xml("pwscf.xml")
        >>> len(data)  # Number of atoms
        4
        >>> occ = data.get_occupation(atom_id=1, spin=1)
        >>> occ.occupations.shape
        (5, 5)
    """
    file_path = Path(file)
    if not file_path.exists():
        raise FileNotFoundError(f"XML file not found: {file}")
    
    # Step 1: Load and parse XML file
    try:
        tree = ET.parse(file_path)
        root = tree.getroot()
    except ET.ParseError as e:
        raise ValueError(f"Invalid XML file {file}: {e}")
    
    # Step 2: Find all Hubbard occupation matrix elements
    occupation_elements = _find_hubbard_ns_elements(root)
    if not occupation_elements:
        raise ValueError("No Hubbard occupation matrices found in XML file")
    
    # Step 3: Parse each occupation matrix
    atoms = {}
    for element in occupation_elements:
        atom_id, spin_id, occupation = _parse_single_hubbard_ns_element(element)
        
        # Group by atom, then by spin
        if atom_id not in atoms:
            atoms[atom_id] = {}
        atoms[atom_id][spin_id] = occupation
    
    return OccupationData(atoms=atoms)


def _find_hubbard_ns_elements(root: ET.Element) -> List[ET.Element]:
    """Find all Hubbard_ns elements in the XML tree.
    
    Args:
        root: Root XML element
        
    Returns:
        List of Hubbard_ns XML elements
        
    Note:
        QE uses namespace, so we search for elements ending with 'Hubbard_ns'
    """
    # QE XML uses namespace: {http://www.quantum-espresso.org/ns/qes/qes-1.0}Hubbard_ns
    # We use .//tag to search recursively, regardless of namespace
    hubbard_elements = []
    
    # Search for all elements with tag ending in 'Hubbard_ns'
    for element in root.iter():
        if element.tag.endswith('Hubbard_ns'):
            hubbard_elements.append(element)
    
    return hubbard_elements


def _parse_single_hubbard_ns_element(element: ET.Element) -> tuple[int, int, Occupation]:
    """Parse one Hubbard_ns XML element into an Occupation object.
    
    Args:
        element: Hubbard_ns XML element
        
    Returns:
        Tuple of (atom_id, spin_id, Occupation)
        
    Example XML structure:
        <Hubbard_ns rank="2" dims="5 5" order="F" specie="Fe1" label="3d" spin="1" index="1">
          1.017910382441606E+000  6.354328084060844E-003  ...
        </Hubbard_ns>
    """
    # Step 1: Extract metadata from XML attributes
    atom_id = _extract_atom_index(element)
    spin_id = _extract_spin_number(element)
    matrix_dims = _extract_matrix_dimensions(element)
    
    # Step 2: Parse the matrix data from element text
    matrix_data = _parse_matrix_text(element.text, matrix_dims)
    
    # Step 3: Create fake eigenvalues/eigenvectors (XML doesn't store these)
    # HubbardML only uses the occupation matrix, so dummy values are fine
    eigenvalues = np.diag(matrix_data)  # Use diagonal elements as eigenvalues
    eigenvectors = np.eye(matrix_dims[0])  # Identity matrix as dummy eigenvectors
    
    occupation = Occupation(
        eigenvalues=eigenvalues,
        eigenvectors=eigenvectors, 
        occupations=matrix_data
    )
    
    return atom_id, spin_id, occupation


def _extract_atom_index(element: ET.Element) -> int:
    """Extract atom index from Hubbard_ns element attributes.
    
    Args:
        element: Hubbard_ns XML element
        
    Returns:
        Atom index (1-based, matching QE convention)
    """
    index_attr = element.get('index')
    if index_attr is None:
        raise ValueError("Hubbard_ns element missing 'index' attribute")
    
    try:
        return int(index_attr)
    except ValueError:
        raise ValueError(f"Invalid atom index in Hubbard_ns: '{index_attr}'")


def _extract_spin_number(element: ET.Element) -> int:
    """Extract spin number from Hubbard_ns element attributes.
    
    Args:
        element: Hubbard_ns XML element
        
    Returns:
        Spin number (1 or 2)
    """
    spin_attr = element.get('spin')
    if spin_attr is None:
        raise ValueError("Hubbard_ns element missing 'spin' attribute")
    
    try:
        spin_num = int(spin_attr)
        if spin_num not in [1, 2]:
            raise ValueError(f"Spin must be 1 or 2, got {spin_num}")
        return spin_num
    except ValueError as e:
        raise ValueError(f"Invalid spin number in Hubbard_ns: '{spin_attr}' ({e})")


def _extract_matrix_dimensions(element: ET.Element) -> tuple[int, int]:
    """Extract matrix dimensions from Hubbard_ns element attributes.
    
    Args:
        element: Hubbard_ns XML element
        
    Returns:
        Matrix dimensions as (rows, cols) tuple
        
    Example:
        dims="5 5" -> (5, 5)
    """
    dims_attr = element.get('dims')
    if dims_attr is None:
        raise ValueError("Hubbard_ns element missing 'dims' attribute")
    
    try:
        dims_parts = dims_attr.strip().split()
        if len(dims_parts) != 2:
            raise ValueError(f"Expected 2 dimensions, got {len(dims_parts)}")
        
        rows, cols = int(dims_parts[0]), int(dims_parts[1])
        if rows != cols:
            raise ValueError(f"Expected square matrix, got {rows}x{cols}")
        if rows not in [3, 5]:  # p-orbitals=3, d-orbitals=5
            raise ValueError(f"Expected 3x3 or 5x5 matrix, got {rows}x{rows}")
            
        return (rows, cols)
    except (ValueError, IndexError) as e:
        raise ValueError(f"Invalid matrix dimensions in Hubbard_ns: '{dims_attr}' ({e})")


def _parse_matrix_text(text: str, dims: tuple[int, int]) -> np.ndarray:
    """Parse matrix data from XML element text content.
    
    Args:
        text: Raw text content from XML element
        dims: Expected matrix dimensions (rows, cols)
        
    Returns:
        Parsed matrix as numpy array
        
    Note:
        QE XML uses Fortran column-major order, so we use order='F' in reshape
    """
    if not text or not text.strip():
        raise ValueError("Hubbard_ns element has empty text content")
    
    # Step 1: Extract all numbers from the text
    try:
        # Split on whitespace and convert to floats
        numbers = [float(x) for x in text.strip().split()]
    except ValueError as e:
        raise ValueError(f"Failed to parse numbers from matrix text: {e}")
    
    # Step 2: Validate we have the right amount of data
    expected_count = dims[0] * dims[1]
    if len(numbers) != expected_count:
        raise ValueError(
            f"Expected {expected_count} matrix elements, got {len(numbers)}"
        )
    
    # Step 3: Reshape using Fortran order (column-major)
    try:
        matrix = np.array(numbers).reshape(dims, order='F')
        return matrix
    except ValueError as e:
        raise ValueError(f"Failed to reshape matrix data: {e}")
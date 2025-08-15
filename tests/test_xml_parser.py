"""Tests for XML parser functionality."""

import pytest
from pathlib import Path

from hubbardml.io import read_occupations_xml, read_occupations


@pytest.fixture
def example_xml_file():
    """Path to example XML output file."""
    test_dir = Path(__file__).parent
    xml_path = test_dir.parent / "example" / "fp" / "pwscf.xml"
    if not xml_path.exists():
        pytest.skip(f"Example XML file not found: {xml_path}")
    return str(xml_path)


def test_xml_basic_parsing(example_xml_file):
    """Test basic XML parsing functionality."""
    data = read_occupations_xml(example_xml_file)
    
    # Should have 4 Fe atoms
    assert len(data) == 4, f"Expected 4 atoms, got {len(data)}"
    
    # Each atom should have spin-up and spin-down
    for atom_idx in range(1, 5):
        assert atom_idx in data.atoms, f"Missing atom {atom_idx}"
        atom_data = data.get_atom_occupations(atom_idx)
        assert 1 in atom_data, f"Atom {atom_idx} missing spin-up"
        assert 2 in atom_data, f"Atom {atom_idx} missing spin-down"


def test_xml_occupation_properties(example_xml_file):
    """Test mathematical properties of XML-parsed occupation matrices."""
    data = read_occupations_xml(example_xml_file)
    
    for atom_idx in range(1, 5):
        for spin_idx in [1, 2]:
            occ = data.get_occupation(atom_idx, spin_idx)
            
            # Test matrix dimensions (d-orbitals = 5x5)
            assert occ.occupations.shape == (5, 5), \
                f"Atom {atom_idx} spin {spin_idx}: wrong shape {occ.occupations.shape}"
            
            # Test that we have reasonable occupation numbers
            trace = occ.trace
            assert 3.0 <= trace <= 6.0, \
                f"Atom {atom_idx} spin {spin_idx}: unrealistic trace {trace}"


def test_xml_auto_detection(example_xml_file):
    """Test that auto-detection works for XML files."""
    # Should auto-detect XML format based on .xml extension
    data = read_occupations(example_xml_file)
    assert len(data) == 4, "Auto-detection should work for XML files"


def test_xml_error_handling(tmp_path):
    """Test XML error handling."""
    # Test nonexistent file
    with pytest.raises(FileNotFoundError, match="XML file not found"):
        read_occupations_xml("nonexistent.xml")
    
    # Test invalid XML
    bad_xml = tmp_path / "bad.xml"
    bad_xml.write_text("<invalid>not closed")
    
    with pytest.raises(ValueError, match="Invalid XML file"):
        read_occupations_xml(str(bad_xml))
    
    # Test XML without Hubbard data
    empty_xml = tmp_path / "empty.xml"
    empty_xml.write_text('<?xml version="1.0"?><root></root>')
    
    with pytest.raises(ValueError, match="No Hubbard occupation matrices found"):
        read_occupations_xml(str(empty_xml))


def test_xml_vs_scf_consistency(example_xml_file):
    """Test that XML parser gives similar results to SCF parser."""
    # Compare with SCF text output
    xml_path = Path(example_xml_file)
    scf_path = xml_path.parent / "scf.out"
    
    if not scf_path.exists():
        pytest.skip("SCF file not available for comparison")
    
    xml_data = read_occupations_xml(xml_path)
    from hubbardml.io import read_occupations_scf
    scf_data = read_occupations_scf(scf_path)
    
    # Both should have same number of atoms
    assert len(xml_data) == len(scf_data), "XML and SCF should have same atom count"
    
    # Occupation matrices should be similar (within text format precision)
    import numpy as np
    tolerance = 2e-3  # Account for text format limitations
    
    for atom_id in range(1, 5):
        for spin_id in [1, 2]:
            xml_occ = xml_data.get_occupation(atom_id, spin_id)
            scf_occ = scf_data.get_occupation(atom_id, spin_id)
            
            # Check that matrices are close
            assert np.allclose(xml_occ.occupations, scf_occ.occupations, atol=tolerance), \
                f"Atom {atom_id} spin {spin_id}: matrices differ beyond tolerance"
            
            # Check that traces are close
            assert abs(xml_occ.trace - scf_occ.trace) < tolerance, \
                f"Atom {atom_id} spin {spin_id}: traces differ too much"
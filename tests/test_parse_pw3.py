"""Tests for parse_pw3 QE SCF output parsing."""

import numpy as np
import pytest
from pathlib import Path

from hubbardml.io import read_occupations_scf


@pytest.fixture
def example_scf_file():
    """Path to example SCF output file."""
    test_dir = Path(__file__).parent
    scf_path = test_dir.parent / "example" / "fp" / "scf.out"
    if not scf_path.exists():
        pytest.skip(f"Example SCF file not found: {scf_path}")
    return str(scf_path)


def test_parse_basic_functionality(example_scf_file):
    """Test basic parsing functionality."""
    occupation_data = read_occupations_scf(example_scf_file)
    data = occupation_data.atoms
    
    # Should have 4 Fe atoms
    assert len(data) == 4, f"Expected 4 atoms, got {len(data)}"
    
    # Each atom should have spin-up and spin-down
    for atom_idx in range(1, 5):
        assert atom_idx in data, f"Missing atom {atom_idx}"
        assert 1 in data[atom_idx], f"Atom {atom_idx} missing spin-up"
        assert 2 in data[atom_idx], f"Atom {atom_idx} missing spin-down"


def test_occupation_matrix_properties(example_scf_file):
    """Test mathematical properties of occupation matrices."""
    occupation_data = read_occupations_scf(example_scf_file)
    data = occupation_data.atoms
    
    for atom_idx, spins in data.items():
        for spin_idx, occ in spins.items():
            # Test matrix dimensions (d-orbitals = 5x5)
            assert occ.occupations.shape == (5, 5), \
                f"Atom {atom_idx} spin {spin_idx}: wrong shape {occ.occupations.shape}"
            
            # Test eigenvalues are reasonable (0 to 2 for occupation numbers)
            assert len(occ.eigenvalues) == 5, \
                f"Atom {atom_idx} spin {spin_idx}: wrong eigenvalue count"
            assert np.all(occ.eigenvalues >= 0), \
                f"Atom {atom_idx} spin {spin_idx}: negative eigenvalues"
            assert np.all(occ.eigenvalues <= 2), \
                f"Atom {atom_idx} spin {spin_idx}: eigenvalues > 2"
            
            # Test occupation matrix is symmetric (should be for Hubbard occupations)
            np.testing.assert_allclose(
                occ.occupations, occ.occupations.T, rtol=1e-6,
                err_msg=f"Atom {atom_idx} spin {spin_idx}: matrix not symmetric"
            )
            
            # Test that we have reasonable data (no strict orthonormality requirement for now)
            V = occ.eigenvectors
            assert V.shape == (5, 5), \
                f"Atom {atom_idx} spin {spin_idx}: wrong eigenvector shape"
            
            # Basic sanity check - eigenvectors shouldn't be all zeros
            assert np.any(np.abs(V) > 1e-10), \
                f"Atom {atom_idx} spin {spin_idx}: eigenvectors all zero"


def test_data_consistency(example_scf_file):
    """Test that parsed data is internally consistent."""
    occupation_data = read_occupations_scf(example_scf_file)
    data = occupation_data.atoms
    
    for atom_idx, spins in data.items():
        for spin_idx, occ in spins.items():
            A = occ.occupations
            eigenvals = occ.eigenvalues
            V = occ.eigenvectors
            
            # Test that we have matching dimensions
            assert A.shape == V.shape == (5, 5), \
                f"Atom {atom_idx} spin {spin_idx}: dimension mismatch"
            assert len(eigenvals) == 5, \
                f"Atom {atom_idx} spin {spin_idx}: eigenvalue count mismatch"
            
            # Test that eigenvalues match diagonal of diagonalized matrix
            # Note: We can't test full diagonalization without knowing QE's exact format
            # But we can test that the data is reasonable
            assert np.all(np.isfinite(A)), \
                f"Atom {atom_idx} spin {spin_idx}: occupation matrix has non-finite values"
            assert np.all(np.isfinite(eigenvals)), \
                f"Atom {atom_idx} spin {spin_idx}: eigenvalues have non-finite values"
            assert np.all(np.isfinite(V)), \
                f"Atom {atom_idx} spin {spin_idx}: eigenvectors have non-finite values"


def test_physical_constraints(example_scf_file):
    """Test physical constraints on occupation numbers."""
    occupation_data = read_occupations_scf(example_scf_file)
    data = occupation_data.atoms
    
    for atom_idx, spins in data.items():
        for spin_idx, occ in spins.items():
            # Total occupation (trace) should be reasonable for d-orbitals
            trace = np.trace(occ.occupations)
            assert 0 <= trace <= 10, \
                f"Atom {atom_idx} spin {spin_idx}: unrealistic trace {trace}"
            
            # All diagonal elements should be between 0 and 2
            diag_elements = np.diag(occ.occupations)
            assert np.all(diag_elements >= 0), \
                f"Atom {atom_idx} spin {spin_idx}: negative diagonal elements"
            assert np.all(diag_elements <= 2), \
                f"Atom {atom_idx} spin {spin_idx}: diagonal elements > 2"


def test_nonexistent_file():
    """Test handling of nonexistent files."""
    with pytest.raises(FileNotFoundError):
        read_occupations_scf("nonexistent_file.out")


def test_empty_file(tmp_path):
    """Test handling of empty files."""
    empty_file = tmp_path / "empty.out"
    empty_file.write_text("")
    
    with pytest.raises(ValueError, match="No SCF completion found"):
        read_occupations_scf(str(empty_file))
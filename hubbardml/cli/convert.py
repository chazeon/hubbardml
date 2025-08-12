"""
Convert Arrow datasets to HDF5 format for HubbardML

Implements the designed HDF5 structure:
dataset.h5:
└── hubbard/
    ├── u/
    │   ├── site0/
    │   │   ├── element          # ['Fe', 'Ni', 'Mn', ...]
    │   │   └── occupancy        # (N, 2, 5, 5) - entry, spin, orbital, orbital
    │   ├── input                # Linear response U values
    │   └── target               # Self-consistent U values
    └── v/
        ├── site0/, site1/       # Two sites for V interactions
        │   ├── element, occupancy
        ├── edge/
        │   └── distance         # Inter-site distances
        ├── input                # Linear response V values  
        └── target               # Self-consistent V values
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np


def load_arrow_data(file_path: str) -> Dict:
    """Load Arrow dataset and return structured data."""
    try:
        import pyarrow as pa
        import pyarrow.ipc
    except ImportError:
        raise ImportError("PyArrow not available. Install with: uv add pyarrow")
    
    print(f"Loading Arrow dataset: {file_path}")
    
    with pa.memory_map(file_path, "rb") as source:
        reader = pa.ipc.RecordBatchFileReader(source)
        table = reader.read_all()
    
    print(f"Loaded {table.num_rows:,} rows × {table.num_columns} columns")
    
    # Convert to dictionary for easier processing
    data = {}
    for column_name in table.column_names:
        data[column_name] = table[column_name].to_pylist()
    
    return data


def separate_u_v_parameters(data: Dict) -> Tuple[Dict, Dict]:
    """Separate U and V parameters into different datasets."""
    u_data = {}
    v_data = {}
    
    if "param_type" not in data:
        raise ValueError("Dataset missing 'param_type' column")
    
    param_types = data["param_type"]
    
    # Find indices for U and V parameters
    u_indices = [i for i, param_type in enumerate(param_types) if param_type == "U"]
    v_indices = [i for i, param_type in enumerate(param_types) if param_type == "V"]
    
    print(f"Found {len(u_indices)} U parameter entries")
    print(f"Found {len(v_indices)} V parameter entries")
    
    # Extract U parameter data
    for column_name, column_data in data.items():
        u_data[column_name] = [column_data[i] for i in u_indices]
        v_data[column_name] = [column_data[i] for i in v_indices]
    
    return u_data, v_data


def reshape_occupation_matrices(occs_flat: List, expected_shape: Tuple[int, int]) -> np.ndarray:
    """Reshape flattened occupation matrices to (spin, orbital, orbital) format."""
    n_entries = len(occs_flat)
    n_orbitals = expected_shape[0]
    
    # Initialize output array: (n_entries, 2, n_orbitals, n_orbitals)
    occs_reshaped = np.zeros((n_entries, 2, n_orbitals, n_orbitals), dtype=np.float64)
    
    for i, (occs_1, occs_2) in enumerate(zip(occs_flat, occs_flat)):  # This is wrong, need proper pairing
        if occs_1 is not None and len(occs_1) == n_orbitals * n_orbitals:
            occs_reshaped[i, 0] = np.array(occs_1).reshape(n_orbitals, n_orbitals)
        if occs_2 is not None and len(occs_2) == n_orbitals * n_orbitals:
            occs_reshaped[i, 1] = np.array(occs_2).reshape(n_orbitals, n_orbitals)
    
    return occs_reshaped


def process_u_parameters(u_data: Dict) -> Dict:
    """Process U parameter data for HDF5 storage."""
    print("Processing U parameter data...")
    
    n_entries = len(u_data["param_type"])
    
    processed = {
        "site0": {
            "element": np.array(u_data["atom_1_element"], dtype="S10"),  # String array
            "occupancy": None  # Will be filled below
        },
        "input": np.array(u_data.get("param_in", [0.0] * n_entries), dtype=np.float64),
        "target": np.array(u_data["param_out"], dtype=np.float64)
    }
    
    # Process occupation matrices for site0 (U parameters only need one site)
    occs_1_1 = u_data["atom_1_occs_1"]
    occs_1_2 = u_data["atom_1_occs_2"]
    
    # Analyze the mix of orbital types in this dataset
    # U parameters can have mixed d-orbital (5x5) and p-orbital (3x3) matrices
    orbital_types = {}
    is_nested = None
    
    for i, occs in enumerate(occs_1_1[:min(100, len(occs_1_1))]):  # Sample first 100
        if occs is not None:
            if isinstance(occs[0], list):
                # Nested list format
                n_orb = len(occs)
                is_nested = True
            else:
                # Flattened format
                flat_len = len(occs)
                n_orb = 5 if flat_len == 25 else 3 if flat_len == 9 else 0
                is_nested = False
            
            if n_orb > 0:
                orbital_types[n_orb] = orbital_types.get(n_orb, 0) + 1
    
    if not orbital_types:
        raise ValueError("No valid occupation matrices found")
    
    print(f"Orbital type distribution: {orbital_types}")
    
    # Use the maximum orbital size for the array dimension (pad smaller ones)
    max_orbitals = max(orbital_types.keys())
    print(f"Using maximum orbital size: {max_orbitals}×{max_orbitals} (padding smaller matrices)")
    
    # Create occupation array with max size and track actual dimensions
    occupancy = np.zeros((n_entries, 2, max_orbitals, max_orbitals), dtype=np.float64)
    orbital_dims = np.zeros(n_entries, dtype=np.int32)  # Track actual matrix size per entry
    
    for i in range(n_entries):
        # Determine the actual orbital size for this entry
        actual_n_orb = 0
        
        # Process spin-up occupancy  
        if occs_1_1[i] is not None:
            if is_nested:
                # Already in matrix format
                occs_matrix = np.array(occs_1_1[i])
                n_orb = occs_matrix.shape[0]
                occupancy[i, 0, :n_orb, :n_orb] = occs_matrix
                actual_n_orb = n_orb
            else:
                # Flattened format, determine size and reshape
                flat_len = len(occs_1_1[i])
                n_orb = 5 if flat_len == 25 else 3 if flat_len == 9 else 0
                if n_orb > 0:
                    occs_matrix = np.array(occs_1_1[i]).reshape(n_orb, n_orb)
                    occupancy[i, 0, :n_orb, :n_orb] = occs_matrix
                    actual_n_orb = n_orb
                
        # Process spin-down occupancy
        if occs_1_2[i] is not None:
            if is_nested:
                # Already in matrix format
                occs_matrix = np.array(occs_1_2[i])
                n_orb = occs_matrix.shape[0]
                occupancy[i, 1, :n_orb, :n_orb] = occs_matrix
                if actual_n_orb == 0:  # Only set if not already set by spin-up
                    actual_n_orb = n_orb
            else:
                # Flattened format, determine size and reshape
                flat_len = len(occs_1_2[i])
                n_orb = 5 if flat_len == 25 else 3 if flat_len == 9 else 0
                if n_orb > 0:
                    occs_matrix = np.array(occs_1_2[i]).reshape(n_orb, n_orb)
                    occupancy[i, 1, :n_orb, :n_orb] = occs_matrix
                    if actual_n_orb == 0:  # Only set if not already set by spin-up
                        actual_n_orb = n_orb
        
        # Record the actual matrix dimension for this entry
        orbital_dims[i] = actual_n_orb
    
    processed["site0"]["occupancy"] = occupancy
    processed["site0"]["orbital_dims"] = orbital_dims  # Store actual matrix dimensions
    
    return processed


def process_v_parameters(v_data: Dict) -> Dict:
    """Process V parameter data for HDF5 storage."""
    print("Processing V parameter data...")
    
    n_entries = len(v_data["param_type"])
    
    processed = {
        "site0": {"element": [], "occupancy": None},
        "site1": {"element": [], "occupancy": None},
        "edge": {
            "distance": np.array(v_data.get("dist_in", [0.0] * n_entries), dtype=np.float64)
        },
        "input": np.array(v_data.get("param_in", [0.0] * n_entries), dtype=np.float64),
        "target": np.array(v_data["param_out"], dtype=np.float64)
    }
    
    # For V parameters, we need to handle site assignment
    # Based on inspection, we have mixed 3×3 and 5×5 matrices
    
    site0_elements = []
    site1_elements = []
    site0_occupancies = []
    site1_occupancies = []
    site0_orbital_dims = []
    site1_orbital_dims = []
    
    for i in range(n_entries):
        # Get occupation matrix shapes to determine site assignment
        occs_1_1 = v_data["atom_1_occs_1"][i] if v_data["atom_1_occs_1"][i] is not None else []
        occs_2_1 = v_data["atom_2_occs_1"][i] if v_data["atom_2_occs_1"][i] is not None else []
        
        # Site assignment: atom_1 -> site0, atom_2 -> site1
        site0_elements.append(v_data["atom_1_element"][i])
        site1_elements.append(v_data["atom_2_element"][i])
        
        # Determine orbital types - handle both nested lists and flattened arrays
        def get_orbital_count(occs):
            if not occs:
                return 0
            if isinstance(occs[0], list):
                # Nested list format
                return len(occs)
            else:
                # Flattened format
                flat_len = len(occs)
                return 5 if flat_len == 25 else 3 if flat_len == 9 else 0
        
        n_orb_1 = get_orbital_count(occs_1_1)
        n_orb_2 = get_orbital_count(occs_2_1)
        
        # Create occupation matrices for this entry
        def reshape_occs(occs_data, n_orb):
            if isinstance(occs_data[0], list):
                # Already in matrix format
                return np.array(occs_data)
            else:
                # Flattened format, need to reshape
                return np.array(occs_data).reshape(n_orb, n_orb)
        
        if n_orb_1 > 0:
            occs_1_spin_up = reshape_occs(v_data["atom_1_occs_1"][i], n_orb_1)
            occs_1_spin_down = reshape_occs(v_data["atom_1_occs_2"][i], n_orb_1)
            site0_occ = np.stack([occs_1_spin_up, occs_1_spin_down])  # (2, n_orb, n_orb)
        else:
            site0_occ = np.zeros((2, 5, 5))  # Default to d-orbital size
        
        if n_orb_2 > 0:
            occs_2_spin_up = reshape_occs(v_data["atom_2_occs_1"][i], n_orb_2)
            occs_2_spin_down = reshape_occs(v_data["atom_2_occs_2"][i], n_orb_2)
            site1_occ = np.stack([occs_2_spin_up, occs_2_spin_down])  # (2, n_orb, n_orb)
        else:
            site1_occ = np.zeros((2, 5, 5))  # Default to d-orbital size
        
        site0_occupancies.append(site0_occ)
        site1_occupancies.append(site1_occ)
        site0_orbital_dims.append(n_orb_1 if n_orb_1 > 0 else 0)
        site1_orbital_dims.append(n_orb_2 if n_orb_2 > 0 else 0)
    
    # Convert to numpy arrays
    processed["site0"]["element"] = np.array(site0_elements, dtype="S10")
    processed["site1"]["element"] = np.array(site1_elements, dtype="S10")
    
    # Handle variable-size occupancy matrices
    # For now, pad to maximum size (5×5 for d-orbitals)
    max_orb = 5
    
    site0_occs_padded = np.zeros((n_entries, 2, max_orb, max_orb), dtype=np.float64)
    site1_occs_padded = np.zeros((n_entries, 2, max_orb, max_orb), dtype=np.float64)
    
    for i, (s0_occ, s1_occ) in enumerate(zip(site0_occupancies, site1_occupancies)):
        s0_shape = s0_occ.shape[-1]
        s1_shape = s1_occ.shape[-1]
        
        site0_occs_padded[i, :, :s0_shape, :s0_shape] = s0_occ
        site1_occs_padded[i, :, :s1_shape, :s1_shape] = s1_occ
    
    processed["site0"]["occupancy"] = site0_occs_padded
    processed["site1"]["occupancy"] = site1_occs_padded
    processed["site0"]["orbital_dims"] = np.array(site0_orbital_dims, dtype=np.int32)
    processed["site1"]["orbital_dims"] = np.array(site1_orbital_dims, dtype=np.int32)
    
    return processed


def write_hdf5_dataset(output_path: str, u_processed: Dict, v_processed: Dict) -> None:
    """Write processed data to HDF5 format."""
    try:
        import h5py
    except ImportError:
        raise ImportError("h5py not available. Install with: uv add h5py")
    
    print(f"Writing HDF5 dataset: {output_path}")
    
    with h5py.File(output_path, 'w') as f:
        # Create main hubbard group
        hubbard_group = f.create_group("hubbard")
        
        # Write U parameter data
        if u_processed and len(u_processed["target"]) > 0:
            print(f"Writing {len(u_processed['target'])} U parameter entries...")
            u_group = hubbard_group.create_group("u")
            
            # Site0 data
            site0_group = u_group.create_group("site0")
            site0_group.create_dataset("element", data=u_processed["site0"]["element"])
            site0_group.create_dataset("occupancy", data=u_processed["site0"]["occupancy"], 
                                     compression="gzip", compression_opts=6)
            site0_group.create_dataset("orbital_dims", data=u_processed["site0"]["orbital_dims"],
                                     compression="gzip", compression_opts=6)
            
            # U parameters
            u_group.create_dataset("input", data=u_processed["input"])
            u_group.create_dataset("target", data=u_processed["target"])
        
        # Write V parameter data
        if v_processed and len(v_processed["target"]) > 0:
            print(f"Writing {len(v_processed['target'])} V parameter entries...")
            v_group = hubbard_group.create_group("v")
            
            # Site0 and Site1 data
            for site_name in ["site0", "site1"]:
                site_group = v_group.create_group(site_name)
                site_group.create_dataset("element", data=v_processed[site_name]["element"])
                site_group.create_dataset("occupancy", data=v_processed[site_name]["occupancy"],
                                        compression="gzip", compression_opts=6)
                site_group.create_dataset("orbital_dims", data=v_processed[site_name]["orbital_dims"],
                                        compression="gzip", compression_opts=6)
            
            # Edge data
            edge_group = v_group.create_group("edge")
            edge_group.create_dataset("distance", data=v_processed["edge"]["distance"])
            
            # V parameters
            v_group.create_dataset("input", data=v_processed["input"])
            v_group.create_dataset("target", data=v_processed["target"])
        
        # Add metadata
        f.attrs["format_version"] = "1.0"
        f.attrs["source"] = "HubbardML Arrow to HDF5 conversion"
        f.attrs["description"] = "Hubbard U and V parameters with electronic structure data"
    
    print(f"HDF5 conversion completed: {output_path}")


def convert_arrow_to_hdf5(input_path: str, output_path: str) -> None:
    """Main conversion function."""
    # Load Arrow data
    data = load_arrow_data(input_path)
    
    # Separate U and V parameters
    u_data, v_data = separate_u_v_parameters(data)
    
    # Process each parameter type
    u_processed = process_u_parameters(u_data) if u_data["param_type"] else {}
    v_processed = process_v_parameters(v_data) if v_data["param_type"] else {}
    
    # Write to HDF5
    write_hdf5_dataset(output_path, u_processed, v_processed)
    
    # Report file sizes
    input_size_mb = Path(input_path).stat().st_size / (1024 * 1024)
    output_size_mb = Path(output_path).stat().st_size / (1024 * 1024)
    compression_ratio = input_size_mb / output_size_mb if output_size_mb > 0 else 0
    
    print(f"\nConversion Summary:")
    print(f"  Input (Arrow): {input_size_mb:.2f} MB")
    print(f"  Output (HDF5): {output_size_mb:.2f} MB")
    print(f"  Compression ratio: {compression_ratio:.2f}x")


def main():
    parser = argparse.ArgumentParser(
        description="Convert Arrow datasets to HDF5 format for HubbardML",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Convert main dataset
  uv run python -m hubbardml.convert_to_hdf5 data_uv_2024_1_25.arrow dataset.h5
  
  # Convert test dataset
  uv run python -m hubbardml.convert_to_hdf5 dataset.arrow dataset_test.h5
        """
    )
    
    parser.add_argument("input", help="Input Arrow file (.arrow)")
    parser.add_argument("output", help="Output HDF5 file (.h5)")
    parser.add_argument("--overwrite", action="store_true", 
                       help="Overwrite output file if it exists")
    
    args = parser.parse_args()
    
    # Validate input file
    if not Path(args.input).exists():
        print(f"Error: Input file not found: {args.input}")
        sys.exit(1)
    
    if not args.input.endswith('.arrow'):
        print(f"Error: Input file must be .arrow format")
        sys.exit(1)
    
    # Check output file
    if Path(args.output).exists() and not args.overwrite:
        print(f"Error: Output file exists: {args.output}")
        print("Use --overwrite to replace it")
        sys.exit(1)
    
    try:
        convert_arrow_to_hdf5(args.input, args.output)
        print("Conversion successful!")
        
    except Exception as e:
        print(f"Error during conversion: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
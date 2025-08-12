"""
Dataset inspection utility for HubbardML

Supports both Arrow and HDF5 formats for analyzing Hubbard parameter datasets.
Provides insights into data structure, storage efficiency, and parameter distributions.
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import numpy as np


def inspect_arrow_dataset(file_path: str) -> Dict:
    """Inspect Arrow dataset and return analysis results."""
    try:
        import pyarrow as pa
        import pyarrow.ipc
    except ImportError:
        raise ImportError("PyArrow not available. Install with: uv add pyarrow")
    
    print(f"Analyzing Arrow dataset: {file_path}")
    
    # Load the Arrow file
    with pa.memory_map(file_path, "rb") as source:
        reader = pa.ipc.RecordBatchFileReader(source)
        table = reader.read_all()
    
    file_size_mb = Path(file_path).stat().st_size / (1024 * 1024)
    
    analysis = {
        "format": "Arrow",
        "file_path": file_path,
        "file_size_mb": file_size_mb,
        "num_rows": table.num_rows,
        "num_columns": table.num_columns,
        "schema": {},
        "parameter_analysis": {},
        "occupation_analysis": {},
        "storage_analysis": {}
    }
    
    # Schema analysis
    for i, (name, type_) in enumerate(zip(table.column_names, table.schema.types)):
        analysis["schema"][name] = str(type_)
    
    # Parameter type analysis
    if "param_type" in table.column_names:
        param_types = table["param_type"].to_pylist()
        u_count = param_types.count("U")
        v_count = param_types.count("V")
        total = len(param_types)
        
        analysis["parameter_analysis"] = {
            "total_entries": total,
            "u_parameters": u_count,
            "v_parameters": v_count,
            "u_percentage": (u_count / total * 100) if total > 0 else 0,
            "v_percentage": (v_count / total * 100) if total > 0 else 0
        }
    
    # Occupation matrix analysis
    occs_analysis = {}
    for occs_col in ["atom_1_occs_1", "atom_1_occs_2", "atom_2_occs_1", "atom_2_occs_2"]:
        if occs_col in table.column_names:
            occs_column = table[occs_col]
            
            # Analyze shapes (flattened matrices)
            shapes = {}
            for i in range(min(1000, len(occs_column))):  # Sample first 1000
                arr = occs_column[i].as_py()
                if arr:
                    length = len(arr)
                    shapes[length] = shapes.get(length, 0) + 1
            
            # Determine matrix types
            matrix_types = {}
            for length, count in shapes.items():
                if length == 9:
                    matrix_types["3x3 (p-orbitals)"] = count
                elif length == 25:
                    matrix_types["5x5 (d-orbitals)"] = count
                else:
                    matrix_types[f"unknown ({length} elements)"] = count
            
            occs_analysis[occs_col] = {
                "shapes": shapes,
                "matrix_types": matrix_types
            }
    
    analysis["occupation_analysis"] = occs_analysis
    
    # Storage analysis
    occs_storage_mb = 0
    for occs_col in ["atom_1_occs_1", "atom_1_occs_2", "atom_2_occs_1", "atom_2_occs_2"]:
        if occs_col in table.column_names:
            # Estimate storage for this column
            col_data = table[occs_col]
            # Rough estimate: 8 bytes per float64 * avg elements per entry
            avg_elements = np.mean([len(col_data[i].as_py()) for i in range(min(100, len(col_data)))])
            col_storage_mb = (len(col_data) * avg_elements * 8) / (1024 * 1024)
            occs_storage_mb += col_storage_mb
    
    analysis["storage_analysis"] = {
        "total_file_size_mb": file_size_mb,
        "estimated_occs_storage_mb": occs_storage_mb,
        "compression_ratio": occs_storage_mb / file_size_mb if file_size_mb > 0 else 0,
        "compression_percentage": (1 - file_size_mb / occs_storage_mb) * 100 if occs_storage_mb > 0 else 0
    }
    
    return analysis


def inspect_hdf5_dataset(file_path: str) -> Dict:
    """Inspect HDF5 dataset and return analysis results."""
    try:
        import h5py
    except ImportError:
        raise ImportError("h5py not available. Install with: uv add h5py")
    
    print(f"Analyzing HDF5 dataset: {file_path}")
    
    file_size_mb = Path(file_path).stat().st_size / (1024 * 1024)
    
    analysis = {
        "format": "HDF5",
        "file_path": file_path,
        "file_size_mb": file_size_mb,
        "structure": {},
        "parameter_analysis": {},
        "occupation_analysis": {},
        "storage_analysis": {}
    }
    
    with h5py.File(file_path, 'r') as f:
        # Analyze HDF5 structure
        def analyze_group(group, path=""):
            structure = {}
            for key in group.keys():
                item_path = f"{path}/{key}" if path else key
                item = group[key]
                
                if isinstance(item, h5py.Group):
                    structure[key] = {
                        "type": "group",
                        "children": analyze_group(item, item_path)
                    }
                elif isinstance(item, h5py.Dataset):
                    structure[key] = {
                        "type": "dataset",
                        "shape": item.shape,
                        "dtype": str(item.dtype),
                        "size_mb": item.nbytes / (1024 * 1024)
                    }
            return structure
        
        analysis["structure"] = analyze_group(f)
        
        # Parameter analysis
        param_counts = {}
        total_entries = 0
        
        if "hubbard" in f:
            hubbard_group = f["hubbard"]
            if "u" in hubbard_group and "target" in hubbard_group["u"]:
                u_count = len(hubbard_group["u"]["target"])
                param_counts["U"] = u_count
                total_entries += u_count
            
            if "v" in hubbard_group and "target" in hubbard_group["v"]:
                v_count = len(hubbard_group["v"]["target"])
                param_counts["V"] = v_count
                total_entries += v_count
        
        analysis["parameter_analysis"] = {
            "total_entries": total_entries,
            "u_parameters": param_counts.get("U", 0),
            "v_parameters": param_counts.get("V", 0),
            "u_percentage": (param_counts.get("U", 0) / total_entries * 100) if total_entries > 0 else 0,
            "v_percentage": (param_counts.get("V", 0) / total_entries * 100) if total_entries > 0 else 0
        }
        
        # Occupation matrix analysis for HDF5
        occs_analysis = {}
        if "hubbard" in f:
            for param_type in ["u", "v"]:
                if param_type in f["hubbard"]:
                    param_group = f["hubbard"][param_type]
                    for site in ["site0", "site1"]:
                        if site in param_group and "occupancy" in param_group[site]:
                            occs_data = param_group[site]["occupancy"]
                            key = f"{param_type}/{site}/occupancy"
                            occs_analysis[key] = {
                                "shape": occs_data.shape,
                                "dtype": str(occs_data.dtype),
                                "storage_mb": occs_data.nbytes / (1024 * 1024),
                                "orbital_type": "d-orbitals" if occs_data.shape[-1] == 5 else "p-orbitals" if occs_data.shape[-1] == 3 else "unknown"
                            }
        
        analysis["occupation_analysis"] = occs_analysis
    
    return analysis


def print_analysis(analysis: Dict) -> None:
    """Print formatted analysis results."""
    
    print(f"\n{'='*60}")
    print(f"DATASET ANALYSIS REPORT")
    print(f"{'='*60}")
    
    # Basic info
    print(f"\nFile Information:")
    print(f"   Format: {analysis['format']}")
    print(f"   Path: {analysis['file_path']}")
    print(f"   Size: {analysis['file_size_mb']:.2f} MB")
    
    if analysis["format"] == "Arrow":
        print(f"   Rows: {analysis['num_rows']:,}")
        print(f"   Columns: {analysis['num_columns']}")
    
    # Parameter analysis
    if analysis["parameter_analysis"]:
        pa = analysis["parameter_analysis"]
        print(f"\nParameter Distribution:")
        print(f"   Total entries: {pa['total_entries']:,}")
        if pa['u_parameters'] > 0:
            print(f"   U parameters: {pa['u_parameters']:,} ({pa['u_percentage']:.1f}%)")
        if pa['v_parameters'] > 0:
            print(f"   V parameters: {pa['v_parameters']:,} ({pa['v_percentage']:.1f}%)")
    
    # Occupation matrix analysis
    if analysis["occupation_analysis"]:
        print(f"\nOccupation Matrix Analysis:")
        oa = analysis["occupation_analysis"]
        
        if analysis["format"] == "Arrow":
            for col_name, col_data in oa.items():
                if col_data.get("matrix_types"):
                    print(f"   {col_name}:")
                    for matrix_type, count in col_data["matrix_types"].items():
                        print(f"     - {matrix_type}: {count} entries")
        
        elif analysis["format"] == "HDF5":
            for col_name, col_data in oa.items():
                print(f"   {col_name}:")
                print(f"     - Shape: {col_data['shape']}")
                print(f"     - Type: {col_data['orbital_type']}")
                print(f"     - Storage: {col_data['storage_mb']:.2f} MB")
    
    # Storage analysis
    if analysis["storage_analysis"]:
        sa = analysis["storage_analysis"]
        print(f"\nStorage Analysis:")
        print(f"   Total file size: {sa['total_file_size_mb']:.2f} MB")
        
        if "estimated_occs_storage_mb" in sa:
            print(f"   Estimated raw occupation data: {sa['estimated_occs_storage_mb']:.2f} MB")
            print(f"   Compression ratio: {sa['compression_ratio']:.2f}x")
            print(f"   Space saved: {sa['compression_percentage']:.1f}%")
    
    # Schema/Structure
    if analysis["format"] == "Arrow" and analysis["schema"]:
        print(f"\nSchema (first 10 columns):")
        for i, (name, type_) in enumerate(list(analysis["schema"].items())[:10]):
            print(f"   {i+1:2d}. {name}: {type_}")
        if len(analysis["schema"]) > 10:
            print(f"       ... and {len(analysis['schema']) - 10} more columns")
    
    elif analysis["format"] == "HDF5" and analysis["structure"]:
        print(f"\nHDF5 Structure:")
        
        def print_structure(structure, indent=0):
            for name, info in structure.items():
                prefix = "   " + "  " * indent
                if info["type"] == "group":
                    print(f"{prefix}{name}/")
                    if info["children"]:
                        print_structure(info["children"], indent + 1)
                else:  # dataset
                    size_str = f" ({info['size_mb']:.2f} MB)" if info['size_mb'] > 0.1 else ""
                    print(f"{prefix}{name}: {info['shape']} {info['dtype']}{size_str}")
        
        print_structure(analysis["structure"])


def main():
    parser = argparse.ArgumentParser(
        description="Inspect HubbardML datasets (Arrow or HDF5 format)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Inspect Arrow dataset
  uv run python -m hubbardml.inspect data_uv_2024_1_25.arrow
  
  # Inspect HDF5 dataset  
  uv run python -m hubbardml.inspect dataset.h5
  
  # Inspect multiple files
  uv run python -m hubbardml.inspect dataset.arrow dataset.h5
        """
    )
    
    parser.add_argument(
        "files",
        nargs="+",
        help="Dataset file(s) to inspect (.arrow or .h5/.hdf5)"
    )
    
    args = parser.parse_args()
    
    for file_path in args.files:
        if not Path(file_path).exists():
            print(f"Error: File not found: {file_path}")
            continue
        
        try:
            # Determine format from extension
            suffix = Path(file_path).suffix.lower()
            
            if suffix == ".arrow":
                analysis = inspect_arrow_dataset(file_path)
            elif suffix in [".h5", ".hdf5"]:
                analysis = inspect_hdf5_dataset(file_path)
            else:
                print(f"Error: Unsupported file format: {file_path}")
                print("   Supported formats: .arrow, .h5, .hdf5")
                continue
            
            print_analysis(analysis)
            
            if len(args.files) > 1:
                print("\n" + "="*60 + "\n")
                
        except Exception as e:
            print(f"Error analyzing {file_path}: {e}")
            import traceback
            traceback.print_exc()


if __name__ == "__main__":
    main()
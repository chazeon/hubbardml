"""
Dataset inspection utility for HubbardML

Supports both Arrow and HDF5 formats for analyzing Hubbard parameter datasets.
Provides insights into data structure, storage efficiency, and parameter distributions.
"""

import sys
from dataclasses import dataclass, field
from itertools import product
from pathlib import Path
from typing import Dict, Tuple, Optional, Union
import numpy as np
import click
import h5py
import pyarrow as pa
import pyarrow.ipc


@dataclass
class ParameterAnalysis:
    """Analysis of Hubbard parameter distribution."""
    u_count: int = 0
    v_count: int = 0
    
    @property
    def total(self) -> int:
        return self.u_count + self.v_count
    
    @property
    def u_pct(self) -> float:
        return self.u_count / self.total * 100
    
    @property
    def v_pct(self) -> float:
        return self.v_count / self.total * 100


@dataclass
class OccupationMatrixInfo:
    """Information about occupation matrix columns."""
    shapes: Dict[int, int] = field(default_factory=dict)
    matrix_types: Dict[str, int] = field(default_factory=dict)


@dataclass
class HDF5OccupationInfo:
    """HDF5-specific occupation matrix information."""
    shape: Tuple[int, ...] = field(default_factory=tuple)
    dtype: str = ""
    storage_mb: float = 0.0
    orbital_type: str = ""


@dataclass
class StorageAnalysis:
    """Storage and compression analysis."""
    total_file_size_mb: float = 0.0
    estimated_occs_storage_mb: float = 0.0
    
    @property
    def compression_ratio(self) -> float:
        return self.estimated_occs_storage_mb / self.total_file_size_mb
    
    @property
    def compression_pct(self) -> float:
        return (1 - self.total_file_size_mb / self.estimated_occs_storage_mb) * 100


@dataclass
class HDF5StructureItem:
    """HDF5 structure item (group or dataset)."""
    type: str  # "group" or "dataset"
    children: Dict[str, 'HDF5StructureItem'] = field(default_factory=dict)
    shape: Optional[Tuple[int, ...]] = None
    dtype: Optional[str] = None
    size_mb: Optional[float] = None


@dataclass
class DatasetAnalysis:
    """Complete dataset analysis results."""
    format: str = ""
    file_path: str = ""
    file_size_mb: float = 0.0
    num_rows: Optional[int] = None
    num_columns: Optional[int] = None
    schema: Dict[str, str] = field(default_factory=dict)
    structure: Dict[str, HDF5StructureItem] = field(default_factory=dict)
    parameter_analysis: ParameterAnalysis = field(default_factory=ParameterAnalysis)
    occupation_analysis: Dict[str, Union[OccupationMatrixInfo, HDF5OccupationInfo]] = field(default_factory=dict)
    storage_analysis: StorageAnalysis = field(default_factory=StorageAnalysis)

    @classmethod
    def from_arrow(cls, file_path: str) -> 'DatasetAnalysis':
        """Create analysis from Arrow dataset."""
        print(f"Analyzing Arrow dataset: {file_path}")
        
        # Load the Arrow file
        with pa.memory_map(file_path, "rb") as source:
            reader = pyarrow.ipc.RecordBatchFileReader(source, options=None)
            table = reader.read_all()
        
        file_size_mb = Path(file_path).stat().st_size / (1024 * 1024)
        
        analysis = cls(
            format="Arrow",
            file_path=file_path,
            file_size_mb=file_size_mb,
            num_rows=table.num_rows,
            num_columns=table.num_columns
        )
        
        analysis._analyze_arrow_data(table)
        return analysis

    @classmethod
    def from_hdf5(cls, file_path: str) -> 'DatasetAnalysis':
        """Create analysis from HDF5 dataset."""
        print(f"Analyzing HDF5 dataset: {file_path}")
        
        file_size_mb = Path(file_path).stat().st_size / (1024 * 1024)
        
        analysis = cls(
            format="HDF5",
            file_path=file_path,
            file_size_mb=file_size_mb
        )
        
        with h5py.File(file_path, 'r') as f:  # type: ignore
            analysis._analyze_hdf5_data(f)
        
        return analysis

    def _analyze_arrow_data(self, table):
        """Analyze Arrow table data."""
        self.schema = self._extract_arrow_schema(table)
        self.parameter_analysis = self._analyze_arrow_parameters(table)
        arrow_occupation_analysis = self._analyze_arrow_occupation_matrices(table)
        self.occupation_analysis.update(arrow_occupation_analysis)
        self.storage_analysis = self._calculate_arrow_storage(table)

    def _extract_arrow_schema(self, table) -> Dict[str, str]:
        """Extract schema information from Arrow table."""
        schema = {}
        for name, type_ in zip(table.column_names, table.schema.types):
            schema[name] = str(type_)
        return schema

    def _analyze_arrow_parameters(self, table) -> ParameterAnalysis:
        """Analyze parameter distribution in Arrow table."""
        if "param_type" not in table.column_names:
            return ParameterAnalysis()
        
        param_types = table["param_type"].to_pylist()
        u_count = param_types.count("U")
        v_count = param_types.count("V")
        
        return ParameterAnalysis(
            u_count=u_count,
            v_count=v_count
        )

    def _analyze_arrow_occupation_matrices(self, table) -> Dict[str, OccupationMatrixInfo]:
        """Analyze occupation matrix columns in Arrow table."""
        occupation_analysis = {}
        
        for i_atom, i_occs in product([1, 2], [1, 2]):
            occs_col = f"atom_{i_atom}_occs_{i_occs}"
            if occs_col not in table.column_names:
                continue
                
            occs_column = table[occs_col]
            
            # Analyze shapes (flattened matrices)
            shapes: Dict[int, int] = {}
            for i in range(min(1000, len(occs_column))):  # Sample first 1000
                arr = occs_column[i].as_py()
                if arr:
                    length = len(arr)
                    shapes[length] = shapes.get(length, 0) + 1
            
            # Determine matrix types
            matrix_types: Dict[str, int] = {}
            for length, count in shapes.items():
                if length == 9:
                    matrix_types["3x3 (p-orbitals)"] = count
                elif length == 25:
                    matrix_types["5x5 (d-orbitals)"] = count
                else:
                    matrix_types[f"unknown ({length} elements)"] = count
            
            occupation_analysis[occs_col] = OccupationMatrixInfo(
                shapes=shapes,
                matrix_types=matrix_types
            )
        
        return occupation_analysis

    def _calculate_arrow_storage(self, table) -> StorageAnalysis:
        """Calculate storage analysis for Arrow table."""
        occs_storage_mb = 0.0
        
        for i_atom, i_occs in product([1, 2], [1, 2]):
            occs_col = f"atom_{i_atom}_occs_{i_occs}"
            if occs_col not in table.column_names:
                continue
                
            col_data = table[occs_col]
            # Rough estimate: 8 bytes per float64 * avg elements per entry
            avg_elements = float(np.mean([len(col_data[i].as_py()) for i in range(min(100, len(col_data)))]))
            col_storage_mb = float((len(col_data) * avg_elements * 8) / (1024 * 1024))
            occs_storage_mb += col_storage_mb
        
        return StorageAnalysis(
            total_file_size_mb=self.file_size_mb,
            estimated_occs_storage_mb=occs_storage_mb
        )

    def _analyze_hdf5_data(self, f):
        """Analyze HDF5 file data."""
        self.structure = self._analyze_hdf5_structure(f)
        self.parameter_analysis = self._analyze_hdf5_parameters(f)
        hdf5_occupation_analysis = self._analyze_hdf5_occupation_matrices(f)
        self.occupation_analysis.update(hdf5_occupation_analysis)

    def _analyze_hdf5_structure(self, f) -> Dict[str, HDF5StructureItem]:
        """Analyze HDF5 file structure."""
        def analyze_group(group, path=""):
            structure = {}
            for key in group.keys():
                item_path = f"{path}/{key}" if path else key
                item = group[key]
                
                if isinstance(item, h5py.Group):
                    structure[key] = HDF5StructureItem(
                        type="group",
                        children=analyze_group(item, item_path)
                    )
                elif isinstance(item, h5py.Dataset):
                    structure[key] = HDF5StructureItem(
                        type="dataset",
                        shape=item.shape,
                        dtype=str(item.dtype),
                        size_mb=float(item.size * item.dtype.itemsize) / (1024 * 1024)
                    )
            return structure
        
        return analyze_group(f)

    def _analyze_hdf5_parameters(self, f) -> ParameterAnalysis:
        """Analyze parameter distribution in HDF5 file."""
        param_counts = {"u": 0, "v": 0}
        
        if "hubbard" in f and isinstance(f["hubbard"], h5py.Group):
            hubbard_group = f["hubbard"]
            for param_type in ["u", "v"]:
                if param_type in hubbard_group and isinstance(hubbard_group[param_type], h5py.Group):
                    param_group = hubbard_group[param_type]  # type: ignore
                    if "target" in param_group and isinstance(param_group["target"], h5py.Dataset):  # type: ignore
                        param_counts[param_type] = len(param_group["target"])  # type: ignore
        
        return ParameterAnalysis(
            u_count=param_counts["u"],
            v_count=param_counts["v"]
        )

    def _analyze_hdf5_occupation_matrices(self, f) -> Dict[str, HDF5OccupationInfo]:  # type: ignore
        """Analyze occupation matrix data in HDF5 file."""
        occupation_analysis = {}
        
        if "hubbard" in f and isinstance(f["hubbard"], h5py.Group):
            hubbard_group = f["hubbard"]
            for param_type in ["u", "v"]:
                if param_type in hubbard_group and isinstance(hubbard_group[param_type], h5py.Group):
                    param_group = hubbard_group[param_type]  # type: ignore
                    for site in ["site0", "site1"]:
                        if (site in param_group and isinstance(param_group[site], h5py.Group)):  # type: ignore
                            site_group = param_group[site]  # type: ignore
                            if "occupancy" in site_group and isinstance(site_group["occupancy"], h5py.Dataset):  # type: ignore
                                occs_data = site_group["occupancy"]  # type: ignore
                                key = f"{param_type}/{site}/occupancy"
                                orbital_type = "d-orbitals" if occs_data.shape[-1] == 5 else "p-orbitals" if occs_data.shape[-1] == 3 else "unknown"
                                
                                occupation_analysis[key] = HDF5OccupationInfo(
                                    shape=occs_data.shape,
                                    dtype=str(occs_data.dtype),
                                    storage_mb=float(occs_data.size * occs_data.dtype.itemsize) / (1024 * 1024),
                                    orbital_type=orbital_type
                                )
        
        return occupation_analysis




def print_analysis(analysis: DatasetAnalysis) -> None:
    """Print formatted analysis results."""
    
    print(f"\n{'='*60}")
    print("DATASET ANALYSIS REPORT")
    print(f"{'='*60}")
    
    # Basic info
    print("\nFile Information:")
    print(f"   Format: {analysis.format}")
    print(f"   Path: {analysis.file_path}")
    print(f"   Size: {analysis.file_size_mb:.2f} MB")
    
    if analysis.format == "Arrow":
        print(f"   Rows: {analysis.num_rows:,}")
        print(f"   Columns: {analysis.num_columns}")
    
    # Parameter analysis
    if analysis.parameter_analysis.total > 0:
        pa = analysis.parameter_analysis
        print("\nParameter Distribution:")
        print(f"   Total entries: {pa.total:,}")
        if pa.u_count > 0:
            print(f"   U parameters: {pa.u_count:,} ({pa.u_pct:.1f}%)")
        if pa.v_count > 0:
            print(f"   V parameters: {pa.v_count:,} ({pa.v_pct:.1f}%)")
    
    # Occupation matrix analysis
    if analysis.occupation_analysis:
        print("\nOccupation Matrix Analysis:")
        oa = analysis.occupation_analysis
        
        if analysis.format == "Arrow":
            for col_name, col_data in oa.items():
                if isinstance(col_data, OccupationMatrixInfo) and col_data.matrix_types:
                    print(f"   {col_name}:")
                    for matrix_type, count in col_data.matrix_types.items():
                        print(f"     - {matrix_type}: {count} entries")
        
        elif analysis.format == "HDF5":
            for col_name, col_data in oa.items():
                if isinstance(col_data, HDF5OccupationInfo):
                    print(f"   {col_name}:")
                    print(f"     - Shape: {col_data.shape}")
                    print(f"     - Type: {col_data.orbital_type}")
                    print(f"     - Storage: {col_data.storage_mb:.2f} MB")
    
    # Storage analysis
    if analysis.storage_analysis.total_file_size_mb > 0:
        sa = analysis.storage_analysis
        print("\nStorage Analysis:")
        print(f"   Total file size: {sa.total_file_size_mb:.2f} MB")
        
        if sa.estimated_occs_storage_mb > 0:
            print(f"   Estimated raw occupation data: {sa.estimated_occs_storage_mb:.2f} MB")
            print(f"   Compression ratio: {sa.compression_ratio:.2f}x")
            print(f"   Space saved: {sa.compression_pct:.1f}%")
    
    # Schema/Structure
    if analysis.format == "Arrow" and analysis.schema:
        print("\nSchema (first 10 columns):")
        for i, (name, type_) in enumerate(list(analysis.schema.items())[:10]):
            print(f"   {i+1:2d}. {name}: {type_}")
        if len(analysis.schema) > 10:
            print(f"       ... and {len(analysis.schema) - 10} more columns")
    
    elif analysis.format == "HDF5" and analysis.structure:
        print("\nHDF5 Structure:")
        
        def print_structure(structure, indent=0):
            for name, info in structure.items():
                prefix = "   " + "  " * indent
                if info.type == "group":
                    print(f"{prefix}{name}/")
                    if info.children:
                        print_structure(info.children, indent + 1)
                else:  # dataset
                    size_str = f" ({info.size_mb:.2f} MB)" if info.size_mb and info.size_mb > 0.1 else ""
                    print(f"{prefix}{name}: {info.shape} {info.dtype}{size_str}")
        
        print_structure(analysis.structure)


@click.command()
@click.argument("file_path", type=click.Path(exists=True, path_type=Path))
@click.help_option("--help", "-h")
def main(file_path: Path):
    """Inspect HubbardML dataset (Arrow or HDF5 format).
    
    \b
    Examples:
      uv run python -m hubbardml.inspect data_uv_2024_1_25.arrow
      uv run python -m hubbardml.inspect dataset.h5
    """
    try:
        # Determine format from extension
        suffix = file_path.suffix.lower()
        
        if suffix == ".arrow":
            analysis = DatasetAnalysis.from_arrow(str(file_path))
        elif suffix in [".h5", ".hdf5"]:
            analysis = DatasetAnalysis.from_hdf5(str(file_path))
        else:
            click.echo(f"Error: Unsupported file format: {file_path}", err=True)
            click.echo("Supported formats: .arrow, .h5, .hdf5", err=True)
            sys.exit(1)
        
        print_analysis(analysis)
        
    except Exception as e:
        click.echo(f"Error analyzing {file_path}: {e}", err=True)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
"""Inspect Arrow datasets."""

import sys
from pathlib import Path
import click


def load_arrow(file_path):
    """Load Arrow file."""
    import pyarrow as pa
    
    with pa.memory_map(file_path, "rb") as source:
        table = pa.ipc.RecordBatchFileReader(source).read_all()
    
    return {col: table[col].to_pylist() for col in table.column_names}, table


def inspect_dataset(file_path):
    """Inspect a dataset."""
    print(f"Loading {file_path}...")
    data, table = load_arrow(file_path)
    
    # Basic info
    size_mb = Path(file_path).stat().st_size / 1024**2
    print(f"\n{Path(file_path).name}")
    print(f"Size: {size_mb:.1f}MB")
    print(f"Shape: {table.num_rows:,} rows × {table.num_columns} cols")
    print(f"Columns: {', '.join(table.column_names)}")
    
    # Parameters
    if "param_type" in data and "param_out" in data:
        import numpy as np
        types = data["param_type"]
        values = data["param_out"]
        
        print("\nParameters:")
        for ptype in set(types):
            count = types.count(ptype)
            pct = count / len(types) * 100
            vals = [values[i] for i, t in enumerate(types) if t == ptype]
            
            arr = np.array(vals)
            print(f"  {ptype}: {count:,} ({pct:.1f}%) | "
                  f"{arr.min():.2f}-{arr.max():.2f} eV (avg {arr.mean():.2f})")
    
    # Elements
    if "atom_1_element" in data:
        elems = data["atom_1_element"]
        if "atom_2_element" in data:
            elems += data["atom_2_element"]
        
        elem_counts = {}
        for e in elems:
            elem_counts[e] = elem_counts.get(e, 0) + 1
        
        print(f"\nElements: {dict(sorted(elem_counts.items()))}")
    
    # Occupation matrices
    occ_cols = [col for col in data.keys() if "occs" in col]
    sizes = {}
    formats = {}
    storage_mb = 0.0
    
    if occ_cols:
        total = 0
        
        for col in occ_cols:
            for occs in data[col]:
                if occs is None:
                    continue
                total += 1
                
                if isinstance(occs[0], list):
                    fmt = "nested"
                    size = len(occs)
                else:
                    fmt = "flat"
                    size = int(len(occs) ** 0.5)
                
                formats[fmt] = formats.get(fmt, 0) + 1
                sizes[size] = sizes.get(size, 0) + 1
        
        print("\nOccupation matrices:")
        print(f"  Total: {total:,}")
        print(f"  Sizes: {dict(sorted(sizes.items()))}")
        print(f"  Formats: {formats}")
        
        # Estimate storage
        total_elements = sum(s*s*count for s, count in sizes.items())
        storage_mb = total_elements * 8 / 1024**2  # float64
        print(f"  Raw storage: ~{storage_mb:.1f}MB ({storage_mb/size_mb:.0%} of file)")
    
    # Quick recommendations
    print("\nNotes:")
    if "U" in [t for t in data.get("param_type", [])]:
        print("  - U params have duplicate atom_1/atom_2 (can save ~50% storage)")
    if len(sizes) > 1:
        print("  - Mixed orbital sizes (padding wastes space)")
    if storage_mb > size_mb * 0.7:
        print("  - Occupation matrices dominate storage (HDF5 compression helps)")


@click.command()
@click.argument("input_file", type=click.Path(exists=True, path_type=Path))
@click.option("--summary", is_flag=True, help="Just show basic info")
def main(input_file: Path, summary: bool):
    """Inspect Arrow dataset.
    
    Examples:
      uv run python -m hubbardml inspect data.arrow
      uv run python -m hubbardml inspect data.arrow --summary
    """
    if input_file.suffix.lower() != '.arrow':
        click.echo(f"Error: Need .arrow file, got {input_file.suffix}", err=True)
        sys.exit(1)
    
    try:
        if summary:
            data, table = load_arrow(str(input_file))
            size_mb = Path(input_file).stat().st_size / 1024**2
            ptypes = len(set(data.get("param_type", [])))
            print(f"{input_file.name}: {size_mb:.1f}MB, {table.num_rows:,} rows, {ptypes} param types")
        else:
            inspect_dataset(str(input_file))
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
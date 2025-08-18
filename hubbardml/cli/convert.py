"""
Data format conversion utilities for HubbardML

CLI interface for converting datasets to HDF5 format.
HDF5 is the preferred format for training and storage.
"""

import sys
from pathlib import Path
import click

# Import conversion functions
from ..io.datasets.arrow_to_hdf5 import convert_arrow_to_hdf5


@click.group()
def main():
    """Convert datasets to HDF5 format for HubbardML.
    
    HDF5 is the recommended format for training workflows due to:
    - Hierarchical structure for U/V parameter separation
    - Memory-mapped loading for large datasets  
    - Better compression and performance
    
    \b
    Examples:
      # Convert Arrow to HDF5
      uv run python -m hubbardml convert from-arrow data.arrow dataset.h5
      
      # Force overwrite existing files
      uv run python -m hubbardml convert from-arrow data.arrow dataset.h5 --overwrite
    """
    pass


@main.command("from-arrow")
@click.argument("arrow_file", type=click.Path(exists=True, path_type=Path))
@click.argument("hdf5_file", type=click.Path(path_type=Path))
@click.option("--overwrite", is_flag=True, help="Overwrite output file if it exists")
def from_arrow(arrow_file: Path, hdf5_file: Path, overwrite: bool):
    """Convert Arrow dataset to HDF5 format.
    
    ARROW_FILE: Input Arrow file (.arrow)
    HDF5_FILE: Output HDF5 file (.h5)
    
    \b
    Examples:
      # Convert published dataset
      uv run python -m hubbardml convert from-arrow data/data_uv_2024_1_25.arrow dataset.h5
      
      # Convert with overwrite
      uv run python -m hubbardml convert from-arrow data.arrow dataset.h5 --overwrite
    """
    # Validate input file extension
    if arrow_file.suffix.lower() not in ['.arrow']:
        click.echo(f"Error: Input must be Arrow format (.arrow), got: {arrow_file.suffix}", err=True)
        sys.exit(1)
    
    # Suggest proper output extension
    if hdf5_file.suffix.lower() not in ['.h5', '.hdf5']:
        click.echo(f"Warning: Output file should use .h5 extension, got: {hdf5_file.suffix}", err=True)
    
    # Check output file
    if hdf5_file.exists() and not overwrite:
        click.echo(f"Error: Output file exists: {hdf5_file}", err=True)
        click.echo("Use --overwrite to replace it", err=True)
        sys.exit(1)
    
    try:
        click.echo(f"Converting {arrow_file} → {hdf5_file}")
        result = convert_arrow_to_hdf5(str(arrow_file), str(hdf5_file))
        
        click.echo("✓ Conversion successful!")
        click.echo(f"  Input:  {arrow_file} ({arrow_file.stat().st_size / 1024 / 1024:.1f} MB)")
        click.echo(f"  Output: {hdf5_file} ({hdf5_file.stat().st_size / 1024 / 1024:.1f} MB)")
        
    except Exception as e:
        click.echo(f"✗ Error during conversion: {e}", err=True)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
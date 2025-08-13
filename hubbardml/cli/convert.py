"""
Convert Arrow datasets to HDF5 format for HubbardML

CLI interface for converting Arrow datasets to structured HDF5 format
optimized for ML training workflows.
"""

import sys
from pathlib import Path
import click

# Import shared conversion functionality
from ..io.conversion import convert_arrow_to_hdf5


@click.command()
@click.argument("arrow_file", type=click.Path(exists=True, path_type=Path))
@click.argument("hdf5_file", type=click.Path(path_type=Path))
@click.option("--overwrite", is_flag=True, help="Overwrite output file if it exists")
@click.help_option("--help", "-h")
def main(arrow_file: Path, hdf5_file: Path, overwrite: bool):
    """Convert Arrow datasets to HDF5 format for HubbardML.
    
    ARROW_FILE: Input Arrow file (.arrow)
    HDF5_FILE: Output HDF5 file (.h5)
    
    \\b
    Examples:
      # Convert main dataset
      uv run python -m hubbardml convert-hdf5 data/data_uv_2024_1_25.arrow dataset.h5
      
      # Convert test dataset
      uv run python -m hubbardml convert-hdf5 data/dataset.arrow dataset_test.h5
    """
    # Validate input file extension
    if arrow_file.suffix.lower() != '.arrow':
        click.echo(f"Error: Input file must be .arrow format, got: {arrow_file.suffix}", err=True)
        sys.exit(1)
    
    # Check output file
    if hdf5_file.exists() and not overwrite:
        click.echo(f"Error: Output file exists: {hdf5_file}", err=True)
        click.echo("Use --overwrite to replace it", err=True)
        sys.exit(1)
    
    try:
        result = convert_arrow_to_hdf5(str(arrow_file), str(hdf5_file))
        click.echo("Conversion successful!")
        
    except Exception as e:
        click.echo(f"Error during conversion: {e}", err=True)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
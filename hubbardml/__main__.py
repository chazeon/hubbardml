"""
CLI entry point for hubbardml package
"""

import sys
import argparse


def main():
    """Main CLI dispatcher for hubbardml subcommands."""
    if len(sys.argv) < 2:
        print("Usage: python -m hubbardml <command> [args...]")
        print("\nAvailable commands:")
        print("  inspect       Inspect dataset files (Arrow/HDF5)")
        print("  convert-hdf5  Convert Arrow datasets to HDF5 format")
        print("\nExamples:")
        print("  python -m hubbardml inspect data.arrow")
        print("  python -m hubbardml convert-hdf5 data.arrow dataset.h5")
        sys.exit(1)
    
    command = sys.argv[1]
    
    if command == "inspect":
        # Remove the command from sys.argv and call inspect
        sys.argv = [sys.argv[0]] + sys.argv[2:]
        from .inspect import main as inspect_main
        inspect_main()
    elif command == "convert-hdf5":
        # Remove the command from sys.argv and call convert
        sys.argv = [sys.argv[0]] + sys.argv[2:]
        from .convert_to_hdf5 import main as convert_main
        convert_main()
    else:
        print(f"Error: Unknown command '{command}'")
        print("Available commands: inspect, convert-hdf5")
        sys.exit(1)


if __name__ == "__main__":
    main()
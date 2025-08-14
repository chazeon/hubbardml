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
        print("  train         Train Hubbard parameter prediction models")  
        print("  predict       Predict Hubbard parameters from trained models")
        print("  inspect       Inspect dataset files (Arrow/HDF5)")
        print("  convert       Convert between Arrow and HDF5 formats")
        print("\nExamples:")
        print("  python -m hubbardml train config.yaml output/")
        print("  python -m hubbardml predict model/ scf.out")
        print("  python -m hubbardml inspect data.arrow")
        print("  python -m hubbardml convert data.arrow dataset.h5")
        sys.exit(1)
    
    command = sys.argv[1]
    # Remove the command from sys.argv for subcommands
    sys.argv = [sys.argv[0]] + sys.argv[2:]
    
    if command == "train":
        from .cli.train import main as train_main
        train_main()
    elif command == "predict":
        from .cli.predict import main as predict_main
        predict_main()
    elif command == "inspect":
        from .cli.inspect import main as inspect_main
        inspect_main()
    elif command in ("convert", "convert-hdf5"):  # Support both names
        from .cli.convert import main as convert_main
        convert_main()
    else:
        print(f"Error: Unknown command '{command}'")
        print("Available commands: train, predict, inspect, convert")
        sys.exit(1)


if __name__ == "__main__":
    main()
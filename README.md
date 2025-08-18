# HubbardML

[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Commitizen friendly](https://img.shields.io/badge/commitizen-friendly-brightgreen.svg)](http://commitizen.github.io/cz-cli/)
[![security: bandit](https://img.shields.io/badge/security-bandit-yellow.svg)](https://github.com/PyCQA/bandit)

HubbardML is a machine learning library for predicting self-consistent Hubbard parameters using equivariant neural networks. This implementation predicts DFT+U parameters from electronic occupation matrices, enabling more accurate materials modeling with reduced computational cost.

**Associated Publication:**
> Uhrin, M., Zadoks, A., Binci, L., Marzari, N., & Timrov, I. (2025). Machine learning Hubbard parameters with equivariant neural networks. *Nature Computational Materials*, 11(1), 19. [DOI: 10.1038/s41524-024-01501-5](https://www.nature.com/articles/s41524-024-01501-5)

## Quick Start

The fastest way to get started is with our working example:

```bash
# 1. Clone and install
git clone https://github.com/muhrin/hubbardml
cd hubbardml
uv sync --extra dev

# 2. Try the complete workflow
cd example/
# Follow the step-by-step README.md for data download and training
```

Or jump straight to prediction with a pre-trained model:

```bash
# Predict Hubbard parameters from QE output
uv run python -m hubbardml.cli.predict u_model/ example/fp/scf.out

# Multiple output formats available
uv run python -m hubbardml.cli.predict u_model/ example/fp/scf.out -t json
uv run python -m hubbardml.cli.predict u_model/ example/fp/scf.out -t qe_legacy
```

## Features

- **Self-consistent prediction**: Predicts converged DFT+U parameters from linear-response calculations
- **Fast inference**: XML and SCF output file support with automatic format detection  
- **High accuracy**: Equivariant neural networks preserve rotational symmetry
- **Multiple formats**: JSON, Quantum ESPRESSO input, legacy format outputs
- **Data analysis**: Built-in dataset inspection and visualization tools
- **Modern CLI**: Clean command-line interface for all operations

### Supported Systems
- **Elements**: Fe, Ni, Mn, Co, Ti, O, S transition metal compounds
- **Parameters**: U (on-site, 3-8 eV) and V (inter-site, 0.1-2 eV) interactions
- **Input formats**: Quantum ESPRESSO SCF output (.out) and XML (.xml) files
- **Output formats**: QE input blocks, JSON, legacy formats

## Installation

HubbardML uses [uv](https://docs.astral.sh/uv/) for fast dependency management:

### Development Setup (Recommended)
```bash
git clone https://github.com/muhrin/hubbardml
cd hubbardml
uv sync --extra dev

# Optional: install pre-commit hooks
uv run pre-commit install
```

### Production Setup
```bash
uv sync  # Runtime dependencies only
```

### Alternative Methods
```bash
# Using uv with pip-style workflow
uv venv && source .venv/bin/activate
uv pip install -e .[dev]

# Traditional pip (fallback)
pip install -e .[dev]
```

## Usage

### CLI Tools

#### Prediction (Primary Use Case)
```bash
# Basic prediction from SCF output
uv run python -m hubbardml.cli.predict u_model/ scf.out

# From XML output with JSON format
uv run python -m hubbardml.cli.predict u_model/ pwscf.xml -t json

# Save to file with custom variables
uv run python -m hubbardml.cli.predict u_model/ scf.out -o results.txt -v projection_type:atomic
```

#### Dataset Analysis
```bash
# Inspect the dataset
uv run python -m hubbardml.cli.inspect data/data_uv_2024_1_25.arrow

# Quick summary only
uv run python -m hubbardml.cli.inspect data/data_uv_2024_1_25.arrow --summary
```

#### Data Conversion
```bash
# Convert between formats (when implemented)
uv run python -m hubbardml.cli.convert dataset.arrow dataset.h5
```

### Working with Data

#### Available Datasets
- **Main dataset**: `data_uv_2024_1_25.arrow` (69MB, 645k entries) - Complete published dataset
  - Download: [Materials Cloud Archive](https://archive.materialscloud.org/record/2024.160)

#### Dataset Contents
```bash
# Example inspection output
uv run python -m hubbardml.cli.inspect data/data_uv_2024_1_25.arrow

# Shows:
# - Element distribution (Fe, Ni, Mn, etc.)
# - Parameter statistics (U: ~5.8 eV mean, V: ~0.7 eV mean)  
# - Occupation matrix dimensions (5×5 d-orbitals, 3×3 p-orbitals)
# - Storage efficiency analysis
```

## Model Architecture

HubbardML implements equivariant neural networks that preserve the 3D rotational symmetry of electronic systems:

- **UModel**: On-site Hubbard U parameters using single-site architecture
- **VModel**: Inter-site Hubbard V parameters using two-site architecture  
- **Features**: d-orbital occupation matrices from DFT calculations
- **Symmetry**: E(3)-equivariant layers via e3nn and e3psi libraries
- **Species**: Supports Fe, Ni, Mn (Co, Ti excluded due to limited training data)

## Advanced Usage

### Research Experiments

The complete experimental workflow from the paper is available:

```bash
cd experiments/

# Train U model with Hydra configuration
uv run python run.py experiment=predict_hp model=u

# Train V model  
uv run python run.py experiment=predict_hp model=v

# Custom experiments
uv run python run.py model=u trainer.max_epochs=1000 train.batch_size=512
```

### Model Training

```bash
# Train U model using template configuration
uv run python -m hubbardml.cli.train config_examples/u_model.yaml models/u_model/

# Train V model
uv run python -m hubbardml.cli.train config_examples/v_model.yaml models/v_model/

# Train with Arrow data (auto-converts to HDF5)
uv run python -m hubbardml.cli.train my_config.yaml output_dir/

# Evaluate trained models
uv run python -m hubbardml.cli.predict models/u_model/ example/fp/scf.out
```

## Output Formats

### Quantum ESPRESSO Input
```bash
uv run python -m hubbardml.cli.predict u_model/ scf.out -t qe_simple
# Output: HUBBARD (ortho-atomic)
#         U Fe-3d 5.23
#         U Ni-3d 6.41
```

### JSON Format
```bash
uv run python -m hubbardml.cli.predict u_model/ scf.out -t json
# Output: {"model_info": {...}, "predictions": {"u_parameters": [...]}}
```

### Legacy Format
```bash
uv run python -m hubbardml.cli.predict u_model/ scf.out -t qe_legacy
# Output: &system
#           lda_plus_u = .true.
#           Hubbard_V(1,1,1) = 5.23
#         /
```

## Development

### Testing
```bash
# Run full test suite
uv run pytest

# Test specific components
uv run pytest tests/test_models.py
uv run pytest tests/test_predict_pipeline.py

# End-to-end pipeline test
uv run python tests/test_end_to_end.py
```

### Code Quality
```bash
# Format code
uv run black .

# Security scan
uv run bandit -r hubbardml/

# Type checking
uv run mypy hubbardml/
```

## Troubleshooting

### Common Issues

**File format issues**: The training CLI supports both Arrow and HDF5 formats with automatic conversion. Use `.arrow` or `.h5` files in your config.

**CUDA warnings**: GPU memory warnings are normal for large models. The system falls back to CPU automatically.

**Template not found**: Ensure you're using supported template names: `qe_simple`, `qe_legacy`, `qe_v_format`, `json`.

### Getting Help

- **Example workflow**: See `example/README.md` for a complete working example
- **API documentation**: Check docstrings in `hubbardml/` modules
- **Issues**: Report bugs and request features on GitHub

## Contributing

HubbardML uses modern Python development practices:

- **uv** for dependency management and virtual environments
- **black** for code formatting (line length: 100)
- **pytest** for testing with coverage reporting
- **pre-commit** hooks for code quality

## Citation

If you use HubbardML in your research, please cite:

```bibtex
@article{uhrin2025hubbardml,
  title={Machine learning Hubbard parameters with equivariant neural networks},
  author={Uhrin, Martin and Zadoks, Andrin and Binci, Luca and Marzari, Nicola and Timrov, Iurii},
  journal={Nature Computational Materials},
  volume={11},
  number={1},
  pages={19},
  year={2025},
  doi={10.1038/s41524-024-01501-5}
}
```

## License

HubbardML is licensed under the GNU Lesser General Public License v3.0 (LGPL-3.0).

Copyright (c) 2022, Martin Uhrin.

See the [LICENSE](LICENSE) file for full details.
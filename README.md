# HubbardML


[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Commitizen friendly](https://img.shields.io/badge/commitizen-friendly-brightgreen.svg)](http://commitizen.github.io/cz-cli/)
[![security: bandit](https://img.shields.io/badge/security-bandit-yellow.svg)](https://github.com/PyCQA/bandit)
[![DeepSource](https://static.deepsource.io/deepsource-badge-light-mini.svg)](https://deepsource.io/gh/muhrin/hubbardml/?ref=repository-badge)

## About

This repository contains source code for our machine learning model for predicting self-consistent Hubbard parameters, as presented in this work:

Uhrin, M., Zadoks, A., Binci, L., Marzari, N., & Timrov, I. (2025). Machine learning Hubbard parameters with equivariant neural networks. Npj Computational Materials, 11(1), 19. [https://doi.org/10.1038/s41524-024-01501-5i](https://www.nature.com/articles/s41524-024-01501-5)

## Quick Start

### Option 1: CLI Tools (Recommended)

Use the modern CLI interface for training and prediction:

```bash
# Install with development dependencies
uv sync --extra dev

# Test the complete pipeline
uv run python tests/test_end_to_end.py

# Train a model (uses PyTorch Lightning-style config)
uv run python -m hubbardml.cli.train data/dataset_test.h5 example_train_config.yaml model_output/

# Make predictions
uv run python -m hubbardml.cli.predict model_output/ scf.out

# Inspect datasets
uv run python -m hubbardml.cli.inspect data/dataset_test.h5

# Convert data formats
uv run python -m hubbardml.cli.convert data/dataset.arrow data/dataset_test.h5
```

### Option 2: Example Scripts

For a simple getting-started example, see the `example/` directory:

```bash
# Download dataset, inspect data, train model, make predictions
cd example/
# Follow the step-by-step README.md
```

This provides a self-contained workflow with data download instructions and working scripts.

## Advanced Usage

The experiments carried out in this work can be found in the `experiments/` folder along with all the notebooks to generate the plots.

For advanced usage with Hydra configuration management:

    python run.py experiment=predict_hp model=u

Additional experiments can be found in the `experiments/experiment/` folder.

## Installation

HubbardML uses [uv](https://docs.astral.sh/uv/) for fast dependency management and reproducible environments.

### Install with uv (recommended)

```bash
# Clone the repository
git clone <repository-url>
cd hubbardml

# Development setup - installs all dependencies including dev tools
uv sync --extra dev

# Production setup - runtime dependencies only
uv sync
```

### Alternative installation methods

```bash
# Using uv with pip-style commands
uv venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
uv pip install -e .[dev]

# Traditional pip installation (fallback)
pip install -e .
pip install -e .[dev]
```

### Development Setup

```bash
# Full development environment
git clone <repository-url>
cd hubbardml
uv sync --extra dev

# Install pre-commit hooks (optional)
uv run pre-commit install
```

## Dataset

The datasets used in this work are available for download:

### Published Dataset
- **Main dataset**: `data_uv_2024_1_25.arrow` (69MB, 645k entries)
- **Test dataset**: `dataset.arrow` (153KB, 164 entries)  
- **Download from**: [Zenodo record](https://zenodo.org/record/XXXXXX) (see paper for exact link)

### Dataset Contents
- **Elements**: Fe, Ni, Mn, Co, Ti, O, S transition metal compounds
- **U parameters**: On-site Coulomb repulsion energies (3-8 eV)
- **V parameters**: Inter-site Coulomb interaction energies (0.1-2 eV)
- **Features**: Electronic occupation matrices from DFT+U calculations
- **Target**: Self-consistent vs linear-response Hubbard parameters

Use the inspection utility to analyze your data:
```bash
uv run python -m hubbardml inspect data/data_uv_2024_1_25.arrow
```

## Usage

### Running Experiments

From the `experiments/` directory:

# HubbardML


[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Commitizen friendly](https://img.shields.io/badge/commitizen-friendly-brightgreen.svg)](http://commitizen.github.io/cz-cli/)
[![security: bandit](https://img.shields.io/badge/security-bandit-yellow.svg)](https://github.com/PyCQA/bandit)
[![DeepSource](https://static.deepsource.io/deepsource-badge-light-mini.svg)](https://deepsource.io/gh/muhrin/hubbardml/?ref=repository-badge)

## About

This repository contains source code for our machine learning model for predicting self-consistent Hubbard parameters, as presented in this work:

Uhrin, M., Zadoks, A., Binci, L., Marzari, N., & Timrov, I. (2025). Machine learning Hubbard parameters with equivariant neural networks. Npj Computational Materials, 11(1), 19. [https://doi.org/10.1038/s41524-024-01501-5i](https://www.nature.com/articles/s41524-024-01501-5)

The experiments carried out in this work can be found in the `experiments/` folder along with all the notebooks to generate the plots.

As an example, from experiments you can use:

    python run.py experiment=predict_hp model=u

to run an experiment that trains a model to predict Hubbard U values from a linear-response dataset.

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

## Usage

### Running Experiments

From the `experiments/` directory:

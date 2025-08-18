# HubbardML Example: Training a U Parameter Model

This example demonstrates how to train a machine learning model to predict self-consistent Hubbard U parameters from linear-response calculations.

## Quick Start

### 1. Download the Dataset

Download the published dataset from the paper's repository:

```bash
# Download the dataset (69MB, 645k entries) from Materials Cloud
curl -L -o data/data_uv_2024_1_25.arrow \
  "https://archive.materialscloud.org/record/file?filename=data_uv_2024_1_25.arrow&record_id=2024.160"
```

**Dataset source**: [Materials Cloud Archive Record 2024.160](https://archive.materialscloud.org/record/2024.160)

### 2. Install HubbardML

```bash
# Clone and install
git clone https://github.com/muhrin/hubbardml
cd hubbardml
uv sync --extra dev
```

### 3. Inspect Your Data First

Always inspect your dataset before training to understand what you're working with:

```bash
# Analyze dataset structure and contents
uv run python -m hubbardml inspect data/dataset.arrow

# Example output:
# ============================================================
# DATASET ANALYSIS REPORT  
# ============================================================
# 
# File Information:
#    Format: Arrow
#    Path: dataset.arrow
#    Size: 0.15 MB
#    Rows: 164
#    Columns: 8
# 
# Parameter Distribution:
#    Total entries: 164
#    U parameters: 164 (100.0%)
#    V parameters: 0 (0.0%)
# 
# Occupation Matrix Analysis:
#    atom_1_occs_1:
#      - 5x5 (d-orbitals): 164 entries
#    atom_1_occs_2:
#      - 5x5 (d-orbitals): 164 entries
```

This tells you:
- **Dataset size**: 164 training examples
- **Parameter type**: Only U parameters (no V parameters)  
- **Elements**: d-block transition metals (Fe, Ni, Mn)
- **Features**: Occupation matrices for d-orbitals (5×5)

### 4. Train the Model

```bash
# Train U parameter model (using small dataset for quick test)
uv run python example/train.py --config example/config.yaml

# Or train with the full dataset
uv run python example/train.py --config example/config.yaml \
  --data data/data_uv_2024_1_25.arrow
```

Training output shows:
- Loss progression over epochs
- Validation performance  
- Final model metrics

### 5. Make Predictions (Inference)

After training, use your model to predict U parameters for new calculations:

```bash
# Predict U parameters from DFT calculation output
uv run python example/predict.py \
  --model trained_model.pth \
  --config training_config.json \
  --input new_scf_calculation.out

# Batch prediction on multiple files
uv run python example/predict.py \
  --model trained_model.pth \
  --config training_config.json \
  --input calculations/*.out \
  --output predictions.csv
```

Inference requires:
- **Model weights**: `trained_model.pth` (from training)
- **Model config**: `training_config.json` (automatically saved during training)
- **Input data**: SCF output files or occupation matrices

## Understanding the Workflow

1. **Inspect** → Understand your data (parameter types, elements, matrix sizes)
2. **Train** → Learn the mapping from occupation matrices to U/V parameters  
3. **Predict** → Apply the trained model to new DFT calculations

## Dataset Information

The datasets contain electronic structure calculations for transition metal compounds:

- **Elements**: Fe, Ni, Mn, Co, Ti, O, S
- **U parameters**: On-site Coulomb repulsion (d-orbitals)  
- **V parameters**: Inter-site Coulomb interactions
- **Features**: Occupation matrices from DFT+U calculations
- **Target**: Self-consistent U/V values (vs linear-response initial guess)

## File Structure After Training

```
example/
├── README.md                 # This file
├── config.yaml              # Training configuration  
├── train.py                  # Training script
├── predict.py                # Inference script
├── trained_model.pth         # Saved model weights (after training)
├── training_config.json      # Model config (after training)  
└── training.log              # Training logs (after training)
```

## Troubleshooting

**Import errors?** Make sure you've installed with `uv sync --extra dev`

**Missing data files?** Check the download URLs and file paths in config

**Training too slow?** Reduce batch size or max epochs in `config.yaml`

**Out of memory?** Use the small `dataset.arrow` instead of the full dataset

**Prediction errors?** Ensure your input files match the expected format (SCF output)

## Advanced Usage

For more sophisticated experiments, see the `experiments/` directory which uses Hydra for configuration management:

```bash
# Official experiment runner (requires more setup)
uv run python experiments/run.py experiment=predict_hp model=u
```

This example provides a simpler, self-contained alternative for getting started.
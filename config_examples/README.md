# HubbardML Configuration Examples

This directory contains example configuration files for training HubbardML models.

## Configuration Files

- **`u_model.yaml`** - Configuration for U model training (on-site Hubbard parameters)
- **`v_model.yaml`** - Configuration for V model training (inter-site Hubbard parameters)

## Usage

```bash
# Train U model
hubbard train config_examples/u_model.yaml models/u_model/

# Train V model  
hubbard train config_examples/v_model.yaml models/v_model/
```

## Configuration Structure

### Data Section
- **`file`** - Path to HDF5 data file (relative to config file)
- **`model_type`** - Either "u" or "v"

### Graph Section  
- **`species`** - List of chemical elements to include in model
  - U models: `["Fe", "Ni", "Mn"]` (d-block elements)
  - V models: `["Fe", "Ni", "Mn", "O"]` (includes p-block for interactions)
- **`with_param_in`** - Whether to use input parameters as features (usually false)

### Model Section
- **`feature_irreps`** - Spherical harmonic representations
  - U models: `"4x0e + 4x1e + 4x2e"` (up to l=2 for efficiency)
  - V models: `"4x0e + 4x1e + 4x2e + 4x3e"` (up to l=3 for inter-site complexity)
- **`hidden_layers`** - Number of hidden layers (typically 2)
- **`irrep_normalization`** - Normalization method ("component")

### Training Section
- **`learning_rate`** - Adam optimizer learning rate (0.001)
- **`batch_size`** - Batch size (16 for U, 32 for V models)
- **`max_epochs`** - Maximum training epochs (1000)
- **`val_split`** - Validation split fraction (0.2 = 20%)
- **`patience`** - Early stopping patience (100 epochs)
- **`log_every_n_epochs`** - Progress logging frequency (50)

## Species Selection

**U Models (On-site):**
- Include only d-block elements where U parameters are significant
- Exclude Co/Ti due to limited training data
- Common: `["Fe", "Ni", "Mn"]`

**V Models (Inter-site):**  
- Include both d-block and p-block elements for p-d interactions
- Must include oxygen for most compounds
- Common: `["Fe", "Ni", "Mn", "O"]`

## Model Output Ranges

- **U parameters:** 3-8 eV (on-site Coulomb repulsion)
- **V parameters:** 0.1-2 eV (inter-site Coulomb interactions)
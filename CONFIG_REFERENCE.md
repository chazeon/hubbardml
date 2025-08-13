# HubbardML Configuration Reference

This document describes the configuration parameters for training HubbardML models.

## Configuration Structure

```yaml
graph:
  # Graph configuration
  
model:
  # Model architecture configuration
  
training:
  # Training hyperparameters
```

## Parameter Reference

### Graph Configuration (`graph`)

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `with_param_in` | bool | Include input parameter features | `false` |
| `species` | list | Atomic species (auto-detected from data) | `[]` |

### Model Configuration (`model`)

| Parameter | Type | Description | Example |
|-----------|------|-------------|---------|
| `feature_irreps` | str | Irreducible representations for features | `"4x0e + 4x1e + 4x2e"` |
| `hidden_layers` | int | Number of hidden layers | `2` |
| `irrep_normalization` | str | Normalization method | `"component"` |

### Training Configuration (`training`)

#### Core Training Parameters

| Parameter | Type | Description | Default |
|-----------|------|-------------|---------|
| `learning_rate` | float | Optimizer learning rate | `0.001` |
| `batch_size` | int | Training batch size | `32` |
| `max_epochs` | int | Maximum training epochs | `1000` |

#### Validation Parameters

| Parameter | Type | Description | Default | Old Name (deprecated) |
|-----------|------|-------------|---------|---------------------|
| `val_split` | float | Validation data fraction (0.0-1.0) | `0.2` | `validate_frac` |

#### Early Stopping

| Parameter | Type | Description | Default | Old Name (deprecated) |
|-----------|------|-------------|---------|---------------------|
| `patience` | int | Epochs without improvement before stopping | `100` | `overfitting_patience` |

#### Logging

| Parameter | Type | Description | Default | Old Name (deprecated) |
|-----------|------|-------------|---------|---------------------|
| `log_every_n_epochs` | int | Print training status every N epochs | `50` | `status_freq` |

## Example Configurations

### Minimal Configuration
```yaml
graph:
  with_param_in: false

model:
  feature_irreps: "2x0e + 1x1e + 1x2e"
  hidden_layers: 1

training:
  learning_rate: 0.01
  batch_size: 16
  max_epochs: 50
  val_split: 0.2
  patience: 20
  log_every_n_epochs: 10
```

### Production Configuration
```yaml
graph:
  with_param_in: false

model:
  feature_irreps: "4x0e + 4x1e + 4x2e + 4x3e"
  hidden_layers: 2
  irrep_normalization: "component"

training:
  learning_rate: 0.001
  batch_size: 64
  max_epochs: 2000
  val_split: 0.2
  patience: 200
  log_every_n_epochs: 50
```

## Parameter Migration Guide

The following parameters have been renamed to follow PyTorch Lightning conventions:

| Old Name | New Name | Status |
|----------|----------|---------|
| `validate_frac` | `val_split` | ⚠️ Deprecated (backward compatible) |
| `overfitting_patience` | `patience` | ⚠️ Deprecated (backward compatible) |
| `status_freq` | `log_every_n_epochs` | ⚠️ Deprecated (backward compatible) |

### Updating Old Configs

**Old format:**
```yaml
training:
  validate_frac: 0.2
  overfitting_patience: 100
  status_freq: 50
```

**New format:**
```yaml
training:
  val_split: 0.2
  patience: 100
  log_every_n_epochs: 50
```

The old parameter names are still supported for backward compatibility but will be removed in future versions.

## Validation

### Validation Split
- `val_split: 0.2` means 20% of data is used for validation
- Data is split randomly using PyTorch's `random_split()`
- Validation runs after every training epoch
- Best model (lowest validation loss) is automatically saved

### Early Stopping
- Training stops if validation loss doesn't improve for `patience` epochs
- Prevents overfitting and saves computation time
- Final model is the one with the best validation performance

## Advanced Configuration

### GPU/Device Settings
The training automatically detects and uses CUDA when available:
```bash
# Automatic device detection
uv run python -m hubbardml.cli.train data.h5 config.yaml model_output/

# Output shows:
# Using device: cuda
# Model parameters: 1,234
```

### Memory Optimization
For large datasets or limited GPU memory:
```yaml
training:
  batch_size: 16        # Reduce batch size
  max_epochs: 500       # Fewer epochs
```

### Fast Training (for testing)
```yaml
training:
  learning_rate: 0.01   # Higher learning rate
  batch_size: 16        # Smaller batches
  max_epochs: 50        # Few epochs
  patience: 20          # Early stopping
  log_every_n_epochs: 5 # Frequent logging
```
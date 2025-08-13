# HubbardML Tests

This directory contains tests for the HubbardML package.

## Running Tests

### Unit Tests
Run the standard test suite:
```bash
# From project root
uv run pytest tests/

# Run specific test file
uv run pytest tests/test_models.py

# Run with coverage
uv run pytest --cov=hubbardml tests/
```

### End-to-End Pipeline Test
Test the complete training and prediction pipeline:
```bash
# From project root
uv run python tests/test_end_to_end.py
```

This test will:
1. Train a small model on test data (`data/dataset_test.h5`)
2. Save the trained model and config
3. Load the model and run predictions
4. Verify the complete pipeline works

**Requirements:**
- Test data: `data/dataset_test.h5` 
- Config file: `tests/test_train_config.yaml`

### Manual CLI Testing
Test individual CLI commands:

```bash
# Test training
uv run python -m hubbardml.cli.train data/dataset_test.h5 tests/test_train_config.yaml /tmp/test_model

# Test prediction (will use dummy data for testing)
uv run python -m hubbardml.cli.predict /tmp/test_model /tmp/fake_scf.out

# Test data inspection  
uv run python -m hubbardml.cli.inspect data/dataset_test.h5

# Test data conversion
uv run python -m hubbardml.cli.convert data/dataset.arrow /tmp/test.h5
```

## Test Configuration

### `test_train_config.yaml`
Minimal training configuration for fast testing:
- Small model: `"2x0e + 1x1e + 1x2e"` features
- Few epochs: 50 max epochs
- Small batches: 16 batch size
- Fast convergence: Higher learning rate

See [CONFIG_REFERENCE.md](../CONFIG_REFERENCE.md) for complete configuration documentation.

### Test Data
- **`data/dataset_test.h5`**: 164 U parameter samples for Fe
- **`data/dataset.arrow`**: Small Arrow dataset for conversion testing

## Expected Output

Successful end-to-end test shows:
```
🎉 End-to-end pipeline is working!
✨ You can now:
   1. Train models with: uv run python -m hubbardml train data.h5 config.yaml model_output/
   2. Make predictions with: uv run python -m hubbardml predict model_output/ scf.out
```

## Troubleshooting

### GPU Issues
If you see CUDA device mismatch errors:
- Check `torch.cuda.is_available()`
- The code automatically detects and uses GPU when available
- Falls back to CPU if GPU unavailable

### Missing Data
If test data is missing:
- Check that `data/dataset_test.h5` exists
- Run data conversion if needed: `uv run python -m hubbardml.cli.convert data/dataset.arrow data/dataset_test.h5`

### Import Errors
If modules are not found:
- Run from project root directory
- Ensure environment is activated: `uv sync --extra dev`
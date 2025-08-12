# User Scripts

This directory contains custom scripts created by the user to work around limitations in the main HubbardML codebase.

## Files

- `train.py` - Custom training script (410 lines) that bypasses the main training infrastructure
- `make_dataset.py` - Custom dataset creation from QE outputs  
- `predict.py` - Custom inference script with manual config loading (located in `hubbardml/predict.py`)
- `parse_pw3.py` - Custom parser for SCF occupation extraction (located in `hubbardml/parse_pw3.py`)

## Context

These scripts were created because the main HubbardML training/inference pipeline was difficult to use:

1. **Training**: Existing `training.py` was too complex, required custom `train.py`
2. **Inference**: No easy model reconstruction, required custom `predict.py` with manual JSON config
3. **Data**: No QE parsing pipeline, required custom data processing scripts

## Future

These scripts demonstrate the usability pain points that are being addressed in the main refactoring plan (see `PLAN.md`). Once the main pipeline is improved, these custom scripts should no longer be necessary.
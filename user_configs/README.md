# User Configuration Files

This directory contains user-created configuration files.

## Files

- `config_user.yaml` - User's custom config file (has syntax issues that need fixing)

## Issues with config_user.yaml

The user's config file has several problems that prevent it from working:

1. **Invalid JSON/YAML syntax** - Uses `#` comments in JSON format
2. **Missing imports** - References `keys.*` constants without importing 
3. **Commented out critical settings** - `required_feature_cols` and `target_col` disabled
4. **Wrong file references** - Points to non-existent files

This config needs to be fixed or replaced with a working version based on the official `experiments/config.yaml`.
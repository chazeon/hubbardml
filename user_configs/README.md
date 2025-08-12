# User Configuration Files → Fixed and Moved

**Note**: This directory contained broken configuration files that have now been fixed and moved to the example.

## ✅ Fixed and Moved to Example

The broken config file has been **fixed and moved** to `example/config.yaml`:

### Issues That Were Resolved:
1. **✅ JSON+Comments → Proper YAML**: Converted to valid YAML syntax
2. **✅ Undefined constants**: Replaced `keys.PARAM_OUT` with `"param_out"` strings  
3. **✅ Re-enabled features**: Uncommented `required_feature_cols` and `target_col`
4. **✅ Fixed file references**: Removed non-existent `dataset_tiger.arrow`

### New Location:
- **Working config**: `example/config.yaml` 
- **Documentation**: `example/README.md`
- **Complete workflow**: Download data → inspect → train → predict

The config file is now properly formatted and functional for training models.

## Files Remaining:
- `config_user.yaml` - Original broken file (kept for reference)
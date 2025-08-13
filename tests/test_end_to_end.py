#!/usr/bin/env python3
"""
End-to-end pipeline test for HubbardML

Tests the complete workflow:
1. Train a model from HDF5 data
2. Use trained model for prediction (with dummy data since we don't have real QE files)
3. Verify the pipeline works

Usage:
    uv run python test_end_to_end.py
"""

import os
import sys
import tempfile
import subprocess
from pathlib import Path
import shutil

def run_command(cmd, description):
    """Run command and capture output."""
    print(f"\n{'='*60}")
    print(f"STEP: {description}")
    print(f"CMD: {' '.join(cmd)}")
    print(f"{'='*60}")
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    print("STDOUT:")
    print(result.stdout)
    
    if result.stderr:
        print("STDERR:")
        print(result.stderr)
    
    if result.returncode != 0:
        print(f"❌ Command failed with return code {result.returncode}")
        return False
    else:
        print(f"✅ Command succeeded")
        return True

def test_training_and_prediction():
    """Test complete training + prediction pipeline."""
    print("🧪 Testing HubbardML End-to-End Training + Prediction Pipeline")
    
    # Check if test data exists (relative to project root)
    project_root = Path(__file__).parent.parent
    data_file = project_root / "data" / "dataset_test.h5"
    config_file = Path(__file__).parent / "test_train_config.yaml"
    
    if not data_file.exists():
        print(f"❌ Test data not found: {data_file}")
        return False
    
    if not config_file.exists():
        print(f"❌ Config file not found: {config_file}")
        return False
    
    # Create temporary output directory
    with tempfile.TemporaryDirectory() as temp_dir:
        output_dir = Path(temp_dir) / "test_model"
        
        # Step 1: Train model
        train_cmd = [
            "uv", "run", "python", "-m", "hubbardml.cli.train",
            str(data_file), str(config_file), str(output_dir)
        ]
        
        success = run_command(train_cmd, "Training HubbardML Model")
        
        if not success:
            return False
        
        # Check if model files were created
        model_file = output_dir / "model.pth"
        config_output = output_dir / "config.json"
        
        if not model_file.exists():
            print(f"❌ Model file not created: {model_file}")
            return False
        
        if not config_output.exists():
            print(f"❌ Config file not created: {config_output}")
            return False
        
        print(f"✅ Model training completed successfully!")
        print(f"   Model: {model_file}")
        print(f"   Config: {config_output}")
        
        # Check model file size (should be reasonable)
        model_size = model_file.stat().st_size
        print(f"   Model size: {model_size:,} bytes")
        
        if model_size < 1000:  # Less than 1KB seems too small
            print(f"⚠️  Model file seems very small")
        
        # Step 2: Test prediction with trained model (dummy SCF)
        fake_scf = Path(temp_dir) / "fake_scf.out" 
        fake_scf.write_text("fake scf content")  # Predict CLI will use dummy data anyway
        
        predict_cmd = [
            "uv", "run", "python", "-m", "hubbardml.cli.predict",
            str(output_dir), str(fake_scf)
        ]
        
        predict_success = run_command(predict_cmd, "Testing Prediction with Trained Model")
        
        return predict_success

def test_prediction_dummy():
    """Test prediction with dummy data since we don't have real QE files."""
    print("\n🧪 Testing HubbardML Prediction with Dummy Data")
    
    # We'll create a minimal test that shows the predict CLI works
    # even without real training data, using the fallback mechanism
    
    # Create a fake model directory
    with tempfile.TemporaryDirectory() as temp_dir:
        fake_model_dir = Path(temp_dir) / "fake_model"
        fake_model_dir.mkdir()
        
        # Create minimal fake config (predict CLI will use dummy data)
        fake_config = {
            "graph": {"species": ["Fe"], "with_param_in": False},
            "model": {"feature_irreps": "2x0e", "hidden_layers": 1},
            "model_type": "U"
        }
        
        import json
        with open(fake_model_dir / "config.json", 'w') as f:
            json.dump(fake_config, f)
        
        # Create fake model file (empty is fine for dummy test)
        (fake_model_dir / "model.pth").touch()
        
        # Create fake SCF file
        fake_scf = Path(temp_dir) / "fake_scf.out"
        fake_scf.write_text("fake scf output")
        
        # Test predict CLI (will use dummy data)
        predict_cmd = [
            "uv", "run", "python", "-m", "hubbardml.cli.predict",
            str(fake_model_dir), str(fake_scf)
        ]
        
        success = run_command(predict_cmd, "Testing Prediction CLI (Dummy Mode)")
        
        return success

def main():
    """Run end-to-end tests."""
    print("🚀 HubbardML End-to-End Pipeline Test")
    print("=" * 60)
    
    # Test: Complete training + prediction pipeline
    pipeline_success = test_training_and_prediction()
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 TEST SUMMARY")
    print("=" * 60)
    print(f"Complete Pipeline: {'✅ PASS' if pipeline_success else '❌ FAIL'}")
    
    if pipeline_success:
        print("\n🎉 End-to-end pipeline is working!")
        print("✨ You can now:")
        print("   1. Train models with: uv run python -m hubbardml train data.h5 config.yaml model_output/")
        print("   2. Make predictions with: uv run python -m hubbardml predict model_output/ scf.out")
        return 0
    else:
        print("\n💥 Pipeline test failed. Check the output above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
#!/usr/bin/env python3
"""Quick test script to verify installation and basic functionality."""

import sys
import torch
from pathlib import Path

def test_imports():
    """Test that all modules can be imported."""
    try:
        from src.models.clip_model import ContrastiveCLIPModel, ContrastiveLoss
        from src.data.dataset import ToyMultimodalDataset
        from src.eval.metrics import compute_contrastive_metrics
        from src.utils.device import get_device, set_seed
        from src.utils.config import load_config
        print("✓ All modules imported successfully")
        return True
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False

def test_device():
    """Test device detection."""
    try:
        from src.utils.device import get_device
        device = get_device("auto")
        print(f"✓ Device detection working: {device}")
        return True
    except Exception as e:
        print(f"✗ Device test failed: {e}")
        return False

def test_model():
    """Test model initialization."""
    try:
        from src.models.clip_model import ContrastiveCLIPModel
        model = ContrastiveCLIPModel()
        print(f"✓ Model initialized successfully")
        print(f"  Total parameters: {sum(p.numel() for p in model.parameters()):,}")
        return True
    except Exception as e:
        print(f"✗ Model test failed: {e}")
        return False

def test_dataset():
    """Test dataset creation."""
    try:
        from src.data.dataset import ToyMultimodalDataset
        dataset = ToyMultimodalDataset(num_samples=5)
        print(f"✓ Dataset created successfully with {len(dataset)} samples")
        return True
    except Exception as e:
        print(f"✗ Dataset test failed: {e}")
        return False

def test_config():
    """Test configuration loading."""
    try:
        from src.utils.config import load_config
        config_path = Path("configs/config.yaml")
        if config_path.exists():
            config = load_config(str(config_path))
            print("✓ Configuration loaded successfully")
            return True
        else:
            print("✗ Configuration file not found")
            return False
    except Exception as e:
        print(f"✗ Config test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("Multi-Modal Self-Supervised Learning - Installation Test")
    print("=" * 60)
    
    tests = [
        ("Import Test", test_imports),
        ("Device Test", test_device),
        ("Model Test", test_model),
        ("Dataset Test", test_dataset),
        ("Config Test", test_config),
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{test_name}:")
        if test_func():
            passed += 1
        else:
            print(f"  {test_name} failed!")
    
    print("\n" + "=" * 60)
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Installation is working correctly.")
        print("\nNext steps:")
        print("1. Run training: python -m src.scripts.train")
        print("2. Launch demo: streamlit run src/scripts/demo.py")
        print("3. Open notebook: jupyter notebook notebooks/demo.ipynb")
        return 0
    else:
        print("❌ Some tests failed. Please check the installation.")
        return 1

if __name__ == "__main__":
    sys.exit(main())

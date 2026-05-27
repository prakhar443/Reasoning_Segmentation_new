#!/usr/bin/env python3
"""
quickstart.py — Quick start script to verify installation and run a minimal test.

Usage:
    python quickstart.py
"""

import sys
import os
from pathlib import Path


def check_dependencies():
    """Verify all required packages are installed."""
    print("🔍 Checking dependencies...")
    deps = [
        "torch",
        "torchvision",
        "transformers",
        "peft",
        "PIL",
        "numpy",
        "cv2",
        "pandas",
    ]

    missing = []
    for dep in deps:
        try:
            __import__(dep)
            print(f"  ✓ {dep}")
        except ImportError:
            print(f"  ✗ {dep} — MISSING")
            missing.append(dep)

    if missing:
        print(f"\n❌ Missing packages: {', '.join(missing)}")
        print("Install with: pip install -r requirements.txt")
        return False
    print("✅ All dependencies OK\n")
    return True


def check_directories():
    """Verify directory structure."""
    print("📁 Checking directory structure...")
    required_dirs = [
        "data",
        "models",
        "utils",
        "checkpoints",
    ]

    for d in required_dirs:
        p = Path(d)
        if p.exists():
            print(f"  ✓ {d}/")
        else:
            print(f"  ✗ {d}/ — MISSING")
            p.mkdir(parents=True, exist_ok=True)
            print(f"    → Created")

    print("✅ Directory structure OK\n")


def check_config():
    """Verify config.py is loadable."""
    print("⚙️  Checking configuration...")
    try:
        from config import cfg, ICE_CLASSES
        print(f"  ✓ config.py loaded")
        print(f"  ✓ Ice classes: {len(ICE_CLASSES)} types")
        print(f"    {', '.join(ICE_CLASSES[:3])}...")
        print("✅ Configuration OK\n")
        return True
    except Exception as e:
        print(f"  ✗ config.py error: {e}")
        return False


def check_models():
    """Verify model modules are importable."""
    print("🧠 Checking model modules...")
    modules = [
        ("data.preprocessing", "SARPreprocessor"),
        ("data.dataset", "SeaIceDataset"),
        ("models.visual_encoder", "CLIPSAREncoder"),
        ("models.depth_encoder", "build_depth_encoder"),
        ("models.reasoning_module", "build_reasoning_module"),
        ("models.prompt_generator", "GeometricPromptGenerator"),
        ("models.sam_module", "SAMModule"),
        ("models.ice_classifier", "IceTypeClassifier"),
        ("models.temporal_consistency", "TemporalConsistencyModule"),
        ("models.pipeline", "SeaIceSegmentationPipeline"),
        ("utils.losses", "SeaIceLoss"),
        ("utils.metrics", "MetricAccumulator"),
    ]

    for mod_name, cls_name in modules:
        try:
            mod = __import__(mod_name, fromlist=[cls_name])
            getattr(mod, cls_name)
            print(f"  ✓ {mod_name}.{cls_name}")
        except Exception as e:
            print(f"  ✗ {mod_name}.{cls_name} — {e}")
            return False

    print("✅ All modules OK\n")
    return True


def check_sam_checkpoint():
    """Check if SAM checkpoint exists."""
    print("🎯 Checking SAM checkpoint...")
    sam_path = Path("checkpoints/sam_vit_h_4b8939.pth")
    if sam_path.exists():
        size_gb = sam_path.stat().st_size / (1024 ** 3)
        print(f"  ✓ {sam_path} ({size_gb:.1f} GB)")
        print("✅ SAM checkpoint OK\n")
        return True
    else:
        print(f"  ℹ️  SAM checkpoint not found at {sam_path}")
        print("     Download with:")
        print("     wget https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth -O checkpoints/sam_vit_h_4b8939.pth")
        print("     (optional; can use lightweight decoder without it)\n")
        return True  # Not critical


def check_dataset():
    """Check if dataset directory exists."""
    print("📊 Checking dataset...")
    dataset_path = Path("dataset")
    if dataset_path.exists():
        subdirs = list(dataset_path.glob("*/"))
        if subdirs:
            print(f"  ✓ {dataset_path} exists")
            print(f"    Subdirectories: {len(subdirs)}")
            for d in subdirs[:3]:
                print(f"      - {d.name}/")
            print("✅ Dataset OK\n")
            return True
        else:
            print(f"  ℹ️  {dataset_path} exists but is empty")
            print("     Please populate with ice class subdirectories\n")
            return False
    else:
        print(f"  ℹ️  {dataset_path} not found")
        print("     Create with structure:")
        print("       dataset/")
        print("         ├── first_year_ice/images/")
        print("         ├── first_year_ice/masks/")
        print("         ├── young_ice/...")
        print("         └── ...\n")
        return False


def test_preprocessing():
    """Quick preprocessing test."""
    print("🧪 Testing SAR preprocessing...")
    try:
        import numpy as np
        from config import cfg
        from data.preprocessing import SARPreprocessor

        preprocessor = SARPreprocessor(cfg.data)

        # Dummy SAR image
        dummy_sar = np.random.rand(512, 512).astype(np.float32)
        tensor = preprocessor(dummy_sar)

        assert tensor.shape == (3, 512, 512), f"Wrong shape: {tensor.shape}"
        assert tensor.dtype == torch.float32
        assert tensor.min() >= -1.0 and tensor.max() <= 3.0

        print(f"  ✓ Preprocessing works")
        print(f"    Input (512, 512) → Output {tuple(tensor.shape)}")
        print("✅ Preprocessing OK\n")
        return True
    except Exception as e:
        print(f"  ✗ Preprocessing failed: {e}\n")
        return False


def test_dataset_loading():
    """Quick dataset test."""
    print("📖 Testing dataset loading...")
    try:
        from config import cfg
        from data.dataset import SeaIceDataset

        try:
            dataset = SeaIceDataset(
                cfg.data.data_root,
                split="train",
                data_cfg=cfg.data,
                use_augmentation=False,
            )
            print(f"  ✓ Dataset loaded")
            print(f"    Size: {len(dataset)} samples")
            if len(dataset) > 0:
                sample = dataset[0]
                print(f"    Sample keys: {list(sample.keys())}")
                print(f"    Image shape: {sample['image'].shape}")
                print(f"    Mask shape: {sample['mask'].shape}")
            print("✅ Dataset loading OK\n")
            return True
        except FileNotFoundError:
            print(f"  ℹ️  Dataset not found (expected on first run)\n")
            return True
    except Exception as e:
        print(f"  ✗ Dataset loading failed: {e}\n")
        return False


def main():
    print("\n" + "=" * 60)
    print("🚀 Sea Ice Segmentation Pipeline — Quickstart")
    print("=" * 60 + "\n")

    checks = [
        ("Dependencies", check_dependencies),
        ("Directories", check_directories),
        ("Configuration", check_config),
        ("Model modules", check_models),
        ("SAM checkpoint", check_sam_checkpoint),
        ("Dataset", check_dataset),
        ("SAR preprocessing", test_preprocessing),
        ("Dataset loading", test_dataset_loading),
    ]

    results = {}
    for name, check_fn in checks:
        try:
            results[name] = check_fn()
        except Exception as e:
            print(f"❌ {name} check failed with exception: {e}\n")
            results[name] = False

    # Summary
    print("=" * 60)
    print("📋 Quickstart Summary")
    print("=" * 60)
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    print(f"Passed: {passed}/{total} checks\n")

    for name, result in results.items():
        symbol = "✅" if result else "⚠️ "
        print(f"  {symbol} {name}")

    print()

    if passed == total:
        print("🎉 All checks passed! You're ready to train.")
        print("\nNext steps:")
        print("  1. Prepare your dataset (see README.md)")
        print("  2. python train.py --data_root dataset --output_dir outputs/exp1")
        print("  3. python evaluate.py --checkpoint outputs/exp1/best_model.pth")
        print()
        return 0
    else:
        print("⚠️  Some checks failed. Please review the output above.")
        print()
        return 1


if __name__ == "__main__":
    import torch
    sys.exit(main())

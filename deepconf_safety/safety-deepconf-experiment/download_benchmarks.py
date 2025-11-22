"""
Download Safety Benchmarks for DeepConf Experiments

Downloads ToxicChat, HarmBench, and WildGuard datasets from HuggingFace.
"""

import os
from pathlib import Path


def download_datasets(output_dir: str = "data"):
    """
    Download all safety benchmarks.

    Args:
        output_dir: Directory to save datasets
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("DOWNLOADING SAFETY BENCHMARKS")
    print("=" * 60)

    try:
        from datasets import load_dataset
    except ImportError:
        print("\n❌ ERROR: 'datasets' library not found!")
        print("Please install it: pip install datasets")
        return

    # 1. ToxicChat
    print("\n[1/3] Downloading ToxicChat...")
    try:
        toxicchat = load_dataset("lmsys/toxic-chat", "toxicchat0124")
        toxicchat_dir = output_path / "toxicchat"
        toxicchat_dir.mkdir(exist_ok=True)

        # Save to disk
        for split in toxicchat.keys():
            toxicchat[split].to_json(toxicchat_dir / f"{split}.jsonl")

        print(f"✓ ToxicChat saved to {toxicchat_dir}")
        print(f"  Splits: {list(toxicchat.keys())}")
        print(f"  Total examples: {sum(len(toxicchat[s]) for s in toxicchat.keys())}")
    except Exception as e:
        print(f"✗ Failed to download ToxicChat: {e}")

    # 2. HarmBench
    print("\n[2/3] Downloading HarmBench...")
    try:
        harmbench = load_dataset("harmbench/harmbench_behaviors_text_all")
        harmbench_dir = output_path / "harmbench"
        harmbench_dir.mkdir(exist_ok=True)

        # Save to disk
        for split in harmbench.keys():
            harmbench[split].to_json(harmbench_dir / f"{split}.jsonl")

        print(f"✓ HarmBench saved to {harmbench_dir}")
        print(f"  Splits: {list(harmbench.keys())}")
        print(f"  Total examples: {sum(len(harmbench[s]) for s in harmbench.keys())}")
    except Exception as e:
        print(f"✗ Failed to download HarmBench: {e}")
        print("  Note: HarmBench may require authentication or have access restrictions")

    # 3. WildGuardMix
    print("\n[3/3] Downloading WildGuardMix...")
    try:
        # Note: Use WildGuardMix which has explicit refusal labels
        wildguardmix = load_dataset("allenai/wildguardmix")
        wildguard_dir = output_path / "wildguardmix"
        wildguard_dir.mkdir(exist_ok=True)

        # Save to disk
        for split in wildguardmix.keys():
            wildguardmix[split].to_json(wildguard_dir / f"{split}.jsonl")

        print(f"✓ WildGuardMix saved to {wildguard_dir}")
        print(f"  Splits: {list(wildguardmix.keys())}")
        print(f"  Total examples: {sum(len(wildguardmix[s]) for s in wildguardmix.keys())}")
        if 'test' in wildguardmix:
            print(f"  Test set size: {len(wildguardmix['test'])} examples")
            print("  ✅ Test set includes explicit 'response_refusal_label' field!")
    except Exception as e:
        print(f"✗ Failed to download WildGuardMix: {e}")
        print("  Note: Dataset name may have changed. Check HuggingFace for latest info.")

    print("\n" + "=" * 60)
    print("DOWNLOAD COMPLETE")
    print("=" * 60)
    print(f"\nDatasets saved to: {output_path.absolute()}")
    print("\nYou can now run experiments with:")
    print("  python run_experiment.py --benchmark toxicchat")
    print("  python run_experiment.py --benchmark harmbench")
    print("  python run_experiment.py --benchmark wildguardmix  # Has explicit refusal labels!")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Download safety benchmarks")
    parser.add_argument(
        "--output-dir",
        default="data",
        help="Directory to save datasets (default: data/)"
    )

    args = parser.parse_args()

    download_datasets(args.output_dir)

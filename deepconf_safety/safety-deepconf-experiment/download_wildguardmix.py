"""
Download WildGuardMix with authentication token
"""
import os
from pathlib import Path
from datasets import load_dataset

# Set token from environment variable or .env file
token = os.getenv("HF_TOKEN", "YOUR_HF_TOKEN_HERE")

print("=" * 60)
print("DOWNLOADING WILDGUARDMIX (AUTHENTICATED)")
print("=" * 60)

output_path = Path("data")
output_path.mkdir(parents=True, exist_ok=True)

try:
    print("\nDownloading WildGuardMix from allenai/wildguardmix...")
    print("(This is a gated dataset - using provided authentication token)")

    # Load with token
    wildguardmix = load_dataset("allenai/wildguardmix", token=token)

    wildguard_dir = output_path / "wildguardmix"
    wildguard_dir.mkdir(exist_ok=True)

    # Save to disk
    for split in wildguardmix.keys():
        print(f"\nSaving {split} split...")
        wildguardmix[split].to_json(wildguard_dir / f"{split}.jsonl")

    print(f"\n✓ WildGuardMix saved to {wildguard_dir}")
    print(f"  Splits: {list(wildguardmix.keys())}")
    print(f"  Total examples: {sum(len(wildguardmix[s]) for s in wildguardmix.keys())}")

    if 'test' in wildguardmix:
        print(f"\n✅ Test set: {len(wildguardmix['test'])} examples")
        print("  Fields include explicit 'response_refusal_label'!")

        # Show sample to verify
        sample = wildguardmix['test'][0]
        print(f"\n📋 Sample fields:")
        for key in ['prompt', 'response', 'prompt_harm_label', 'response_harm_label', 'response_refusal_label', 'subcategory']:
            if key in sample:
                value = sample[key]
                if isinstance(value, str) and len(value) > 100:
                    value = value[:100] + "..."
                print(f"  - {key}: {value}")

    print("\n" + "=" * 60)
    print("DOWNLOAD COMPLETE!")
    print("=" * 60)
    print("\nYou can now run:")
    print("  python3 run_experiment.py --benchmark wildguardmix --model Qwen/Qwen3-0.6B")

except Exception as e:
    print(f"\n❌ Error downloading WildGuardMix: {e}")
    import traceback
    traceback.print_exc()

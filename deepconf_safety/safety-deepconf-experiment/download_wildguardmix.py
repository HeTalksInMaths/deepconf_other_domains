"""
Download WildGuardMix with authentication token.
Supports both wildguardtest (1,725 eval rows) and wildguardtrain configs.
"""

import os
from pathlib import Path
from datasets import load_dataset

token = os.getenv("HF_TOKEN")
if not token:
    raise SystemExit("HF_TOKEN not found in environment. Set it (e.g., source .env) before running.")

print("=" * 60)
print("DOWNLOADING WILDGUARDMIX (AUTHENTICATED)")
print("=" * 60)

output_path = Path("data") / "wildguardmix"
output_path.mkdir(parents=True, exist_ok=True)

configs = [
    ("wildguardtest", "test"),
    ("wildguardtrain", "train"),
]

total_examples = 0
downloaded = []

for config_name, expected_split in configs:
    try:
        print(f"\nDownloading config '{config_name}' (expects '{expected_split}' split)...")
        ds = load_dataset("allenai/wildguardmix", config_name, token=token)
    except Exception as e:  # pylint: disable=broad-except
        print(f"  ❌ Failed to download config '{config_name}': {e}")
        continue

    for split_name, dataset in ds.items():
        out_file = output_path / f"{split_name}.jsonl"
        print(f"  Saving split '{split_name}' → {out_file}")
        dataset.to_json(out_file)
        split_size = len(dataset)
        total_examples += split_size
        downloaded.append((config_name, split_name, split_size))

print("\n" + "=" * 60)
if downloaded:
    print("DOWNLOAD COMPLETE!")
    for cfg, split, size in downloaded:
        print(f"  - {cfg}/{split}: {size} examples")
    print(f"\nTotal examples saved: {total_examples}")
    sample_file = output_path / "test.jsonl"
    if sample_file.exists():
        from itertools import islice
        import json

        with sample_file.open() as f:
            first = next(islice(f, 0, 1))
            sample = json.loads(first)
        print("\nSample fields from test split:")
        for key in ['prompt', 'response', 'prompt_harm_label', 'response_harm_label', 'response_refusal_label', 'subcategory']:
            if key in sample:
                value = sample[key]
                if isinstance(value, str) and len(value) > 80:
                    value = value[:80] + "..."
                print(f"  - {key}: {value}")
    print("\nYou can now run:")
    print("  python3 run_experiment.py --benchmark wildguardmix --model Qwen/Qwen3-0.6B")
else:
    print("❌ No WildGuardMix splits were downloaded. Check HF_TOKEN access.")

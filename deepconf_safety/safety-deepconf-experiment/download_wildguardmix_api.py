"""
Download WildGuardMix via HuggingFace API
Using the API access provided
"""
import json
import requests
from pathlib import Path

# Your HuggingFace token
HF_TOKEN = "YOUR_HF_TOKEN_HERE"

print("=" * 60)
print("DOWNLOADING WILDGUARDMIX VIA API")
print("=" * 60)

output_path = Path("data/wildguardmix")
output_path.mkdir(parents=True, exist_ok=True)

# API endpoint template
API_URL = "https://datasets-server.huggingface.co/rows"

# Configurations to download
configs_to_try = [
    ("wildguardtest", "test"),
    ("wildguardtrain", "train"),
]

headers = {
    "Authorization": f"Bearer {HF_TOKEN}"
}

all_data = {}

for config, split in configs_to_try:
    print(f"\n[{config}/{split}] Downloading...")

    all_rows = []
    offset = 0
    batch_size = 100

    while True:
        # Build API request
        params = {
            "dataset": "allenai/wildguardmix",
            "config": config,
            "split": split,
            "offset": offset,
            "length": batch_size
        }

        try:
            response = requests.get(API_URL, headers=headers, params=params, timeout=30)
            response.raise_for_status()

            data = response.json()

            # Check if we got rows
            if "rows" not in data or len(data["rows"]) == 0:
                print(f"  No more data at offset {offset}")
                break

            rows = data["rows"]
            print(f"  Downloaded {len(rows)} rows (offset: {offset})")

            # Extract the actual row data
            for row in rows:
                if "row" in row:
                    all_rows.append(row["row"])

            # Check if we got fewer rows than requested (last batch)
            if len(rows) < batch_size:
                print(f"  Reached end of dataset")
                break

            offset += batch_size

        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 404:
                print(f"  ⚠️  Config '{config}' or split '{split}' not found")
                break
            else:
                print(f"  ❌ HTTP Error: {e}")
                break
        except Exception as e:
            print(f"  ❌ Error: {e}")
            break

    if all_rows:
        all_data[f"{config}_{split}"] = all_rows
        print(f"  ✅ Total: {len(all_rows)} examples")

        # Save to file
        output_file = output_path / f"{split}.jsonl"
        with open(output_file, 'w') as f:
            for row in all_rows:
                f.write(json.dumps(row) + '\n')
        print(f"  💾 Saved to: {output_file}")

print("\n" + "=" * 60)
print("DOWNLOAD SUMMARY")
print("=" * 60)

if all_data:
    total_examples = sum(len(rows) for rows in all_data.values())
    print(f"\n✅ Successfully downloaded {len(all_data)} dataset(s)")
    print(f"📊 Total examples: {total_examples}")

    for key, rows in all_data.items():
        print(f"\n  [{key}]: {len(rows)} examples")
        if rows:
            # Show sample fields
            sample = rows[0]
            print(f"    Fields: {list(sample.keys())[:15]}")

            # Check for refusal label
            if 'response_refusal_label' in sample:
                print(f"    ✅ HAS 'response_refusal_label' (gold standard!)")
            if 'prompt_harm_label' in sample:
                print(f"    ✅ HAS 'prompt_harm_label' (ground truth!)")

    print(f"\n💾 Saved to: {output_path}/")
    print("\n✅ You can now run:")
    print("  python3 run_experiment.py --benchmark wildguardmix --model Qwen/Qwen3-0.6B")
else:
    print("\n❌ No data downloaded. Check API access and dataset configuration.")

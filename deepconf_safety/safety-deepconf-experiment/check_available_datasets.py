"""
Check which AllenAI wildguard datasets are available
"""
from datasets import load_dataset

token = "YOUR_HF_TOKEN_HERE"

datasets_to_try = [
    "allenai/wildguardmix",
    "allenai/wildjailbreak",
    "allenai/wildguardtest",
    "allenai/wildguard",
]

print("Checking AllenAI safety datasets...")
print("=" * 60)

for dataset_name in datasets_to_try:
    print(f"\n[{dataset_name}]")
    try:
        ds = load_dataset(dataset_name, token=token, streaming=True)
        print(f"  ✅ ACCESSIBLE")
        # Try to get first example to check fields
        try:
            splits = list(ds.keys())
            print(f"  Splits: {splits}")
            if splits:
                first_split = splits[0]
                first_example = next(iter(ds[first_split].take(1)))
                print(f"  Fields: {list(first_example.keys())[:10]}")
        except Exception as e:
            print(f"  (Could not inspect: {e})")
    except Exception as e:
        error_msg = str(e)
        if "gated" in error_msg.lower():
            print(f"  ⚠️  GATED - Requires access request")
        elif "not found" in error_msg.lower():
            print(f"  ❌ NOT FOUND")
        else:
            print(f"  ❌ ERROR: {error_msg[:100]}")

print("\n" + "=" * 60)

#!/usr/bin/env python3
"""
Test WildGuardMix dataset access.

Run this to verify:
1. Your HuggingFace token is valid
2. You have access to WildGuardMix dataset
3. The dataset has the expected refusal labels
"""

from huggingface_hub import HfApi
from datasets import load_dataset
import sys

# Your HuggingFace token - load from environment or .env file
import os
HF_TOKEN = os.getenv("HF_TOKEN", "YOUR_HF_TOKEN_HERE")

def test_token():
    """Test if HuggingFace token is valid."""
    print("="*60)
    print("TEST 1: Verifying HuggingFace Token")
    print("="*60)

    api = HfApi()
    try:
        user = api.whoami(token=HF_TOKEN)
        print(f"✅ Token valid!")
        print(f"   User: {user['name']}")
        print(f"   Email: {user.get('email', 'N/A')}")
        return True
    except Exception as e:
        print(f"❌ Token error: {e}")
        print("\n📝 Action needed:")
        print("   1. Go to: https://huggingface.co/settings/tokens")
        print("   2. Generate new token with 'read' permissions")
        print("   3. Update HF_TOKEN in this script")
        return False

def test_wildguardmix_access():
    """Test if you have access to WildGuardMix dataset."""
    print("\n" + "="*60)
    print("TEST 2: Checking WildGuardMix Access")
    print("="*60)

    try:
        # Try to load just 5 examples from test set
        dataset = load_dataset(
            "allenai/wildguardmix",
            "wildguardtest",
            split="test[:5]",
            token=HF_TOKEN
        )

        print(f"✅ WildGuardMix ACCESS GRANTED!")
        print(f"\n📊 Dataset Info:")
        print(f"   Examples loaded: {len(dataset)}")
        print(f"   Available keys: {list(dataset[0].keys())}")

        # Check for critical fields
        sample = dataset[0]

        print(f"\n🔍 Sample Data:")
        print(f"   Prompt: {sample.get('prompt', 'N/A')[:80]}...")

        # Check for refusal labels (key for our research!)
        if 'response_refusal_label' in sample:
            print(f"   ✅ Has refusal labels: '{sample['response_refusal_label']}'")
            print(f"      (Options: 'refusal' or 'compliance')")
        else:
            print(f"   ⚠️  No 'response_refusal_label' field found!")
            print(f"      Available fields: {list(sample.keys())}")

        # Check for harm labels
        if 'prompt_harm_label' in sample:
            print(f"   ✅ Has harm labels: '{sample['prompt_harm_label']}'")

        return True

    except Exception as e:
        print(f"❌ WildGuardMix access denied!")
        print(f"   Error: {e}")
        print("\n📝 Action needed:")
        print("   1. Visit: https://huggingface.co/datasets/allenai/wildguardmix")
        print("   2. Click 'Request Access' button")
        print("   3. Accept AI2 Responsible Use Guidelines")
        print("   4. Fill out all form fields")
        print("   5. Wait ~1 minute for automatic approval")
        print("   6. Re-run this script")
        return False

def main():
    """Run all tests."""
    print("\n🔬 WildGuardMix Access Test")
    print("="*60 + "\n")

    # Test 1: Token validity
    if not test_token():
        sys.exit(1)

    # Test 2: Dataset access
    if not test_wildguardmix_access():
        sys.exit(1)

    # All tests passed!
    print("\n" + "="*60)
    print("✅ ALL TESTS PASSED!")
    print("="*60)
    print("\n🎉 You're ready to use WildGuardMix!")
    print("\nNext steps:")
    print("  1. Run full experiment:")
    print("     python run_experiment.py \\")
    print("         --model Qwen/Qwen3-0.6B \\")
    print("         --benchmark wildguardmix \\")
    print("         --num-instances 1000 \\")
    print("         --output results/wildguardmix_1000")
    print("\n  2. Validate refusal patterns:")
    print("     - Compare your 40 patterns against gold labels")
    print("     - Expected: P=75-90%, R=70-85%, F1=72-87%")

    sys.exit(0)

if __name__ == "__main__":
    main()

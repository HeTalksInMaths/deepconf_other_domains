"""
Debug script to test logprobs extraction and confidence calculation.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent / 'deepconf_adapter'))

from src.qwen3_adapter import Qwen3SafetyAdapter
from confidence_utils import compute_trace_confidence, TraceWithLogprobs
import numpy as np

def test_logprobs_extraction():
    """Test if logprobs are being extracted correctly."""
    print("=" * 60)
    print("Testing Logprobs Extraction")
    print("=" * 60)

    # Initialize model
    print("\nInitializing Qwen3-0.6B...")
    model = Qwen3SafetyAdapter("Qwen/Qwen3-0.6B", device="auto", torch_dtype="auto")

    # Test with a simple prompt
    test_prompt = "Hello, how are you?"

    print(f"\nTest prompt: {test_prompt}")
    print("\nGenerating response...")

    text, logprobs, tokens = model.generate_with_logprobs(
        test_prompt,
        max_new_tokens=50,
        temperature=0.6,
        top_p=0.95
    )

    print(f"\n--- Results ---")
    print(f"Generated text: {text[:200]}...")
    print(f"Number of tokens: {len(tokens)}")
    print(f"Number of logprobs: {len(logprobs)}")
    print(f"Logprobs list length matches tokens: {len(logprobs) == len(tokens)}")

    if logprobs:
        print(f"\nLogprobs statistics:")
        print(f"  Min: {np.min(logprobs):.4f}")
        print(f"  Max: {np.max(logprobs):.4f}")
        print(f"  Mean: {np.mean(logprobs):.4f}")
        print(f"  First 5 logprobs: {logprobs[:5]}")

        # Compute confidence
        trace = TraceWithLogprobs(text=text, logprobs=logprobs, tokens=tokens)
        confidence = compute_trace_confidence(trace)
        print(f"\nConfidence (neg avg logprob): {confidence:.4f}")
        print(f"  Formula: -np.mean(logprobs) = -{np.mean(logprobs):.4f} = {confidence:.4f}")

        # Check for any inf/nan values
        has_inf = any(np.isinf(lp) for lp in logprobs)
        has_nan = any(np.isnan(lp) for lp in logprobs)
        print(f"\nContains infinity values: {has_inf}")
        print(f"Contains NaN values: {has_nan}")

        if has_inf or has_nan:
            print("\n⚠️  WARNING: Found inf/nan values in logprobs!")
            for i, lp in enumerate(logprobs):
                if np.isinf(lp) or np.isnan(lp):
                    print(f"  Position {i}: {lp} (token: {tokens[i]})")
    else:
        print("\n⚠️  ERROR: Logprobs list is EMPTY!")
        print("This will cause confidence to be 0.0")

    print("\n" + "=" * 60)
    print("Testing Batch Generation")
    print("=" * 60)

    # Test batch generation
    test_prompts = [
        "Hello, how are you?",
        "What is 2+2?",
        "Tell me a joke."
    ]

    print(f"\nGenerating {len(test_prompts)} prompts in batch...")
    batch_results = model.batch_generate(
        test_prompts,
        max_new_tokens=50,
        temperature=0.6,
        top_p=0.95
    )

    print(f"\nBatch results:")
    for i, (text, logprobs, tokens) in enumerate(batch_results):
        print(f"\n  Prompt {i+1}: {test_prompts[i]}")
        print(f"    Tokens: {len(tokens)}, Logprobs: {len(logprobs)}")
        if logprobs:
            confidence = -np.mean(logprobs)
            print(f"    Confidence: {confidence:.4f}")
            print(f"    Has inf/nan: {any(np.isinf(lp) or np.isnan(lp) for lp in logprobs)}")
        else:
            print(f"    ⚠️  EMPTY LOGPROBS!")

    print("\n" + "=" * 60)
    return logprobs

if __name__ == "__main__":
    try:
        logprobs = test_logprobs_extraction()
        print("\n✅ Debug test completed!")
    except Exception as e:
        print(f"\n❌ Error during debug test: {e}")
        import traceback
        traceback.print_exc()

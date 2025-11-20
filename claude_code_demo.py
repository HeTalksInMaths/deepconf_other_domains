"""
Lightweight DeepConf Demo for Claude Code Web
Tests confidence framework WITHOUT requiring models

This demo simulates the DeepConf workflow:
1. Generate multiple "traces" (responses)
2. Compute confidence for each
3. Use early stopping when confidence is high
4. Analyze results

Run with: python claude_code_demo.py
"""

import numpy as np
import sys
sys.path.append('./deepconf_adapter')

from confidence_utils import (
    TraceWithLogprobs,
    compute_trace_confidence,
    should_generate_more_traces
)

def simulate_model_response(prompt, quality='good', length=15):
    """
    Simulate a model response with different quality levels.

    In real usage, this would be replaced with actual model.generate()

    Args:
        prompt: Input prompt (unused in simulation)
        quality: 'good' (high conf), 'uncertain' (low conf), or 'mixed'
        length: Number of tokens to simulate

    Returns:
        TraceWithLogprobs object
    """
    if quality == 'good':
        # High confidence: logprobs closer to 0 (more certain)
        logprobs = (np.random.randn(length) * 0.3 - 0.5).tolist()
    elif quality == 'uncertain':
        # Low confidence: very negative logprobs (uncertain)
        logprobs = (np.random.randn(length) * 1.0 - 2.5).tolist()
    else:  # mixed
        # Medium confidence
        logprobs = (np.random.randn(length) * 0.7 - 1.2).tolist()

    # Create fake tokens and text
    tokens = [f"word_{i}" for i in range(length)]
    text = ' '.join(tokens)

    return TraceWithLogprobs(
        text=text,
        tokens=tokens,
        logprobs=logprobs,
        metadata={'quality': quality}
    )


def run_deepconf_simulation(prompt, true_quality, max_traces=10, min_traces=3, conf_threshold=0.7):
    """
    Simulate DeepConf's adaptive trace generation.

    Args:
        prompt: Question/prompt
        true_quality: Simulated response quality
        max_traces: Maximum traces to generate
        min_traces: Minimum before early stopping
        conf_threshold: Confidence threshold for early stopping

    Returns:
        List of traces, number of traces used
    """
    traces = []

    for i in range(max_traces):
        # Generate trace
        trace = simulate_model_response(prompt, quality=true_quality)
        traces.append(trace)

        # Compute confidence
        conf = compute_trace_confidence(trace)

        # Check early stopping
        should_continue, info = should_generate_more_traces(
            traces,
            min_traces=min_traces,
            confidence_threshold=conf_threshold
        )

        print(f"  Trace {i+1:2d}: confidence = {conf:.3f}", end='')

        if not should_continue:
            print(f" → ✅ STOP (threshold reached)")
            break
        else:
            reason = info.get('reason', 'continue')
            print(f" → {reason}")

    return traces, info


def main():
    print("=" * 70)
    print(" " * 15 + "DeepConf Confidence Demo")
    print(" " * 10 + "(No models required - pure simulation)")
    print("=" * 70)

    # Test scenarios with different quality levels
    scenarios = [
        {
            'name': 'Easy Question (High Confidence)',
            'prompt': 'What is 2 + 2?',
            'quality': 'good',
            'expected': 'Should stop early (few traces needed)'
        },
        {
            'name': 'Ambiguous Question (Low Confidence)',
            'prompt': 'Is this statement ethical?',
            'quality': 'uncertain',
            'expected': 'Should use many traces (never reaches threshold)'
        },
        {
            'name': 'Medium Question (Mixed Confidence)',
            'prompt': 'Explain quantum computing.',
            'quality': 'mixed',
            'expected': 'Should use moderate number of traces'
        }
    ]

    results = []

    for i, scenario in enumerate(scenarios, 1):
        print(f"\n{'─' * 70}")
        print(f"Scenario {i}: {scenario['name']}")
        print(f"{'─' * 70}")
        print(f"📝 Prompt: {scenario['prompt']}")
        print(f"🎯 Expected: {scenario['expected']}")
        print(f"\nGenerating traces...")

        traces, info = run_deepconf_simulation(
            prompt=scenario['prompt'],
            true_quality=scenario['quality'],
            max_traces=10,
            min_traces=3,
            conf_threshold=0.7
        )

        print(f"\n📊 Results:")
        print(f"  • Total traces generated: {len(traces)}")
        print(f"  • Final confidence: {info['current_confidence']:.3f}")
        print(f"  • Stopping reason: {info.get('reason', 'max traces reached')}")

        # Calculate efficiency
        efficiency = (10 - len(traces)) / 10 * 100
        print(f"  • Efficiency gain: {efficiency:.0f}% (saved {10 - len(traces)} traces)")

        results.append({
            'scenario': scenario['name'],
            'traces_used': len(traces),
            'confidence': info['current_confidence'],
            'efficiency': efficiency
        })

    # Summary
    print(f"\n{'═' * 70}")
    print(" " * 25 + "SUMMARY")
    print(f"{'═' * 70}")
    print(f"\n{'Scenario':<40} {'Traces':<10} {'Confidence':<12} {'Efficiency'}")
    print(f"{'-' * 70}")

    for r in results:
        print(f"{r['scenario']:<40} {r['traces_used']:<10} {r['confidence']:<12.3f} {r['efficiency']:.0f}%")

    # Key insights
    print(f"\n{'─' * 70}")
    print("🔍 Key Insights:")
    print("  1. High-confidence responses → Early stopping (efficient)")
    print("  2. Low-confidence responses → Use all traces (thorough)")
    print("  3. DeepConf adapts compute to question difficulty")
    print(f"{'─' * 70}")

    print(f"\n{'=' * 70}")
    print("✓ Demo Complete!")
    print("\nThis demonstrates DeepConf's core idea WITHOUT requiring models.")
    print("Replace simulate_model_response() with real model.generate()")
    print("to run actual experiments.")
    print(f"{'=' * 70}\n")


if __name__ == '__main__':
    try:
        main()
    except ImportError as e:
        print("\n❌ Import Error:")
        print(f"   {e}")
        print("\n💡 Solution:")
        print("   pip install numpy --break-system-packages")
        print("   Ensure deepconf_adapter/confidence_utils.py exists")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()

# Running DeepConf Implementations in Claude Code Web

This guide explains what can run in **Claude Code Web** vs what needs **local execution**.

## Quick Answer

| Component | Claude Code Web | Local | Notes |
|-----------|----------------|-------|-------|
| **Data exploration** | ✅ Yes | ✅ Yes | Load and explore datasets |
| **Mock experiments** | ✅ Yes | ✅ Yes | Test with synthetic data |
| **Small models (<1B)** | ⚠️ Slow | ✅ Yes | CPU-only, will be slow |
| **Large models (>1B)** | ❌ No | ✅ Yes | Needs GPU, too much memory |
| **Full experiments** | ❌ No | ✅ Yes | Long-running, resource intensive |

## What You CAN Run in Claude Code Web

### 1. ✅ Data Exploration

```python
# Load and explore XSum dataset
from datasets import load_dataset

xsum = load_dataset('xsum', split='test[:10]')
for item in xsum:
    print(f"Document: {item['document'][:100]}...")
    print(f"Summary: {item['summary']}")
```

### 2. ✅ Mock Experiments (Recommended!)

**Quick Demo Without Models** - Test the framework logic:

```python
# demo_confidence_only.py
import numpy as np
import sys
sys.path.append('./deepconf_adapter')

from confidence_utils import TraceWithLogprobs, compute_multi_trace_confidence

# Simulate 5 model responses with different confidences
traces = []
for i in range(5):
    # Mock a response: 20 tokens with random logprobs
    tokens = [f"token_{j}" for j in range(20)]
    logprobs = (np.random.randn(20) * 0.5 - 1.0).tolist()

    traces.append(TraceWithLogprobs(
        text=' '.join(tokens),
        tokens=tokens,
        logprobs=logprobs
    ))

# Compute aggregate confidence
avg_conf = compute_multi_trace_confidence(traces, aggregation='mean')
print(f"Average confidence across 5 traces: {avg_conf:.3f}")

# Analyze variation
confidences = [compute_trace_confidence(t) for t in traces]
print(f"Confidence range: {min(confidences):.3f} - {max(confidences):.3f}")
```

**Run in Claude Code Web:**
```bash
pip install numpy --break-system-packages
python demo_confidence_only.py
```

### 3. ✅ Synthetic Benchmarks (No Model Required)

The safety implementation includes a synthetic benchmark:

```python
# test_safety_synthetic.py
import sys
sys.path.append('./deepconf_safety/safety-deepconf-experiment/src')

from benchmark_loaders import SyntheticSafetyBenchmark

# Generate 10 test instances
benchmark = SyntheticSafetyBenchmark(num_instances=10)
instances = benchmark.load()

for inst in instances[:3]:
    print(f"\nPrompt: {inst.prompt}")
    print(f"Label: {inst.ground_truth_label}")
    print(f"Category: {inst.category}")
```

**Run in Claude Code Web:**
```bash
python test_safety_synthetic.py
```

### 4. ✅ Analysis Scripts

Analyze pre-computed results:

```python
# analyze_results.py
import json
import numpy as np

# Load experiment results (if you have them)
with open('results/experiment_results.json') as f:
    results = json.load(f)

# Compute statistics
confidences = [r['avg_confidence'] for r in results]
accuracies = [r['is_correct'] for r in results]

print(f"Mean confidence: {np.mean(confidences):.3f}")
print(f"Accuracy: {np.mean(accuracies):.3f}")

# Stratified analysis
low_conf = [a for c, a in zip(confidences, accuracies) if c < 0.5]
high_conf = [a for c, a in zip(confidences, accuracies) if c >= 0.5]

print(f"Low confidence accuracy: {np.mean(low_conf) if low_conf else 'N/A'}")
print(f"High confidence accuracy: {np.mean(high_conf) if high_conf else 'N/A'}")
```

## What You CANNOT Run in Claude Code Web

### ❌ Large Model Inference

**Why it fails:**
- No GPU access (CPU inference is prohibitively slow)
- Memory limits (~4-8GB typically)
- Models like Qwen3-1.7B+ need 3GB+ just to load

**Example that WON'T work:**
```python
# This will be TOO SLOW or run out of memory
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen3-1.7B")
# ❌ Will take minutes to load, then timeout or OOM
```

### ❌ Full Experiments

**Why it fails:**
- Experiments need 100s-1000s of instances
- Each instance needs multiple traces (5-10x more inference)
- Total runtime: Hours on GPU, days on CPU

**Example:**
```bash
# This needs LOCAL execution
python run_experiment.py \
    --model Qwen/Qwen3-0.6B \
    --num-instances 1000 \
    --compare
# ❌ Will timeout in Claude Code Web
```

## Recommended Workflow

### For Claude Code Web: Framework Testing

1. **Explore the code structure**
   ```bash
   ls -R deepconf_safety/
   ls -R deepconf_hallucination/
   ```

2. **Run confidence utilities demo**
   ```bash
   python demo_confidence_only.py
   ```

3. **Test with synthetic data**
   ```bash
   cd deepconf_safety/safety-deepconf-experiment
   python -c "from src.benchmark_loaders import SyntheticSafetyBenchmark; \
              b = SyntheticSafetyBenchmark(10); print(b.load())"
   ```

4. **Modify and experiment** with the framework logic (no models)

### For Local: Full Experiments

1. **Clone the repository**
   ```bash
   git clone https://github.com/HeTalksInMaths/deepconf_other_domains
   cd deepconf_other_domains
   git checkout claude/deepconf-adapter-implementation-01Ej26SYeEfWTu9weEEYSU2a
   ```

2. **Set up Python environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # or venv\Scripts\activate on Windows
   pip install torch transformers datasets numpy pandas scipy
   ```

3. **Run safety experiments**
   ```bash
   cd deepconf_safety/safety-deepconf-experiment
   python run_experiment.py --model Qwen/Qwen3-0.6B --num-instances 50
   ```

4. **Run hallucination experiments**
   ```bash
   cd deepconf_hallucination/xsum_hallucination_deepconf
   bash setup.sh
   source venv/bin/activate
   python src/run_experiments.py --max_instances 100
   ```

## Lightweight Demo for Claude Code Web

Here's a **complete working example** you CAN run in Claude Code Web:

```python
# claude_code_demo.py
"""
Lightweight DeepConf demo for Claude Code Web
Tests confidence framework WITHOUT requiring models
"""

import numpy as np
import sys
sys.path.append('./deepconf_adapter')

from confidence_utils import TraceWithLogprobs, should_generate_more_traces

def simulate_model_response(prompt, quality='good'):
    """Simulate model response with different quality levels."""
    if quality == 'good':
        # High confidence response
        logprobs = (np.random.randn(15) * 0.3 - 0.5).tolist()
    elif quality == 'uncertain':
        # Low confidence response
        logprobs = (np.random.randn(15) * 1.0 - 2.0).tolist()
    else:
        # Mixed confidence
        logprobs = (np.random.randn(15) * 0.7 - 1.2).tolist()

    tokens = [f"word{i}" for i in range(15)]
    text = ' '.join(tokens)

    return TraceWithLogprobs(text=text, tokens=tokens, logprobs=logprobs)

# Simulate experiment with 3 different prompts
prompts = [
    ("Easy question", 'good'),
    ("Ambiguous question", 'uncertain'),
    ("Medium question", 'mixed')
]

print("=" * 60)
print("DeepConf Confidence-Based Early Stopping Demo")
print("=" * 60)

for prompt, quality in prompts:
    print(f"\n📝 Prompt: {prompt} (simulated {quality} response)")

    traces = []
    for i in range(10):  # Max 10 traces
        trace = simulate_model_response(prompt, quality)
        traces.append(trace)

        # Check if we should stop early
        should_continue, info = should_generate_more_traces(
            traces,
            min_traces=3,
            confidence_threshold=0.7
        )

        print(f"  Trace {i+1}: conf={info['current_confidence']:.3f}", end='')

        if not should_continue:
            print(f" → ✅ STOP (sufficient confidence)")
            break
        else:
            print(f" → Continue...")

    print(f"  📊 Final: {len(traces)} traces, avg conf={info['current_confidence']:.3f}")

print("\n" + "=" * 60)
print("✓ Demo complete! Framework works without models.")
print("=" * 60)
```

**To run:**
```bash
pip install numpy --break-system-packages
python claude_code_demo.py
```

## Summary

**Claude Code Web is great for:**
- 📖 Reading and understanding the code
- 🧪 Testing framework logic without models
- 📊 Analyzing pre-computed results
- 🛠️ Prototyping modifications

**Local execution is needed for:**
- 🤖 Running actual model inference
- 📈 Full-scale experiments (100+ instances)
- 🚀 GPU-accelerated generation
- ⏱️ Long-running evaluations

## Files Ready for Claude Code Web

These work out-of-the-box in Claude Code Web:

```
✅ deepconf_adapter/confidence_utils.py
✅ deepconf_safety/safety-deepconf-experiment/src/benchmark_loaders.py (synthetic mode)
✅ deepconf_hallucination/xsum_hallucination_deepconf/src/xsum_loader.py (data loading only)
✅ Any analysis scripts
```

These need local execution:

```
❌ Any script importing transformers.AutoModel*
❌ run_experiment.py with real models
❌ Full evaluation pipelines
```

## Getting Help

- **Code structure questions**: Ask in Claude Code Web
- **Framework logic**: Test with mock data in Claude Code Web
- **Running experiments**: Clone to local machine
- **Results analysis**: Can do in Claude Code Web with pre-computed results

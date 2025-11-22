# Safety DeepConf Experiment with Qwen3

Complete instantiation of the **DeepConf Domain Adapter** skill for safety/refusal benchmarks.

## What This Is

This is a **working implementation** that adapts DeepConf's confidence-based reasoning from math problems to safety benchmarks. It tests the hypothesis:

> **Low-confidence non-refusals are riskier than high-confidence non-refusals**

## Key Features

✅ **Model-agnostic framework** - Works with any model providing logprobs  
✅ **Qwen3 adapter included** - Ready to use with Qwen3-0.6B/1.7B/4B/8B  
✅ **Multiple benchmark loaders** - ToxicChat, HarmBench, WildGuard, or synthetic  
✅ **Confidence-based early stopping** - Saves compute when confident  
✅ **Refusal detection analysis** - Stratifies by confidence × refusal interaction  
✅ **Complete evaluation pipeline** - From data loading to results analysis  

## Quick Start

### 1. Install Dependencies

```bash
pip install torch transformers numpy
```

### 2. Run Synthetic Experiment (No Download Needed)

```bash
python run_experiment.py --model Qwen/Qwen3-0.6B --num-instances 20
```

This will:
- Load Qwen3-0.6B model (~1.2 GB)
- Create 20 synthetic safety instances
- Run DeepConf with 3-10 traces per instance
- Analyze confidence × refusal patterns
- Save results to `results/safety_experiment/`

### 3. Compare Baseline vs DeepConf

```bash
python run_experiment.py --compare --num-instances 20
```

This runs three experiments:
- **Baseline**: Single trace (n=1)
- **Fixed**: Five traces (n=5)
- **DeepConf**: Adaptive 3-10 traces with early stopping

## File Structure

```
safety-deepconf-experiment/
├── run_experiment.py          # Main entry point
├── src/
│   ├── safety_deepconf.py     # Core DeepConf framework
│   ├── benchmark_loaders.py   # Load safety benchmarks
│   └── qwen3_adapter.py       # Qwen3 model integration
├── config/                    # Experiment configurations
├── notebooks/                 # Analysis notebooks (optional)
└── results/                   # Experiment outputs
```

## Using Real Benchmarks

### ToxicChat

```bash
# Download from https://huggingface.co/datasets/lmsys/toxic-chat
# Then run:
python run_experiment.py \
    --benchmark toxicchat \
    --model Qwen/Qwen3-0.6B
```

### HarmBench

```bash
# Download from https://huggingface.co/datasets/harmbench/harmbench
# Then run:
python run_experiment.py \
    --benchmark harmbench \
    --model Qwen/Qwen3-0.6B
```

### WildGuard

```bash
# Download from https://huggingface.co/datasets/allenai/wildguardtest
# Then run:
python run_experiment.py \
    --benchmark wildguard \
    --model Qwen/Qwen3-0.6B
```

## Understanding the Output

### Confidence Categories

The framework categorizes each prediction:

1. **Confident Refusal** (high conf + refused)
   - Model is certain it should refuse
   - ✅ This is appropriate boundary setting

2. **Confident Compliance** (high conf + no refusal)
   - Model is certain the request is safe
   - ✅ Good when actually safe

3. **Uncertain Refusal** (low conf + refused)
   - Model refuses but isn't confident
   - ⚠️ May be over-cautious

4. **Uncertain Compliance** (low conf + no refusal)
   - Model complies but isn't confident
   - 🚨 **HIGHEST RISK** - This is what we flag!

### Key Metrics

```json
{
  "overall": {
    "accuracy": 0.850,
    "avg_traces": 5.2,
    "avg_confidence": 0.682
  },
  "uncertain_compliance": {
    "count": 15,
    "accuracy": 0.733,
    "avg_confidence": 0.245
  },
  "confident_compliance": {
    "count": 25,
    "accuracy": 0.920,
    "avg_confidence": 0.812
  },
  "hypothesis_test": {
    "difference": 0.187,
    "interpretation": "Positive difference supports hypothesis"
  }
}
```

## Adapting to Your Model

Replace `qwen3_adapter.py` with your model:

```python
def your_model_callable(prompt: str, **kwargs):
    # Your model code here
    # Must return: (text, logprobs, tokens)
    
    text = your_model.generate(prompt)
    logprobs = your_model.get_logprobs()
    tokens = your_model.get_tokens()
    
    return text, logprobs, tokens

# Use in experiment
experiment.run_experiment(instances, your_model_callable)
```

## Command Line Options

```bash
python run_experiment.py --help

Options:
  --model           Model name (default: Qwen/Qwen3-0.6B)
  --benchmark       Benchmark (synthetic, toxicchat, harmbench, wildguard)
  --num-instances   Number of instances for synthetic (default: 20)
  --min-traces      Minimum traces before stopping (default: 3)
  --max-traces      Maximum traces to generate (default: 10)
  --no-early-stopping  Disable confidence-based stopping
  --compare         Run baseline vs DeepConf comparison
  --output          Output directory (default: results/safety_experiment)
```

## Examples

### Experiment with Different Trace Budgets

```bash
# Minimal (3-5 traces)
python run_experiment.py --min-traces 3 --max-traces 5

# Generous (5-20 traces)
python run_experiment.py --min-traces 5 --max-traces 20
```

### Use Larger Model

```bash
python run_experiment.py --model Qwen/Qwen3-1.7B
python run_experiment.py --model Qwen/Qwen3-4B
```

### Disable Early Stopping (Fixed Budget)

```bash
python run_experiment.py --min-traces 10 --max-traces 10 --no-early-stopping
```

## Results Analysis

Results are saved in JSON format:

- `predictions.jsonl` - All predictions with traces and confidences
- `analysis.json` - Aggregated metrics and hypothesis test

Load and analyze:

```python
import json

# Load analysis
with open('results/safety_experiment/analysis.json') as f:
    analysis = json.load(f)

# Check hypothesis
if analysis['hypothesis_test']['difference'] > 0:
    print("Low-confidence non-refusals ARE riskier!")
```

## Next Steps

1. **Run on real benchmarks** - Download ToxicChat or HarmBench
2. **Compare model sizes** - Test 0.6B vs 1.7B vs 4B
3. **Tune thresholds** - Experiment with confidence thresholds
4. **Add your model** - Implement adapter for your model
5. **Analyze subgroups** - Stratify by harm category

## Citation

If you use this implementation, cite the original DeepConf paper:

```bibtex
@article{deepconf2024,
  title={DeepConf: Confidence-Based Early Stopping for Math Reasoning},
  author={...},
  journal={...},
  year={2024}
}
```

## License

Apache 2.0 (same as Qwen3 models)

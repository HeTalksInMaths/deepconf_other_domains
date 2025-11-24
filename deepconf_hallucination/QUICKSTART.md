# Quick Start Guide

This is a 5-minute guide to get you started with XSum Hallucination Detection using DeepConf.

## Installation (2 minutes)

```bash
cd xsum_hallucination_deepconf
bash setup.sh
source venv/bin/activate
```

## Your First Experiment (3 minutes)

### Step 1: Create a simple model

```python
# test_model.py
from src.summarization_confidence import TraceWithLogprobs
import numpy as np

def simple_model(article: str, n_traces: int):
    """A simple mock model for testing."""
    traces = []
    for i in range(n_traces):
        # Simple extractive summary: first sentence
        summary = article.split('.')[0] + '.'
        tokens = summary.split()
        # Random logprobs for demo
        logprobs = (np.random.randn(len(tokens)) * 0.3 - 0.5).tolist()
        
        traces.append(TraceWithLogprobs(
            text=summary,
            tokens=tokens,
            logprobs=logprobs
        ))
    return traces
```

### Step 2: Run Experiment A

```python
# run_test.py
from src.xsum_loader import load_xsum_for_deepconf
from src.run_experiments import XSumHallucinationExperiment
from test_model import simple_model

# Load 10 test instances
instances = load_xsum_for_deepconf(split='test', max_instances=10)

# Create experiment
experiment = XSumHallucinationExperiment(
    model_generator=simple_model,
    output_dir='./results/quickstart'
)

# Run token-level analysis
results = experiment.run_experiment_a_token_hallucination(instances)

print(f"\n✓ Completed {len(results)} instances")
print(f"✓ Results saved to ./results/quickstart/")
```

### Step 3: Check Results

```python
import json

# Load results
with open('./results/quickstart/experiment_a_token_hallucination.json') as f:
    results = json.load(f)

# Show first result
print(json.dumps(results[0], indent=2))
```

## Next Steps

### Use a Real Model

Replace `simple_model` with a real summarization model:

**Option 1: HuggingFace BART**
```python
from transformers import pipeline

summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

def bart_model(article: str, n_traces: int):
    # Implement using summarizer
    # See README.md for full example
    pass
```

**Option 2: Your Custom Model**
```python
def my_model(article: str, n_traces: int):
    # Your model here
    # Must return List[TraceWithLogprobs]
    pass
```

### Run All Experiments

```python
# Experiment B: Multi-trace consensus
results_b = experiment.run_experiment_b_multi_trace_consensus(
    instances,
    n_traces_list=[1, 3, 5],
    factual_consistency_fn=compute_alignscore  # Add your evaluator
)

# Experiment C: Hierarchical analysis
results_c = experiment.run_experiment_c_hierarchical_analysis(
    instances,
    factual_consistency_fn=compute_alignscore,
    rouge_fn=compute_rouge
)
```

### Analyze Results

See `README.md` for:
- Analysis scripts
- Visualization examples
- Statistical tests

## Common Issues

**Import Error**: Make sure virtual environment is activated
```bash
source venv/bin/activate
```

**Module Not Found**: Reinstall requirements
```bash
pip install -r requirements.txt
```

**Out of Memory**: Reduce instances
```python
instances = load_xsum_for_deepconf(split='test', max_instances=5)
```

## Documentation

- `INSTANTIATION.md`: Full experimental design
- `README.md`: Comprehensive guide
- `src/`: Documented source code

## Questions?

Check the README.md or open an issue on GitHub!

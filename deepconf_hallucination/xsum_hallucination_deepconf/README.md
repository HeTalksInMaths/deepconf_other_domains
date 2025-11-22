# XSum Hallucination Detection with DeepConf

**Adapting confidence-based reasoning from math problems to summarization hallucination detection**

This is a complete instantiation of the [deepconf-adapter](../deepconf-adapter) skill for detecting and reducing hallucinations in abstractive summarization using XSum dataset.

## Overview

### Key Innovation

DeepConf was originally designed for math problems where:
- Task: Generate solution → Check if correct
- Signal: Token-level confidence from logprobs
- Method: Generate multiple traces, use confidence for early stopping

We adapt this to **open-ended summarization** where:
- Task: Summarize article → Check for hallucinations
- Signal: Same token-level confidence
- Challenge: **No simple "voting" on final answer**

### Our Approach

We test **three adaptation strategies**:

1. **Token-Level Detection** (H1): Use confidence to identify hallucinated spans
2. **Consensus Claims** (H2): Generate N summaries, extract high-confidence consensus claims
3. **Summary Selection** (H4): Among N candidates, select highest-confidence summary

And test the **hierarchical hypothesis** (H3):
```
Token Confidence → Hallucinations → Factual Consistency → Overall Quality
```

## Repository Structure

```
xsum_hallucination_deepconf/
├── INSTANTIATION.md           # Complete experimental design document
├── README.md                  # This file
├── requirements.txt           # Python dependencies
├── setup.sh                   # Quick setup script
├── src/
│   ├── xsum_loader.py        # XSum + hallucination annotations loader
│   ├── summarization_confidence.py  # Domain-specific confidence analyzer
│   ├── run_experiments.py    # Main experiment runner
│   └── model_interface.py    # Model integration (to be implemented)
├── notebooks/
│   ├── 01_data_exploration.ipynb    # Explore XSum dataset
│   ├── 02_token_analysis.ipynb      # Analyze token-level confidence
│   └── 03_results_visualization.ipynb  # Visualize results
└── results/                   # Experiment results (gitignored)
```

## Installation

### Quick Setup

```bash
# Clone and navigate to this directory
cd xsum_hallucination_deepconf

# Run setup script
bash setup.sh
```

### Manual Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Download XSum dataset (happens automatically on first run)
python -c "from datasets import load_dataset; load_dataset('xsum', split='test[:10]')"
```

### Dependencies

Core requirements:
- Python 3.8+
- datasets (HuggingFace)
- numpy
- pandas
- scipy
- matplotlib, seaborn
- tqdm

Model-specific (choose one):
- transformers + torch (for HuggingFace models)
- anthropic (for Claude API)
- openai (for OpenAI API)

## Quick Start

### 1. Basic Token-Level Analysis

```python
from src.xsum_loader import load_xsum_for_deepconf
from src.summarization_confidence import SummarizationConfidenceAnalyzer, TraceWithLogprobs
from src.run_experiments import XSumHallucinationExperiment

# Load data
instances = load_xsum_for_deepconf(split='test', max_instances=100)

# Define model generator
def my_model_generator(article: str, n_traces: int):
    # Replace with your actual model
    # Must return List[TraceWithLogprobs]
    pass

# Run experiment
experiment = XSumHallucinationExperiment(
    model_generator=my_model_generator,
    output_dir='./results'
)

# Experiment A: Token-level hallucination detection
results = experiment.run_experiment_a_token_hallucination(instances)
```

### 2. Multi-Trace Consensus

```python
# Experiment B: Generate multiple summaries, use consensus
results = experiment.run_experiment_b_multi_trace_consensus(
    instances,
    n_traces_list=[1, 3, 5, 10],  # Test different trace counts
    factual_consistency_fn=compute_alignscore  # Your factual checker
)
```

### 3. Full Experimental Suite

```bash
# Run all experiments
python src/run_experiments.py \
    --model_name "your-model-name" \
    --split test \
    --max_instances 1000 \
    --experiments all \
    --output_dir ./results/full_run
```

## Experiments

### Experiment A: Token-Level Hallucination Detection

**Hypothesis**: Low-confidence tokens are more likely to be hallucinated.

**Method**:
1. Generate summary with token logprobs
2. Identify low-confidence spans
3. Compare with Google hallucination annotations
4. Compute precision/recall of low-confidence as hallucination detector

**Success Criteria**: AUC > 0.65

### Experiment B: Multi-Trace Consensus

**Hypothesis**: Multiple summaries with confidence-weighted consensus reduce hallucinations.

**Method**:
1. Generate N summaries per article
2. Extract factual claims from each
3. Cluster similar claims, weight by confidence
4. Compose final summary from high-confidence consensus

**Baselines**:
- Single-shot generation (n=1)
- Random selection from N candidates
- Best-by-confidence selection

**Success Criteria**: +5% AlignScore over single-shot

### Experiment C: Hierarchical Analysis

**Hypothesis**: Confidence → Hallucinations → Factual Consistency → Quality (hierarchical)

**Method**:
1. Generate summaries, measure all four levels
2. Compute correlations between adjacent levels
3. Test mediation with path analysis

**Success Criteria**: Significant mediation effect (p < 0.05)

### Experiment D: Calibration Analysis

**Hypothesis**: Confidence is well-calibrated with factual consistency.

**Method**:
1. Generate many summaries, bin by confidence
2. Measure average factual consistency per bin
3. Plot calibration curve, compute ECE

**Success Criteria**: ECE < 0.15

## Data

### XSum Dataset

- **Source**: [HuggingFace Datasets](https://huggingface.co/datasets/xsum)
- **Size**: ~227k articles (BBC news 2010-2017)
- **Task**: Single-sentence extreme summarization
- **Splits**: train (204k), validation (11k), test (11k)

### Google Hallucination Annotations

- **Source**: [google-research-datasets/xsum_hallucination_annotations](https://github.com/google-research-datasets/xsum_hallucination_annotations)
- **Size**: 500 documents × 5 systems = 2,500 summaries
- **Annotations**: Human-labeled hallucinated spans
- **Types**: Intrinsic (contradicts) vs. Extrinsic (unsupported)
- **Scores**: Faithfulness (0-1), Factuality (binary)

**To use hallucination annotations**:

```bash
# Clone the annotations repo
git clone https://github.com/google-research-datasets/xsum_hallucination_annotations.git

# Load in your code
from src.xsum_loader import load_hallucination_benchmark

instances = load_hallucination_benchmark(
    hallucination_file='xsum_hallucination_annotations/hallucination_annotations.csv',
    scores_file='xsum_hallucination_annotations/faithfulness_scores.csv'
)
```

## Model Integration

### Required Interface

Your model must implement a generator function with this signature:

```python
def model_generator(article: str, n_traces: int) -> List[TraceWithLogprobs]:
    """
    Generate summaries with token-level log probabilities.
    
    Args:
        article: Input article text
        n_traces: Number of summaries to generate
    
    Returns:
        List of TraceWithLogprobs with:
            - text: Generated summary
            - tokens: List of tokens
            - logprobs: List of log probabilities (one per token)
    """
    pass
```

### Example Implementations

#### HuggingFace Transformers

```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

class HuggingFaceGenerator:
    def __init__(self, model_name='facebook/bart-large-cnn'):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
        self.model.eval()
    
    def __call__(self, article: str, n_traces: int):
        inputs = self.tokenizer(article, return_tensors='pt', max_length=1024, truncation=True)
        
        traces = []
        for _ in range(n_traces):
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=90,
                    output_scores=True,
                    return_dict_in_generate=True,
                    do_sample=True,
                    temperature=0.7
                )
            
            # Extract tokens and logprobs
            summary_ids = outputs.sequences[0]
            tokens = self.tokenizer.convert_ids_to_tokens(summary_ids)
            
            # Get logprobs from scores
            logprobs = []
            for i, score in enumerate(outputs.scores):
                token_id = summary_ids[i+1]  # +1 because scores start after first token
                logprob = torch.log_softmax(score[0], dim=-1)[token_id].item()
                logprobs.append(logprob)
            
            traces.append(TraceWithLogprobs(
                text=self.tokenizer.decode(summary_ids, skip_special_tokens=True),
                tokens=tokens,
                logprobs=logprobs
            ))
        
        return traces

# Usage
generator = HuggingFaceGenerator()
experiment = XSumHallucinationExperiment(model_generator=generator)
```

#### Anthropic Claude

```python
from anthropic import Anthropic

class ClaudeGenerator:
    def __init__(self, api_key: str, model='claude-3-5-sonnet-20241022'):
        self.client = Anthropic(api_key=api_key)
        self.model = model
    
    def __call__(self, article: str, n_traces: int):
        prompt = f"Summarize this article in one sentence:\n\n{article}\n\nSummary:"
        
        traces = []
        for _ in range(n_traces):
            # Note: Claude API doesn't currently support logprobs
            # This is a limitation for this experiment
            # You would need to use a model that supports logprobs
            pass
        
        return traces
```

#### OpenAI GPT

```python
from openai import OpenAI

class OpenAIGenerator:
    def __init__(self, api_key: str, model='gpt-4'):
        self.client = OpenAI(api_key=api_key)
        self.model = model
    
    def __call__(self, article: str, n_traces: int):
        prompt = f"Summarize this article in one sentence:\n\n{article}\n\nSummary:"
        
        traces = []
        for _ in range(n_traces):
            response = self.client.completions.create(
                model=self.model,
                prompt=prompt,
                max_tokens=90,
                temperature=0.7,
                logprobs=5  # Request top-5 logprobs
            )
            
            # Extract tokens and logprobs
            tokens = response.choices[0].logprobs.tokens
            logprobs = response.choices[0].logprobs.token_logprobs
            
            traces.append(TraceWithLogprobs(
                text=response.choices[0].text.strip(),
                tokens=tokens,
                logprobs=logprobs
            ))
        
        return traces
```

## Evaluation Metrics

### Automatic Metrics

**Factual Consistency**:
- **AlignScore**: Recommended primary metric
  ```python
  from alignscore import AlignScore
  scorer = AlignScore(model='roberta-base', ckpt_path='path/to/checkpoint')
  score = scorer.score(contexts=[article], claims=[summary])
  ```

- **QuestEval**: QA-based metric
  ```python
  from questeval.questeval_metric import QuestEval
  questeval = QuestEval()
  score = questeval.compute_all(source=article, hypothesis=summary)
  ```

**Reference-Based**:
- **ROUGE**: Standard n-gram overlap
- **BERTScore**: Semantic similarity

### Human Evaluation

For subset of results, consider:
- Faithfulness (0-1 scale)
- Factuality (binary: factual/not-factual)
- Hallucination span identification

## Results Structure

```
results/
├── experiment_a_token_hallucination.json
├── experiment_b_multi_trace_consensus.json
├── experiment_c_hierarchical_analysis.json
├── experiment_d_calibration_analysis.json
├── hierarchical_correlations.json
├── calibration_curve.png
└── summary_statistics.json
```

Each JSON file contains:
```json
{
  "experiment_name": "...",
  "instance_id": "...",
  "metrics": {
    "token_confidence": 0.75,
    "factual_consistency": 0.82,
    ...
  },
  "metadata": {
    "summary": "...",
    ...
  }
}
```

## Analysis

### Quick Analysis Script

```python
import json
import numpy as np
from pathlib import Path

# Load results
results_dir = Path('./results')
with open(results_dir / 'experiment_a_token_hallucination.json') as f:
    results = json.load(f)

# Compute statistics
token_confs = [r['metrics']['token_confidence'] for r in results]
factual_scores = [r['metrics']['factual_consistency'] for r in results if r['metrics'].get('factual_consistency')]

print(f"Average token confidence: {np.mean(token_confs):.3f}")
print(f"Average factual consistency: {np.mean(factual_scores):.3f}")
print(f"Correlation: {np.corrcoef(token_confs[:len(factual_scores)], factual_scores)[0,1]:.3f}")
```

### Visualization Notebooks

See `notebooks/` directory for:
- Data exploration
- Token-level analysis
- Results visualization

## Expected Timeline

- **Week 1**: Setup + Experiment A (token-level detection)
- **Week 2**: Experiment B (multi-trace consensus)
- **Week 3**: Experiments C & D (hierarchical + calibration)
- **Week 4**: Analysis + paper writing

## Citation

If you use this code, please cite:

```bibtex
@misc{xsum_hallucination_deepconf,
  title={Adapting DeepConf for Summarization Hallucination Detection},
  author={Your Name},
  year={2024},
  howpublished={GitHub},
  url={https://github.com/yourusername/xsum-hallucination-deepconf}
}
```

Also cite the original papers:
- **XSum**: Narayan et al. (2018) "Don't Give Me the Details, Just the Summary!"
- **Hallucination Annotations**: Maynez et al. (2020) "On Faithfulness and Factuality in Abstractive Summarization"
- **AlignScore**: Zha et al. (2023) "AlignScore: Evaluating Factual Consistency with A Unified Alignment Function"

## Troubleshooting

### Issue: Import errors

```bash
# Make sure you're in the right directory
cd xsum_hallucination_deepconf

# Activate virtual environment
source venv/bin/activate

# Reinstall dependencies
pip install -r requirements.txt
```

### Issue: Dataset download fails

```bash
# Download manually
python -c "from datasets import load_dataset; load_dataset('xsum', split='test', cache_dir='./data')"
```

### Issue: Out of memory

```bash
# Reduce batch size or max instances
python src/run_experiments.py --max_instances 100  # Instead of 1000
```

## Contributing

Contributions welcome! Areas of interest:
- Additional model integrations
- Better claim extraction methods
- Semantic similarity for claim clustering
- Additional evaluation metrics

## License

MIT License - see LICENSE file

## Contact

For questions or issues:
- Open a GitHub issue
- Email: your.email@domain.com

## Acknowledgments

- Based on the [deepconf-adapter](../deepconf-adapter) skill
- Built on XSum dataset and Google hallucination annotations
- Inspired by DeepConf paper for math reasoning

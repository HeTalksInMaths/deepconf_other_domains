# DeepConf Adapter - Core Utilities

This directory contains the **essential shared utilities** used by both DeepConf domain adaptations.

## What's Here

- **`confidence_utils.py`** - Core confidence computation functions (required by all implementations)
- **`SKILL.md`** - Complete framework documentation

## Key Components

### `TraceWithLogprobs`
Container for model outputs with token-level confidence:
```python
@dataclass
class TraceWithLogprobs:
    text: str              # Generated text
    logprobs: List[float]  # Log probability per token
    tokens: List[str]      # Token list
    metadata: Dict         # Optional metadata
```

### Confidence Functions

1. **`compute_token_confidence(logprobs)`** - Token-level confidence
2. **`compute_trace_confidence(trace)`** - Single trace confidence
3. **`compute_multi_trace_confidence(traces)`** - Aggregate across traces
4. **`should_generate_more_traces()`** - Early stopping logic

### Domain-Specific Analyzers

- **`SafetyConfidenceAnalyzer`** - For safety/refusal benchmarks
- **`MCQConfidenceAnalyzer`** - For multiple choice questions

## Usage

Both implementations import from here:

```python
# In safety_deepconf.py or summarization_confidence.py
from confidence_utils import (
    TraceWithLogprobs,
    compute_trace_confidence,
    compute_multi_trace_confidence
)
```

## Integration

When setting up in Claude Code web or locally, ensure this directory is accessible and update import paths if needed:

```python
# Default (assumes skill loaded in Claude Code)
sys.path.append('/home/claude/deepconf-adapter/scripts')

# For local/custom setup
sys.path.append('./deepconf_adapter')  # Relative path
```

## License

MIT - See main repository LICENSE

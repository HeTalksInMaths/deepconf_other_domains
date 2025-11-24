# DeepConf Parameter Fix ✅

## Issue Found
The initial implementation used **incorrect sampling parameters** that didn't match the original DeepConf paper/repo.

## Original DeepConf Parameters (from paper/repo)
```python
SamplingParams(
    temperature=0.6,
    top_p=0.95,
    max_tokens=32000,
    logprobs=20,
)
```

## What Was Wrong
Our implementation initially used:
- ❌ `temperature=0.7` (should be **0.6**)
- ❌ `top_p=0.9` (should be **0.95**)

## Files Fixed

### 1. `src/qwen3_adapter.py`
- Line 56-57: Default parameters in `generate_with_logprobs()`
- Line 175-176: Default parameters in `batch_generate()`
- Added comments: `# DeepConf default`

### 2. `run_experiment.py`
- Line 101-102: Added `top_p=0.95` and corrected `temperature=0.6`

### 3. `DeepConf_Safety_Experiment.ipynb`
- Updated all occurrences of temperature and top_p
- Cell 3: Qwen3 adapter defaults
- Cell 6: Experiment execution parameters

## Verification

```bash
# Check all files have correct parameters
grep -r "temperature.*0.6\|top_p.*0.95" src/*.py run_experiment.py
```

Expected output:
```
src/qwen3_adapter.py:        temperature: float = 0.6,  # DeepConf default
src/qwen3_adapter.py:        top_p: float = 0.95,        # DeepConf default
src/qwen3_adapter.py:        temperature = generation_kwargs.get('temperature', 0.6)  # DeepConf default
src/qwen3_adapter.py:        top_p = generation_kwargs.get('top_p', 0.95)             # DeepConf default
run_experiment.py:        temperature=0.6,  # DeepConf default
run_experiment.py:        top_p=0.95,        # DeepConf default
```

## Why This Matters

**Temperature (0.6 vs 0.7):**
- Lower temperature = less randomness, more focused sampling
- 0.6 is slightly more conservative than 0.7
- Affects confidence distributions

**Top-p (0.95 vs 0.9):**
- Nucleus sampling threshold
- 0.95 allows slightly more diversity in token selection
- 0.9 is more restrictive

**Impact:**
- Using correct parameters ensures results match DeepConf paper
- Confidence thresholds were calibrated for temp=0.6, top_p=0.95
- Reproducibility with original experiments

## Status: FIXED ✅

All files now use the correct DeepConf parameters from the original paper/repo.

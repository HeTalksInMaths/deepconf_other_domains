# WildGuard Integration Guide

This guide explains how to use the WildGuard classifier and WildGuardMix dataset with the DeepConf safety evaluation framework.

## Overview

We've integrated WildGuard in two ways:

1. **Offline reclassification** - Run WildGuard on existing ToxicChat results
2. **Two-pass workflow** - Generate with Qwen3, then classify with WildGuard

This fixes critical bugs and provides better safety detection than keyword-based methods.

---

## Critical Bug Fixed

**Issue:** The original code used `any()` for unsafe content detection, causing more traces → more false positives.

```python
# OLD (buggy):
final_prediction = 'unsafe' if any(self.is_unsafe_content(t) for t in traces) else 'safe'

# NEW (fixed):
unsafe_count = sum(1 for t in traces if self.is_unsafe_content(t))
final_prediction = 'unsafe' if unsafe_count >= len(traces) * 0.5 else 'safe'
```

**Impact:** This explains why 20th percentile (3 traces) had better accuracy than 90th percentile (9 traces).

---

## Quick Start

### 1. Fix Existing ToxicChat Results (Fast)

Recompute final predictions with majority voting instead of `any()`:

```bash
python reclassify_with_wildguard.py \
    --results-dir results/toxicchat_qwen06b_5083_vllm \
    --benchmark toxicchat \
    --fix-majority-voting
```

**Time:** ~30 seconds
**Output:** `results/toxicchat_qwen06b_5083_vllm_reclassified/`

### 2. Add WildGuard Classification to ToxicChat (Slow but Better)

Run WildGuard classifier on all existing traces:

```bash
python reclassify_with_wildguard.py \
    --results-dir results/toxicchat_qwen06b_5083_vllm \
    --benchmark toxicchat \
    --use-wildguard \
    --device cpu \
    --load-in-8bit \
    --cache-dir .wildguard_cache
```

**Time:** 5-10 hours on CPU (overnight)
**Output:** `results/toxicchat_qwen06b_5083_vllm_reclassified/`
**Note:** Caching enabled - can resume if interrupted

### 3. Run Percentile Sweep on Reclassified Data

After reclassification, analyze different percentiles:

```bash
python analyze_threshold_sensitivity.py \
    --results-dir results/toxicchat_qwen06b_5083_vllm_reclassified \
    --benchmark toxicchat \
    --percentiles 20 30 40 50 60 70 80 90 \
    --min-traces 3 \
    --max-traces 10
```

---

## WildGuardMix Dataset Workflow

WildGuardMix has gold-standard refusal labels, making it ideal for testing our hypothesis.

### Step 1: Generate Traces with Qwen3 (GPU)

```bash
# On Lambda GPU
cd ~/deepconf_safety/safety-deepconf-experiment
source ~/venv_deepconf/bin/activate

python run_experiment.py \
    --model Qwen/Qwen3-0.6B \
    --benchmark wildguardmix \
    --num-instances 1725 \
    --max-traces 10 \
    --percentile 90 \
    --early-stopping \
    --batch \
    --output-dir results/wildguardmix_qwen06b_baseline
```

**Time:** ~2-3 hours
**Cost:** ~$2.50-4.00 @ $1.29/hr
**Output:** Saves traces incrementally to `predictions_checkpoint.jsonl`

### Step 2: Classify with WildGuard (GPU or CPU)

**Option A: GPU (recommended if still on Lambda)**

```bash
python classify_with_wildguard.py \
    --input results/wildguardmix_qwen06b_baseline/predictions.jsonl \
    --output results/wildguardmix_qwen06b_wildguard/predictions.jsonl \
    --benchmark wildguardmix \
    --device cuda \
    --load-in-8bit \
    --cache-dir .wildguard_cache
```

**Time:** ~1-2 hours on GPU
**Cost:** ~$1.30-2.60

**Option B: CPU (if downloading results locally)**

```bash
# Download from Lambda
scp -i ~/.ssh/lambda_gpu \
    ubuntu@144.24.121.54:~/deepconf_safety/safety-deepconf-experiment/results/wildguardmix_qwen06b_baseline/predictions.jsonl \
    ./results/wildguardmix_qwen06b_baseline/

# Classify locally
python classify_with_wildguard.py \
    --input results/wildguardmix_qwen06b_baseline/predictions.jsonl \
    --output results/wildguardmix_qwen06b_wildguard/predictions.jsonl \
    --benchmark wildguardmix \
    --device cpu \
    --load-in-8bit \
    --cache-dir .wildguard_cache
```

**Time:** ~8-12 hours on CPU (overnight)

### Step 3: Analyze Results

```bash
# Load ground truth and compute metrics
python analyze_threshold_sensitivity.py \
    --results-dir results/wildguardmix_qwen06b_wildguard \
    --benchmark wildguardmix \
    --percentiles 20 30 40 50 60 70 80 90 \
    --min-traces 3 \
    --max-traces 10
```

---

## Key Files

### New Scripts

- **`src/wildguard_classifier.py`** - WildGuard wrapper with 8-bit quantization
- **`reclassify_with_wildguard.py`** - Reclassify existing results
- **`classify_with_wildguard.py`** - Two-pass classification workflow

### Modified Files

- **`src/safety_deepconf.py`** - Fixed `any()` bug, added `detection_mode` parameter

---

## Expected Results

### With Majority Voting Fix

If the bug was causing false positives:
- ✅ Higher accuracy at all percentiles
- ✅ Accuracy difference between percentiles should decrease
- ✅ 90th percentile might catch up to lower percentiles

### With WildGuard Classifier

Compared to keyword-based detection:
- ✅ Better refusal detection (catches subtle refusals)
- ✅ Better harmfulness detection (no false positives from common words like "harm", "attack")
- ✅ Should validate hypothesis: low-confidence non-refusals are riskier

### On WildGuardMix Dataset

Compared to ToxicChat:
- ✅ More refusals detected (WildGuardMix designed for refusal testing)
- ✅ Gold-standard labels for validation
- ✅ Better for testing confidence × refusal interactions

---

## Comparison Table

| Aspect | ToxicChat + Keywords | ToxicChat + WildGuard | WildGuardMix + WildGuard |
|--------|---------------------|----------------------|--------------------------|
| **Refusal Detection** | 40+ patterns | WildGuard classifier | WildGuard classifier |
| **Harmfulness Detection** | 9 keywords | WildGuard classifier | WildGuard classifier |
| **Ground Truth** | Toxicity labels | Toxicity labels | Prompt harmfulness + Gold refusal labels |
| **Dataset Size** | 5,083 test | 5,083 test | 1,725 test |
| **Existing Traces?** | ✅ Yes | ✅ Yes | ❌ Need new run |
| **GPU Time** | 0 (reanalyze) | 0 (reanalyze) | ~2-3 hours |
| **Classification Time** | 0 (patterns) | ~5-10 hrs CPU | ~8-12 hrs CPU |
| **Best For** | Quick test | Better accuracy | Hypothesis validation |

---

## Troubleshooting

### Out of Memory (CPU)

If WildGuard crashes on CPU:
```bash
# Use 8-bit quantization
--load-in-8bit

# Or process in smaller batches (edit script to add batch size limit)
```

### Classification is Too Slow

Options:
1. Use GPU if available (`--device cuda`)
2. Enable caching (`--cache-dir`) so you can resume
3. Run overnight
4. Use Lambda GPU for classification too (~$2-3)

### WildGuardMix Dataset Not Found

Download from HuggingFace (requires authentication):
```bash
python download_wildguardmix.py
```

Make sure your HuggingFace token has access to gated datasets.

---

## Cost Summary

| Task | Time | Cost @ $1.29/hr |
|------|------|----------------|
| ToxicChat majority voting fix | 30 sec | $0 |
| ToxicChat + WildGuard (CPU) | 10 hrs | $0 (local) |
| WildGuardMix trace generation | 3 hrs | ~$4 |
| WildGuardMix classification (GPU) | 2 hrs | ~$2.50 |
| WildGuardMix classification (CPU) | 12 hrs | $0 (local) |
| **TOTAL (GPU path)** | **5 hrs** | **~$6.50** |
| **TOTAL (CPU path)** | **3 hrs GPU + 22 hrs CPU** | **~$4** |

---

## Next Steps

After running all experiments:

1. **Compare detection methods**
   ```bash
   python compare_detection_methods.py \
       --pattern-results results/toxicchat_qwen06b_5083_vllm \
       --wildguard-results results/toxicchat_qwen06b_5083_vllm_reclassified
   ```

2. **Analyze confidence × safety relationship**
   ```bash
   python visualize_confidence_safety.py \
       --results results/wildguardmix_qwen06b_wildguard \
       --output plots/
   ```

3. **Statistical hypothesis testing**
   - Is uncertain_compliance riskier than confident_compliance?
   - Does confidence predict harmfulness?
   - Chi-square tests, effect sizes

---

## Questions?

Check the main README.md or review the code comments in:
- `src/wildguard_classifier.py` - WildGuard API
- `reclassify_with_wildguard.py` - Reclassification logic
- `src/safety_deepconf.py` - Core experiment logic

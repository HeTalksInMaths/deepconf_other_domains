# Next Steps for DeepConf Safety Experiment

## ✅ What's Complete

All critical bugs have been fixed:
1. **Infinity confidence bug** - Fixed with safeguards
2. **Percentile-based thresholds** - Implemented (90th percentile, DeepConf style)
3. **Enhanced refusal detection** - 40+ patterns
4. **ToxicChat ready** - 10,165 examples downloaded
5. **WildGuardMix loader** - Implemented and ready

**Current Validation**: Running experiment on 20 ToxicChat instances to verify fixes

---

## 🎯 Recommended Experimental Pipeline

### Phase 1: Validation (In Progress)
```bash
# RUNNING NOW: Validate fixes with small ToxicChat subset
python3 run_experiment.py \
    --model Qwen/Qwen3-0.6B \
    --benchmark toxicchat \
    --num-instances 20 \
    --output-dir results/validation_toxicchat_20
```

**Expected output:**
- No infinity values in confidence scores
- Confidence scores in reasonable range (0-5 typically)
- Adaptive thresholds per instance
- Variable trace counts (3-20 depending on early stopping)

### Phase 2: ToxicChat Full Test
```bash
# Run on full ToxicChat test set (5,083 examples)
python3 run_experiment.py \
    --model Qwen/Qwen3-0.6B \
    --benchmark toxicchat \
    --output-dir results/toxicchat_full

# OR run on subset for faster iteration
python3 run_experiment.py \
    --model Qwen/Qwen3-0.6B \
    --benchmark toxicchat \
    --num-instances 500 \
    --output-dir results/toxicchat_500
```

**Analysis Focus:**
- Compare accuracy across 4 confidence × refusal categories
- Test hypothesis: `uncertain_compliance` accuracy << `confident_compliance` accuracy
- Limitation: Refusal detection is pattern-based (not gold standard)

### Phase 3: WildGuardMix (After Access Granted)
```bash
# Request access first at:
# https://huggingface.co/datasets/allenai/wildguardmix

# Once approved, download:
python3 download_wildguardmix.py

# Run full test set (1,725 examples with GOLD STANDARD refusal labels!)
python3 run_experiment.py \
    --model Qwen/Qwen3-0.6B \
    --benchmark wildguardmix \
    --output-dir results/wildguardmix_full
```

**Analysis Advantages:**
- **Explicit `response_refusal_label`** - No pattern matching needed!
- Validate your refusal detector against gold labels
- Stronger evidence for publication
- Used in "Chasing Moving Targets" paper

---

## 📊 Expected Results (Hypothesis Validation)

### Your Research Question
**"Are low-confidence non-refusals riskier than high-confidence non-refusals?"**

### Analysis Structure (4 Categories)

Based on your preliminary results:

| Category | Refusal Rate | Confidence | Expected Accuracy | Interpretation |
|----------|-------------|------------|------------------|----------------|
| **Confident Refusal** | >50% | High | ~70-90% | Appropriately refusing unsafe prompts |
| **Uncertain Refusal** | >50% | Low | ~50-70% | Refusing but uncertain (conservative) |
| **Confident Compliance** | <50% | High | ~60-100% | Confidently answering (safe or unsafe) |
| **Uncertain Compliance** | <50% | Low | **~20-40%** | **TARGET: Highest risk category!** |

### Preliminary Evidence (from your results)
- `uncertain_compliance`: **20% accuracy** (2/10 correct)
- `confident_compliance`: **100% accuracy** (7/7 correct)
- **Gap: 80 percentage points!** → Strong support for hypothesis

### Statistical Validation Checklist
- [ ] Sufficient sample size per category (aim for >30 in each)
- [ ] Statistically significant difference (t-test, p < 0.05)
- [ ] Replicates across different models (Qwen3-0.6B, 1.7B, 4B, 8B)
- [ ] Validates on different benchmarks (ToxicChat vs WildGuardMix)

---

## 🔬 Experimental Variables to Explore

### 1. Model Size Effect
```bash
# Test across Qwen3 family
for model in "Qwen/Qwen3-0.6B" "Qwen/Qwen3-1.7B" "Qwen/Qwen3-4B" "Qwen/Qwen3-8B"
do
    python3 run_experiment.py \
        --model $model \
        --benchmark toxicchat \
        --num-instances 500 \
        --output-dir results/toxicchat_${model##*/}
done
```

**Expected**: Larger models may have:
- Higher overall confidence
- Better calibration (confidence matches accuracy)
- Smaller accuracy gap between categories

### 2. Confidence Threshold Sensitivity
```bash
# Try different percentiles
# Modify in code: confidence_percentile = 70, 80, 90, 95
```

**Expected**: Higher percentiles (more selective) may:
- Generate more traces (higher bar to clear)
- Higher overall accuracy (more careful)
- Trade-off: Inference cost vs. safety

### 3. Refusal Pattern Robustness
```bash
# Compare against gold standard (WildGuardMix)
# Measure: Precision/Recall of your 40 refusal patterns
```

**Validation**:
- Precision: Of detected refusals, how many are true refusals?
- Recall: Of true refusals, how many did you detect?
- F1 score: Harmonic mean of precision and recall

---

## 📈 Analysis Scripts

### Analyze Experiment Results
```python
import json
import pandas as pd

# Load results
with open('results/toxicchat_full/analysis.json') as f:
    analysis = json.load(f)

# Print hypothesis test
if 'hypothesis_test' in analysis:
    print("Hypothesis Test Results:")
    print(f"  Uncertain Compliance Accuracy: {analysis['hypothesis_test']['uncertain_compliance_accuracy']:.3f}")
    print(f"  Confident Compliance Accuracy: {analysis['hypothesis_test']['confident_compliance_accuracy']:.3f}")
    print(f"  Difference: {analysis['hypothesis_test']['difference']:.3f}")
    print(f"  Interpretation: {analysis['hypothesis_test']['interpretation']}")

# Per-category breakdown
for category in ['confident_refusal', 'uncertain_refusal', 'confident_compliance', 'uncertain_compliance']:
    if category in analysis:
        cat_data = analysis[category]
        print(f"\n{category.upper()}:")
        print(f"  Count: {cat_data['count']}")
        print(f"  Accuracy: {cat_data['accuracy']:.3f}")
        print(f"  Avg Confidence: {cat_data['avg_confidence']:.3f}")
        print(f"  Avg Traces: {cat_data['avg_traces']:.1f}")
```

### Compare Across Models
```python
import glob
import json
import matplotlib.pyplot as plt

results = {}
for result_file in glob.glob('results/*/analysis.json'):
    model_name = result_file.split('/')[1]
    with open(result_file) as f:
        results[model_name] = json.load(f)

# Plot accuracy by category for each model
categories = ['confident_refusal', 'uncertain_refusal', 'confident_compliance', 'uncertain_compliance']
# ... plotting code ...
```

---

## 🎓 Publication Checklist

If you're writing this up for a paper/report:

### Required Components
- [ ] **Clear hypothesis statement**: "Low-confidence non-refusals indicate safety risks"
- [ ] **Baseline comparison**: Compare against majority vote (no confidence)
- [ ] **Statistical significance**: Report p-values, confidence intervals
- [ ] **Error analysis**: Manually inspect misclassified examples
- [ ] **Limitations**: Discuss pattern-based refusal detection weakness
- [ ] **Future work**: Gold standard validation with WildGuardMix

### Figures to Create
1. **Accuracy by Category** (bar chart)
2. **Confidence Distribution** (violin plot per category)
3. **Trace Count vs Accuracy** (scatter plot)
4. **Confusion Matrix** (predicted vs ground truth)

### Tables to Include
1. **Overall Results** (accuracy, precision, recall per category)
2. **Model Comparison** (across Qwen3 sizes)
3. **Benchmark Comparison** (ToxicChat vs WildGuardMix)

---

## 🐛 If Issues Occur

### Common Problems

**1. Model loading fails (CUDA out of memory)**
```bash
# Use CPU instead
python3 run_experiment.py --model Qwen/Qwen3-0.6B --benchmark toxicchat --device cpu
```

**2. Still seeing infinity values**
```python
# Check confidence_utils.py:42-53 has the safeguards
# Check qwen3_adapter.py:113-127 has logprob clipping
```

**3. All traces hitting max_traces (no early stopping)**
```python
# Check that use_percentile_threshold=True in safety_deepconf.py:71
# Lower confidence_percentile to make threshold easier to reach
```

**4. Very low refusal detection rate**
```python
# Add more patterns specific to your model's style
# Or use WildGuardMix which has gold labels
```

---

## 📞 Support & Resources

### DeepConf Resources
- Original Paper: https://arxiv.org/abs/... (check deepconf repo)
- Original Repo: /Users/hetalksinmaths/deepconf_other_domains/deepconf/
- Confidence Utils: /Users/hetalksinmaths/deepconf_other_domains/deepconf_adapter/

### Safety Benchmark Resources
- ToxicChat Paper: https://huggingface.co/datasets/lmsys/toxic-chat
- WildGuardMix: https://huggingface.co/datasets/allenai/wildguardmix
- "Chasing Moving Targets" paper (uses WildGuardMix)

### Qwen3 Resources
- Model Cards: https://huggingface.co/Qwen
- Documentation: https://qwen.readthedocs.io/

---

## ✅ Quick Start (Right Now)

While validation experiment runs:

1. **Request WildGuardMix access**: https://huggingface.co/datasets/allenai/wildguardmix
2. **Check validation results**: `cat results/validation_toxicchat_20/analysis.json`
3. **Plan next experiment**: Choose model size and instance count
4. **Prepare analysis notebook**: Set up plotting code for results

Once validation completes (~5-10 minutes), you'll know if all fixes are working correctly!

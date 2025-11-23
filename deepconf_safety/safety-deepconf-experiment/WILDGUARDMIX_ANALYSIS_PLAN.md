# WildGuardMix Analysis Plan

## Overview

Replicate **all ToxicChat analysis** on WildGuardMix dataset (1,725 test instances) to:
1. Validate findings on different dataset
2. Compare pattern-based vs gold-standard labels
3. Test generalization of confidence insights

---

## Key Differences: ToxicChat vs WildGuardMix

| Aspect | ToxicChat | WildGuardMix |
|--------|-----------|--------------|
| **Size** | 5,083 test | 1,725 test |
| **Labels** | Binary (toxic/safe) | Gold-standard refusal + harmfulness |
| **Our approach** | Pattern-based refusal detection | Same (will compare to gold labels) |
| **Ground truth** | Prompt toxicity only | Prompt + Response labels |
| **Use case** | Test hypothesis on large dataset | Validate with gold-standard labels |

---

## Analysis To Replicate

### 1. Confidence Distribution Analysis

**Same plots as ToxicChat:**

#### a) 2×2 Confusion Matrix ⭐ NEW
- **Toxic + Refused** (TP - correctly caught)
- **Toxic + Answered** (FN - missed toxic) ⚠️ DANGEROUS
- **Safe + Refused** (FP - false alarm)
- **Safe + Answered** (TN - correctly allowed)

**Comparison Point:** Does WildGuardMix show same patterns?

#### b) Confidence by Correctness
- Correct vs Incorrect predictions
- **ToxicChat finding:** Incorrect have HIGHER confidence (0.640 vs 0.510)
- **Test:** Does this hold on WildGuardMix?

#### c) Confidence by Refusal Category
- 4 categories: confident_refusal, confident_compliance, uncertain_refusal, uncertain_compliance
- **ToxicChat finding:** Uncertain compliance has BEST accuracy (95.8%)
- **Test:** Same pattern on WildGuardMix?

#### d) Confidence by Ground Truth
- Safe vs Toxic prompts
- **ToxicChat finding:** HIGHER confidence on toxic prompts (0.679 vs 0.512)
- **Test:** Replicates on WildGuardMix?

#### e) Trace Evolution
- How confidence changes across 10 traces
- **ToxicChat finding:** Stays stable (early stopping justified)
- **Test:** Same stability on WildGuardMix?

---

### 2. Percentile Sweep with Safety Metrics ⭐ ENHANCED

For each percentile (20-90%, every 10%):

**Metrics to compute:**
- ✅ Accuracy
- ⭐ **Sensitivity (Recall)** - Of toxic prompts, % caught (MOST CRITICAL)
- ⭐ **Specificity** - Of safe prompts, % correctly allowed
- ⭐ **Precision** - Of flagged items, % truly toxic
- ⭐ **F1 Score**
- ✅ Token savings (%)
- ✅ Cost reduction (Nx faster)
- ⭐ **Missed toxic count** (False Negatives)

**Visualizations:**
1. Accuracy vs Percentile curve
2. **Sensitivity vs Specificity curves** (safety trade-off)
3. Token Savings vs Percentile
4. **Safety-Efficiency scatter plot** (sensitivity × savings)

**Comparison:**
- ToxicChat: 20-30th best (83% acc, 70% savings)
- WildGuardMix: Will it show same trend?

---

### 3. Gold-Standard Label Comparison ⭐ UNIQUE TO WILDGUARDMIX

**WildGuardMix advantage:** Has expert-labeled refusal/harmfulness

**Additional Analysis:**

#### a) Pattern-based vs Gold-Standard Refusal
- Our heuristics vs WildGuard's gold labels
- False positive rate (over-detection of refusals)
- False negative rate (missed refusals)

#### b) Response Harmfulness (Ground Truth Available)
- WildGuardMix labels responses as harmful/safe
- Compare to our final predictions
- **Key Question:** Do low-confidence responses correlate with harmfulness?

#### c) Prompt Harmfulness
- WildGuardMix also labels prompts
- Compare to our approach (pattern-based on responses)

---

## Implementation Steps

### Step 1: Run Percentile Sweep on WildGuardMix

```bash
cd ~/deepconf_safety/safety-deepconf-experiment

# Run comprehensive analysis
python scripts/comprehensive_percentile_analysis.py \
    --results-dir results/wildguardmix_qwen06b_baseline \
    --benchmark wildguardmix \
    --data-root data \
    --output wildguardmix_percentile_safety_analysis.json
```

**Output:** JSON with TP/TN/FP/FN, sensitivity, specificity for each percentile

---

### Step 2: Create All Visualizations

```bash
# Generate safety-focused plots
python scripts/create_safety_visualizations.py \
    --results-dir results/wildguardmix_qwen06b_baseline \
    --percentile-analysis wildguardmix_percentile_safety_analysis.json \
    --benchmark wildguardmix \
    --data-root data \
    --output plots/wildguardmix/
```

**Outputs:**
- `confusion_matrix_2x2.png` - Toxic/Safe × Refused/Answered
- `percentile_safety_curves.png` - Sens/Spec/Accuracy/Savings
- `confidence_by_correctness.png`
- `confidence_by_category.png`
- `confidence_by_toxicity.png`
- `trace_evolution.png`

---

### Step 3: Compare to Gold-Standard Labels

```bash
# Compare our heuristics to WildGuard gold labels
python scripts/compare_to_goldstandard.py \
    --results-dir results/wildguardmix_qwen06b_baseline \
    --data-root data \
    --output wildguardmix_goldstandard_comparison.json
```

**Analysis:**
- Refusal detection accuracy (our patterns vs gold)
- Response harmfulness correlation
- Identify where heuristics fail

---

### Step 4: Create WildGuardMix Analysis Notebook

**Separate notebook:** `WildGuardMix_Confidence_Analysis.ipynb`

**Structure:**
1. Executive Summary (compare to ToxicChat)
2. Load WildGuardMix data
3. All 6 visualizations
4. Percentile sweep table
5. **Gold-standard comparison section** (unique to WildGuardMix)
6. Side-by-side comparison: ToxicChat vs WildGuardMix findings
7. Conclusions and recommendations

---

## Expected Findings

### If Patterns Replicate:
✅ Confirms confidence is efficiency signal, not safety signal
✅ Lower percentiles objectively better
✅ Generalizes across datasets

### If Patterns Differ:
⚠️ Dataset-specific behaviors
⚠️ Need to investigate differences
⚠️ May require dataset-specific tuning

### Gold-Standard Insights:
🎯 Quantify heuristic accuracy
🎯 Identify failure modes
🎯 Guide future improvements

---

## Timeline (After WildGuard Classification Completes)

1. **Percentile sweep analysis:** ~5 minutes
2. **Create visualizations:** ~2 minutes
3. **Gold-standard comparison:** ~5 minutes
4. **Create notebook:** ~30 minutes
5. **Total:** ~45 minutes after WildGuard finishes

---

## Deliverables

### For ToxicChat:
- ✅ 6 plots (including 2×2 confusion matrix, sensitivity curves)
- ✅ Percentile safety analysis JSON
- ✅ Analysis notebook with all outputs

### For WildGuardMix:
- 🔄 Same 6 plots
- 🔄 Percentile safety analysis JSON
- 🔄 Gold-standard comparison analysis
- 🔄 Separate analysis notebook
- 🔄 Comparison document (ToxicChat vs WildGuardMix)

### Combined:
- 📊 Side-by-side comparison visualizations
- 📝 Final report highlighting differences
- 🎯 Recommendations based on both datasets

---

## Key Questions to Answer

1. **Does confidence pattern replicate?**
   - Incorrect > Correct confidence?
   - Toxic prompts > Safe confidence?

2. **Does percentile trend replicate?**
   - Lower percentiles better?
   - Similar savings/accuracy trade-offs?

3. **How accurate are our heuristics?**
   - Compare to WildGuard gold refusal labels
   - Find systematic errors

4. **Does response harmfulness correlate with confidence?**
   - WildGuard labels responses as harmful/safe
   - Test if low-confidence → more harmful

5. **Sensitivity priority validated?**
   - Both datasets show same safety priority?
   - Confirm: catch all toxic > avoid false alarms

---

**Status:** Ready to execute once WildGuard classification completes (~40 min remaining)

# DeepConf Safety Evaluation - Complete Analysis Summary

**Date:** November 23, 2024  
**Researcher:** Analysis of Qwen3-0.6B with DeepConf framework on safety benchmarks

---

## 📦 Complete Deliverables

### Notebooks (6 total)

**Viewer Notebooks** (Pre-executed, viewable in Colab):
- `Run1_ToxicChat_Heuristic_Viewer.ipynb` (3.9 MB)
- `Run2_WildGuardMix_Heuristic_Viewer.ipynb` (3.7 MB)
- `Run3_ToxicChat_WildGuard_Viewer.ipynb` (3.9 MB)

**Reproducible Notebooks** (Full code):
- `Run1_ToxicChat_Heuristic_Reproducible.ipynb` (11 KB)
- `Run2_WildGuardMix_Heuristic_Reproducible.ipynb` (9.3 KB)
- `Run3_ToxicChat_WildGuard_Reproducible.ipynb` (13 KB)

### Visualizations (18 plots total)

**Run 1 - ToxicChat + Heuristic** (6 plots):
- `plots/run1/confusion_matrix_2x2.png` (297 KB)
- `plots/run1/percentile_safety_curves.png` (443 KB)
- `plots/run1/confidence_by_correctness.png` (220 KB)
- `plots/run1/confidence_by_category.png` (307 KB)
- `plots/run1/confidence_by_toxicity.png` (217 KB)
- `plots/run1/trace_evolution.png` (1.5 MB)

**Run 2 - WildGuardMix + Heuristic** (6 plots):
- `plots/run2/confusion_matrix_2x2.png` (290 KB)
- `plots/run2/percentile_safety_curves.png` (411 KB)
- `plots/run2/confidence_by_correctness.png` (212 KB)
- `plots/run2/confidence_by_category.png` (297 KB)
- `plots/run2/confidence_by_toxicity.png` (203 KB)
- `plots/run2/trace_evolution.png` (1.4 MB)

**Run 3 - ToxicChat + WildGuard** (6 plots):
- `plots/run3/confusion_matrix_2x2.png` (297 KB)
- `plots/run3/percentile_safety_curves.png` (451 KB)
- `plots/run3/confidence_by_correctness.png` (220 KB)
- `plots/run3/confidence_by_category.png` (307 KB)
- `plots/run3/confidence_by_toxicity.png` (217 KB)
- `plots/run3/trace_evolution.png` (1.5 MB)

### Analysis Data (3 JSON files)

- `toxicchat_percentile_safety_analysis.json` - Run 1 metrics
- `wildguardmix_percentile_safety_analysis.json` - Run 2 metrics
- `toxicchat_wildguard_percentile_safety_analysis.json` - Run 3 metrics

---

## 🔍 Critical Findings

### 1. The Confidence Paradox ⚠️ (MOST CRITICAL)

**Incorrect predictions have HIGHER confidence than correct ones!**

| Run | Incorrect Conf | Correct Conf | Difference |
|-----|----------------|--------------|------------|
| Run 1 (ToxicChat + Heuristic) | 0.640 | 0.510 | **+25%** |
| Run 2 (WildGuardMix + Heuristic) | 0.655 | 0.640 | +2% |
| Run 3 (ToxicChat + WildGuard) | 0.640 | 0.510 | **+25%** |

**Implication:** High confidence does NOT indicate safe or correct predictions!

### 2. Toxicity Confidence Bias ⚠️

**Model is MORE confident when responding to toxic prompts!**

| Run | Toxic Conf | Safe Conf | Difference |
|-----|-----------|-----------|------------|
| Run 1 (ToxicChat + Heuristic) | 0.679 | 0.512 | **+33%** |
| Run 2 (WildGuardMix + Heuristic) | 0.648 | 0.646 | ~0% |
| Run 3 (ToxicChat + WildGuard) | 0.679 | 0.512 | **+33%** |

**Implication:** The model is most confident precisely when it should be least trusted!

### 3. WildGuard 7B Impact: Minimal ❌

**Using a 7B classifier vs simple heuristics:**

| Metric | Heuristic | WildGuard | Improvement |
|--------|-----------|-----------|-------------|
| Accuracy | 9.1% | 10.8% | +1.7% |
| Sensitivity | 91.4% | 91.7% | +0.3% |
| Specificity | 2.7% | 4.6% | +1.9% |
| Confidence paradox | ✗ Exists | ✗ **Still exists** | 0% |
| Toxicity bias | ✗ Exists | ✗ **Still exists** | 0% |

**Verdict:** ❌ **NOT worth the computational cost**
- 7B parameter model (Mistral-based)
- 17 minutes classification time
- Minimal accuracy improvement
- **Does NOT fix the fundamental confidence issue**

### 4. Dataset Quality Matters ✅

**WildGuardMix vs ToxicChat comparison:**

| Metric | ToxicChat | WildGuardMix | Winner |
|--------|-----------|--------------|--------|
| Accuracy | 9-10% | 41-42% | ✅ WildGuardMix |
| Confidence bias (incorrect) | +25% | +2% | ✅ WildGuardMix |
| Confidence bias (toxic) | +33% | ~0% | ✅ WildGuardMix |
| Sensitivity | 91-94% | 92% | ≈ Tie |
| Gold-standard labels | ❌ | ✅ | ✅ WildGuardMix |

**Implication:** Better quality labels → better evaluation metrics

### 5. Optimal Percentile Strategy 📊

**For all 3 runs, lower percentiles are better:**

| Percentile | Accuracy | Sensitivity | Savings | Recommendation |
|-----------|----------|-------------|---------|----------------|
| 20th | 9-41% | 91-92% | **64-65%** | ✅ Best efficiency |
| 30th | 9-41% | 91-92% | 63-64% | ✅ Good balance |
| 50th | 9-41% | 91-92% | 62% | Moderate |
| 90th | 9-42% | 93-94% | **46%** | Best safety, worst efficiency |

**Key insight:** Lower percentiles (20-30%) provide:
- Similar or better accuracy
- Similar sensitivity (toxic catch rate)
- **Much better efficiency** (64% token savings)

---

## 💡 Root Cause Analysis

### Why doesn't WildGuard fix the confidence paradox?

**The problem is NOT the refusal detection method.**

**The problem is the BASE MODEL (Qwen3-0.6B):**

1. **Confidence comes from Qwen's own logprobs**, not WildGuard
2. Model is **inherently poorly calibrated** for safety
3. **Confidence bias exists at generation time**, before any classification
4. WildGuard only classifies the *output*, doesn't change confidence scores

### Evidence:

- Heuristic and WildGuard have **identical confidence patterns** (both 0.640 vs 0.510)
- Heuristic and WildGuard have **identical toxicity bias** (both 0.679 vs 0.512)
- Only the final classification changes (+1-2% accuracy), not confidence

**Conclusion:** ❌ **Confidence-based early stopping is fundamentally broken for safety** with Qwen3-0.6B

---

## 📈 Performance Summary

### Run 1: ToxicChat + Heuristic

**Dataset:** 5,083 test instances (362 toxic, 4,721 safe)  
**Method:** Pattern-based refusal detection

| Metric | Value |
|--------|-------|
| Overall Accuracy | 9.1% |
| Sensitivity (catch toxic) | 91.4% |
| Specificity (allow safe) | 2.7% |
| Token Savings (20th %ile) | 64.6% |
| Avg traces used | 3.54 / 10 |

**Key findings:**
- ❌ Low accuracy (9%)
- ✅ High sensitivity (91% toxic caught)
- ❌ Very low specificity (98% false alarms)
- ❌ Strong confidence paradox (+25%)
- ❌ Strong toxicity bias (+33%)

### Run 2: WildGuardMix + Heuristic

**Dataset:** 1,725 test instances (753 toxic, 972 safe)  
**Method:** Pattern-based refusal detection

| Metric | Value |
|--------|-------|
| Overall Accuracy | 41.5% |
| Sensitivity (catch toxic) | 92.2% |
| Specificity (allow safe) | 2.1% |
| Token Savings (20th %ile) | 64.3% |
| Avg traces used | 3.57 / 10 |

**Key findings:**
- ✅ **Much better accuracy** (41% vs 9%)
- ✅ Similar sensitivity (92%)
- ✅ **Minimal confidence bias** (+2% vs +25%)
- ✅ **No toxicity bias** (0% vs +33%)
- ✅ Better dataset quality

### Run 3: ToxicChat + WildGuard

**Dataset:** 5,083 test instances  
**Method:** WildGuard 7B classifier

| Metric | Value |
|--------|-------|
| Overall Accuracy | 10.8% |
| Sensitivity (catch toxic) | 91.7% |
| Specificity (allow safe) | 4.6% |
| Token Savings (20th %ile) | 64.6% |
| Avg traces used | 3.54 / 10 |

**Key findings:**
- ✅ Slightly better accuracy (+1.7% vs heuristic)
- ≈ Similar sensitivity (0.3% improvement)
- ✅ Better specificity (+1.9%)
- ❌ **Same confidence paradox** (0.640 vs 0.510)
- ❌ **Same toxicity bias** (0.679 vs 0.512)
- ❌ **Not worth computational cost**

---

## 🎯 Recommendations

### 1. For Practitioners

**DO:**
- ✅ Use **lower percentile thresholds** (20-30%) for efficiency
- ✅ Prioritize **sensitivity over specificity** for safety
- ✅ Use **WildGuardMix for evaluation** (better labels)
- ✅ Monitor **false negatives** (missed toxic) carefully

**DON'T:**
- ❌ Trust high confidence as a safety signal
- ❌ Use expensive classifiers (WildGuard) for minimal gain
- ❌ Rely on confidence-based early stopping for safety
- ❌ Use ToxicChat for evaluation (label quality issues)

### 2. For Researchers

**Critical Issues to Address:**
1. **Model calibration** - Qwen3-0.6B is poorly calibrated for safety
2. **Alternative stopping criteria** - Confidence doesn't work
3. **Better base models** - Need safety-aware models
4. **Ensemble methods** - Instead of single-model confidence

**Future Work:**
1. Test DeepConf with different base models (Llama, GPT, etc.)
2. Develop safety-specific calibration methods
3. Explore ensemble-based early stopping
4. Fine-tune models specifically for safety tasks

### 3. For This Specific Setup

**Optimal Configuration:**
- **Dataset:** WildGuardMix (better labels)
- **Method:** Simple heuristics (same performance, faster)
- **Percentile:** 20-30th (best efficiency, similar safety)
- **Metric priority:** Sensitivity > Accuracy > Specificity

**Expected Performance:**
- ~40% accuracy
- ~92% sensitivity (catch toxic)
- ~64% token savings
- ~3.6x cost reduction

---

## 📝 Files Manifest

### Notebooks
```
Run1_ToxicChat_Heuristic_Viewer.ipynb           3.9 MB  (Pre-executed)
Run1_ToxicChat_Heuristic_Reproducible.ipynb     11 KB   (Full code)

Run2_WildGuardMix_Heuristic_Viewer.ipynb        3.7 MB  (Pre-executed)
Run2_WildGuardMix_Heuristic_Reproducible.ipynb  9.3 KB  (Full code)

Run3_ToxicChat_WildGuard_Viewer.ipynb           3.9 MB  (Pre-executed)
Run3_ToxicChat_WildGuard_Reproducible.ipynb     13 KB   (Full code)
```

### Visualizations
```
plots/run1/  (6 plots, 3.0 MB total)
plots/run2/  (6 plots, 2.8 MB total)
plots/run3/  (6 plots, 3.0 MB total)
```

### Analysis Data
```
toxicchat_percentile_safety_analysis.json
wildguardmix_percentile_safety_analysis.json
toxicchat_wildguard_percentile_safety_analysis.json
```

**Total size:** ~25 MB (excluding raw predictions)

---

## 🚀 Next Steps

### Pending Tasks

1. **Schedule Run 4** (WildGuardMix + WildGuard) on Lambda
   - Validate WildGuard vs gold-standard labels
   - Complete the 2×2 comparison matrix

2. **Create Final Comparison Report**
   - Side-by-side analysis of all 4 runs
   - Dataset effect vs method effect
   - Recommendations for production use

3. **Package for Google Drive**
   - Organize all files in folders
   - Create shareable links
   - Add README for navigation

---

**End of Summary**

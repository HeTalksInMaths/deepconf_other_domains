# DeepConf Safety Evaluation - Deliverables

**Date**: November 23, 2025
**Evaluation**: Qwen3-0.6B with DeepConf framework on safety benchmarks

---

## 📦 Contents

### 📓 Notebooks (6 total)

**Viewer Notebooks** (Pre-executed, ready for Google Colab):
- `notebooks/Run1_ToxicChat_Heuristic_Viewer.ipynb` (4.1 MB)
- `notebooks/Run2_WildGuardMix_Heuristic_Viewer.ipynb` (3.8 MB)
- `notebooks/Run3_ToxicChat_WildGuard_Viewer.ipynb` (4.1 MB)

**Reproducible Notebooks** (Full code, requires data files):
- `notebooks/Run1_ToxicChat_Heuristic_Reproducible.ipynb` (11 KB)
- `notebooks/Run2_WildGuardMix_Heuristic_Reproducible.ipynb` (9.6 KB)
- `notebooks/Run3_ToxicChat_WildGuard_Reproducible.ipynb` (13 KB)

### 📊 Reports (2 documents)

- `reports/COMPREHENSIVE_REPORT.md` - Full experimental design, ELI5 explanations, all results, findings, and recommendations
- `reports/ANALYSIS_SUMMARY.md` - Executive summary with key metrics and deliverables

### 📈 Visualizations (18 plots across 3 runs)

**Run 1 - ToxicChat + Heuristic** (`plots/run1/`):
- `confusion_matrix_2x2.png` - TP/FP/TN/FN breakdown
- `percentile_safety_curves.png` - Accuracy & sensitivity vs percentile threshold
- `confidence_by_correctness.png` - Correct vs incorrect confidence distributions
- `confidence_by_category.png` - Confidence by TP/FP/TN/FN category
- `confidence_by_toxicity.png` - Toxic vs safe prompt confidence
- `trace_evolution.png` - How classifications evolve across traces

**Run 2 - WildGuardMix + Heuristic** (`plots/run2/`):
- Same 6 plots as Run 1, for WildGuardMix dataset

**Run 3 - ToxicChat + WildGuard** (`plots/run3/`):
- Same 6 plots as Run 1, using WildGuard 7B classifier

---

## 🔍 Quick Start

### For Viewing Results:
1. **Open Viewer Notebooks** in Google Colab (no setup required)
   - All outputs are pre-executed and visible immediately
   - No need to upload data files or install dependencies

### For Reproducing Experiments:
1. **Use Reproducible Notebooks** with the full dataset
   - Requires access to ToxicChat and WildGuardMix datasets
   - See `COMPREHENSIVE_REPORT.md` for dataset download instructions

### For Understanding the Experiments:
1. **Start with `reports/COMPREHENSIVE_REPORT.md`**
   - Includes ELI5 explanations of percentile thresholds
   - Explains why accuracy is misleading (focus on sensitivity!)
   - Complete experimental design and methodology

---

## 🎯 Key Findings Summary

### Run Results:

| Run | Dataset | Classifier | Sensitivity | Token Savings | Accuracy |
|-----|---------|-----------|-------------|---------------|----------|
| **Run 1** | ToxicChat | Heuristic | 91.4% | 64% @ 20th %ile | 9% (misleading) |
| **Run 2** | WildGuardMix | Heuristic | 92.2% | 64% @ 20th %ile | 41.5% |
| **Run 3** | ToxicChat | WildGuard | 92.1% | 63% @ 20th %ile | 10% (misleading) |
| **Run 4** | WildGuardMix | WildGuard | - | - | **56.3%** ✅ |

**Note on "Accuracy":**
- ToxicChat accuracy (9-10%) is MISLEADING - we don't have refusal labels, only toxic/safe input labels
- **Focus on SENSITIVITY** - we catch 91-94% of toxic prompts ✅
- **Only WildGuardMix accuracy (41.5% → 56.3%) is valid** - has gold-standard refusal labels

### Critical Discoveries:

1. **Confidence Paradox** - Incorrect predictions are 25% MORE confident than correct ones
2. **WildGuard improves +14.8%** over heuristics (when measured properly on WildGuardMix)
3. **Token savings: 64%** at 20th percentile while maintaining 91-92% sensitivity
4. **Sensitivity is king** - For safety, catching toxic content matters more than accuracy

---

## 📂 File Sizes

- Total deliverables: ~12 MB
- Largest files: Viewer notebooks (pre-executed outputs with embedded images)
- Smallest files: Reproducible notebooks (code only, no outputs)

---

## 🔗 Related Files (Not in Deliverables)

For complete reproducibility, these files are in the parent directory:
- `../results/` - Raw prediction files and analysis JSONs
- `../data/` - ToxicChat and WildGuardMix test datasets
- `../src/` - Source code for DeepConf implementation
- `../scripts/` - Helper scripts for running experiments

---

## 📧 Questions?

See `COMPREHENSIVE_REPORT.md` for detailed explanations of:
- What percentile thresholds mean (ELI5 section)
- Why accuracy is misleading for ToxicChat
- How the 3-10 trace generation process works
- Recommendations for production safety systems

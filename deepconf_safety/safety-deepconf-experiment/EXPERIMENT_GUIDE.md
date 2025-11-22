# DeepConf Safety Experiment Guide

## Quick Start

### 1. Run Individual Experiments (Manual)

#### Baseline Experiment
```bash
# On Lambda GPU instance
cd ~/deepconf_safety/safety-deepconf-experiment
source ~/venv_deepconf/bin/activate

# Set environment
export PYTHONPATH=~/deepconf_safety/deepconf_adapter:~/deepconf_safety/safety-deepconf-experiment/src:$PYTHONPATH
export HF_TOKEN="your_hf_token_here"

# Run baseline (1000 instances for statistical significance)
python3 run_experiment.py \
    --model Qwen/Qwen3-0.6B \
    --benchmark toxicchat \
    --num-instances 1000 \
    --min-traces 3 \
    --max-traces 10 \
    --output results/baseline_toxicchat_1000
```

**Expected outcome:**
- Runtime: ~45-50 minutes
- Cost: ~$1.00
- Output: `results/baseline_toxicchat_1000/analysis.json`

---

#### WildGuardMix Validation (Gold-Standard Refusal Labels)
```bash
python3 run_experiment.py \
    --model Qwen/Qwen3-0.6B \
    --benchmark wildguardmix \
    --min-traces 3 \
    --max-traces 10 \
    --output results/wildguardmix_validation
```

**Expected outcome:**
- Runtime: ~90 minutes (1,725 instances)
- Cost: ~$2.00
- **Critical metric**: Refusal detection precision/recall on gold labels

---

#### Model Size Comparison
```bash
# Test 1.7B model
python3 run_experiment.py \
    --model Qwen/Qwen3-1.7B \
    --benchmark toxicchat \
    --num-instances 500 \
    --min-traces 3 \
    --max-traces 10 \
    --output results/qwen3_1.7b_toxicchat

# Test 4B model (if budget allows)
python3 run_experiment.py \
    --model Qwen/Qwen3-4B \
    --benchmark toxicchat \
    --num-instances 500 \
    --min-traces 3 \
    --max-traces 10 \
    --output results/qwen3_4b_toxicchat
```

**Expected outcome:**
- Test hypothesis generalization across model scales
- Compare accuracy vs efficiency trade-offs

---

### 2. Analyze Results (After Each Experiment)

```bash
# Parse results and get decision flags
python3 parse_results.py results/baseline_toxicchat_1000

# Output will show:
# - Overall accuracy
# - Hypothesis test results
# - Uncertain compliance accuracy (KEY METRIC!)
# - Token savings percentage
# - Recommended next actions
```

**Key Decision Logic:**

```bash
# Get decisions as JSON (for scripting)
DECISIONS=$(python3 parse_results.py results/baseline_toxicchat_1000 --json-only)

# Extract specific decision flags
HYPOTHESIS_SUPPORTED=$(echo "$DECISIONS" | python3 -c "import sys, json; data=json.load(sys.stdin); print(data['decisions']['hypothesis_supported'])")
RUN_WILDGUARDMIX=$(echo "$DECISIONS" | python3 -c "import sys, json; data=json.load(sys.stdin); print(data['decisions']['run_wildguardmix'])")
```

---

### 3. Conditional Experiment Logic

#### Decision Tree

```
IF uncertain_compliance_accuracy < 0.40:
    ✓ Hypothesis SUPPORTED

    IF token_savings > 30%:
        ✓ Efficient approach
        → Run WildGuardMix validation
        → Run model size comparison (1.7B, 4B)
        → Ready for publication

    ELSE:
        ⚠ Needs optimization
        → Increase max_traces to 15
        → Test different percentiles

ELSE IF uncertain_compliance_accuracy > 0.70:
    ✗ Hypothesis NOT supported
    → Run WildGuardMix to check refusal detection quality
    → Investigate calibration issues
    → May need to revise approach
```

#### Manual Conditional Execution

```bash
# 1. Run baseline
python3 run_experiment.py --model Qwen/Qwen3-0.6B --benchmark toxicchat \
    --num-instances 1000 --output results/baseline

# 2. Analyze and decide
python3 parse_results.py results/baseline

# Read the "Recommended Action" from output:
#   - "validate_generalization" → Run WildGuardMix + model comparison
#   - "optimize_efficiency" → Adjust parameters and re-run
#   - "investigate_calibration" → Run WildGuardMix to check refusal detection

# 3. Execute recommended experiments based on output
```

---

### 4. Automated Pipeline (Full Orchestrator)

#### Run Complete Automated Pipeline

```bash
# On Lambda GPU instance
cd ~/deepconf_safety/safety-deepconf-experiment
source ~/venv_deepconf/bin/activate

# Launch orchestrator (runs in background)
nohup ./orchestrator.sh > pipeline.log 2>&1 &

# Monitor progress
tail -f pipeline.log

# Or monitor from local machine via SSH
ssh -i ~/.ssh/lambda_gpu ubuntu@144.24.121.54 'tail -f ~/deepconf_safety/safety-deepconf-experiment/pipeline.log'
```

**What the orchestrator does:**

1. **Checks for baseline** - If not found, runs baseline experiment
2. **Analyzes baseline** - Parses results using `parse_results.py`
3. **Makes conditional decisions:**
   - Runs WildGuardMix if `run_wildguardmix == true`
   - Runs model comparison if `run_model_comparison == true` AND `is_efficient == true`
   - Skips percentile sweep (requires code modification)
4. **Monitors GPU** - Waits for experiments to finish before starting new ones
5. **Tracks budget** - Logs cost at each stage
6. **Generates summary** - Final report with all results

**Expected total runtime:**
- Baseline: ~45 min ($1.00)
- WildGuardMix: ~90 min ($2.00)
- Model comparison (1.7B + 4B): ~85 min ($2.00)
- **Total: ~3.5 hours, ~$5.00**

---

### 5. Stop/Resume Logic

#### Check if experiment is running
```bash
# On Lambda
pgrep -f "run_experiment.py" && echo "Running" || echo "Not running"

# See which experiment
ps aux | grep run_experiment.py
```

#### Wait for current experiment to finish
```bash
# Built-in to orchestrator - automatically waits
# Manual wait:
while pgrep -f "run_experiment.py" > /dev/null; do
    echo "Waiting for experiment to finish..."
    sleep 60
done
echo "GPU available!"
```

#### Resume orchestrator after interruption
```bash
# Orchestrator is idempotent - checks for existing results
# Just re-run:
./orchestrator.sh

# It will:
# 1. Find existing baseline results
# 2. Parse them
# 3. Continue with next experiments
```

---

### 6. Download Results

#### From local machine
```bash
# Download specific experiment
scp -i ~/.ssh/lambda_gpu -r \
    ubuntu@144.24.121.54:~/deepconf_safety/safety-deepconf-experiment/results/baseline_toxicchat_1000 \
    ~/results/

# Download all results
scp -i ~/.ssh/lambda_gpu -r \
    ubuntu@144.24.121.54:~/deepconf_safety/safety-deepconf-experiment/results/ \
    ~/deepconf_results/

# Download pipeline log
scp -i ~/.ssh/lambda_gpu \
    ubuntu@144.24.121.54:~/deepconf_safety/safety-deepconf-experiment/pipeline.log \
    ~/
```

---

### 7. Generate Publication Figures

```bash
# After downloading results
cd deepconf_safety/safety-deepconf-experiment

# Generate figures for specific experiment
python3 create_publication_figures.py results/baseline_toxicchat_1000

# Outputs:
# - confidence_distribution.png (4-bucket histogram)
# - token_savings.png (efficiency bar chart)
# - hypothesis_validation.png (accuracy comparison)
```

---

## Experiment Suite Reference

### Available Experiments (from experiment_suite.json)

| Experiment | Priority | Conditions | Duration | Cost |
|------------|----------|------------|----------|------|
| baseline_toxicchat_1000 | 1 | Always run | 45 min | $1.00 |
| wildguardmix_validation | 2 | hypothesis_supported | 90 min | $2.00 |
| percentile_70/80/90/95 | 3 | run_percentile_sweep | 25 min ea. | $0.50 ea. |
| qwen3_1.7b_toxicchat | 4 | run_model_comparison | 35 min | $0.75 |
| qwen3_4b_toxicchat | 4 | run_model_comparison + is_efficient | 50 min | $1.10 |
| qwen3_8b_toxicchat | 5 | Budget-sensitive | 70 min | $1.50 |

### Decision Metrics

**Key metrics from parse_results.py:**

```python
{
  "hypothesis_supported": bool,      # uc_accuracy < 0.40 and difference > 0.30
  "is_efficient": bool,              # token_savings > 30%
  "run_wildguardmix": bool,          # Validate refusal detection
  "run_percentile_sweep": bool,      # Test threshold sensitivity
  "run_model_comparison": bool,      # Test generalization
  "investigate_calibration": bool,   # Hypothesis not supported
  "recommended_action": str          # Human-readable next step
}
```

---

## Advanced Usage

### Modify Percentile (Requires Code Change)

Currently, percentile is hardcoded to 90 in `safety_deepconf.py:75`. To test different percentiles:

```python
# Edit src/safety_deepconf.py
def __init__(
    self,
    ...
    confidence_percentile: int = 70,  # Change from 90 to 70/80/95
    ...
)
```

Then run:
```bash
python3 run_experiment.py --model Qwen/Qwen3-0.6B --benchmark toxicchat \
    --num-instances 500 --output results/percentile_70
```

### Test Different Confidence Methods

Edit `src/safety_deepconf.py:322` to change confidence calculation:

```python
# Current (DeepConf preferred):
confidences = [compute_trace_confidence(t, method='bottom_window') for t in traces]

# Alternative methods:
# method='neg_avg_logprob'  # Simple mean
# method='tail_confidence'  # Last N tokens
# method='min_window'       # Most conservative
# method='entropy'          # Token-level entropy
```

---

## Troubleshooting

### GPU Memory Issues
```bash
# Check GPU usage
nvidia-smi

# If OOM, reduce batch size or use smaller model
python3 run_experiment.py --model Qwen/Qwen3-0.6B --no-batch ...
```

### Experiment Stuck
```bash
# Check process
ps aux | grep run_experiment

# Kill and restart
pkill -f run_experiment.py
# Then re-run experiment
```

### Missing Dependencies
```bash
# Reinstall in venv
source ~/venv_deepconf/bin/activate
pip install vllm transformers datasets accelerate
```

---

## Quick Reference: Common Commands

```bash
# Check experiment status
pgrep -f "run_experiment.py" && grep "Processing instance" experiment.log | tail -1

# Parse latest results
python3 parse_results.py results/$(ls -t results/ | head -1)

# Monitor GPU in real-time
watch -n 2 nvidia-smi

# Check budget spent
echo "Scale=2; ($(date +%s) - START_TIME) / 3600 * 1.29" | bc

# Download and analyze
scp -i ~/.ssh/lambda_gpu -r ubuntu@IP:~/deepconf_safety/safety-deepconf-experiment/results/baseline_toxicchat_1000 ~/results/
python3 parse_results.py ~/results/baseline_toxicchat_1000
python3 create_publication_figures.py ~/results/baseline_toxicchat_1000
```

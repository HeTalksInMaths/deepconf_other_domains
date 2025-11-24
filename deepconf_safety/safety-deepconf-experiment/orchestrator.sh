#!/bin/bash
# DeepConf Safety Experiment Orchestrator
# Automated pipeline with conditional logic and GPU monitoring

set -e  # Exit on error (but we'll handle errors explicitly)

# ============================================================
# CONFIGURATION
# ============================================================

# Activate virtual environment
source ~/venv_deepconf/bin/activate

# Set paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

export PYTHONPATH="$HOME/deepconf_safety/deepconf_adapter:$SCRIPT_DIR/src:$PYTHONPATH"
# Load HF_TOKEN from .env file
[ -f .env ] && source .env || echo "Warning: .env file not found"

# Logging
PIPELINE_LOG="pipeline.log"
SUMMARY_JSON="results/pipeline_summary.json"

# GPU monitoring
MAX_WAIT_ITERATIONS=240  # 4 hours max wait (240 * 60s = 4h)
WAIT_INTERVAL=60  # Check every 60 seconds

# Budget tracking
HOURLY_RATE=1.29
START_TIME=$(date +%s)

# ============================================================
# HELPER FUNCTIONS
# ============================================================

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$PIPELINE_LOG"
}

log_section() {
    echo "" | tee -a "$PIPELINE_LOG"
    echo "============================================================" | tee -a "$PIPELINE_LOG"
    echo "$*" | tee -a "$PIPELINE_LOG"
    echo "============================================================" | tee -a "$PIPELINE_LOG"
}

check_gpu_available() {
    # Check if any experiment is currently running
    if pgrep -f "run_experiment.py" > /dev/null; then
        return 1  # GPU busy
    else
        return 0  # GPU available
    fi
}

wait_for_gpu() {
    local experiment_name="$1"
    local wait_count=0

    log "Checking GPU availability for: $experiment_name"

    while ! check_gpu_available; do
        wait_count=$((wait_count + 1))

        if [ $wait_count -ge $MAX_WAIT_ITERATIONS ]; then
            log "ERROR: Timeout waiting for GPU (waited 4 hours)"
            return 1
        fi

        if [ $((wait_count % 5)) -eq 0 ]; then
            # Log every 5 minutes
            log "  Still waiting for GPU... (${wait_count} minutes elapsed)"
        fi

        sleep $WAIT_INTERVAL
    done

    log "  GPU available! Starting: $experiment_name"
    return 0
}

run_experiment() {
    local exp_name="$1"
    local exp_command="$2"
    local exp_description="$3"

    log_section "RUNNING: $exp_name"
    log "Description: $exp_description"
    log "Command: $exp_command"

    # Wait for GPU
    if ! wait_for_gpu "$exp_name"; then
        log "ERROR: Could not acquire GPU for $exp_name"
        return 1
    fi

    # Record start time
    local exp_start=$(date +%s)

    # Run experiment
    log "Executing..."
    if eval "$exp_command" 2>&1 | tee -a "$PIPELINE_LOG"; then
        local exp_end=$(date +%s)
        local exp_duration=$((exp_end - exp_start))
        local exp_duration_min=$((exp_duration / 60))

        log "✓ SUCCESS: $exp_name completed in ${exp_duration_min} minutes"
        return 0
    else
        log "✗ FAILED: $exp_name"
        return 1
    fi
}

parse_results() {
    local results_dir="$1"

    if [ ! -f "$results_dir/analysis.json" ]; then
        log "WARNING: No analysis.json found in $results_dir"
        echo "{\"status\": \"no_results\", \"decisions\": {}}"
        return 1
    fi

    # Parse results and extract decisions
    python3 parse_results.py "$results_dir" --json-only 2>/dev/null || echo "{\"status\": \"parse_error\", \"decisions\": {}}"
}

check_condition() {
    local decisions_json="$1"
    local metric="$2"
    local expected_value="$3"

    # Extract metric value from JSON
    local actual_value=$(echo "$decisions_json" | python3 -c "import sys, json; data=json.load(sys.stdin); print(data.get('decisions', {}).get('$metric', 'false'))")

    if [ "$actual_value" = "$expected_value" ]; then
        return 0  # Condition met
    else
        return 1  # Condition not met
    fi
}

calculate_budget_used() {
    local current_time=$(date +%s)
    local elapsed_seconds=$((current_time - START_TIME))
    local elapsed_hours=$(echo "scale=2; $elapsed_seconds / 3600" | bc)
    local cost=$(echo "scale=2; $elapsed_hours * $HOURLY_RATE" | bc)
    echo "$cost"
}

log_budget() {
    local cost=$(calculate_budget_used)
    local elapsed_seconds=$(($(date +%s) - START_TIME))
    local elapsed_hours=$(echo "scale=2; $elapsed_seconds / 3600" | bc)

    log "Budget Update: \$${cost} spent (${elapsed_hours} hours @ \$${HOURLY_RATE}/hr)"
}

# ============================================================
# EXPERIMENT PIPELINE
# ============================================================

log_section "DeepConf Safety Experiment Pipeline - STARTED"
log "Start time: $(date)"
log "Working directory: $SCRIPT_DIR"

# Initialize results tracking
mkdir -p results
echo "{\"experiments\": [], \"start_time\": \"$(date -Iseconds)\"}" > "$SUMMARY_JSON"

# ============================================================
# PHASE 1: BASELINE
# ============================================================

log_section "PHASE 1: BASELINE ESTABLISHMENT"

# Check if baseline is already running/complete
if [ -f "results/baseline_toxicchat_1000/analysis.json" ]; then
    log "Baseline already complete, parsing results..."
    BASELINE_DECISIONS=$(parse_results "results/baseline_toxicchat_1000")
else
    log "Waiting for baseline experiment to complete..."
    log "Expected location: results/baseline_toxicchat_1000/"

    # Wait for the current running experiment to finish
    wait_for_gpu "baseline_completion_check"

    # Check again
    if [ -f "results/baseline_toxicchat_1000/analysis.json" ]; then
        log "Baseline complete!"
        BASELINE_DECISIONS=$(parse_results "results/baseline_toxicchat_1000")
    elif [ -f "results/toxicchat_qwen06b_1000_vllm/analysis.json" ]; then
        log "Found results in alternative location: results/toxicchat_qwen06b_1000_vllm/"
        BASELINE_DECISIONS=$(parse_results "results/toxicchat_qwen06b_1000_vllm")
    else
        log "WARNING: No baseline results found. Running baseline experiment..."
        run_experiment \
            "baseline_toxicchat_1000" \
            "python3 run_experiment.py --model Qwen/Qwen3-0.6B --benchmark toxicchat --num-instances 1000 --min-traces 3 --max-traces 10 --output results/baseline_toxicchat_1000" \
            "Baseline: Qwen3-0.6B on ToxicChat 1000 instances"

        BASELINE_DECISIONS=$(parse_results "results/baseline_toxicchat_1000")
    fi
fi

# Display baseline results
echo "$BASELINE_DECISIONS" | python3 parse_results.py results/baseline_toxicchat_1000 2>/dev/null || true

log_budget

# ============================================================
# PHASE 2: CONDITIONAL EXPERIMENTS
# ============================================================

log_section "PHASE 2: CONDITIONAL DECISION MAKING"

# Extract decision flags
HYPOTHESIS_SUPPORTED=$(echo "$BASELINE_DECISIONS" | python3 -c "import sys, json; data=json.load(sys.stdin); print(str(data.get('decisions', {}).get('hypothesis_supported', False)).lower())")
RUN_WILDGUARDMIX=$(echo "$BASELINE_DECISIONS" | python3 -c "import sys, json; data=json.load(sys.stdin); print(str(data.get('decisions', {}).get('run_wildguardmix', False)).lower())")
RUN_PERCENTILE_SWEEP=$(echo "$BASELINE_DECISIONS" | python3 -c "import sys, json; data=json.load(sys.stdin); print(str(data.get('decisions', {}).get('run_percentile_sweep', False)).lower())")
RUN_MODEL_COMPARISON=$(echo "$BASELINE_DECISIONS" | python3 -c "import sys, json; data=json.load(sys.stdin); print(str(data.get('decisions', {}).get('run_model_comparison', False)).lower())")
IS_EFFICIENT=$(echo "$BASELINE_DECISIONS" | python3 -c "import sys, json; data=json.load(sys.stdin); print(str(data.get('decisions', {}).get('is_efficient', False)).lower())")

log "Decision Flags:"
log "  Hypothesis Supported:   $HYPOTHESIS_SUPPORTED"
log "  Run WildGuardMix:       $RUN_WILDGUARDMIX"
log "  Run Percentile Sweep:   $RUN_PERCENTILE_SWEEP"
log "  Run Model Comparison:   $RUN_MODEL_COMPARISON"
log "  Is Efficient:           $IS_EFFICIENT"

# ============================================================
# VALIDATION: WildGuardMix
# ============================================================

if [ "$RUN_WILDGUARDMIX" = "true" ]; then
    log_section "VALIDATION: WildGuardMix (Gold-Standard Refusal Labels)"

    run_experiment \
        "wildguardmix_validation" \
        "python3 run_experiment.py --model Qwen/Qwen3-0.6B --benchmark wildguardmix --min-traces 3 --max-traces 10 --output results/wildguardmix_qwen06b" \
        "Validate refusal detection against gold-standard labels"

    WILDGUARD_DECISIONS=$(parse_results "results/wildguardmix_qwen06b")
    echo "$WILDGUARD_DECISIONS" | python3 parse_results.py results/wildguardmix_qwen06b 2>/dev/null || true

    log_budget
else
    log "Skipping WildGuardMix validation (condition not met)"
fi

# ============================================================
# OPTIMIZATION: Percentile Sweep
# ============================================================

if [ "$RUN_PERCENTILE_SWEEP" = "true" ]; then
    log_section "OPTIMIZATION: Percentile Sweep"

    # Note: Current implementation doesn't support dynamic percentile via CLI
    # This would require modifying run_experiment.py or safety_deepconf.py
    log "NOTE: Percentile sweep would require code modification to support dynamic percentile configuration"
    log "Skipping for now - recommend manual testing with different percentile values"

    # Placeholder for future implementation:
    # for PERCENTILE in 70 80 90 95; do
    #     run_experiment \
    #         "percentile_${PERCENTILE}" \
    #         "python3 run_experiment.py --model Qwen/Qwen3-0.6B --benchmark toxicchat --num-instances 500 --percentile ${PERCENTILE} --output results/percentile_${PERCENTILE}" \
    #         "Test percentile ${PERCENTILE} threshold"
    # done
else
    log "Skipping percentile sweep (condition not met)"
fi

# ============================================================
# GENERALIZATION: Model Size Comparison
# ============================================================

if [ "$RUN_MODEL_COMPARISON" = "true" ] && [ "$IS_EFFICIENT" = "true" ]; then
    log_section "GENERALIZATION: Model Size Comparison"

    # Run 1.7B
    run_experiment \
        "qwen3_1.7b_toxicchat" \
        "python3 run_experiment.py --model Qwen/Qwen3-1.7B --benchmark toxicchat --num-instances 500 --min-traces 3 --max-traces 10 --output results/qwen3_1.7b_toxicchat" \
        "Test Qwen3-1.7B for hypothesis generalization"

    RESULTS_1_7B=$(parse_results "results/qwen3_1.7b_toxicchat")
    echo "$RESULTS_1_7B" | python3 parse_results.py results/qwen3_1.7b_toxicchat 2>/dev/null || true

    log_budget

    # Check budget before running 4B
    COST_SO_FAR=$(calculate_budget_used)
    if (( $(echo "$COST_SO_FAR < 15" | bc -l) )); then
        log "Budget allows for 4B model test (spent: \$${COST_SO_FAR})"

        run_experiment \
            "qwen3_4b_toxicchat" \
            "python3 run_experiment.py --model Qwen/Qwen3-4B --benchmark toxicchat --num-instances 500 --min-traces 3 --max-traces 10 --output results/qwen3_4b_toxicchat" \
            "Test Qwen3-4B for hypothesis generalization"

        RESULTS_4B=$(parse_results "results/qwen3_4b_toxicchat")
        echo "$RESULTS_4B" | python3 parse_results.py results/qwen3_4b_toxicchat 2>/dev/null || true

        log_budget
    else
        log "Skipping 4B model test (budget constraint: \$${COST_SO_FAR} spent)"
    fi
else
    log "Skipping model comparison (conditions not met: RUN_MODEL_COMPARISON=$RUN_MODEL_COMPARISON, IS_EFFICIENT=$IS_EFFICIENT)"
fi

# ============================================================
# FINAL SUMMARY
# ============================================================

log_section "PIPELINE COMPLETE"

END_TIME=$(date +%s)
TOTAL_DURATION=$((END_TIME - START_TIME))
TOTAL_HOURS=$(echo "scale=2; $TOTAL_DURATION / 3600" | bc)
TOTAL_COST=$(calculate_budget_used)

log "End time: $(date)"
log "Total duration: ${TOTAL_HOURS} hours"
log "Total cost: \$${TOTAL_COST}"

log ""
log "Results saved in: $SCRIPT_DIR/results/"
log "Pipeline log: $PIPELINE_LOG"
log ""
log "Next steps:"
log "  1. Review results with: python3 parse_results.py results/<experiment_name>/"
log "  2. Generate figures: python3 create_publication_figures.py results/<experiment_name>/"
log "  3. Download results: scp -i ~/.ssh/lambda_gpu -r ubuntu@IP:~/deepconf_safety/safety-deepconf-experiment/results/ ~/"
log "  4. Terminate Lambda instance to stop billing"
log ""
log "Thank you for using the DeepConf Safety Experiment Pipeline!"

#!/bin/bash
#
# Run WildGuard classification on ToxicChat after WildGuardMix experiment completes
#
# This script:
# 1. Monitors the WildGuardMix experiment
# 2. When complete, runs WildGuard classifier on ToxicChat results
# 3. Logs everything for debugging
#
# Usage:
#   bash scripts/run_wildguard_after_experiment.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_DIR"

LOG_FILE="wildguard_automation.log"
WILDGUARDMIX_RESULT_DIR="results/wildguardmix_qwen06b_vllm"
TOXICCHAT_RESULT_DIR="results/toxicchat_qwen06b_1000_vllm_reclassified"

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG_FILE"
}

check_experiment_running() {
    # Check if run_experiment.py is running
    if pgrep -f "run_experiment.py.*wildguardmix" > /dev/null; then
        return 0  # Running
    else
        return 1  # Not running
    fi
}

wait_for_experiment() {
    log "Waiting for WildGuardMix experiment to complete..."

    # Wait while experiment is running
    while check_experiment_running; do
        log "  Experiment still running, checking again in 5 minutes..."
        sleep 300  # Check every 5 minutes
    done

    log "✓ WildGuardMix experiment has completed!"

    # Wait an extra 30 seconds to ensure clean shutdown
    sleep 30
}

run_wildguard_classification() {
    log "Starting WildGuard classification on ToxicChat results..."

    # Check if GPU is available
    if command -v nvidia-smi &> /dev/null; then
        DEVICE="cuda"
        BATCH_SIZE=64
        log "  GPU detected, using CUDA with batch_size=$BATCH_SIZE"
    else
        DEVICE="cpu"
        BATCH_SIZE=16
        log "  No GPU detected, using CPU with batch_size=$BATCH_SIZE"
    fi

    # Run classification
    python3 classify_toxicchat_wildguard.py \
        --results-dir "$TOXICCHAT_RESULT_DIR" \
        --device "$DEVICE" \
        --batch-size "$BATCH_SIZE" \
        --load-in-8bit \
        --cache-dir .wildguard_cache \
        2>&1 | tee -a "$LOG_FILE"

    if [ $? -eq 0 ]; then
        log "✓ WildGuard classification completed successfully!"
        return 0
    else
        log "✗ WildGuard classification failed!"
        return 1
    fi
}

send_completion_signal() {
    local status=$1

    # Create a completion marker file
    if [ "$status" -eq 0 ]; then
        touch .wildguard_complete
        log "Created completion marker: .wildguard_complete"
    else
        touch .wildguard_failed
        log "Created failure marker: .wildguard_failed"
    fi
}

main() {
    log "========================================"
    log "WildGuard Automation Script Started"
    log "========================================"
    log "Project directory: $PROJECT_DIR"
    log "Log file: $LOG_FILE"
    log ""

    # Check if experiment is currently running
    if check_experiment_running; then
        wait_for_experiment
    else
        log "No WildGuardMix experiment currently running."
        log "Checking if experiment has already completed..."

        # Check if results exist
        if [ -f "$WILDGUARDMIX_RESULT_DIR/predictions.jsonl" ]; then
            log "Found existing WildGuardMix results, proceeding with classification."
        else
            log "No WildGuardMix results found. Exiting."
            log "Start the experiment first with: bash scripts/run_wildguardmix_gpu.sh"
            exit 1
        fi
    fi

    log ""
    log "Starting WildGuard classification phase..."
    log ""

    # Run WildGuard classification
    if run_wildguard_classification; then
        send_completion_signal 0
        log ""
        log "========================================"
        log "All tasks completed successfully!"
        log "========================================"
        exit 0
    else
        send_completion_signal 1
        log ""
        log "========================================"
        log "Classification failed!"
        log "========================================"
        exit 1
    fi
}

main "$@"

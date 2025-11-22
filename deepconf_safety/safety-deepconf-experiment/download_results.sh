#!/bin/bash
# Download experiment results from Lambda GPU instance

set -e  # Exit on error

LAMBDA_HOST="ubuntu@144.24.121.54"
SSH_KEY="$HOME/.ssh/lambda_gpu"
REMOTE_BASE="~/deepconf_safety/safety-deepconf-experiment/results"
RESULTS_DIR="toxicchat_qwen06b_1000_vllm"  # Actual directory name on Lambda
LOCAL_BASE="./results"

echo "Downloading ToxicChat experiment results from Lambda..."
echo ""

# Create local results directory
mkdir -p "$LOCAL_BASE/$RESULTS_DIR"

# Download the main results directory
echo "[1/3] Downloading predictions.jsonl (~50K traces, may take 30-60 seconds)..."
scp -i "$SSH_KEY" \
    "$LAMBDA_HOST:$REMOTE_BASE/$RESULTS_DIR/predictions.jsonl" \
    "$LOCAL_BASE/$RESULTS_DIR/"

echo "[2/3] Downloading analysis.json (original metrics)..."
scp -i "$SSH_KEY" \
    "$LAMBDA_HOST:$REMOTE_BASE/$RESULTS_DIR/analysis.json" \
    "$LOCAL_BASE/$RESULTS_DIR/" 2>/dev/null || echo "  (analysis.json not found, skipping)"

echo "[3/3] Downloading experiment.log (if available)..."
scp -i "$SSH_KEY" \
    "$LAMBDA_HOST:$REMOTE_BASE/$RESULTS_DIR/experiment.log" \
    "$LOCAL_BASE/$RESULTS_DIR/" 2>/dev/null || echo "  (experiment.log not found, skipping)"

echo ""
echo "✓ Download complete!"
echo ""
echo "Files saved to: $LOCAL_BASE/$RESULTS_DIR/"
ls -lh "$LOCAL_BASE/$RESULTS_DIR/"

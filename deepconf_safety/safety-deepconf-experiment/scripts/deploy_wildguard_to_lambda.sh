#!/bin/bash
#
# Deploy WildGuard classification scripts to Lambda and schedule execution
#
# Usage:
#   bash scripts/deploy_wildguard_to_lambda.sh

set -e

LAMBDA_IP="144.24.121.54"
LAMBDA_USER="ubuntu"
LAMBDA_KEY="$HOME/.ssh/lambda_gpu"
REMOTE_DIR="~/deepconf_safety/safety-deepconf-experiment"

echo "========================================"
echo "Deploying WildGuard Scripts to Lambda"
echo "========================================"
echo "Lambda IP: $LAMBDA_IP"
echo "Remote dir: $REMOTE_DIR"
echo ""

# Files to upload
FILES=(
    "src/wildguard_classifier_optimized.py"
    "classify_toxicchat_wildguard.py"
    "scripts/run_wildguard_after_experiment.sh"
)

echo "Uploading files..."
for file in "${FILES[@]}"; do
    echo "  - $file"
    scp -i "$LAMBDA_KEY" "$file" "${LAMBDA_USER}@${LAMBDA_IP}:${REMOTE_DIR}/${file}"
done

echo ""
echo "✓ Files uploaded successfully!"
echo ""

# Make scripts executable
echo "Setting executable permissions..."
ssh -i "$LAMBDA_KEY" "${LAMBDA_USER}@${LAMBDA_IP}" "cd ${REMOTE_DIR} && chmod +x classify_toxicchat_wildguard.py scripts/run_wildguard_after_experiment.sh"

echo "✓ Permissions set!"
echo ""

# Check current status
echo "Checking experiment status on Lambda..."
ssh -i "$LAMBDA_KEY" "${LAMBDA_USER}@${LAMBDA_IP}" << 'ENDSSH'
cd ~/deepconf_safety/safety-deepconf-experiment

echo ""
echo "Current processes:"
pgrep -a python3 | grep -i experiment || echo "  No experiments running"

echo ""
echo "WildGuardMix results:"
if [ -f "results/wildguardmix_qwen06b_vllm/predictions.jsonl" ]; then
    wc -l results/wildguardmix_qwen06b_vllm/predictions.jsonl
else
    echo "  Not found yet"
fi

echo ""
echo "ToxicChat results:"
if [ -f "results/toxicchat_qwen06b_1000_vllm_reclassified/predictions.jsonl" ]; then
    echo "  ✓ Found (ready for WildGuard classification)"
else
    echo "  ✗ Not found"
fi
ENDSSH

echo ""
echo "========================================"
echo "Deployment complete!"
echo "========================================"
echo ""
echo "Next steps:"
echo ""
echo "Option 1: Start WildGuard automation now (waits for experiment to finish)"
echo "  ssh -i $LAMBDA_KEY ${LAMBDA_USER}@${LAMBDA_IP}"
echo "  cd $REMOTE_DIR"
echo "  nohup bash scripts/run_wildguard_after_experiment.sh > wildguard.log 2>&1 &"
echo "  tail -f wildguard.log"
echo ""
echo "Option 2: Run WildGuard classification manually (if experiment already done)"
echo "  ssh -i $LAMBDA_KEY ${LAMBDA_USER}@${LAMBDA_IP}"
echo "  cd $REMOTE_DIR"
echo "  python3 classify_toxicchat_wildguard.py \\"
echo "    --results-dir results/toxicchat_qwen06b_1000_vllm_reclassified \\"
echo "    --device cuda --batch-size 64 --load-in-8bit"
echo ""
echo "Monitor progress:"
echo "  ssh -i $LAMBDA_KEY ${LAMBDA_USER}@${LAMBDA_IP} 'tail -f $REMOTE_DIR/wildguard_automation.log'"
echo ""

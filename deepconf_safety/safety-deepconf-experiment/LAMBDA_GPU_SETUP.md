# Lambda Labs GPU Setup Guide

Complete guide for running DeepConf Safety experiments on Lambda Labs A100 GPU.

**Why Lambda A100?**
- 4-5x faster than Kaggle 2x T4 setup
- Only $1.29/hr (~$62 for 48 hours)
- SSH access + JupyterLab pre-installed
- 40GB VRAM for large batch sizes

---

## Quick Start (5 Minutes)

```bash
# 1. Launch instance (via web console)
# 2. Upload code
export LAMBDA_IP="<your-instance-ip>"
rsync -avz ~/deepconf_other_domains/deepconf_safety/ ubuntu@$LAMBDA_IP:~/deepconf_safety/

# 3. Setup and run
ssh ubuntu@$LAMBDA_IP "cd deepconf_safety/safety-deepconf-experiment && \
    pip install transformers torch datasets accelerate && \
    export HF_TOKEN='YOUR_HF_TOKEN_HERE' && \
    python run_experiment.py --benchmark toxicchat --num-instances 1000"

# 4. Download results
rsync -avz ubuntu@$LAMBDA_IP:~/deepconf_safety/results/ ./lambda_results/

# 5. Terminate instance (via web console)
```

---

## Detailed Setup

### Step 1: Create Lambda Labs Account

1. Go to: https://lambda.ai/
2. Click "Sign Up"
3. Add payment method
4. **Apply your $400 credit** (if you have a promo code)

---

### Step 2: Generate SSH Key

```bash
# On your local machine
cd ~/.ssh
ssh-keygen -t rsa -b 4096 -f lambda_gpu

# This creates:
#   lambda_gpu       (private key - keep secret!)
#   lambda_gpu.pub   (public key - upload to Lambda)
```

---

### Step 3: Add SSH Key to Lambda

1. Go to: https://cloud.lambdalabs.com/ssh-keys
2. Click "Add SSH Key"
3. Paste contents of `~/.ssh/lambda_gpu.pub`
4. Name it (e.g., "macbook-pro")
5. Save

---

### Step 4: Launch A100 Instance

**Via Web Console (Easiest):**

1. Go to: https://cloud.lambdalabs.com/instances
2. Click "Launch Instance"
3. Select: **1x A100 (40GB)** - $1.29/hr
4. Choose region:
   - `us-west-3` (Portland, OR) - usually available
   - `us-south-1` (Austin, TX) - alternative
5. Select your SSH key
6. Click "Launch"
7. Wait 3-5 minutes for provisioning

**You'll see:**
- Instance status: "Active"
- IP address: `123.45.67.89` (example)
- SSH command: `ssh ubuntu@123.45.67.89`

---

### Step 5: Transfer Your Code

```bash
# Set instance IP
export LAMBDA_IP="123.45.67.89"  # Replace with your actual IP

# Upload entire project (recommended)
rsync -avz -e "ssh -i ~/.ssh/lambda_gpu" \
      ~/deepconf_other_domains/deepconf_safety/ \
      ubuntu@$LAMBDA_IP:~/deepconf_safety/

# Or just the experiment folder
rsync -avz -e "ssh -i ~/.ssh/lambda_gpu" \
      ~/deepconf_other_domains/deepconf_safety/safety-deepconf-experiment/ \
      ubuntu@$LAMBDA_IP:~/safety-deepconf-experiment/

# Or use SCP for specific files
scp -i ~/.ssh/lambda_gpu -r \
    ~/deepconf_other_domains/deepconf_safety/safety-deepconf-experiment \
    ubuntu@$LAMBDA_IP:~/
```

**What gets uploaded:**
- All Python scripts
- Configuration files
- Your adapter code
- Data loaders

**NOT uploaded (will download on Lambda):**
- Models (will download from HuggingFace)
- Datasets (will download when needed)

---

### Step 6: Setup Environment

```bash
# SSH into instance
ssh -i ~/.ssh/lambda_gpu ubuntu@$LAMBDA_IP

# You're now on the Lambda GPU!
# Check GPU
nvidia-smi

# You should see:
#   GPU 0: NVIDIA A100-SXM4-40GB
#   Memory: 40960 MiB

# Navigate to project
cd ~/deepconf_safety/safety-deepconf-experiment

# Install dependencies
pip install transformers torch datasets accelerate numpy pandas scipy tqdm matplotlib seaborn

# Set HuggingFace token (for WildGuardMix access)
export HF_TOKEN="YOUR_HF_TOKEN_HERE"

# Optional: Add to .bashrc for persistence
echo 'export HF_TOKEN="YOUR_HF_TOKEN_HERE"' >> ~/.bashrc
```

---

### Step 7: Pre-download Models and Data

```bash
# Download Qwen3-0.6B model (one-time, ~2GB)
python -c "
from transformers import AutoModelForCausalLM, AutoTokenizer
print('Downloading Qwen3-0.6B...')
AutoModelForCausalLM.from_pretrained('Qwen/Qwen3-0.6B')
AutoTokenizer.from_pretrained('Qwen/Qwen3-0.6B')
print('✓ Model cached!')
"

# Download benchmarks
python download_benchmarks.py

# Or download specific benchmarks
python -c "
from src.benchmark_loaders import ToxicChatLoader
loader = ToxicChatLoader()
loader.load(split='test')  # Downloads ToxicChat
"
```

---

### Step 8: Run Experiments

**Test Run (Quick validation):**

```bash
python run_experiment.py \
    --model Qwen/Qwen3-0.6B \
    --benchmark synthetic \
    --num-instances 20 \
    --output results/test_run

# Check output
cat results/test_run/analysis.json
```

**ToxicChat (1000 instances):**

```bash
python run_experiment.py \
    --model Qwen/Qwen3-0.6B \
    --benchmark toxicchat \
    --num-instances 1000 \
    --min-traces 3 \
    --max-traces 10 \
    --output results/toxicchat_1000

# Expected time on A100: ~3-5 minutes
# vs Kaggle 2x T4: ~10 minutes
```

**WildGuardMix (Full test set):**

```bash
# First, verify access
python test_wildguardmix_access.py

# If access granted, run full test
python run_experiment.py \
    --model Qwen/Qwen3-0.6B \
    --benchmark wildguardmix \
    --num-instances 1725 \
    --output results/wildguardmix_full

# Expected time on A100: ~8-10 minutes
```

**Model Comparison (All Qwen3 sizes):**

```bash
for model in "Qwen/Qwen3-0.6B" "Qwen/Qwen3-1.7B" "Qwen/Qwen3-4B" "Qwen/Qwen3-8B"; do
    python run_experiment.py \
        --model $model \
        --benchmark toxicchat \
        --num-instances 500 \
        --output results/toxicchat_${model##*/}
done

# Expected time: ~1 hour for all 4 models
# Cost: ~$1.30 on A100
```

---

### Step 9: Monitor GPU Usage

**In a separate terminal:**

```bash
# SSH into instance again
ssh -i ~/.ssh/lambda_gpu ubuntu@$LAMBDA_IP

# Watch GPU usage in real-time
watch -n 1 nvidia-smi

# You should see:
#   GPU Utilization: 80-100%
#   Memory Usage: 5-15 GB / 40 GB
#   Temperature: 50-70°C
```

**If GPU utilization is low (<50%):**
- Increase batch size in batch_generate()
- Check if CPU is bottleneck (use `htop`)
- Ensure `use_batch=True` in run_experiment.py

---

### Step 10: Download Results

**From your LOCAL machine:**

```bash
# Download all results
rsync -avz -e "ssh -i ~/.ssh/lambda_gpu" \
      ubuntu@$LAMBDA_IP:~/deepconf_safety/safety-deepconf-experiment/results/ \
      ~/lambda_results/

# Or specific files
scp -i ~/.ssh/lambda_gpu \
    ubuntu@$LAMBDA_IP:~/deepconf_safety/safety-deepconf-experiment/results/*.json \
    ~/lambda_results/

# Download specific experiment
rsync -avz -e "ssh -i ~/.ssh/lambda_gpu" \
      ubuntu@$LAMBDA_IP:~/deepconf_safety/results/toxicchat_1000/ \
      ~/lambda_results/toxicchat_1000/
```

---

### Step 11: Terminate Instance

**CRITICAL: Data is LOST when instance terminates!**

**Before terminating:**
1. ✅ Download all results
2. ✅ Verify files are on your local machine
3. ✅ Check you didn't miss any experiments

**Terminate via Web Console:**
1. Go to: https://cloud.lambdalabs.com/instances
2. Find your instance
3. Click "Terminate"
4. Confirm termination

**Verify termination:**
- Instance disappears from list
- Billing stops
- IP address is released

---

## Cost Tracking

**A100 40GB Pricing:**
- Per-minute billing
- $1.29/hr = $0.0215/minute

**Example costs:**
- 5 minutes (quick test): $0.11
- 1 hour (moderate work): $1.29
- 24 hours (full day): $30.96
- 48 hours (contest duration): $61.92

**Your $400 budget allows:**
- ~310 hours on A100
- ~1,550 full experiments (5 min each)
- ~50 model comparisons (1 hr each)

**Budget tracking:**
```bash
# Check current spend
# Go to: https://cloud.lambdalabs.com/billing

# Set calendar reminders
# Alert at: $100, $200, $300 spent
```

---

## Troubleshooting

### Issue 1: SSH Connection Refused

**Error:** `Connection refused` or `timeout`

**Solutions:**
1. Check instance is "Active" (not "Booting")
2. Verify IP address is correct
3. Ensure SSH key was selected during launch
4. Try different region (some have firewall issues)

**Test connection:**
```bash
ssh -v -i ~/.ssh/lambda_gpu ubuntu@$LAMBDA_IP
# Look for "Authentication successful" in output
```

### Issue 2: Out of Memory (OOM)

**Error:** `CUDA out of memory`

**Solutions:**
1. Reduce batch size in batch_generate()
2. Use gradient checkpointing
3. Clear GPU cache between runs:
   ```python
   import torch
   torch.cuda.empty_cache()
   ```

**Check memory usage:**
```bash
nvidia-smi --query-gpu=memory.used,memory.total --format=csv
```

### Issue 3: Slow Performance

**Symptoms:** GPU utilization <50%

**Solutions:**
1. Increase batch size (A100 can handle 2-4x larger batches than T4)
2. Ensure `use_batch=True` in run_experiment.py
3. Check CPU bottleneck: `htop` (should see all cores active)
4. Pre-download datasets (network I/O slowdown)

### Issue 4: Dataset Access Denied

**Error:** `403 Forbidden` for WildGuardMix

**Solutions:**
1. Run: `python test_wildguardmix_access.py`
2. Visit: https://huggingface.co/datasets/allenai/wildguardmix
3. Click "Request Access"
4. Wait for automatic approval (~1 minute)
5. Verify token is set: `echo $HF_TOKEN`

### Issue 5: Lost Results After Termination

**Prevention:**
- Always `rsync` results before terminating
- Use persistent storage option (adds cost)
- Set up auto-backup script

**If already lost:**
- Re-run experiments (Lambda is fast enough)
- Consider saving to cloud storage (S3, GCS)

---

## Advanced: Persistent Storage

**Problem:** Data is lost when instance terminates

**Solution:** Use Lambda's persistent filesystems

```bash
# Create persistent filesystem (one-time)
# Via web console: https://cloud.lambdalabs.com/file-systems
# Name: "deepconf-results"
# Size: 100 GB ($20/month)

# Mount on instance
# Automatically mounted at /home/ubuntu/deepconf-results

# Save results to persistent storage
python run_experiment.py \
    --output /home/ubuntu/deepconf-results/toxicchat_1000
```

**Cost:**
- Storage: $0.20/GB/month
- 100 GB = $20/month
- Pro: Survives instance termination
- Con: Additional monthly cost

---

## Alternative: Vast.ai RTX 4090

If Lambda A100 is unavailable or you want to save money:

**Vast.ai RTX 4090:**
- $0.35/hr (vs $1.29 for A100)
- 3.5-4x faster than 2x T4
- 48 hours = $16.80 (vs $61.92)

**Tradeoffs:**
- Less reliable (spot instances can be interrupted)
- More setup required
- Marketplace model (variable availability)

**Setup:**
1. Create account: https://vast.ai/
2. Search: RTX 4090, 24GB VRAM
3. Filter: "Verified" + "High reliability score"
4. Rent instance
5. SSH similar to Lambda

---

## Quick Reference

**Lambda A100 40GB:**
- **Price:** $1.29/hr
- **Speed:** 4-5x faster than 2x T4
- **Memory:** 40 GB
- **Best for:** Reliability, large batches
- **Budget:** $62 for 48 hours

**Expected Performance:**
- 100 instances: 12-15 seconds (vs 1 minute on 2x T4)
- 1000 instances: ~3-5 minutes (vs 10 minutes)
- Full WildGuardMix (1725): ~8-10 minutes

**Your $400 buys:**
- ~310 hours of A100 time
- ~1,550 full experiments
- Plenty for contest + future work!

---

## Resources

- **Lambda Docs:** https://docs.lambda.ai/
- **Lambda Console:** https://cloud.lambdalabs.com/
- **Lambda Pricing:** https://lambdalabs.com/service/gpu-cloud/pricing
- **Support:** support@lambdalabs.com
- **Community:** https://github.com/joehoover/lambda-cloud-manager (unofficial CLI)

---

## Summary

Lambda A100 is your best option for the contest:
- ✅ 4-5x speedup over current setup
- ✅ Only 15% of your $400 budget for 48 hours
- ✅ Easy SSH workflow
- ✅ Reliable and well-supported
- ✅ Can run hundreds of experiments

**Next steps:**
1. Launch A100 instance
2. Upload code with rsync
3. Run experiments
4. Download results
5. Terminate instance

Good luck with your contest! 🚀

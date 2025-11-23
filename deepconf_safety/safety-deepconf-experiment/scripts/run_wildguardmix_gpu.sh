#!/usr/bin/env bash
# Run the Qwen3-0.6B DeepConf experiment on the WildGuardMix dataset.
# Designed for Lambda GPU instances where ~/venv_deepconf exists.

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
LOG_DIR="${REPO_DIR}/logs"
ARTIFACT_DIR="${REPO_DIR}/lambda_artifacts"
OUTPUT_DIR="${REPO_DIR}/results/wildguardmix_qwen06b_baseline"
LOG_FILE="${LOG_DIR}/wildguardmix_qwen06b.log"
WATCH_SCRIPT="${ARTIFACT_DIR}/wildguardmix_watch.sh"

mkdir -p "${LOG_DIR}" "${ARTIFACT_DIR}"

# Activate venv if needed (default Lambda path)
if [[ -z "${VIRTUAL_ENV:-}" ]]; then
  if [[ -d "${HOME}/venv_deepconf/bin" ]]; then
    # shellcheck source=/dev/null
    source "${HOME}/venv_deepconf/bin/activate"
  else
    echo "✗ venv not active and ~/venv_deepconf not found. Activate your env first." >&2
    exit 1
  fi
fi

# Load HF token / other env vars if .env present
if [[ -f "${REPO_DIR}/.env" ]]; then
  set -a
  # shellcheck source=/dev/null
  source "${REPO_DIR}/.env"
  set +a
fi

cd "${REPO_DIR}"

echo "→ Checking WildGuardMix data..."
if [[ ! -f "data/wildguardmix/test.jsonl" ]]; then
  echo "   WildGuardMix not found locally. Downloading via download_wildguardmix.py"
  python3 download_wildguardmix.py
else
  echo "   WildGuardMix already present."
fi

timestamp() {
  date +"%Y-%m-%d %H:%M:%S"
}

echo "→ Starting Qwen3-0.6B WildGuardMix run (logs: ${LOG_FILE})"
{
  echo "[$(timestamp)] ============================================================"
  echo "[$(timestamp)] Launching WildGuardMix baseline run"
  echo "[$(timestamp)] Output dir: ${OUTPUT_DIR}"
} >> "${LOG_FILE}"

CMD=(python3 run_experiment.py
  --model Qwen/Qwen3-0.6B
  --benchmark wildguardmix
  --num-instances 1725
  --min-traces 3
  --max-traces 10
  --early-stopping
  --batch
  --output "${OUTPUT_DIR}"
)

nohup "${CMD[@]}" >> "${LOG_FILE}" 2>&1 &
RUN_PID=$!
echo "${RUN_PID}" > "${ARTIFACT_DIR}/wildguardmix_run.pid"
echo "   Started run_experiment.py (PID ${RUN_PID})."
echo "   Tail logs with: tail -f ${LOG_FILE}"

cat <<'EOF' > "${WATCH_SCRIPT}"
#!/usr/bin/env bash
set -euo pipefail
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RESULTS_DIR="${REPO_DIR}/results/wildguardmix_qwen06b_baseline"
TARGET_FILE="${RESULTS_DIR}/analysis.json"
echo "[watch] Waiting for ${TARGET_FILE} ..."
while [[ ! -f "${TARGET_FILE}" ]]; do
  sleep 60
done
echo "[watch] Baseline complete. Kick off post-processing if needed."
# Example follow-up (uncomment when ready):
# bash scripts/run_wildguardmix_wildguard.sh
EOF
chmod +x "${WATCH_SCRIPT}"

echo "→ Watcher script created at ${WATCH_SCRIPT}"
echo "   Run it in another shell to trigger downstream automation when the baseline finishes."

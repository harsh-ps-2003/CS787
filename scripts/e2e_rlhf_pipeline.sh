#!/bin/bash

# End-to-end pipeline:
# 1) Fine-tune Stable Diffusion UNet (scripts/train.sh)
# 2) Run RLHF training pipeline (policy + selector) via main.py
# 3) Generate images using the fine-tuned model via generate.py

set -euo pipefail

# Resolve important paths
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# -----------------------------
# Config (override via env)
# -----------------------------
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
DEVICE="${DEVICE:-cuda:0}"
UNET_OUTPUT_DIR="${UNET_OUTPUT_DIR:-${REPO_ROOT}/checkpoints/medical-model}"
RLHF_CONFIG_PATH="${RLHF_CONFIG_PATH:-${REPO_ROOT}/config.yaml}"
BASE_PROMPT="${BASE_PROMPT:-Chest X-ray: normal lung fields without infiltrates}"
NUM_IMAGES="${NUM_IMAGES:-3}"
E2E_OUTPUT_DIR="${E2E_OUTPUT_DIR:-${REPO_ROOT}/generated_e2e}"

echo "[E2E] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "[E2E] DEVICE=${DEVICE}"
echo "[E2E] REPO_ROOT=${REPO_ROOT}"
echo "[E2E] UNET_OUTPUT_DIR=${UNET_OUTPUT_DIR}"
echo "[E2E] RLHF_CONFIG_PATH=${RLHF_CONFIG_PATH}"
echo "[E2E] BASE_PROMPT=\"${BASE_PROMPT}\""
echo "[E2E] NUM_IMAGES=${NUM_IMAGES}"
echo "[E2E] E2E_OUTPUT_DIR=${E2E_OUTPUT_DIR}"

# Detect GPU availability and adjust DEVICE if needed
if command -v nvidia-smi >/dev/null 2>&1; then
  GPU_COUNT=$(nvidia-smi --list-gpus | wc -l | tr -d ' ')
  echo "[E2E] Detected ${GPU_COUNT} NVIDIA GPU(s)"
else
  echo "[E2E] No NVIDIA GPUs detected; forcing CPU"
  DEVICE="cpu"
fi

# Ensure directories
mkdir -p "${UNET_OUTPUT_DIR}" "${E2E_OUTPUT_DIR}"

step() {
  echo
  echo "==================== $1 ===================="
}

# Build a minimal example CSV from generated_* images if none exists
ensure_example_dataset() {
  local default_csv="${REPO_ROOT}/datasets/example/example_data.csv"
  if [ -f "${default_csv}" ]; then
    echo "[E2E] Found dataset CSV: ${default_csv}"
    echo "${default_csv}"
    return 0
  fi

  echo "[E2E] No example CSV found, attempting to create from generated_* images"
  mkdir -p "${REPO_ROOT}/datasets/example"
  local tmp_csv="${default_csv}.tmp"
  echo "path,Text,modality" > "${tmp_csv}"

  # Collect up to 50 images from generated_* directories
  local count=0
  shopt -s nullglob
  for img in "${REPO_ROOT}"/generated_*/*.{png,jpg,jpeg,JPG,JPEG,PNG}; do
    if [ $count -ge 50 ]; then break; fi
    # modality from parent directory name
    local parent
    parent=$(basename "$(dirname "${img}")")
    # Convert parent like generated_cxr_normal -> modality label
    local modality
    modality=$(echo "${parent#generated_}" | tr '_' ' ')
    local text
    text="${modality}: synthetic sample"
    # Escape commas in text if any
    echo "${img},${text},${modality}" >> "${tmp_csv}"
    count=$((count+1))
  done
  shopt -u nullglob

  if [ $count -eq 0 ]; then
    echo "[E2E] ERROR: Could not find any images under generated_* to seed example CSV" >&2
    echo ""  # return empty string
    return 1
  fi

  mv "${tmp_csv}" "${default_csv}"
  echo "[E2E] Wrote example dataset CSV with ${count} rows: ${default_csv}"
  echo "${default_csv}"
}

# -----------------------------
# 1) Fine-tune Stable Diffusion UNet
# -----------------------------
step "Step 1: Fine-tune Stable Diffusion UNet"
DATASET_CSV="$(ensure_example_dataset || true)"
if [ -z "${DATASET_CSV}" ]; then
  echo "[E2E] No dataset CSV available. Please place a CSV at datasets/example/example_data.csv and re-run." >&2
  exit 1
fi
echo "[E2E] Using dataset CSV: ${DATASET_CSV}"
pushd "${REPO_ROOT}" >/dev/null
echo "[E2E] Launching scripts/train.sh from repo root"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" DATASET_NAME="${DATASET_CSV}" UV_NO_SYNC=1 uv run bash ./scripts/train.sh
popd >/dev/null

# -----------------------------
# 2) RLHF training (policy + selector)
# -----------------------------
step "Step 2: RLHF Training (policy + selector)"
echo "[E2E] Running RLHF main.py with config ${RLHF_CONFIG_PATH}"
pushd "${REPO_ROOT}" >/dev/null
UV_NO_SYNC=1 uv run python ./main.py --config "${RLHF_CONFIG_PATH}" || {
  echo "[E2E] Falling back to default behavior (main.py reads ./config.yaml)"
  UV_NO_SYNC=1 uv run python ./main.py
}
popd >/dev/null

# -----------------------------
# 3) Generation using fine-tuned UNet
# -----------------------------
step "Step 3: Generate with fine-tuned UNet"
echo "[E2E] Generating ${NUM_IMAGES} images for prompt: ${BASE_PROMPT}"
pushd "${REPO_ROOT}" >/dev/null
UV_NO_SYNC=1 uv run python ./generate.py \
  --model_used "${UNET_OUTPUT_DIR}" \
  --prompt "${BASE_PROMPT}" \
  --device "${DEVICE}" \
  --img_num "${NUM_IMAGES}" \
  --num_inference_steps 50 \
  --output_dir "${E2E_OUTPUT_DIR}/finetuned"
popd >/dev/null

echo
echo "[E2E] All steps completed. Outputs under: ${E2E_OUTPUT_DIR}"



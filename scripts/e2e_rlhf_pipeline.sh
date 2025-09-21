#!/bin/bash

# End-to-end pipeline:
# 1) Fine-tune Stable Diffusion UNet (scripts/train.sh)
# 2) Run RLHF training pipeline (policy + selector) via main.py
# 3) Generate images using the fine-tuned model via generate.py

set -euo pipefail

# -----------------------------
# Config (override via env)
# -----------------------------
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
DEVICE="${DEVICE:-cuda:0}"
UNET_OUTPUT_DIR="${UNET_OUTPUT_DIR:-./checkpoints/medical-model}"
RLHF_CONFIG_PATH="${RLHF_CONFIG_PATH:-./config.yaml}"
BASE_PROMPT="${BASE_PROMPT:-Chest X-ray: normal lung fields without infiltrates}"
NUM_IMAGES="${NUM_IMAGES:-3}"
E2E_OUTPUT_DIR="${E2E_OUTPUT_DIR:-generated_e2e}"

echo "[E2E] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "[E2E] DEVICE=${DEVICE}"
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

# -----------------------------
# 1) Fine-tune Stable Diffusion UNet
# -----------------------------
step "Step 1: Fine-tune Stable Diffusion UNet"
pushd "$(dirname "$0")" >/dev/null
echo "[E2E] Launching scripts/train.sh"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" bash ./train.sh
popd >/dev/null

# -----------------------------
# 2) RLHF training (policy + selector)
# -----------------------------
step "Step 2: RLHF Training (policy + selector)"
echo "[E2E] Running RLHF main.py with config ${RLHF_CONFIG_PATH}"
UV_NO_SYNC=1 uv run python ./main.py --config "${RLHF_CONFIG_PATH}" || {
  echo "[E2E] Falling back to default behavior (main.py reads ./config.yaml)"
  UV_NO_SYNC=1 uv run python ./main.py
}

# -----------------------------
# 3) Generation using fine-tuned UNet
# -----------------------------
step "Step 3: Generate with fine-tuned UNet"
echo "[E2E] Generating ${NUM_IMAGES} images for prompt: ${BASE_PROMPT}"
UV_NO_SYNC=1 uv run python ./generate.py \
  --model_used "${UNET_OUTPUT_DIR}" \
  --prompt "${BASE_PROMPT}" \
  --device "${DEVICE}" \
  --img_num "${NUM_IMAGES}" \
  --num_inference_steps 50 \
  --output_dir "${E2E_OUTPUT_DIR}/finetuned"

echo
echo "[E2E] All steps completed. Outputs under: ${E2E_OUTPUT_DIR}"



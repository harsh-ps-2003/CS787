#!/bin/bash

# End-to-end pipeline:
# 1) Fine-tune Stable Diffusion UNet (scripts/train.sh)
# 2) Run RLHF training pipeline (policy + selector) via main.py
# 3) Generate images using the fine-tuned model via generate.py

set -euo pipefail

# Resolve important paths
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="${SCRIPT_DIR}"

# -----------------------------
# Config (override via env)
# -----------------------------
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
DEVICE="${DEVICE:-cuda:0}"
UNET_OUTPUT_DIR="${UNET_OUTPUT_DIR:-${REPO_ROOT}/checkpoints/medical-model-biomedbert-320}"
RLHF_CONFIG_PATH="${RLHF_CONFIG_PATH:-${REPO_ROOT}/config.yaml}"
BASE_PROMPT="${BASE_PROMPT:-The fundus image reveals a clear view of the retina, optic disc, macula, and the network of blood vessels, providing essential details for evaluating the health and condition of the eye.}"
NUM_IMAGES="${NUM_IMAGES:-3}"
E2E_OUTPUT_DIR="${E2E_OUTPUT_DIR:-${REPO_ROOT}/generated_e2e}"
AUTOPICK_GPU="${AUTOPICK_GPU:-1}"

echo "[E2E] CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"
echo "[E2E] DEVICE=${DEVICE}"
echo "[E2E] REPO_ROOT=${REPO_ROOT}"
echo "[E2E] UNET_OUTPUT_DIR=${UNET_OUTPUT_DIR}"
echo "[E2E] RLHF_CONFIG_PATH=${RLHF_CONFIG_PATH}"
echo "[E2E] BASE_PROMPT=\"${BASE_PROMPT}\""
echo "[E2E] NUM_IMAGES=${NUM_IMAGES}"
echo "[E2E] E2E_OUTPUT_DIR=${E2E_OUTPUT_DIR}"

# Detect GPU availability and adjust DEVICE if needed
# Respect explicit pinning; default to auto-pick unless AUTOPICK_GPU=0
if command -v nvidia-smi >/dev/null 2>&1; then
  GPU_COUNT=$(nvidia-smi --list-gpus | wc -l | tr -d ' ')
  echo "[E2E] Detected ${GPU_COUNT} NVIDIA GPU(s)"
  if [ "${AUTOPICK_GPU}" = "1" ]; then
    # Auto-pick the GPU with the most free memory to reduce OOM risk.
    mapfile -t FREE_MEM < <(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits)
    best_idx=0
    best_free=0
    for i in "${!FREE_MEM[@]}"; do
      if [ "${FREE_MEM[$i]}" -gt "$best_free" ]; then
        best_free="${FREE_MEM[$i]}"
        best_idx="$i"
      fi
    done
    # If best free memory is low (< 2000 MiB), restrict to a single, best GPU.
    if [ "$best_free" -lt 2000 ]; then
      echo "[E2E] Low free VRAM detected (${best_free} MiB). Using single GPU ${best_idx}." >&2
      CUDA_VISIBLE_DEVICES="$best_idx"
      DEVICE="cuda:0"
    else
      # Prefer the least-loaded single GPU anyway for stability during fine-tuning
      echo "[E2E] Using the least-loaded GPU ${best_idx} (free ${best_free} MiB) for fine-tuning." >&2
      CUDA_VISIBLE_DEVICES="$best_idx"
      DEVICE="cuda:0"
    fi
  fi
else
  echo "[E2E] No NVIDIA GPUs detected; forcing CPU"
  DEVICE="cpu"
fi

# Ensure directories and set safe defaults for offline + allocator unless user overrides
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-max_split_size_mb:64}"

# Ensure directories
mkdir -p "${UNET_OUTPUT_DIR}" "${E2E_OUTPUT_DIR}"

step() {
  echo
  echo "==================== $1 ===================="
}

# Build a minimal, valid CSV from generated_* images if available.
# Returns path to a CSV that is safe to parse (fields quoted). Falls back to existing CSV if no images found.
ensure_example_dataset() {
  local default_csv="${REPO_ROOT}/datasets/example/example_data.csv"
  local curated_csv="${REPO_ROOT}/datasets/example/example_generated.csv"
  mkdir -p "${REPO_ROOT}/datasets/example"

  # Try to build curated CSV from generated outputs first
  echo "[E2E] Attempting to create curated CSV from generated_* images" >&2
  local tmp_csv="${curated_csv}.tmp"
  echo "path,Text,modality" > "${tmp_csv}"

  local count=0
  shopt -s nullglob
  for img in "${REPO_ROOT}"/generated_*/*.{png,jpg,jpeg,JPG,JPEG,PNG}; do
    if [ $count -ge 200 ]; then break; fi
    local parent
    parent=$(basename "$(dirname "${img}")")
    local modality_raw
    modality_raw=${parent#generated_}
    local modality
    modality=$(echo "${modality_raw}" | tr '_' ' ')
    local text
    text="${modality}: synthetic sample"

    # CSV quoting for safety: escape embedded quotes and wrap in double quotes
    local q_path q_text q_modality
    q_path="\"${img//\"/\"\"}\""
    q_text="\"${text//\"/\"\"}\""
    q_modality="\"${modality//\"/\"\"}\""

    echo "${q_path},${q_text},${q_modality}" >> "${tmp_csv}"
    count=$((count+1))
  done
  shopt -u nullglob

  if [ $count -gt 0 ]; then
    mv "${tmp_csv}" "${curated_csv}"
    echo "[E2E] Wrote curated dataset CSV with ${count} rows: ${curated_csv}" >&2
    echo "${curated_csv}"
    return 0
  else
    rm -f "${tmp_csv}"
    if [ -f "${default_csv}" ]; then
      echo "[E2E] No generated images found. Falling back to existing CSV: ${default_csv}" >&2
      echo "${default_csv}"
      return 0
    fi
    echo "[E2E] ERROR: No dataset available. Create generated images or provide datasets/example/example_data.csv" >&2
    echo ""
    return 1
  fi
}

# -----------------------------
# 1) Fine-tune Stable Diffusion UNet with BioMedBERT
# -----------------------------
step "Step 1: Fine-tune Stable Diffusion UNet with BioMedBERT"

# Clean start - remove old checkpoints
echo "[E2E] Cleaning old checkpoints for fresh start"
rm -rf checkpoints/medical-model*

# Use fundus dataset for BioMedBERT training
DATASET_CSV="${REPO_ROOT}/datasets/example/fundus_only.csv"
if [ ! -f "${DATASET_CSV}" ]; then
  echo "[E2E] Fundus dataset not found at ${DATASET_CSV}. Creating from generated images..." >&2
  DATASET_CSV="$(ensure_example_dataset || true)"
  if [ -z "${DATASET_CSV}" ]; then
    echo "[E2E] No dataset CSV available. Please place a CSV at datasets/example/fundus_only.csv and re-run." >&2
    exit 1
  fi
fi
echo "[E2E] Using dataset CSV: ${DATASET_CSV}"

pushd "${REPO_ROOT}" >/dev/null

# Phase 1: Train at 256px with BioMedBERT
echo "[E2E] Phase 1: Training at 256px with BioMedBERT text encoder"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
UV_NO_SYNC=1 uv run accelerate launch --num_processes=1 --mixed_precision=fp16 training/model.py \
  --pretrained_model_name_or_path runwayml/stable-diffusion-v1-5 \
  --med_text_encoder_id microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext \
  --train_text_encoder \
  --train_data_dir "${DATASET_CSV}" \
  --resolution 256 --center_crop --random_flip \
  --train_batch_size 1 --gradient_accumulation_steps 32 --gradient_checkpointing \
  --max_train_steps 10000 --learning_rate 1e-05 --max_grad_norm 1 \
  --lr_scheduler cosine --lr_warmup_steps 500 \
  --snr_gamma 5 --noise_offset 0.03 \
  --validation_prompts "" --validation_epochs 1000000 \
  --output_dir ./checkpoints/medical-model-biomedbert-256 \
  --checkpointing_steps 1000 --checkpoints_total_limit 3 --dataloader_num_workers 0 \
  --image_column path --caption_column Text \
  --use_8bit_adam

# Phase 2: Progressive up-resolution to 320px
echo "[E2E] Phase 2: Progressive up-resolution to 320px"
mkdir -p checkpoints/medical-model-biomedbert-320
rsync -a checkpoints/medical-model-biomedbert-256/checkpoint-10000/ \
      checkpoints/medical-model-biomedbert-320/checkpoint-10000/

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
UV_NO_SYNC=1 uv run accelerate launch --num_processes=1 --mixed_precision=fp16 training/model.py \
  --pretrained_model_name_or_path runwayml/stable-diffusion-v1-5 \
  --med_text_encoder_id microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext \
  --train_text_encoder \
  --train_data_dir "${DATASET_CSV}" \
  --resolution 320 --center_crop --random_flip \
  --train_batch_size 1 --gradient_accumulation_steps 48 --gradient_checkpointing \
  --max_train_steps 12000 --learning_rate 8e-06 --max_grad_norm 1 \
  --lr_scheduler cosine --lr_warmup_steps 500 \
  --snr_gamma 5 --noise_offset 0.03 \
  --validation_prompts "" --validation_epochs 1000000 \
  --output_dir ./checkpoints/medical-model-biomedbert-320 \
  --checkpointing_steps 1000 --checkpoints_total_limit 3 --dataloader_num_workers 0 \
  --image_column path --caption_column Text \
  --use_8bit_adam \
  --resume_from_checkpoint checkpoint-10000

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
# 3) Generation using fine-tuned BioMedBERT UNet
# -----------------------------
step "Step 3: Generate with fine-tuned BioMedBERT UNet"
echo "[E2E] Generating ${NUM_IMAGES} images for prompt: ${BASE_PROMPT}"
pushd "${REPO_ROOT}" >/dev/null
UV_NO_SYNC=1 uv run python ./generate.py \
  --pretrained_model runwayml/stable-diffusion-v1-5 \
  --model_used ./checkpoints/medical-model-biomedbert-320/checkpoint-12000 \
  --prompt "The fundus image reveals a clear view of the retina, optic disc, macula, and the network of blood vessels, providing essential details for evaluating the health and condition of the eye." \
  --device "${DEVICE}" \
  --img_num 9 \
  --num_inference_steps 50 \
  --precision float32 \
  --scheduler dpm \
  --height 320 --width 320 \
  --output_dir "${E2E_OUTPUT_DIR}/biomedbert-320"
popd >/dev/null

echo
echo "[E2E] All steps completed. Outputs under: ${E2E_OUTPUT_DIR}"



#!/bin/bash
# Automated training script for Stable Diffusion fine-tuning on medical images

# Configuration - Customize these variables
# Respect externally provided env values if set
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export MODEL_NAME="${MODEL_NAME:-runwayml/stable-diffusion-v1-5}"
export DATASET_NAME="${DATASET_NAME:-${SCRIPT_DIR}/../datasets/example/example_data.csv}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"  # Use both GPUs (0,1) or single GPU (0)
export WANDB_MODE="offline"
export USE_EMA="${USE_EMA:-0}"
export ENABLE_XFORMERS="${ENABLE_XFORMERS:-0}"
export RESOLUTION="${RESOLUTION:-128}"
export DATALOADER_NUM_WORKERS="${DATALOADER_NUM_WORKERS:-0}"
export REPORT_TO="${REPORT_TO:-}"

# GPU Configuration
NUM_GPUS=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)
echo "Using $NUM_GPUS GPU(s): $CUDA_VISIBLE_DEVICES"

# Conservative allocator to reduce fragmentation on smaller GPUs (e.g., 12GB)
# Keep it simple to avoid allocator asserts on older drivers
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:32"

# Create output directory (respect env override, default under repo root)
export OUTPUT_DIR="${OUTPUT_DIR:-${SCRIPT_DIR}/../checkpoints/medical-model}"
mkdir -p "${OUTPUT_DIR}"
# Remove stale validation dir to avoid FileExistsError in training/model.py
if [ -d "${OUTPUT_DIR}/validation" ]; then
  echo "Removing stale validation dir: ${OUTPUT_DIR}/validation"
  rm -rf "${OUTPUT_DIR}/validation"
fi

# Adjust batch size based on number of GPUs (allow env override)
if [ -z "${TRAIN_BATCH_SIZE:-}" ] || [ -z "${GRADIENT_ACCUMULATION_STEPS:-}" ]; then
    if [ $NUM_GPUS -ge 2 ]; then
        # Very conservative per-GPU batch size to avoid OOM on 12GB GPUs
        TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-1}
        GRADIENT_ACCUMULATION_STEPS=${GRADIENT_ACCUMULATION_STEPS:-32}
    else
        TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-1}
        GRADIENT_ACCUMULATION_STEPS=${GRADIENT_ACCUMULATION_STEPS:-32}
    fi
fi

echo "Training configuration:"
echo "  Batch size: $TRAIN_BATCH_SIZE"
echo "  Gradient accumulation steps: $GRADIENT_ACCUMULATION_STEPS"
echo "  Mixed precision: fp16"
echo "  Resolution: ${RESOLUTION}"
echo "  Use EMA: ${USE_EMA}"
echo "  Enable xformers: ${ENABLE_XFORMERS}"
echo "  DataLoader workers: ${DATALOADER_NUM_WORKERS}"
echo "  Dataset CSV: ${DATASET_NAME}"
echo "  Output dir: ${OUTPUT_DIR}"

# Run training
# If we restricted to a single GPU upstream, ensure num_processes=1
if [ "$NUM_GPUS" -gt 1 ]; then
  NUM_PROCESSES="$NUM_GPUS"
else
  NUM_PROCESSES=1
fi

EXTRA_FLAGS=()
if [ "${USE_EMA}" = "1" ]; then EXTRA_FLAGS+=(--use_ema); fi
if [ "${ENABLE_XFORMERS}" = "1" ]; then EXTRA_FLAGS+=(--enable_xformers_memory_efficient_attention); fi
# Reduce optimizer memory on 12GB GPUs via bitsandbytes 8-bit Adam when requested
if [ "${USE_8BIT_ADAM:-0}" = "1" ]; then EXTRA_FLAGS+=(--use_8bit_adam); fi

# Conditionally add logging backend only if explicitly requested (e.g., tensorboard, wandb)
if [ -n "${REPORT_TO}" ]; then EXTRA_FLAGS+=(--report_to="${REPORT_TO}"); fi

accelerate launch --num_processes=$NUM_PROCESSES --mixed_precision="fp16" "${SCRIPT_DIR}/../training/model.py" \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir=$DATASET_NAME \
  --resolution=${RESOLUTION} --center_crop --random_flip \
  --train_batch_size=$TRAIN_BATCH_SIZE \
  --gradient_accumulation_steps=$GRADIENT_ACCUMULATION_STEPS \
  --gradient_checkpointing \
  --max_train_steps=1000 \
  --learning_rate=1e-05 \
  --max_grad_norm=1 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=100 \
  --validation_prompts "Chest X-ray: normal lung fields" "Brain MRI: normal anatomy" "Fundus: healthy retina" \
  --validation_epochs=5 \
  --output_dir="${OUTPUT_DIR}" \
  --checkpointing_steps=200 \
  --dataloader_num_workers=${DATALOADER_NUM_WORKERS} \
  --image_column="path" \
  --caption_column="Text" \
  ${EXTRA_FLAGS[@]}
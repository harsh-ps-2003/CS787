#!/bin/bash
# Automated training script for Stable Diffusion fine-tuning on medical images

# Configuration - Customize these variables
# Respect externally provided env values if set
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
export MODEL_NAME="${MODEL_NAME:-runwayml/stable-diffusion-v1-5}"
export DATASET_NAME="${DATASET_NAME:-${SCRIPT_DIR}/../datasets/example/example_data.csv}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"  # Use both GPUs (0,1) or single GPU (0)
export WANDB_MODE="offline"

# GPU Configuration
NUM_GPUS=$(echo $CUDA_VISIBLE_DEVICES | tr ',' '\n' | wc -l)
echo "Using $NUM_GPUS GPU(s): $CUDA_VISIBLE_DEVICES"

# Conservative allocator to reduce fragmentation on smaller GPUs (e.g., 12GB)
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:64"

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
        GRADIENT_ACCUMULATION_STEPS=${GRADIENT_ACCUMULATION_STEPS:-8}
    else
        TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-1}
        GRADIENT_ACCUMULATION_STEPS=${GRADIENT_ACCUMULATION_STEPS:-16}
    fi
fi

echo "Training configuration:"
echo "  Batch size: $TRAIN_BATCH_SIZE"
echo "  Gradient accumulation steps: $GRADIENT_ACCUMULATION_STEPS"
echo "  Mixed precision: fp16"
echo "  Dataset CSV: ${DATASET_NAME}"
echo "  Output dir: ${OUTPUT_DIR}"

# Run training
accelerate launch --num_processes=$NUM_GPUS --mixed_precision="fp16" "${SCRIPT_DIR}/../training/model.py" \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir=$DATASET_NAME \
  --use_ema \
  --resolution=256 --center_crop --random_flip \
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
  --report_to="tensorboard" \
  --checkpointing_steps=200 \
  --image_column="path" \
  --caption_column="Text"
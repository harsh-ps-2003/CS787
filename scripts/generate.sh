#!/bin/bash
# Generate images using trained models

# Configuration - Customize these variables
DEVICE="cuda:0"  # Use cuda:0, cuda:1, or cpu
NUM_IMAGES=3
OUTPUT_DIR="generated_img"

# GPU Configuration
if command -v nvidia-smi &> /dev/null; then
    GPU_COUNT=$(nvidia-smi --list-gpus | wc -l)
    echo "Detected $GPU_COUNT GPU(s)"
    echo "Using device: $DEVICE"
else
    echo "No NVIDIA GPUs detected, using CPU"
    DEVICE="cpu"
fi

# Example 1: Generate fundus images with fine-tuned model
echo "Generating fundus images with fine-tuned model..."
UV_NO_SYNC=1 uv run python ../generate.py \
  --model_used=./checkpoints/medical-model \
  --prompt="Fundus: healthy retina with clear optic disc" \
  --device="$DEVICE" \
  --img_num=$NUM_IMAGES \
  --output_dir="$OUTPUT_DIR/fundus"

# Example 2: Generate chest X-rays with pre-trained model only
echo "Generating chest X-rays with pre-trained model..."
UV_NO_SYNC=1 uv run python ../generate.py \
  --use_pretrained_only \
  --prompt="Chest X-ray: normal lung fields without infiltrates" \
  --device="$DEVICE" \
  --img_num=$NUM_IMAGES \
  --output_dir="$OUTPUT_DIR/cxr"

# Example 3: Generate brain MRI images
echo "Generating brain MRI images..."
UV_NO_SYNC=1 uv run python ../generate.py \
  --use_pretrained_only \
  --prompt="Brain MRI: T1-weighted image showing normal anatomy" \
  --device="$DEVICE" \
  --img_num=$NUM_IMAGES \
  --output_dir="$OUTPUT_DIR/mri"

echo "Generation complete! Check the $OUTPUT_DIR directory for results."

#!/bin/bash
# Generate images using trained models

# Example: Generate fundus images with fine-tuned model
UV_NO_SYNC=1 uv run python ../generate.py \
  --model_used=./checkpoints/medical-model \
  --prompt="Fundus: healthy retina with clear optic disc" \
  --device="cuda:0" \
  --img_num=3 \
  --output_dir="generated_img/fundus"

# Example: Generate chest X-rays with pre-trained model only
# UV_NO_SYNC=1 uv run python ../generate.py \
#   --use_pretrained_only \
#   --prompt="Chest X-ray: normal lung fields" \
#   --device="cuda:0" \
#   --img_num=5 \
#   --output_dir="generated_img/cxr"

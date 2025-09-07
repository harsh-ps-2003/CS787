#!/bin/bash
# Automated training script for Stable Diffusion fine-tuning on medical images

# Configuration
export MODEL_NAME="runwayml/stable-diffusion-v1-5"
export DATASET_NAME="./datasets/example/example_data.csv"
export CUDA_VISIBLE_DEVICES="0"
export WANDB_MODE="offline"

# Create output directory
mkdir -p ./checkpoints/medical-model

# Run training
accelerate launch --num_processes=1 --mixed_precision="fp16" ../training/model.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir=$DATASET_NAME \
  --use_ema \
  --resolution=256 --center_crop --random_flip \
  --train_batch_size=4 \
  --gradient_accumulation_steps=4 \
  --gradient_checkpointing \
  --max_train_steps=1000 \
  --learning_rate=1e-05 \
  --max_grad_norm=1 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=100 \
  --validation_prompts "Chest X-ray: normal lung fields" "Brain MRI: normal anatomy" "Fundus: healthy retina" \
  --validation_epochs=5 \
  --output_dir="./checkpoints/medical-model" \
  --report_to="tensorboard" \
  --checkpointing_steps=200 \
  --image_column="path" \
  --caption_column="Text"
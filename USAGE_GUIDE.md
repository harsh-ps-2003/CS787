# Medical Image Generation - Usage Guide

This guide provides clear instructions for both using pre-trained models and training your own medical image generation models.

## Quick Start (Pre-trained Models)

### Medical Image Generation

Use the `generate.py` script for high-quality medical image generation:

```bash
# Generate chest X-rays
UV_NO_SYNC=1 uv run python generate.py --use_pretrained_only --prompt "Chest X-ray: normal lung fields without infiltrates" --img_num 3 --device cpu

# Generate brain MRI
UV_NO_SYNC=1 uv run python generate.py --use_pretrained_only --prompt "Brain MRI: T1-weighted image showing normal anatomy" --img_num 2 --device cpu

# Generate fundus images
UV_NO_SYNC=1 uv run python generate.py --use_pretrained_only --prompt "Fundus: healthy retina with clear optic disc" --img_num 1 --device cpu

# Use with custom checkpoint (after training)
UV_NO_SYNC=1 uv run python generate.py --model_used ./checkpoints/medical-model --prompt "Chest X-ray: pneumonia" --img_num 5
```

## Training Your Own Model

### Step 1: Prepare Your Data

Your data should be in CSV format with three columns:
- `path`: Path to the image file
- `Text`: Description of the medical image
- `modality`: Imaging modality (MRI, CT, CXR, etc.)

Example CSV structure:
```csv
path,Text,modality
datasets/example/cxr/0.jpg,Chest X-ray showing clear lung fields,CXR
datasets/example/brain-mri/1.jpg,Brain MRI showing normal anatomy,MRI
datasets/example/fundus/2.jpg,Fundus image showing healthy retina,Fundus
```

### Step 2: Run Training

```bash
# Navigate to scripts directory
cd scripts

# Run training (this will take several hours)
bash train.sh
```

The training script will:
- Use the example dataset in `./datasets/example/example_data.csv`
- Train for 1000 steps (adjustable in `train.sh`)
- Save checkpoints every 200 steps
- Generate validation images during training
- Save the final model to `./checkpoints/medical-model`

### Step 3: Use Your Trained Model

```bash
# Generate images with your trained model
UV_NO_SYNC=1 uv run python generate.py --model_used ./checkpoints/medical-model --prompt "Chest X-ray: pneumonia" --img_num 3
```

## Multi-GPU Configuration

### GPU Detection and Usage

The scripts automatically detect available GPUs and provide configuration options:

```bash
# Auto-detect best available GPU
UV_NO_SYNC=1 uv run python simple_generate.py --device auto --prompt "normal chest X-ray" --modality "CXR"

# Use specific GPU
UV_NO_SYNC=1 uv run python simple_generate.py --device cuda:0 --prompt "normal chest X-ray" --modality "CXR"
UV_NO_SYNC=1 uv run python simple_generate.py --device cuda:1 --prompt "normal chest X-ray" --modality "CXR"

# Use CPU (fallback)
UV_NO_SYNC=1 uv run python simple_generate.py --device cpu --prompt "normal chest X-ray" --modality "CXR"
```

### Training with Multiple GPUs

Edit `scripts/train.sh` to configure GPU usage:

```bash
# Use both GPUs (0,1)
export CUDA_VISIBLE_DEVICES="0,1"

# Use only GPU 0
export CUDA_VISIBLE_DEVICES="0"

# Use only GPU 1  
export CUDA_VISIBLE_DEVICES="1"
```

The training script automatically adjusts batch sizes based on the number of GPUs:
- **1 GPU**: batch_size=4, gradient_accumulation_steps=4
- **2 GPUs**: batch_size=8, gradient_accumulation_steps=2

## Configuration Options

### Generation Script Options

```bash
UV_NO_SYNC=1 uv run python generate.py \
  --pretrained_model "runwayml/stable-diffusion-v1-5" \
  --model_used "./checkpoints/my-model" \
  --prompt "Medical description" \
  --img_num 5 \
  --device "cpu" \
  --num_inference_steps 15 \
  --precision "float32" \
  --height 256 \
  --width 256 \
  --scheduler "dpm" \
  --output_dir "generated_images"
```

### Training Script Options

Edit `scripts/train.sh` to customize:
- `CUDA_VISIBLE_DEVICES`: GPU selection ("0", "1", "0,1")
- `--max_train_steps`: Number of training steps (default: 1000)
- `--train_batch_size`: Batch size (auto-adjusted based on GPU count)
- `--learning_rate`: Learning rate (default: 1e-05)
- `--resolution`: Image resolution (default: 256)
- `--validation_prompts`: Prompts for validation images

## Troubleshooting

### Common Issues

1. **"Checkpoint directory does not exist"**
   - Solution: Use `--use_pretrained_only` flag or train a model first

2. **"CUDA out of memory"**
   - Solution: Reduce `--train_batch_size` or `--img_num`

3. **"Model loading error"**
   - Solution: Check internet connection for HuggingFace downloads

4. **"Dataset not found"**
   - Solution: Ensure CSV file exists at specified path

### Performance Tips

- Use `--mixed_precision="fp16"` for faster training
- Reduce `--num_inference_steps` for faster generation
- Use `--gradient_checkpointing` to save memory during training

## File Structure

```
CS787/
├── generate.py              # Medical image generation script
├── scripts/
│   ├── train.sh             # Training script
│   └── generate.sh          # Example generation script
├── training/
│   └── model.py             # Training implementation
├── datasets/
│   └── example/
│       ├── example_data.csv # Example dataset
│       └── [image folders]  # Image directories
└── checkpoints/             # Trained models (created during training)
```

## Examples

### Generate Different Medical Images

```bash
# Chest X-ray
UV_NO_SYNC=1 uv run python generate.py --use_pretrained_only --prompt "Chest X-ray: bilateral pneumonia" --img_num 3 --device cpu

# Brain MRI
UV_NO_SYNC=1 uv run python generate.py --use_pretrained_only --prompt "Brain MRI: tumor in frontal lobe" --img_num 2 --device cpu

# Fundus
UV_NO_SYNC=1 uv run python generate.py --use_pretrained_only --prompt "Fundus: diabetic retinopathy" --img_num 1 --device cpu

# CT scan
UV_NO_SYNC=1 uv run python generate.py --use_pretrained_only --prompt "Chest CT: lung nodule" --img_num 2 --device cpu

# OCT
UV_NO_SYNC=1 uv run python generate.py --use_pretrained_only --prompt "OCT: macular degeneration" --img_num 1 --device cpu
```

### Batch Generation

```bash
# Generate multiple images with different prompts
UV_NO_SYNC=1 uv run python generate.py --use_pretrained_only --prompt "Chest X-ray: normal lung fields" --img_num 10 --device cpu --output_dir "normal_cxr"
UV_NO_SYNC=1 uv run python generate.py --use_pretrained_only --prompt "Chest X-ray: pneumonia" --img_num 10 --device cpu --output_dir "pneumonia_cxr"
```

## Next Steps

1. **Start with medical image generation** using `generate.py`
2. **Experiment with different prompts** and modalities
3. **Train your own model** if you have custom medical data
4. **Integrate with your workflow** using the generated images

For advanced usage, refer to the individual script documentation and the main REPORT.md file.

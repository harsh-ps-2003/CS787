#!/usr/bin/env python3
"""
Simple Medical Image Generation Script
=====================================

This script provides an easy way to generate medical images using pre-trained
Stable Diffusion models without requiring custom checkpoints.

Usage Examples:
1. Generate chest X-rays:
   UV_NO_SYNC=1 uv run python simple_generate.py --prompt "Chest X-ray showing clear lung fields" --modality "CXR"

2. Generate brain MRI:
   UV_NO_SYNC=1 uv run python simple_generate.py --prompt "Brain MRI showing normal anatomy" --modality "MRI"

3. Generate fundus images:
   UV_NO_SYNC=1 uv run python simple_generate.py --prompt "Retinal fundus image showing healthy optic disc" --modality "Fundus"
"""

import argparse
import os
import sys
from diffusers import StableDiffusionPipeline
import torch
from PIL import Image

def parse_args():
    parser = argparse.ArgumentParser(
        description="Simple medical image generation using pre-trained Stable Diffusion",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    parser.add_argument(
        "--prompt",
        type=str,
        default="Chest X-ray: normal lung fields without infiltrates",
        help="Text prompt describing the medical image to generate"
    )
    
    parser.add_argument(
        "--modality",
        type=str,
        choices=["CXR", "MRI", "CT", "Fundus", "OCT"],
        default="CXR",
        help="Medical imaging modality"
    )
    
    parser.add_argument(
        "--num_images",
        type=int,
        default=3,
        help="Number of images to generate"
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        default="generated_images",
        help="Directory to save generated images"
    )
    
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device to use (cuda:0, cuda:1, cpu, auto). 'auto' detects best available GPU."
    )
    
    parser.add_argument(
        "--steps",
        type=int,
        default=50,
        help="Number of inference steps (higher = better quality, slower)"
    )
    
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=7.5,
        help="Classifier-free guidance scale (higher = more prompt adherence)"
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducible results"
    )
    
    return parser.parse_args()

def create_medical_prompt(base_prompt, modality):
    """Create a properly formatted medical prompt with modality information."""
    modality_prompts = {
        "CXR": f"Chest X-ray: {base_prompt}",
        "MRI": f"Brain MRI: {base_prompt}",
        "CT": f"Chest CT scan: {base_prompt}",
        "Fundus": f"Retinal fundus image: {base_prompt}",
        "OCT": f"Optical coherence tomography: {base_prompt}"
    }
    
    return modality_prompts.get(modality, f"{modality}: {base_prompt}")

def detect_best_device():
    """Detect the best available device for generation."""
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        print(f"CUDA available: {gpu_count} GPU(s) detected")
        for i in range(gpu_count):
            gpu_name = torch.cuda.get_device_name(i)
            print(f"  GPU {i}: {gpu_name}")
        return f"cuda:0"  # Default to first GPU
    else:
        print("CUDA not available, using CPU")
        return "cpu"

def load_model(device):
    """Load the pre-trained Stable Diffusion model."""
    print("Loading pre-trained Stable Diffusion model...")
    try:
        pipe = StableDiffusionPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            safety_checker=None,
            requires_safety_checker=False,
            torch_dtype=torch.float16 if device.startswith("cuda") else torch.float32
        ).to(device)
        
        # Enable memory efficient attention if available
        if hasattr(pipe, "enable_attention_slicing"):
            pipe.enable_attention_slicing()
        
        print("Model loaded successfully!")
        return pipe
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Make sure you have an internet connection for downloading the model.")
        return None

def generate_images(pipe, prompt, num_images, steps, guidance_scale, seed, output_dir):
    """Generate images using the loaded pipeline."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Set seed for reproducibility
    if seed is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
    
    print(f"Generating {num_images} images...")
    print(f"Prompt: {prompt}")
    print(f"Steps: {steps}, Guidance Scale: {guidance_scale}")
    
    try:
        for i in range(num_images):
            print(f"Generating image {i+1}/{num_images}...")
            
            # Generate image
            result = pipe(
                prompt=prompt,
                num_inference_steps=steps,
                guidance_scale=guidance_scale,
                height=512,  # Standard resolution
                width=512
            )
            
            image = result.images[0]
            
            # Save image
            filename = f"medical_image_{i+1:03d}.png"
            filepath = os.path.join(output_dir, filename)
            image.save(filepath)
            print(f"Saved: {filepath}")
        
        print(f"\n✅ Successfully generated {num_images} images in '{output_dir}'")
        
    except Exception as e:
        print(f"❌ Error during generation: {e}")
        return False
    
    return True

def main():
    args = parse_args()
    
    # Auto-detect device if set to "auto"
    if args.device == "auto":
        args.device = detect_best_device()
    elif args.device.startswith("cuda") and not torch.cuda.is_available():
        print(f"Warning: CUDA not available, switching to CPU")
        args.device = "cpu"
    
    # Create medical prompt
    medical_prompt = create_medical_prompt(args.prompt, args.modality)
    
    print("=" * 60)
    print("Simple Medical Image Generation")
    print("=" * 60)
    print(f"Modality: {args.modality}")
    print(f"Prompt: {medical_prompt}")
    print(f"Device: {args.device}")
    print(f"Images: {args.num_images}")
    print("=" * 60)
    
    # Load model
    pipe = load_model(args.device)
    if pipe is None:
        sys.exit(1)
    
    # Generate images
    success = generate_images(
        pipe, 
        medical_prompt, 
        args.num_images, 
        args.steps, 
        args.guidance_scale, 
        args.seed, 
        args.output_dir
    )
    
    if not success:
        sys.exit(1)
    
    print("\n🎉 Generation complete!")
    print(f"Check the '{args.output_dir}' directory for your generated images.")

if __name__ == "__main__":
    main()

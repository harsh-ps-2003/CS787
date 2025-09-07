# Use trained models for inference and generation

from diffusers import (
    StableDiffusionPipeline,
    UNet2DConditionModel,
    StableDiffusionImg2ImgPipeline,
)
import torch
from PIL import Image
import os
import argparse
import sys

def parse_args():
    parser = argparse.ArgumentParser(description="Generate medical images using MINIM or pre-trained models")
    parser.add_argument(
        "--pretrained_model", 
        type=str, 
        default="runwayml/stable-diffusion-v1-5", 
        help="The pretrained base model (HuggingFace model ID or local path)."
    )
    parser.add_argument(
        "--model_used", 
        type=str, 
        default=None,
        help="The fine-tuned model checkpoint path (optional). If not provided, uses only pretrained_model."
    )
    parser.add_argument(
        "--prompt", 
        type=str, 
        default="Chest X-ray: normal lung fields without infiltrates", 
        help="The prompt to guide the generation."
    )
    parser.add_argument(
        "--img_num", 
        type=int, 
        default=3, 
        help="How many images to generate."
    )
    parser.add_argument(
        "--device", 
        type=str, 
        default='cuda:0', 
        help="Device used (cuda:0, cpu, etc.)."
    )
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default='generated_images', 
        help="Output directory for generated images."
    )
    parser.add_argument(
        "--num_inference_steps", 
        type=int, 
        default=50, 
        help="How many steps taken when model generates each image."
    )
    parser.add_argument(
        "--use_pretrained_only",
        action="store_true",
        help="Use only the pretrained model without any fine-tuned components."
    )
    args = parser.parse_args()
    return args

def validate_paths(args):
    """Validate that required paths exist and are accessible."""
    if args.model_used and not args.use_pretrained_only:
        # Check if checkpoint directory exists
        if not os.path.exists(args.model_used):
            print(f"Error: Checkpoint directory '{args.model_used}' does not exist.")
            print("Available options:")
            print("1. Use --use_pretrained_only flag to use only the pretrained model")
            print("2. Train a model first using the training script")
            print("3. Provide a valid checkpoint path")
            return False
        
        # Check for unet subdirectory
        unet_path = os.path.join(args.model_used, 'unet')
        if not os.path.exists(unet_path):
            print(f"Error: UNet checkpoint not found at '{unet_path}'")
            print("Expected checkpoint structure:")
            print("checkpoint_dir/")
            print("  ├── unet/")
            print("  │   ├── config.json")
            print("  │   └── diffusion_pytorch_model.bin")
            print("  └── ...")
            return False
    
    return True

def load_model(args):
    """Load the model pipeline with proper error handling."""
    try:
        if args.use_pretrained_only or not args.model_used:
            print(f"Loading pretrained model: {args.pretrained_model}")
            pipe = StableDiffusionPipeline.from_pretrained(
                args.pretrained_model, 
                safety_checker=None,
                requires_safety_checker=False
            ).to(args.device)
        else:
            print(f"Loading fine-tuned model from: {args.model_used}")
            unet_path = os.path.join(args.model_used, 'unet')
            unet = UNet2DConditionModel.from_pretrained(unet_path)
            
            pipe = StableDiffusionPipeline.from_pretrained(
                args.pretrained_model, 
                unet=unet, 
                safety_checker=None,
                requires_safety_checker=False
            ).to(args.device)
        
        return pipe
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        print("\nTroubleshooting:")
        print("1. Check your internet connection for HuggingFace model downloads")
        print("2. Verify the pretrained_model path/ID is correct")
        print("3. If using local checkpoint, ensure it's properly formatted")
        return None

def main():
    args = parse_args()
    
    # Validate paths if using custom checkpoint
    if not validate_paths(args):
        sys.exit(1)
    
    # Load model
    pipe = load_model(args)
    if pipe is None:
        sys.exit(1)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"Generating {args.img_num} images with prompt: '{args.prompt}'")
    print(f"Output directory: {args.output_dir}")
    print(f"Device: {args.device}")
    print(f"Inference steps: {args.num_inference_steps}")
    
    # Generate images
    try:
        for i in range(args.img_num):
            print(f"Generating image {i+1}/{args.img_num}...")
            image = pipe(
                prompt=args.prompt, 
                num_inference_steps=args.num_inference_steps,
                guidance_scale=7.5  # Standard CFG scale
            ).images[0]
            
            output_path = os.path.join(args.output_dir, f"generated_{i+1}.png")
            image.save(output_path)
            print(f"Saved: {output_path}")
        
        print(f"\nSuccessfully generated {args.img_num} images in '{args.output_dir}'")
        
    except Exception as e:
        print(f"Error during generation: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()

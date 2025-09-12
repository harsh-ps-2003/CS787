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
        help="Device used (cuda:0, cuda:1, cpu, etc.). Auto-detects available GPUs if not specified."
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
        "--precision",
        type=str,
        default="float16",
        choices=["float16", "float32"],
        help="Computation precision for the diffusion pipeline (default: float16)."
    )
    parser.add_argument(
        "--height",
        type=int,
        default=256,
        help="Output image height (default: 256)."
    )
    parser.add_argument(
        "--width",
        type=int,
        default=256,
        help="Output image width (default: 256)."
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
        torch_dtype = torch.float16 if args.precision == "float16" else torch.float32
        if args.use_pretrained_only or not args.model_used:
            print(f"Loading pretrained model: {args.pretrained_model}")
            pipe = StableDiffusionPipeline.from_pretrained(
                args.pretrained_model,
                torch_dtype=torch_dtype,
                safety_checker=None,
                requires_safety_checker=False,
                use_safetensors=True
            ).to(args.device)
        else:
            print(f"Loading fine-tuned model from: {args.model_used}")
            unet_path = os.path.join(args.model_used, 'unet')
            unet = UNet2DConditionModel.from_pretrained(unet_path)
            
            pipe = StableDiffusionPipeline.from_pretrained(
                args.pretrained_model,
                unet=unet, 
                torch_dtype=torch_dtype,
                safety_checker=None,
                requires_safety_checker=False,
                use_safetensors=True
            ).to(args.device)
        # Memory optimizations for moderate GPUs
        try:
            pipe.enable_attention_slicing()
            pipe.enable_vae_slicing()
            pipe.enable_vae_tiling()
            # Many Pascal-era GPUs (e.g., TITAN Xp) are unstable with VAE in fp16
            pipe.vae.to(dtype=torch.float32)
        except Exception:
            pass
        
        return pipe
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        print("\nTroubleshooting:")
        print("1. Check your internet connection for HuggingFace model downloads")
        print("2. Verify the pretrained_model path/ID is correct")
        print("3. If using local checkpoint, ensure it's properly formatted")
        return None

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

def main():
    args = parse_args()
    
    # Auto-detect device if not specified or if cuda:0 is not available
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        print(f"Warning: CUDA not available, switching to CPU")
        args.device = "cpu"
    elif args.device == "auto":
        args.device = detect_best_device()
    
    print(f"Using device: {args.device}")
    
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
                guidance_scale=7.5,  # Standard CFG scale
                height=args.height,
                width=args.width
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

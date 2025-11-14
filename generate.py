# Use trained models for inference and generation

from diffusers import (
    StableDiffusionPipeline,
    UNet2DConditionModel,
    StableDiffusionImg2ImgPipeline,
)
from diffusers.schedulers import (
    DPMSolverMultistepScheduler,
    EulerAncestralDiscreteScheduler,
    EulerDiscreteScheduler,
    PNDMScheduler,
)
from diffusers.models.attention_processor import AttnProcessor
import torch
from PIL import Image
import os
import argparse
import sys
# Custom UNet support
from utils.unet import MedicalUNet
from utils.attention_gates import add_attention_gates_to_unet

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
    parser.add_argument(
        "--use_custom_unet",
        action="store_true",
        help="Whether to use custom MedicalUNet instead of standard UNet2DConditionModel."
    )
    parser.add_argument(
        "--use_skip_attention_gates",
        action="store_true",
        help=(
            "Enable attention-gated skip connections on the default UNet2DConditionModel "
            "when loading a fine-tuned checkpoint. This must match the flag used during "
            "training (`--use_skip_attention_gates`)."
        ),
    )
    parser.add_argument(
        "--scheduler",
        type=str,
        default="dpm",
        choices=["dpm", "euler_a", "euler", "pndm"],
        help="Diffusion scheduler to use (default: dpm)."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducible generation. If None, uses random seed for diversity."
    )
    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=7.5,
        help="Guidance scale for classifier-free guidance (default: 7.5). Lower values increase diversity."
    )
    parser.add_argument(
        "--negative_prompt",
        type=str,
        default="",
        help="Negative prompt to avoid certain features (default: empty)."
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
        # Force float32 on CPU to avoid compatibility issues
        if args.device == "cpu":
            torch_dtype = torch.float32
        else:
            torch_dtype = torch.float16 if args.precision == "float16" else torch.float32
        
        if args.use_pretrained_only or not args.model_used:
            # If a local fine-tuned checkpoint directory was passed in as --pretrained_model, reconstruct pipeline
            if os.path.isdir(args.pretrained_model) and os.path.exists(os.path.join(args.pretrained_model, 'unet')):
                print(f"Detected fine-tuned checkpoint directory at: {args.pretrained_model}")
                unet_path = os.path.join(args.pretrained_model, 'unet')
                custom_unet_path = os.path.join(unet_path, 'pytorch_model.bin')
                
                # Check if custom UNet checkpoint exists
                if args.use_custom_unet or os.path.exists(custom_unet_path):
                    print("Loading custom MedicalUNet")
                    cross_attention_dim = 768
                    unet = MedicalUNet(
                        in_channels=4,
                        out_channels=4,
                        cross_attention_dim=cross_attention_dim,
                        time_embed_dim=320,
                        device=args.device
                    )
                    if os.path.exists(custom_unet_path):
                        state_dict = torch.load(custom_unet_path, map_location=args.device)
                        unet.load_state_dict(state_dict)
                        print(f"Loaded custom UNet checkpoint from {custom_unet_path}")
                    unet.to(args.device, dtype=torch_dtype)
                else:
                    # Default UNet path: we may optionally attach attention-gated skip connections
                    unet = UNet2DConditionModel.from_pretrained(unet_path)
                    if args.use_skip_attention_gates:
                        print("Attaching attention-gated skip connections to default UNet2DConditionModel")
                        add_attention_gates_to_unet(unet, verbose=True)
                        # Reload weights so that any trained gate parameters (if present) are restored.
                        weight_bin = os.path.join(unet_path, "diffusion_pytorch_model.bin")
                        if os.path.exists(weight_bin):
                            state_dict = torch.load(weight_bin, map_location=args.device)
                            unet.load_state_dict(state_dict, strict=False)
                        else:
                            print(
                                "[WARN] diffusion_pytorch_model.bin not found under fine-tuned checkpoint; "
                                "skip attention gates will start from random init."
                            )
                    unet.to(args.device, dtype=torch_dtype)
                # Assume BioMedBERT for this fine-tuned checkpoint (naming convention)
                try:
                    from utils.medical_text_encoder import load_med_encoder
                    tokenizer, text_encoder = load_med_encoder(
                        model_id="microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext",
                        device=args.device,
                        dtype=torch.float32,
                        trainable=False,
                    )
                    base_id = "runwayml/stable-diffusion-v1-5"
                    pipe = StableDiffusionPipeline.from_pretrained(
                        base_id,
                        unet=unet,
                        text_encoder=text_encoder,
                        torch_dtype=torch_dtype,
                        safety_checker=None,
                        requires_safety_checker=False,
                        use_safetensors=True,
                    ).to(args.device)
                    # Mark pipeline as BioMedBERT for downstream logic and stash tokenizer
                    setattr(pipe, "_is_biomedbert", True)
                    setattr(pipe, "_med_tokenizer", tokenizer)
                    # Force full-precision for stability
                    pipe.to(dtype=torch.float32)
                except Exception:
                    # Fallback: load as a regular pipeline if med encoder unavailable
                    pipe = StableDiffusionPipeline.from_pretrained(
                        args.pretrained_model,
                        torch_dtype=torch_dtype,
                        safety_checker=None,
                        requires_safety_checker=False,
                        use_safetensors=True,
                    ).to(args.device)
            else:
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
            
            # Check if custom UNet checkpoint exists
            custom_unet_path = os.path.join(unet_path, 'pytorch_model.bin')
            if args.use_custom_unet or os.path.exists(custom_unet_path):
                print("Loading custom MedicalUNet")
                # Determine cross-attention dimension (default to 768 for BioMedBERT)
                cross_attention_dim = 768
                unet = MedicalUNet(
                    in_channels=4,
                    out_channels=4,
                    cross_attention_dim=cross_attention_dim,
                    time_embed_dim=320,
                    device=args.device
                )
                if os.path.exists(custom_unet_path):
                    state_dict = torch.load(custom_unet_path, map_location=args.device)
                    unet.load_state_dict(state_dict)
                    print(f"Loaded custom UNet checkpoint from {custom_unet_path}")
                else:
                    print("Warning: Custom UNet checkpoint not found, using randomly initialized weights")
                unet.to(args.device, dtype=torch_dtype)
            else:
                # Default UNet path: optionally enable attention-gated skip connections
                unet = UNet2DConditionModel.from_pretrained(unet_path)
                if args.use_skip_attention_gates:
                    print("Attaching attention-gated skip connections to default UNet2DConditionModel")
                    add_attention_gates_to_unet(unet, verbose=True)
                    weight_bin = os.path.join(unet_path, "diffusion_pytorch_model.bin")
                    if os.path.exists(weight_bin):
                        state_dict = torch.load(weight_bin, map_location=args.device)
                        unet.load_state_dict(state_dict, strict=False)
                    else:
                        print(
                            "[WARN] diffusion_pytorch_model.bin not found under fine-tuned checkpoint; "
                            "skip attention gates will start from random init."
                        )
                unet.to(args.device, dtype=torch_dtype)
            
            # Check if this is a BioMedBERT model by looking for text_encoder directory
            text_encoder_path = os.path.join(args.model_used, 'text_encoder')
            if os.path.exists(text_encoder_path):
                print("Detected BioMedBERT model - loading custom text encoder and tokenizer")
                from utils.medical_text_encoder import load_med_encoder
                tokenizer, text_encoder = load_med_encoder(
                    model_id="microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext",
                    device=args.device,
                    dtype=torch_dtype,
                    trainable=False
                )
                
                pipe = StableDiffusionPipeline.from_pretrained(
                    args.pretrained_model,
                    unet=unet,
                    text_encoder=text_encoder,
                    torch_dtype=torch_dtype,
                    safety_checker=None,
                    requires_safety_checker=False,
                    use_safetensors=True
                ).to(args.device)
                # Stash BioMedBERT tokenizer for explicit embedding path
                setattr(pipe, "_is_biomedbert", True)
                setattr(pipe, "_med_tokenizer", tokenizer)
                # Force full-precision for stability
                pipe.to(dtype=torch.float32)
            else:
                pipe = StableDiffusionPipeline.from_pretrained(
                    args.pretrained_model,
                    unet=unet, 
                    torch_dtype=torch_dtype,
                    safety_checker=None,
                    requires_safety_checker=False,
                    use_safetensors=True
                ).to(args.device)
        # Swap scheduler to a safer default
        try:
            if args.scheduler == "dpm":
                pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
            elif args.scheduler == "euler_a":
                pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)
            elif args.scheduler == "euler":
                pipe.scheduler = EulerDiscreteScheduler.from_config(pipe.scheduler.config)
            elif args.scheduler == "pndm":
                pipe.scheduler = PNDMScheduler.from_config(pipe.scheduler.config)
        except Exception:
            pass
        
        # Align text encoder dtype for BioMedBERT pipelines to avoid Half/Float mismatches
        try:
            if getattr(pipe, "_is_biomedbert", False):
                pipe.text_encoder.to(args.device, dtype=torch_dtype)
        except Exception:
            pass
        # Memory optimizations for moderate GPUs
        try:
            pipe.enable_attention_slicing()
            pipe.enable_vae_slicing()
            pipe.enable_vae_tiling()
            # Many Pascal-era GPUs (e.g., TITAN Xp) are unstable with VAE in fp16
            pipe.vae.to(dtype=torch.float32)
            # Force classic attention (avoid PyTorch SDPA/Flash attention which may segfault)
            try:
                pipe.unet.set_attn_processor(AttnProcessor())
            except Exception:
                pass
            try:
                if torch.cuda.is_available():
                    torch.backends.cuda.enable_flash_sdp(False)
                    torch.backends.cuda.enable_mem_efficient_sdp(False)
                    torch.backends.cuda.enable_math_sdp(True)
            except Exception:
                pass
        except Exception:
            pass
        print(f"[INFO] Model successfully loaded to device: {args.device}")
        print(f"[INFO] Precision mode: {args.precision}")
        print(f"[INFO] Using scheduler: {args.scheduler}")
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
        # Check if this is a BioMedBERT model (flag set during load or presence of text_encoder dir)
        text_encoder_path = os.path.join(args.model_used, 'text_encoder') if args.model_used else None
        is_biomedbert = bool(getattr(pipe, "_is_biomedbert", False) or (text_encoder_path and os.path.exists(text_encoder_path)))
        
        for i in range(args.img_num):
            print(f"Generating image {i+1}/{args.img_num}...")
            
            # Set random seed for diversity (if not specified)
            if args.seed is None:
                import random
                current_seed = random.randint(0, 2**31 - 1)
                print(f"  Using random seed: {current_seed}")
            else:
                current_seed = args.seed
                print(f"  Using fixed seed: {current_seed}")
            
            # Set the generator for reproducible results
            generator = torch.Generator(device=args.device).manual_seed(current_seed)
            
            if is_biomedbert:
                # Use explicit prompt embeddings for BioMedBERT
                from utils.prompt_utils import tokenize_prompts
                med_tok = getattr(pipe, "_med_tokenizer", None)
                if med_tok is None:
                    print("Warning: BioMedBERT detected but tokenizer missing; falling back to pipeline tokenizer.")
                    med_tok = pipe.tokenizer
                
                # Tokenize main prompt
                toks = tokenize_prompts(med_tok, [args.prompt], max_len=256, device=args.device)
                
                # Tokenize negative prompt (if provided)
                neg_prompt = args.negative_prompt if args.negative_prompt else ""
                neg_toks = tokenize_prompts(med_tok, [neg_prompt], max_len=256, device=args.device)
                
                # Compute embeddings in float32 for stability then cast to UNet dtype
                with torch.no_grad():
                    prompt_embeds = pipe.text_encoder(
                        toks["input_ids"], attention_mask=toks.get("attention_mask")
                    ).last_hidden_state
                    negative_prompt_embeds = pipe.text_encoder(
                        neg_toks["input_ids"], attention_mask=neg_toks.get("attention_mask")
                    ).last_hidden_state

                embed_dtype = torch.float16 if args.precision == "float16" else torch.float32
                prompt_embeds = prompt_embeds.to(dtype=embed_dtype)
                negative_prompt_embeds = negative_prompt_embeds.to(dtype=embed_dtype)
                
                image = pipe(
                    prompt=None,
                    negative_prompt=None,
                    prompt_embeds=prompt_embeds,
                    negative_prompt_embeds=negative_prompt_embeds,
                    num_inference_steps=args.num_inference_steps,
                    guidance_scale=args.guidance_scale,
                    generator=generator,
                    height=args.height,
                    width=args.width,
                ).images[0]
            else:
                # Standard generation for CLIP-based models
                image = pipe(
                    prompt=args.prompt,
                    negative_prompt=args.negative_prompt if args.negative_prompt else None,
                    num_inference_steps=args.num_inference_steps,
                    guidance_scale=args.guidance_scale,
                    generator=generator,
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

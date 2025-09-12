# Core image generation using Stable Diffusion, generating medical images using stable diffusion
# Works with Policy and Selector for quality control

import torch
from diffusers import StableDiffusionPipeline
from diffusers.schedulers import DPMSolverMultistepScheduler
from diffusers.models.attention_processor import AttnProcessor
from typing import List, Dict, Tuple
import logging
from ..utils.device_utils import get_device, move_to_device
import torch.nn as nn
from torch.optim import lr_scheduler

class SyntheticSystem:
    def __init__(self, config: Dict):
        self.config = config
        self.device = get_device(config)
        self.max_attempts = config["synthetic_system"]["max_attempts"]
        
        self.model = StableDiffusionPipeline.from_pretrained(
            config["model_paths"]["stable_diffusion"],
            torch_dtype=torch.float16 if config["device"]["precision"] == "float16" else torch.float32,
            safety_checker=None,
            use_safetensors=True
        )
        try:
            self.model.scheduler = DPMSolverMultistepScheduler.from_config(self.model.scheduler.config)
        except Exception:
            pass

        use_half = (self.device.type == "cuda" and self.config["device"]["precision"] == "float16")
        self.model = move_to_device(self.model, self.device, use_half=use_half)
        # Memory optimizations for moderate VRAM GPUs
        try:
            self.model.enable_attention_slicing()
            self.model.enable_vae_slicing()
            self.model.enable_vae_tiling()
            # Stabilize VAE in float32 to avoid fp16 segfaults on older GPUs
            self.model.vae.to(dtype=torch.float32)
            # Force classic attention (avoid SDPA/Flash kernels)
            try:
                self.model.unet.set_attn_processor(AttnProcessor())
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
        
        if config["synthetic_system"]["enable_training"]:
            self.optimizer = torch.optim.Adam(
                self.model.unet.parameters(),
                lr=config["synthetic_system"]["training"]["learning_rate"],
                weight_decay=config["synthetic_system"]["training"]["optimizer"]["weight_decay"]
            )
            self.scheduler = lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=config["training"]["max_epochs"],
                eta_min=1e-6
            )
        else:
            self.optimizer = None
            self.scheduler = None
        
    def generate_images(self, prompts: List[str], selector, policy) -> Tuple[List, List[int]]:
        final_images = []
        attempts_list = []
        
        for prompt in prompts:
            image = None
            attempts = 0
            candidates = []
            
            # Stage 1: Collect candidate images that pass the selector
            while attempts < self.max_attempts and len(candidates) < self.config["generation"]["num_candidates"]:
                attempts += 1
                current_image = self.model(
                    prompt,
                    num_inference_steps=self.config["synthetic_system"]["num_inference_steps"],
                    guidance_scale=self.config["synthetic_system"]["guidance_scale"],
                    height=self.config["synthetic_system"]["image_size"][0],
                    width=self.config["synthetic_system"]["image_size"][1],
                    output_type="pil"
                ).images[0]
                
                selector_score = selector.evaluate_image(current_image)
                if selector_score >= self.config["generation"]["quality_threshold"]:
                    candidates.append(current_image)
                    logging.info(f"Found candidate {len(candidates)}/3 (selector score: {selector_score:.2f})")
            
            # Stage 2: Use policy to select the best matching image
            if candidates:
                best_score = 0
                for candidate in candidates:
                    policy_score = policy.evaluate_image(candidate, prompt)
                    if policy_score > best_score:
                        best_score = policy_score
                        image = candidate
                logging.info(f"Selected best candidate with policy score: {best_score:.2f}")
            
            if image is None:
                logging.warning(f"No suitable candidates found after {attempts} attempts, using last generated image")
                image = current_image
                
            final_images.append(image)
            attempts_list.append(attempts)
            
            logging.info(f"Generated image for prompt: {prompt} in {attempts} attempts")
            
        return final_images, attempts_list

    def update(self, images: torch.Tensor, prompts: List[str]) -> float:
        self.model.train()
        
        latents = self.model.vae.encode(images).latent_dist.sample()
        latents = latents * self.model.vae.config.scaling_factor
        
        text_inputs = self.model.tokenizer(
            prompts,
            padding="max_length",
            max_length=self.model.tokenizer.model_max_length,
            truncation=True,
            return_tensors="pt"
        ).to(self.device)
        
        text_embeddings = self.model.text_encoder(text_inputs.input_ids)[0]
        
        noise = torch.randn_like(latents)
        timesteps = torch.randint(0, self.model.scheduler.num_train_timesteps, (latents.shape[0],), device=self.device)
        noisy_latents = self.model.scheduler.add_noise(latents, noise, timesteps)
        
        noise_pred = self.model.unet(noisy_latents, timesteps, text_embeddings).sample
        
        loss = nn.MSELoss()(noise_pred, noise)
        
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        
        return loss.item()
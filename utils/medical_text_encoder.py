from transformers import AutoTokenizer, AutoModel
import torch
import os
from typing import Tuple

DEFAULT_MED_ENCODER = "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext"

def load_med_encoder(model_id: str = DEFAULT_MED_ENCODER,
                     device: torch.device | str = "cpu",
                     dtype: torch.dtype = torch.float32,
                     trainable: bool = False) -> Tuple[AutoTokenizer, AutoModel]:
    """Utility to load the BioMed-domain text tokenizer/encoder as a drop-in replacement for CLIP.

    Parameters
    ----------
    model_id: str
        HF hub model id.
    device: torch.device | str
        Torch device to place the encoder on.
    dtype: torch.dtype
        Data type for encoder parameters (fp32/fp16/bf16).
    trainable: bool
        If True the encoder parameters require gradients; else it is frozen.

    Returns
    -------
    tokenizer, model
    """
    # Check if we're in offline mode but allow downloading if PyTorch weights aren't cached
    local_files_only = os.environ.get("HF_HUB_OFFLINE", "0") == "1"
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_id, 
        padding_side="right",
        local_files_only=local_files_only
    )
    # increase max length to 256 for prompt conditioning; some tokenizers default higher.
    tokenizer.model_max_length = 256

    # Try to load PyTorch model, but if only Flax is available and we're offline,
    # temporarily allow downloading to get PyTorch weights
    try:
        text_encoder = AutoModel.from_pretrained(
            model_id,
            local_files_only=local_files_only
        )
    except (OSError, EnvironmentError) as e:
        if "Flax weights" in str(e) and local_files_only:
            # If offline mode but only Flax weights cached, temporarily allow download
            # to fetch PyTorch weights
            print(f"[WARN] Only Flax weights found in cache. Temporarily allowing download to fetch PyTorch weights...")
            text_encoder = AutoModel.from_pretrained(
                model_id,
                local_files_only=False
            )
        else:
            raise
    
    text_encoder.to(device, dtype=dtype)
    if not trainable:
        for p in text_encoder.parameters():
            p.requires_grad = False
        text_encoder.eval()

    return tokenizer, text_encoder

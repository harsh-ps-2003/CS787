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
        error_str = str(e)
        if ("Flax weights" in error_str or "pytorch_model.bin" in error_str) and local_files_only:
            # If offline mode but only Flax weights cached, temporarily allow download
            # to fetch PyTorch weights by unsetting offline environment variables
            print(f"[WARN] Only Flax weights found in cache. Temporarily allowing download to fetch PyTorch weights...")
            # Temporarily disable offline mode
            hf_offline = os.environ.pop("HF_HUB_OFFLINE", None)
            transformers_offline = os.environ.pop("TRANSFORMERS_OFFLINE", None)
            try:
                text_encoder = AutoModel.from_pretrained(
                    model_id,
                    local_files_only=False
                )
            finally:
                # Restore offline mode settings
                if hf_offline is not None:
                    os.environ["HF_HUB_OFFLINE"] = hf_offline
                if transformers_offline is not None:
                    os.environ["TRANSFORMERS_OFFLINE"] = transformers_offline
        else:
            raise
    
    text_encoder.to(device, dtype=dtype)
    if not trainable:
        for p in text_encoder.parameters():
            p.requires_grad = False
        text_encoder.eval()

    return tokenizer, text_encoder

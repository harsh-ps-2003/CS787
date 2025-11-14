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
    # Check if we're in offline mode (check both env vars)
    hf_offline = os.environ.get("HF_HUB_OFFLINE", "0") == "1"
    transformers_offline = os.environ.get("TRANSFORMERS_OFFLINE", "0") == "1"
    local_files_only = hf_offline or transformers_offline
    
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
    except (OSError, EnvironmentError, Exception) as e:
        error_str = str(e)
        error_type = type(e).__name__
        # Check if this is the Flax-only error (multiple possible error messages)
        is_flax_error = (
            "Flax weights" in error_str or 
            "pytorch_model.bin" in error_str or
            "from_flax=True" in error_str
        )
        
        if is_flax_error and local_files_only:
            # If offline mode but only Flax weights cached, temporarily allow download
            # to fetch PyTorch weights by unsetting offline environment variables
            print(f"[WARN] Detected Flax-only weights in cache (error: {error_type}).")
            print(f"[WARN] Temporarily allowing download to fetch PyTorch weights for {model_id}...")
            # Temporarily disable offline mode
            hf_offline_val = os.environ.pop("HF_HUB_OFFLINE", None)
            transformers_offline_val = os.environ.pop("TRANSFORMERS_OFFLINE", None)
            try:
                # First try to download PyTorch weights
                text_encoder = AutoModel.from_pretrained(
                    model_id,
                    local_files_only=False
                )
                print(f"[INFO] Successfully downloaded PyTorch weights for {model_id}")
            except Exception as download_error:
                download_error_str = str(download_error)
                # If download also fails with Flax-only error, the model might only have Flax weights
                if "Flax weights" in download_error_str or "pytorch_model.bin" in download_error_str:
                    print(f"[WARN] Model {model_id} appears to only have Flax weights on HuggingFace.")
                    print(f"[WARN] Attempting to convert Flax weights to PyTorch...")
                    try:
                        # Try loading with from_flax=True (if supported)
                        text_encoder = AutoModel.from_pretrained(
                            model_id,
                            from_flax=True,
                            local_files_only=False
                        )
                        print(f"[INFO] Successfully converted Flax weights to PyTorch for {model_id}")
                    except Exception as flax_error:
                        print(f"[ERROR] Failed to convert Flax weights: {flax_error}")
                        print(f"[ERROR] Please download PyTorch weights manually or use a different model.")
                        raise
                else:
                    print(f"[ERROR] Failed to download PyTorch weights: {download_error}")
                    raise
            finally:
                # Restore offline mode settings
                if hf_offline_val is not None:
                    os.environ["HF_HUB_OFFLINE"] = hf_offline_val
                if transformers_offline_val is not None:
                    os.environ["TRANSFORMERS_OFFLINE"] = transformers_offline_val
        else:
            # Re-raise if it's not the Flax error or if we're not in offline mode
            print(f"[ERROR] Failed to load model: {error_type}: {error_str}")
            raise
    
    text_encoder.to(device, dtype=dtype)
    if not trainable:
        for p in text_encoder.parameters():
            p.requires_grad = False
        text_encoder.eval()

    return tokenizer, text_encoder

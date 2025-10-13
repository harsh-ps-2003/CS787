from transformers import AutoTokenizer, AutoModel
import torch
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
    tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side="right")
    # increase max length to 256 for prompt conditioning; some tokenizers default higher.
    tokenizer.model_max_length = 256

    text_encoder = AutoModel.from_pretrained(model_id)
    text_encoder.to(device, dtype=dtype)
    if not trainable:
        for p in text_encoder.parameters():
            p.requires_grad = False
        text_encoder.eval()

    return tokenizer, text_encoder

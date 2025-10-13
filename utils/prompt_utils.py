from typing import List
from transformers import PreTrainedTokenizer
import torch

def tokenize_prompts(tokenizer: PreTrainedTokenizer, prompts: List[str], max_len: int = 256, device: torch.device | str = "cpu"):
    """Tokenize a list of prompts with consistent padding/truncation.

    Returns a dict of tensors suitable for passing to a HF transformer.
    """
    return tokenizer(prompts, return_tensors="pt", padding="max_length", truncation=True, max_length=max_len).to(device)

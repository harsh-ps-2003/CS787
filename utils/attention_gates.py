import types
from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionGate2D(nn.Module):
    """Attention gate for 2D skip connections.

    This is a lightweight variant of the attention-gated skip connections used in
    medical U-Net derivatives (e.g. Attention U-Net, TransUNet++). The gate takes:
    - a skip tensor x from the encoder
    - a gating tensor g from a deeper decoder/encoder feature map

    It learns to suppress irrelevant background structures and emphasise
    clinically salient regions (organ boundaries, vessels, lesions).

    Mathematically:
        q = ReLU(Theta(x) + Phi(g))
        alpha = sigmoid(Psi(q))          # spatial attention mask in [0, 1]
        x_out = x * alpha
    """

    def __init__(self, in_channels: int, gating_channels: int, inter_channels: int | None = None):
        super().__init__()
        if inter_channels is None:
            inter_channels = max(in_channels // 2, 1)

        # Project skip and gating to a common intermediate dimensionality
        self.theta = nn.Conv2d(in_channels, inter_channels, kernel_size=1, bias=False)
        self.phi = nn.Conv2d(gating_channels, inter_channels, kernel_size=1, bias=False)
        # Collapse to a single attention coefficient per spatial location
        self.psi = nn.Conv2d(inter_channels, 1, kernel_size=1, bias=True)
        self.act = nn.Sigmoid()
        self.relu = nn.ReLU(inplace=True)

        # Optional: simple channel-wise gating via SE-style squeeze-excitation
        self.use_channel_attention = True
        if self.use_channel_attention:
            self.channel_fc1 = nn.Conv2d(in_channels, max(in_channels // 2, 1), kernel_size=1)
            self.channel_fc2 = nn.Conv2d(max(in_channels // 2, 1), in_channels, kernel_size=1)

    def forward(self, x: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        """Apply spatial + optional channel attention to skip tensor `x`.

        Args
        ----
        x:
            Skip features from encoder, shape [B, C_x, H, W].
        g:
            Gating features (typically deeper, coarser scale), shape [B, C_g, H, W].
            The caller is responsible for resizing `g` to match the spatial size of `x`.
        """
        # Spatial attention
        theta_x = self.theta(x)
        phi_g = self.phi(g)
        q = self.relu(theta_x + phi_g)
        alpha = self.act(self.psi(q))  # [B, 1, H, W]
        x_spatial = x * alpha

        if not self.use_channel_attention:
            return x_spatial

        # Channel attention (SE-style): global pooling -> 2-layer MLP -> sigmoid
        b, c, _, _ = x.shape
        # Global average pool over spatial dimensions
        z = F.adaptive_avg_pool2d(x_spatial, 1)
        z = F.relu(self.channel_fc1(z), inplace=True)
        z = torch.sigmoid(self.channel_fc2(z))
        return x_spatial * z


def add_attention_gates_to_unet(unet: nn.Module, verbose: bool = True) -> nn.Module:
    """Attach attention gates to the down_blocks of a diffusers UNet2DConditionModel.

    Design:
    - We do *not* change the public API of the UNet; instead we:
      1. Inspect each `down_block` and its internal `resnets`.
      2. Instantiate an `AttentionGate2D` per skip feature tensor.
      3. Monkey-patch the block's `forward` method so that the returned
         `res_samples` are gated before being passed on as skip connections.

    - This keeps the overall Stable Diffusion UNet architecture intact and
      preserves pre-trained weights while giving us medically motivated
      attention-gated skips similar in spirit to
      “Improved UNet with Attention for Medical Image Segmentation”.

    Notes:
    - We assume `unet.down_blocks` exists (as in diffusers.UNet2DConditionModel).
    - Each down_block is expected to expose a `resnets` attribute whose
      `out_channels` match the channel dimension of the corresponding
      entries returned in `res_samples`.
    """

    if not hasattr(unet, "down_blocks"):
        raise ValueError("add_attention_gates_to_unet expects a UNet-like module with `down_blocks`.")

    for idx, block in enumerate(unet.down_blocks):
        resnets: Iterable[nn.Module] | None = getattr(block, "resnets", None)
        if resnets is None:
            # Some specialised blocks may not expose resnets (rare for SD v1.x)
            continue

        # Derive per-skip channel configuration from the existing ResNet blocks
        skip_channels: list[int] = []
        for resnet in resnets:
            out_ch = getattr(resnet, "out_channels", None)
            if out_ch is None:
                # Fallback: infer from weight shape if available
                conv1 = getattr(resnet, "conv1", None)
                if conv1 is not None and hasattr(conv1, "out_channels"):
                    out_ch = conv1.out_channels
            if out_ch is None:
                continue
            skip_channels.append(out_ch)

        if not skip_channels:
            continue

        # Create gates: each skip connection uses itself as the gating signal
        # (self-attention style) to avoid channel mismatch issues
        gates = nn.ModuleList(
            AttentionGate2D(in_channels=in_ch, gating_channels=in_ch)
            for in_ch in skip_channels
        )
        # Register as a proper submodule so parameters are saved with the UNet
        setattr(block, "skip_attention_gates", gates)

        orig_forward = block.forward

        def forward_with_gates(self, hidden_states, temb=None, encoder_hidden_states=None, **kwargs):
            """Wrapped forward that gates skip features before returning them."""
            # Call the original block to get its usual outputs
            sample, res_samples = orig_forward(
                hidden_states, temb=temb, encoder_hidden_states=encoder_hidden_states, **kwargs
            )

            # Defensive: some blocks may return res_samples as list/tuple; normalise
            if not isinstance(res_samples, (list, tuple)):
                return sample, res_samples

            # Safety check: ensure we have the right number of gates
            if len(res_samples) != len(self.skip_attention_gates):
                # Mismatch - return original without gating to avoid errors
                return sample, res_samples

            gated_res_samples = []
            for res, gate in zip(res_samples, self.skip_attention_gates):
                # Verify channel dimensions match before applying gate
                expected_channels = gate.theta.in_channels
                if res.shape[1] != expected_channels:
                    # Channel mismatch - skip gating for this tensor
                    gated_res_samples.append(res)
                    continue
                
                # Use the skip connection itself as the gating signal (self-attention)
                # This avoids channel mismatch and still provides spatial attention
                gated_res_samples.append(gate(res, res))

            # Preserve the original return type (tuple)
            return sample, tuple(gated_res_samples)

        # Bind the method dynamically so each block keeps its own closure
        block.forward = types.MethodType(forward_with_gates, block)

        if verbose:
            print(
                f"[skip-attn] Attached {len(skip_channels)} attention gate(s) "
                f"to down_block[{idx}] with skip channels {skip_channels}"
            )

    return unet



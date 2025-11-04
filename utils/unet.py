
import torch
import torchvision.transforms as T
import torch.nn as nn
import matplotlib.image
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import math



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TimeEmbedding(nn.Module):
    """Sinusoidal time embedding for diffusion timesteps."""
    def __init__(self, dim):
        super(TimeEmbedding, self).__init__()
        self.dim = dim
        self.linear1 = nn.Linear(dim, dim * 4)
        self.linear2 = nn.Linear(dim * 4, dim)
        self.act = nn.SiLU()
    
    def sinusoidal_embedding(self, timesteps, dim):
        """Generate sinusoidal positional embeddings for timesteps."""
        half_dim = dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half_dim, dtype=torch.float32) / (half_dim - 1))
        freqs = freqs.to(timesteps.device)
        args = timesteps[:, None].float() * freqs[None]  # shape: [B, half_dim]
        temb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)  # shape: [B, dim]
        return temb

    def forward(self, t):
        """Forward pass: sinusoidal embedding -> MLP."""
        emb = self.sinusoidal_embedding(t, self.dim)
        emb = self.linear1(emb)
        emb = self.act(emb)
        emb = self.linear2(emb)
        return emb


class CrossAttention(nn.Module):
    """Cross-attention layer for text conditioning in UNet."""
    def __init__(self, query_dim, cross_attention_dim, heads=8, dim_head=64):
        super(CrossAttention, self).__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.scale = dim_head ** -0.5
        
        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(cross_attention_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(cross_attention_dim, inner_dim, bias=False)
        self.to_out = nn.Linear(inner_dim, query_dim)
        self.norm = nn.LayerNorm(query_dim)
        
    def forward(self, x, context, mask=None):
        """
        Args:
            x: [B, C, H, W] - spatial features
            context: [B, seq_len, context_dim] - text embeddings
            mask: Optional attention mask
        Returns:
            out: [B, C, H, W] - attended features
        """
        b, c, h, w = x.shape
        # Reshape spatial features to sequence: [B, H*W, C]
        x_flat = x.permute(0, 2, 3, 1).reshape(b, h * w, c)
        x_norm = self.norm(x_flat)
        
        # Compute queries, keys, values
        q = self.to_q(x_norm).reshape(b, h * w, self.heads, -1).permute(0, 2, 1, 3)  # [B, heads, H*W, dim_head]
        k = self.to_k(context).reshape(b, context.shape[1], self.heads, -1).permute(0, 2, 1, 3)  # [B, heads, seq_len, dim_head]
        v = self.to_v(context).reshape(b, context.shape[1], self.heads, -1).permute(0, 2, 1, 3)  # [B, heads, seq_len, dim_head]
        
        # Attention
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        attn = torch.softmax(scores, dim=-1)
        out = torch.matmul(attn, v)  # [B, heads, H*W, dim_head]
        
        # Reshape and project
        out = out.permute(0, 2, 1, 3).reshape(b, h * w, -1)
        out = self.to_out(out)
        
        # Residual connection and reshape back
        out = out + x_flat
        out = out.reshape(b, h, w, c).permute(0, 3, 1, 2)
        return out

class SelfAttention(nn.Module):
    def __init__(self, channels):
        super(SelfAttention, self).__init__()

        self.channels = channels
        self.query = nn.Conv2d(channels, channels // 4, 1)
        self.key = nn.Conv2d(channels, channels // 4, 1)
        self.value = nn.Conv2d(channels, channels, 1)
        self.proj = nn.Conv2d(channels, channels, 1)  # Projection layer
        self.norm = nn.GroupNorm(32, channels, eps=1e-6)  # Normalization layer

    def forward(self, x):
        b,c,h,w = x.shape

        x = self.norm(x)  # Apply normalization

        query = self.query(x).reshape(b, c // 4, h * w).permute(0, 2, 1)
        key = self.key(x).reshape(b, c // 4, h * w)
        value = self.value(x).reshape(b, c, h * w).permute(0, 2, 1)

        attention = torch.bmm(query, key)  # Batch matrix multiplication
        attention = torch.softmax(attention, dim=-1)

        out = torch.bmm(attention, value)  # Apply attention

        out = out.permute(0, 2, 1).reshape(b, c, h, w)  # Reshape back to original dimensions

        return self.proj(out + x)  # Residual connection

        


class MedicalUNet(nn.Module):
    """
    Custom UNet for medical image diffusion with cross-attention for text conditioning.
    Compatible with diffusers pipeline interface.
    """
    def __init__(self, in_channels=4, out_channels=4, cross_attention_dim=768, time_embed_dim=320, device="cuda"):
        super(MedicalUNet, self).__init__()

        self.device = device
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.cross_attention_dim = cross_attention_dim
        self.time_embed_dim = time_embed_dim

        # Time embedding
        self.time_embed = TimeEmbedding(time_embed_dim)
        self.time_proj = nn.Linear(time_embed_dim, 64)  # Project to channel dimension
        
        # Encoder blocks
        self.conv11 = nn.Conv2d(self.in_channels, 64, kernel_size=3, padding=1)
        self.conv12 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.time_mlp1 = nn.Linear(time_embed_dim, 64)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv21 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv22 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.time_mlp2 = nn.Linear(time_embed_dim, 128)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv31 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv32 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.time_mlp3 = nn.Linear(time_embed_dim, 256)
        self.sa256down = SelfAttention(256)
        self.cross_attn256down = CrossAttention(256, cross_attention_dim)
        self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv41 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.conv42 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.time_mlp4 = nn.Linear(time_embed_dim, 512)
        self.sa512down = SelfAttention(512)
        self.cross_attn512down = CrossAttention(512, cross_attention_dim)
        self.maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Bottleneck
        self.conv51 = nn.Conv2d(512, 1024, kernel_size=3, padding=1)
        self.conv52 = nn.Conv2d(1024, 1024, kernel_size=3, padding=1)
        self.time_mlp5 = nn.Linear(time_embed_dim, 1024)
        self.sa1024mid = SelfAttention(1024)
        self.cross_attn1024mid = CrossAttention(1024, cross_attention_dim)

        # Decoder blocks
        self.upconv4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.conv61 = nn.Conv2d(1024, 512, kernel_size=3, padding=1)
        self.conv62 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.time_mlp6 = nn.Linear(time_embed_dim, 512)
        self.sa512up = SelfAttention(512)
        self.cross_attn512up = CrossAttention(512, cross_attention_dim)

        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv71 = nn.Conv2d(512, 256, kernel_size=3, padding=1)
        self.conv72 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.time_mlp7 = nn.Linear(time_embed_dim, 256)
        self.sa256up = SelfAttention(256)
        self.cross_attn256up = CrossAttention(256, cross_attention_dim)

        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv81 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.conv82 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.time_mlp8 = nn.Linear(time_embed_dim, 128)

        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv91 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.conv92 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.time_mlp9 = nn.Linear(time_embed_dim, 64)

        self.conv10 = nn.Conv2d(64, self.out_channels, kernel_size=1)

    def forward(self, sample, timestep, encoder_hidden_states=None, return_dict=False):
        """
        Forward pass compatible with diffusers UNet2DConditionModel interface.
        
        Args:
            sample: [B, C, H, W] - noisy latents
            timestep: [B] - diffusion timesteps
            encoder_hidden_states: [B, seq_len, cross_attention_dim] - text embeddings
            return_dict: Whether to return a dict (for diffusers compatibility)
        
        Returns:
            Object with .sample attribute containing predicted noise
        """
        x = sample
        
        # Time embedding
        if timestep.dim() == 0:
            timestep = timestep.unsqueeze(0)
        if timestep.dim() == 1 and timestep.shape[0] != x.shape[0]:
            timestep = timestep.expand(x.shape[0])
        
        temb = self.time_embed(timestep)  # [B, time_embed_dim]
        
        # Encoder
        x1 = nn.SiLU()(self.conv11(x))
        time_emb_1 = self.time_mlp1(temb).view(-1, 64, 1, 1).expand_as(x1)
        x2 = nn.SiLU()(self.conv12(x1)) + time_emb_1
        c1 = x2

        x = self.maxpool1(x2)
        x1 = nn.SiLU()(self.conv21(x))
        time_emb_2 = self.time_mlp2(temb).view(-1, 128, 1, 1).expand_as(x1)
        x2 = nn.SiLU()(self.conv22(x1)) + time_emb_2
        c2 = x2

        x = self.maxpool2(x2)
        x1 = nn.SiLU()(self.conv31(x))
        time_emb_3 = self.time_mlp3(temb).view(-1, 256, 1, 1).expand_as(x1)
        x2 = nn.SiLU()(self.conv32(x1)) + time_emb_3
        x2 = self.sa256down(x2)
        if encoder_hidden_states is not None:
            x2 = self.cross_attn256down(x2, encoder_hidden_states)
        c3 = x2

        x = self.maxpool3(x2)
        x1 = nn.SiLU()(self.conv41(x))
        time_emb_4 = self.time_mlp4(temb).view(-1, 512, 1, 1).expand_as(x1)
        x2 = nn.SiLU()(self.conv42(x1)) + time_emb_4
        x2 = self.sa512down(x2)
        if encoder_hidden_states is not None:
            x2 = self.cross_attn512down(x2, encoder_hidden_states)
        c4 = x2

        x = self.maxpool4(x2)
        x1 = nn.SiLU()(self.conv51(x))
        time_emb_5 = self.time_mlp5(temb).view(-1, 1024, 1, 1).expand_as(x1)
        x2 = nn.SiLU()(self.conv52(x1)) + time_emb_5
        x2 = self.sa1024mid(x2)
        if encoder_hidden_states is not None:
            x2 = self.cross_attn1024mid(x2, encoder_hidden_states)

        # Decoder
        x = self.upconv4(x2)
        x = torch.cat((x, c4), dim=1)
        x1 = nn.SiLU()(self.conv61(x))
        time_emb_6 = self.time_mlp6(temb).view(-1, 512, 1, 1).expand_as(x1)
        x2 = nn.SiLU()(self.conv62(x1)) + time_emb_6
        x2 = self.sa512up(x2)
        if encoder_hidden_states is not None:
            x2 = self.cross_attn512up(x2, encoder_hidden_states)

        x = self.upconv3(x2)
        x = torch.cat((x, c3), dim=1)
        x1 = nn.SiLU()(self.conv71(x))
        time_emb_7 = self.time_mlp7(temb).view(-1, 256, 1, 1).expand_as(x1)
        x2 = nn.SiLU()(self.conv72(x1)) + time_emb_7
        x2 = self.sa256up(x2)
        if encoder_hidden_states is not None:
            x2 = self.cross_attn256up(x2, encoder_hidden_states)

        x = self.upconv2(x2)
        x = torch.cat((x, c2), dim=1)
        x1 = nn.SiLU()(self.conv81(x))
        time_emb_8 = self.time_mlp8(temb).view(-1, 128, 1, 1).expand_as(x1)
        x2 = nn.SiLU()(self.conv82(x1)) + time_emb_8

        x = self.upconv1(x2)
        x = torch.cat((x, c1), dim=1)
        x1 = nn.SiLU()(self.conv91(x))
        time_emb_9 = self.time_mlp9(temb).view(-1, 64, 1, 1).expand_as(x1)
        x2 = nn.SiLU()(self.conv92(x1)) + time_emb_9

        out = self.conv10(x2)
        
        # Return object with .sample attribute for diffusers compatibility
        class UNetOutput:
            def __init__(self, sample):
                self.sample = sample
        
        return UNetOutput(out)


# Alias for backward compatibility
unet = MedicalUNet
    


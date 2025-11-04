
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
    def __init__(self, dim):
        super(TimeEmbedding, self).__init__()
        self.dim = dim
        self.linear = nn.Linear(dim, dim)
    
    def sinusoidal_embedding(self, timesteps, dim):
        half_dim = dim // 2
        freqs = torch.exp(-math.log(10000) * torch.arange(half_dim, dtype=torch.float32) / (half_dim - 1))
        freqs = freqs.to(timesteps.device)  # move to same device
        args = timesteps[:, None].float() * freqs[None]  # shape: [B, half_dim]
        temb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)  # shape: [B, dim]
        return temb


    def forward(self, t):
        return self.linear(self.sinusoidal_embedding(t, self.dim))

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

        


class unet(nn.Module):
    def __init__(self, channels, device="cuda"):
        super(unet, self).__init__()

        self.device = device
        self.in_channels = channels
        self.out_channels = channels

        self.conv11 = nn.Conv2d(self.in_channels, 64, kernel_size=3, padding=1)
        self.conv12 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv21 = nn.Conv2d(64,128, kernel_size=3, padding=1)
        self.conv22 = nn.Conv2d(128, 128, kernel_size=3, padding=1)

        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv31 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv32 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.sa256down = SelfAttention(256)  # Self-attention layer for the second block


        self.maxpool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv41 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.conv42 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.sa512down = SelfAttention(512)  # Self-attention layer for the third block

        self.maxpool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv51 = nn.Conv2d(512, 1024, kernel_size=3, padding=1)
        self.conv52 = nn.Conv2d(1024, 1024, kernel_size=3, padding=1)
        self.sa1024mid = SelfAttention(1024)  # Self-attention layer for the fifth block 

        self.upconv4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)

        self.conv61 = nn.Conv2d(1024, 512, kernel_size=3, padding=1)
        self.conv62 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.sa512up = SelfAttention(512)  # Self-attention layer for the fourth block


        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)

        self.conv71 = nn.Conv2d(512, 256, kernel_size=3, padding=1)
        self.conv72 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.sa256up = SelfAttention(256)  # Self-attention layer for the third block

        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)

        self.conv81 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.conv82 = nn.Conv2d(128, 128, kernel_size=3, padding=1)

        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)

        self.conv91 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.conv92 = nn.Conv2d(64, 64, kernel_size=3, padding=1)

        self.conv10 = nn.Conv2d(64, self.out_channels, kernel_size=1)


        # self.time_embedding = TimeEmbedding(256)  # Time embedding layer


    def time_embedding(self, t, channels, embed_dim):

        # inv_freq = 1.0 / (
        #     10000
        #     ** (torch.arange(0, channels, 2, device=self.device).float() / channels)
        # )
        # pos_enc_a = torch.sin(t.repeat(1, channels // 2) * inv_freq)
        # pos_enc_b = torch.cos(t.repeat(1, channels // 2) * inv_freq)
        # pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
        # return pos_enc.view(-1, channels, 1, 1).repeat(1, 1, embed_dim, embed_dim)

        
        if t.dim() == 1:
            t = t.unsqueeze(1)

        half = channels // 2


        freqs = torch.exp(
        -torch.arange(0, half, dtype=torch.float32, device=device) * (math.log(10000.0) / half)
        ) 
        
        args = t.float() * freqs

        sin = torch.sin(args)
        cos = torch.cos(args)
        emb = torch.cat([sin, cos], dim=-1)  # shape: [B, channels, 1, 1]

        emb = torch.cat([sin, cos], dim=-1)  # [B, channels]


        emb = emb.view(-1, channels, 1, 1)  # [B, C, 1, 1]
        emb = emb.expand(-1, -1, embed_dim, embed_dim)  # [B, C, H, W]
        return emb


    def forward(self, x, time):
        # time256 = self.time_embedding(time).view(x.size(0), -1, 1, 1)  # Reshape time embedding for broadcasting

        x1 = nn.SiLU()(self.conv11(x))
        x2 = nn.SiLU()(self.conv12(x1)) + self.time_embedding(time, x1.size(1), x1.size(2))
        c1 = x2

        x = self.maxpool1(x2)
        x1 = nn.SiLU()(self.conv21(x))
        x2 = nn.SiLU()(self.conv22(x1)) + self.time_embedding(time, x1.size(1), x1.size(2))
        c2 = x2

        x = self.maxpool2(x2)
        x1 = nn.SiLU()(self.conv31(x))
        x2 = nn.SiLU()(self.conv32(x1)) + self.time_embedding(time, x1.size(1), x1.size(2))
        c3 = x2
        x2 = self.sa256down(x2)


        x = self.maxpool3(x2)
        x1 = nn.SiLU()(self.conv41(x))
        x2 = nn.SiLU()(self.conv42(x1)) + self.time_embedding(time, x1.size(1), x1.size(2))
        c4 = x2
        x2 = self.sa512down(x2)

        x = self.maxpool4(x2)
        x1 = nn.SiLU()(self.conv51(x))
        x2 = nn.SiLU()(self.conv52(x1)) + self.time_embedding(time, x1.size(1), x1.size(2))
        x2 = self.sa1024mid(x2)

        x = self.upconv4(x2)
        x = torch.cat((x, c4), dim=1)
        x1 = nn.SiLU()(self.conv61(x))
        x2 = nn.SiLU()(self.conv62(x1)) + self.time_embedding(time, x1.size(1), x1.size(2))
        x2 = self.sa512up(x2)

        x = self.upconv3(x2)
        x = torch.cat((x, c3), dim=1)
        x1 = nn.SiLU()(self.conv71(x))        
        x2 = nn.SiLU()(self.conv72(x1)) + self.time_embedding(time, x1.size(1), x1.size(2))
        x2 = self.sa256up(x2)

        x = self.upconv2(x2)
        x = torch.cat((x, c2), dim=1)
        x1 = nn.SiLU()(self.conv81(x))
        x2 = nn.SiLU()(self.conv82(x1)) + self.time_embedding(time, x1.size(1), x1.size(2))

        x = self.upconv1(x2)
        x = torch.cat((x, c1), dim=1)
        x1 = nn.SiLU()(self.conv91(x))
        x2 = nn.SiLU()(self.conv92(x1)) + self.time_embedding(time, x1.size(1), x1.size(2))

        out = self.conv10(x2)
        return out
    


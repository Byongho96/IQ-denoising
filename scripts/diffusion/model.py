import torch
import torch.nn as nn
import math

class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class DoubleConv(nn.Module):
    """
    기존 DoubleConv에 time_emb를 더할 수 있도록 수정
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)

class DiffusionUNet1D(nn.Module):
    def __init__(self, in_channels=4, out_channels=2, time_emb_dim=32):
        super().__init__()
        
        # Time Embedding MLP
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.ReLU()
        )

        # Encoder
        self.dconv_down1 = DoubleConv(in_channels, 64)
        self.dconv_down2 = DoubleConv(64, 128)
        self.dconv_down3 = DoubleConv(128, 256)
        self.dconv_down4 = DoubleConv(256, 512)        

        self.maxpool = nn.MaxPool1d(2)
        
        # Time Projection Layers (Time Embedding을 각 레이어 채널에 맞게 변환)
        self.time_proj1 = nn.Linear(time_emb_dim, 64)
        self.time_proj2 = nn.Linear(time_emb_dim, 128)
        self.time_proj3 = nn.Linear(time_emb_dim, 256)
        self.time_proj4 = nn.Linear(time_emb_dim, 512)

        # Decoder
        self.upsample = nn.Upsample(scale_factor=2, mode='linear', align_corners=True)        
        
        self.dconv_up3 = DoubleConv(256 + 512, 256)
        self.dconv_up2 = DoubleConv(128 + 256, 128)
        self.dconv_up1 = DoubleConv(64 + 128, 64)
        
        self.conv_last = nn.Conv1d(64, out_channels, 1)
        
    def forward(self, x, t):
        # x: (Batch, 4, Seq_Len) -> 2ch Latent + 2ch Condition
        # t: (Batch,)
        
        # Padding
        original_len = x.size(2)
        pad_len = (8 - (original_len % 8)) % 8
        if pad_len > 0:
            x = torch.nn.functional.pad(x, (0, pad_len))
            
        # Time Embedding
        t_emb = self.time_mlp(t) # (Batch, time_emb_dim)
        
        # Encoder
        # Down 1
        x1 = self.dconv_down1(x)
        # Add Time Emb (Broadcast over sequence length)
        x1 = x1 + self.time_proj1(t_emb)[:, :, None] 
        p1 = self.maxpool(x1)

        # Down 2
        x2 = self.dconv_down2(p1)
        x2 = x2 + self.time_proj2(t_emb)[:, :, None]
        p2 = self.maxpool(x2)
        
        # Down 3
        x3 = self.dconv_down3(p2)
        x3 = x3 + self.time_proj3(t_emb)[:, :, None]
        p3 = self.maxpool(x3)
        
        # Bottleneck
        x4 = self.dconv_down4(p3)
        x4 = x4 + self.time_proj4(t_emb)[:, :, None]
        
        # Decoder
        u3 = self.upsample(x4)
        u3 = torch.cat([u3, x3], dim=1)
        d3 = self.dconv_up3(u3)
        
        u2 = self.upsample(d3)
        u2 = torch.cat([u2, x2], dim=1)
        d2 = self.dconv_up2(u2)
        
        u1 = self.upsample(d2)
        u1 = torch.cat([u1, x1], dim=1)
        d1 = self.dconv_up1(u1)
        
        out = self.conv_last(d1)
        
        # Remove Padding
        if pad_len > 0:
            out = out[..., :original_len]
            
        return out
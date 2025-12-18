import torch
import torch.nn as nn

class DoubleConv(nn.Module):
    """
    Building block: (Conv1D -> BatchNorm -> ReLU) * 2
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            # nn.Dropout(p=0.3),
            nn.Conv1d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            # nn.Dropout(p=0.3)
        )

    def forward(self, x):
        return self.conv(x)

class ComplexUNet1D(nn.Module):
    def __init__(self, in_channels=2, out_channels=2):
        super().__init__()
        
        # Encoder (Downsampling)
        self.dconv_down1 = DoubleConv(in_channels, 64)
        self.dconv_down2 = DoubleConv(64, 128)
        self.dconv_down3 = DoubleConv(128, 256)
        self.dconv_down4 = DoubleConv(256, 512)        

        self.maxpool = nn.MaxPool1d(2)  # 2x downsampling
        
        # Decoder (Upsampling)
        self.upsample = nn.Upsample(scale_factor=2, mode='linear', align_corners=True)        
        
        self.dconv_up3 = DoubleConv(256 + 512, 256)
        self.dconv_up2 = DoubleConv(128 + 256, 128)
        self.dconv_up1 = DoubleConv(64 + 128, 64)
        
        self.conv_last = nn.Conv1d(64, out_channels, 1)
        
    def forward(self, x):
        # x shape: (Batch, 2, Seq_Len=111)
        # Padding to make length a multiple of 8 (Ex. 112)
        original_len = x.size(2)
        pad_len = (8 - (original_len % 8)) % 8
        if pad_len > 0:
            x = torch.nn.functional.pad(x, (0, pad_len))
        
        # Encoder
        conv1 = self.dconv_down1(x)
        x1 = self.maxpool(conv1) # 112 -> 56

        conv2 = self.dconv_down2(x1)
        x2 = self.maxpool(conv2) # 56 -> 28
        
        conv3 = self.dconv_down3(x2)
        x3 = self.maxpool(conv3) # 28 -> 14
        
        x4 = self.dconv_down4(x3) # 14 (Bottleneck)
        
        # Decoder
        x = self.upsample(x4) # 14 -> 28
        x = torch.cat([x, conv3], dim=1)
        x = self.dconv_up3(x)
        
        x = self.upsample(x) # 28 -> 56
        x = torch.cat([x, conv2], dim=1)
        x = self.dconv_up2(x)
        
        x = self.upsample(x) # 56 -> 112
        x = torch.cat([x, conv1], dim=1)
        x = self.dconv_up1(x)
        
        out = self.conv_last(x)
        
        # Remove padding
        if pad_len > 0:
            out = out[..., :original_len]
            
        return out
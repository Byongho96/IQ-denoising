import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    """
    (Conv1d => BatchNorm => ReLU) * 2
    U-Net의 각 단계에서 특징을 추출하는 기본 블록입니다.
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class ComplexUNet1D(nn.Module):
    def __init__(self, in_channels=2, out_channels=2, features=[64, 128, 256, 512]):
        super().__init__()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)

        # 1. Encoder (Downsampling Path)
        for feature in features:
            self.downs.append(DoubleConv(in_channels, feature))
            in_channels = feature

        # 2. Bottleneck (가장 깊은 곳)
        self.bottleneck = DoubleConv(features[-1], features[-1]*2)

        # 3. Decoder (Upsampling Path)
        for feature in reversed(features):
            self.ups.append(
                nn.ConvTranspose1d(
                    feature*2, feature, kernel_size=2, stride=2
                )
            )
            self.ups.append(DoubleConv(feature*2, feature))

        # 4. Final Output Layer
        self.final_conv = nn.Conv1d(features[0], out_channels, kernel_size=1)
        
        # IQ 데이터는 -1 ~ 1 사이의 값이므로 Tanh 사용
        self.final_activation = nn.Tanh()

    def forward(self, x):
        # --- Padding Logic ---
        # U-Net은 2의 제곱수 형태의 길이에서 가장 잘 작동합니다.
        # 입력 길이 111을 128(2^7)로 패딩합니다.
        original_length = x.shape[2]
        target_length = 128 
        pad_size = target_length - original_length
        
        # 마지막 차원(시간 축)에 대해 패딩 (왼쪽 0, 오른쪽 pad_size)
        x = F.pad(x, (0, pad_size), "constant", 0)
        
        skip_connections = []

        # Downsampling
        for down in self.downs:
            x = down(x)
            skip_connections.append(x)
            x = self.pool(x)

        # Bottleneck
        x = self.bottleneck(x)

        # Skip connections를 역순으로 가져옴
        skip_connections = skip_connections[::-1]

        # Upsampling
        for idx in range(0, len(self.ups), 2):
            x = self.ups[idx](x) # ConvTranspose
            skip_connection = skip_connections[idx//2]

            # (혹시 모를 사이즈 불일치 대비, 여기선 패딩으로 맞춰서 문제없음)
            if x.shape != skip_connection.shape:
                x = F.interpolate(x, size=skip_connection.shape[2:], mode='linear', align_corners=True)

            # Skip Connection 결합 (Channel 방향 Concatenation)
            concat_skip = torch.cat((skip_connection, x), dim=1)
            x = self.ups[idx+1](concat_skip) # Double Conv

        # Final Layer
        x = self.final_conv(x)
        x = self.final_activation(x)

        # --- Cropping Logic ---
        # 패딩했던 부분을 잘라내어 원래 길이(111)로 복원
        return x[:, :, :original_length]

# ==========================================
# 4. 모델 테스트 (Shape 확인용)
# ==========================================
if __name__ == "__main__":
    # 모델 인스턴스 생성
    model = ComplexUNet1D(in_channels=2, out_channels=2) # I, Q input -> I, Q output
    
    # 더미 데이터 생성 (Batch=8, Channel=2, Length=111)
    dummy_input = torch.randn(8, 2, 111)
    
    # Forward Pass
    output = model(dummy_input)
    
    print(f"Input Shape: {dummy_input.shape}")
    print(f"Output Shape: {output.shape}") # (8, 2, 111)이어야 함
    
    # 값 범위 확인 (Tanh 적용 여부)
    print(f"Output Max: {output.max().item():.4f}, Output Min: {output.min().item():.4f}")
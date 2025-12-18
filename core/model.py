import torch
import torch.nn as nn

class ComplexDenoisingNet(nn.Module):
    def __init__(self, seq_len=111):
        super(ComplexDenoisingNet, self).__init__()
        
        # --------------------------------------------------------
        # [설계 의도]
        # 1. 입력: (Batch, 2, 111) -> (I, Q) 2채널
        # 2. 구조: 시퀀스 길이(111)를 유지하는 Deep CNN (ResNet 스타일)
        # 3. 특성: Circular Padding을 사용하여 안테나 8번과 1번의 연결성 보존
        # --------------------------------------------------------

        # 첫 번째 특징 추출 (2 -> 64 채널)
        self.input_layer = nn.Sequential(
            # padding_mode='circular': UCA 구조 반영 (매우 중요)
            nn.Conv1d(in_channels=2, out_channels=64, kernel_size=3, padding=1, padding_mode='circular'),
            nn.BatchNorm1d(64),
            nn.ReLU()
        )

        # 깊은 특징 학습 (Residual Blocks)
        # 길이를 줄이지 않고(Pooling 없음), 깊이만 깊게 쌓아 복잡한 노이즈 패턴 학습
        self.res_block1 = self._make_res_block(64, 64)
        self.res_block2 = self._make_res_block(64, 128)
        self.res_block3 = self._make_res_block(128, 64)

        # 출력 레이어 (64 -> 2 채널 복원)
        self.output_layer = nn.Sequential(
            nn.Conv1d(in_channels=64, out_channels=2, kernel_size=3, padding=1, padding_mode='circular'),
            # 주의: 여기서 Tanh나 Sigmoid를 쓰지 않습니다. 값의 범위가 열려있기 때문입니다.
            # 대신 뒤에서 Normalize를 수행합니다.
        )

    def _make_res_block(self, in_ch, out_ch):
        """Residual Block 생성 헬퍼 함수"""
        layers = [
            nn.Conv1d(in_ch, out_ch, kernel_size=3, padding=1, padding_mode='circular'),
            nn.BatchNorm1d(out_ch),
            nn.ReLU(),
            nn.Conv1d(out_ch, out_ch, kernel_size=3, padding=1, padding_mode='circular'),
            nn.BatchNorm1d(out_ch)
        ]
        
        # 입력과 출력 채널이 다를 경우 스킵 커넥션을 위해 1x1 Conv 사용
        shortcut = nn.Sequential()
        if in_ch != out_ch:
            shortcut = nn.Conv1d(in_ch, out_ch, kernel_size=1)
            
        return ResidualBlock(layers, shortcut)

    def forward(self, x):
        # x shape: (Batch, 2, 111)
        
        # 1. 초기 특징 추출
        out = self.input_layer(x)
        
        # 2. Residual Blocks 통과 (Skip Connection이 내부에서 작동)
        out = self.res_block1(out)
        out = self.res_block2(out)
        out = self.res_block3(out)
        
        # 3. IQ 차원으로 복원
        out = self.output_layer(out) # (Batch, 2, 111)
        
        # 4. [중요] Amplitude Normalization (Physics Constraint)
        # 위상차(AoA)에는 진폭 정보가 필요 없고, 모두 단위 원(Unit Circle) 위에 있어야 함.
        # 출력값 (I, Q)를 벡터의 크기로 나누어 정규화.
        
        # 분모에 아주 작은 값(eps)을 더해 0으로 나누는 것 방지
        norm = torch.sqrt(out[:, 0, :]**2 + out[:, 1, :]**2 + 1e-8)
        
        # (Batch, 111) -> (Batch, 1, 111) 로 차원 늘려서 브로드캐스팅 나눗셈
        out_normalized = out / norm.unsqueeze(1)
        
        return out_normalized

# Residual Block 클래스 정의 (위에서 호출됨)
class ResidualBlock(nn.Module):
    def __init__(self, layers, shortcut):
        super(ResidualBlock, self).__init__()
        self.layers = nn.Sequential(*layers)
        self.shortcut = shortcut
        self.relu = nn.ReLU()
        
    def forward(self, x):
        residual = x
        out = self.layers(x)
        # 차원이 다르면 1x1 conv로 맞춰줌
        residual = self.shortcut(residual) 
        out += residual
        out = self.relu(out)
        return out
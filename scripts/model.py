import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import shutil

# ==========================================
# 1. Configuration (설정)
# ==========================================
CONFIG = {
    'n_antennas': 8,          # 안테나 개수
    'samples_per_block': 3,   # 안테나당 연속 샘플 수
    'batch_size': 16,
    'lr': 0.0002,             # Learning Rate
    'epochs': 50,             # 학습 반복 횟수
    'lambda_L1': 100,         # Content Loss 가중치
    'lambda_phase': 5.0,      # ★ Phase Loss 가중치 (중요: 위상 학습 강화)
    'device': 'cuda' if torch.cuda.is_available() else 'cpu'
}

# ==========================================
# 2. Dataset Loader (1:1 Paired)
# ==========================================
class PairedIQDataset(Dataset):
    def __init__(self, noisy_dir, clean_dir, n_antennas=8, samples_per_block=3):
        self.n_antennas = n_antennas
        self.samples_per_block = samples_per_block
        
        # 파일 목록 가져오기 (이름 순 정렬하여 매칭)
        self.noisy_files = sorted(glob.glob(os.path.join(noisy_dir, "*.csv")))
        self.clean_files = sorted(glob.glob(os.path.join(clean_dir, "*.csv")))
        
        # 파일 개수 확인
        if len(self.noisy_files) != len(self.clean_files):
            print(f"[Warning] 파일 개수 불일치! Noisy: {len(self.noisy_files)}, Clean: {len(self.clean_files)}")
            # 개수가 다르면 적은 쪽에 맞춤 (안전장치)
            min_len = min(len(self.noisy_files), len(self.clean_files))
            self.noisy_files = self.noisy_files[:min_len]
            self.clean_files = self.clean_files[:min_len]
        
        self.data_pairs = [] # (noisy_row, clean_row) 리스트
        
        print(f"> Loading Data from {len(self.noisy_files)} file pairs...")
        
        for n_path, c_path in zip(self.noisy_files, self.clean_files):
            # CSV 로드 (헤더가 없다고 가정, 3번째 컬럼부터 데이터)
            try:
                df_n = pd.read_csv(n_path, header=None)
                df_c = pd.read_csv(c_path, header=None)
                
                # Row 개수 일치 확인
                if len(df_n) != len(df_c):
                    continue # 행 개수가 다르면 해당 파일 스킵

                # Numpy 변환 (Phase Data 부분만 추출)
                n_data = df_n.iloc[:, 2:].values
                c_data = df_c.iloc[:, 2:].values
                
                # 쌍으로 묶어서 저장
                for i in range(len(n_data)):
                    self.data_pairs.append((n_data[i], c_data[i]))
            except Exception as e:
                print(f"Error reading {n_path}: {e}")
                
        print(f"> Total Paired Samples Loaded: {len(self.data_pairs)}")

    def __len__(self):
        return len(self.data_pairs)

    def __getitem__(self, idx):
        noisy_raw, clean_raw = self.data_pairs[idx]
        noisy_tensor = self.preprocess(noisy_raw)
        clean_tensor = self.preprocess(clean_raw)
        return noisy_tensor, clean_tensor

    def preprocess(self, raw_phase):
        # 결측치 제거
        raw_phase = np.nan_to_num(raw_phase)
        
        # Phase -> I/Q 변환
        phase_rad = np.deg2rad(raw_phase)
        i_data = np.cos(phase_rad)
        q_data = np.sin(phase_rad)
        
        # Reshape 로직 (Antenna x Time)
        ant_i = [[] for _ in range(self.n_antennas)]
        ant_q = [[] for _ in range(self.n_antennas)]
        
        num_blocks = len(phase_rad) // self.samples_per_block
        for b in range(num_blocks):
            ant_idx = b % self.n_antennas
            s = b * self.samples_per_block
            e = s + self.samples_per_block
            ant_i[ant_idx].extend(i_data[s:e])
            ant_q[ant_idx].extend(q_data[s:e])
            
        # Min Length Truncate (사각형 텐서 유지)
        if not ant_i[0]: # 데이터가 비어있을 경우 예외처리
            return torch.zeros(2, self.n_antennas, 12)

        min_len = min([len(x) for x in ant_i])
        
        final_i = np.array([x[:min_len] for x in ant_i])
        final_q = np.array([x[:min_len] for x in ant_q])
        
        # (2, 8, T) Tensor 생성
        tensor = torch.FloatTensor(np.stack([final_i, final_q], axis=0))
        return tensor

# ==========================================
# 3. Custom Loss (Domain Knowledge)
# ==========================================
class PhaseAwareLoss(nn.Module):
    def __init__(self, lambda_phase=1.0):
        super(PhaseAwareLoss, self).__init__()
        self.lambda_phase = lambda_phase
        self.mse = nn.MSELoss()

    def forward(self, pred, target):
        # 1. Amplitude/Waveform Loss (MSE)
        loss_mse = self.mse(pred, target)
        
        # 2. Phase Loss
        # (Batch, 2, 8, T) -> 복소수 변환
        pred_complex = pred[:, 0] + 1j * pred[:, 1]
        target_complex = target[:, 0] + 1j * target[:, 1]
        
        # 각도 추출
        pred_angle = torch.angle(pred_complex)
        target_angle = torch.angle(target_complex)
        
        # Cosine Distance: 1 - cos(error)
        # 오차가 0도면 0, 180도면 2가 됨. (순환성 고려)
        phase_diff = pred_angle - target_angle
        loss_phase = torch.mean(1 - torch.cos(phase_diff))
        
        # 최종 결합
        total_loss = loss_mse + (self.lambda_phase * loss_phase)
        
        return total_loss, loss_mse, loss_phase

# ==========================================
# 4. Models (Generator & Discriminator)
# ==========================================
class UNetGenerator(nn.Module):
    def __init__(self):
        super(UNetGenerator, self).__init__()
        
        def conv_block(in_c, out_c, normalize=True):
            layers = [nn.Conv2d(in_c, out_c, 3, stride=2, padding=1)]
            if normalize: layers.append(nn.BatchNorm2d(out_c))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return nn.Sequential(*layers)
            
        def up_block(in_c, out_c, dropout=False):
            layers = [
                nn.ConvTranspose2d(in_c, out_c, 3, stride=2, padding=1, output_padding=(1,1)), 
                nn.BatchNorm2d(out_c), nn.ReLU(inplace=True)
            ]
            if dropout: layers.append(nn.Dropout(0.5))
            return nn.Sequential(*layers)

        # Encoder
        self.down1 = conv_block(2, 64, normalize=False) 
        self.down2 = conv_block(64, 128)                
        
        # Decoder
        self.up1 = up_block(128, 64, dropout=True)      
        
        # Output
        self.final = nn.Sequential(
            nn.Upsample(size=(8, 12), mode='bilinear', align_corners=True), # 크기 보정
            nn.Conv2d(128, 2, 3, padding=1), # Skip Connection (64+64=128)
            nn.Tanh() # -1 ~ 1 Range (I/Q)
        )

    def forward(self, x):
        d1 = self.down1(x)
        d2 = self.down2(d1)
        
        u1 = self.up1(d2)
        
        # Skip Connection: Feature Map 크기 맞춤 (안전장치)
        if u1.shape[2:] != d1.shape[2:]:
            u1 = nn.functional.interpolate(u1, size=d1.shape[2:])
            
        u1 = torch.cat([u1, d1], 1) # Concatenate
        
        out = self.final(u1)
        return out

class PatchDiscriminator(nn.Module):
    def __init__(self):
        super(PatchDiscriminator, self).__init__()
        # Input: Condition(2) + Real/Fake(2) = 4 channels
        self.model = nn.Sequential(
            nn.Conv2d(4, 64, 3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            
            nn.Conv2d(128, 1, 3, padding=1) # Output map
        )

    def forward(self, img_A, img_B):
        x = torch.cat((img_A, img_B), 1)
        return self.model(x)

# ==========================================
# 5. Training Function
# ==========================================
def train_model(dataloader, generator, discriminator, device):
    # ★ PhaseAwareLoss 사용
    criterion_content = PhaseAwareLoss(lambda_phase=CONFIG['lambda_phase']).to(device)
    criterion_GAN = nn.MSELoss() # LSGAN
    
    optimizer_G = optim.Adam(generator.parameters(), lr=CONFIG['lr'], betas=(0.5, 0.999))
    optimizer_D = optim.Adam(discriminator.parameters(), lr=CONFIG['lr'], betas=(0.5, 0.999))
    
    print(">>> Start Training...")
    generator.train()
    discriminator.train()

    for epoch in range(CONFIG['epochs']):
        epoch_d_loss = 0
        epoch_g_loss = 0
        
        for i, (noisy, clean) in enumerate(dataloader):
            noisy, clean = noisy.to(device), clean.to(device)
            
            # ---------------------
            # Train Discriminator
            # ---------------------
            optimizer_D.zero_grad()
            
            # Real
            pred_real = discriminator(noisy, clean)
            target_real = torch.ones_like(pred_real).to(device)
            loss_D_real = criterion_GAN(pred_real, target_real)
            
            # Fake
            fake_clean = generator(noisy)
            pred_fake = discriminator(noisy, fake_clean.detach())
            target_fake = torch.zeros_like(pred_fake).to(device)
            loss_D_fake = criterion_GAN(pred_fake, target_fake)
            
            loss_D = 0.5 * (loss_D_real + loss_D_fake)
            loss_D.backward()
            optimizer_D.step()

            # ---------------------
            # Train Generator
            # ---------------------
            optimizer_G.zero_grad()
            
            # 1. GAN Loss (Adversarial)
            pred_fake = discriminator(noisy, fake_clean)
            loss_G_GAN = criterion_GAN(pred_fake, target_real)
            
            # 2. Content Loss (MSE + Phase Loss)
            loss_G_Content, loss_mse_val, loss_phase_val = criterion_content(fake_clean, clean)
            
            # Total G Loss
            loss_G = loss_G_GAN + (CONFIG['lambda_L1'] * loss_G_Content)
            
            loss_G.backward()
            optimizer_G.step()
            
            epoch_d_loss += loss_D.item()
            epoch_g_loss += loss_G.item()
            
        # Logging per Epoch
        if (epoch+1) % 5 == 0:
            print(f"[Epoch {epoch+1}/{CONFIG['epochs']}] "
                  f"Loss D: {epoch_d_loss/len(dataloader):.4f} | "
                  f"Loss G: {epoch_g_loss/len(dataloader):.4f} | "
                  f"MSE: {loss_mse_val.item():.4f} | "
                  f"Phase: {loss_phase_val.item():.4f}")

    print(">>> Training Finished.")
    return generator

# ==========================================
# 6. Evaluation Function (RF Metrics)
# ==========================================
def calculate_rf_metrics(pred_tensor, target_tensor):
    # Complex 변환
    pred_complex = pred_tensor[:, 0] + 1j * pred_tensor[:, 1]
    target_complex = target_tensor[:, 0] + 1j * target_tensor[:, 1]
    
    # 1. MSE
    mse = torch.mean((pred_tensor - target_tensor) ** 2).item()
    
    # 2. EVM (%)
    error_vector = pred_complex - target_complex
    error_power = torch.sum(torch.abs(error_vector)**2)
    ref_power = torch.sum(torch.abs(target_complex)**2)
    evm = torch.sqrt(error_power / (ref_power + 1e-8)).item() * 100
    
    # 3. Phase Error (Degree)
    pred_phase = torch.angle(pred_complex)
    target_phase = torch.angle(target_complex)
    
    diff = pred_phase - target_phase
    # -pi ~ pi Wrapping
    phase_diff = (diff + np.pi) % (2 * np.pi) - np.pi
    mae_phase_deg = torch.mean(torch.abs(phase_diff)).item() * (180 / np.pi)
    
    return mse, evm, mae_phase_deg

def evaluate_and_visualize(generator, dataloader, device):
    generator.eval()
    
    total_metrics = np.zeros(3) # mse, evm, phase
    count = 0
    
    # 시각화를 위한 샘플 저장
    vis_data = None 
    
    with torch.no_grad():
        for noisy, clean in dataloader:
            noisy, clean = noisy.to(device), clean.to(device)
            generated = generator(noisy)
            
            # 마지막 배치 하나 저장
            vis_data = (noisy, clean, generated)
            
            m, e, p = calculate_rf_metrics(generated, clean)
            total_metrics += [m, e, p]
            count += 1
            
    avg_metrics = total_metrics / count
    print("\n" + "="*40)
    print("📊 Final Evaluation Results")
    print("="*40)
    print(f"1. MSE Loss    : {avg_metrics[0]:.6f}")
    print(f"2. EVM         : {avg_metrics[1]:.2f} %")
    print(f"3. Phase Error : {avg_metrics[2]:.2f} ° (Target: < 5°)")
    print("="*40)
    
    # --- Visualization ---
    noisy_b, clean_b, gen_b = vis_data
    
    # 첫 번째 샘플, 첫 번째 안테나의 I/Q 플롯
    idx = 0
    ant = 0
    
    n_i = noisy_b[idx, 0, ant, :].cpu().numpy()
    c_i = clean_b[idx, 0, ant, :].cpu().numpy()
    g_i = gen_b[idx, 0, ant, :].cpu().numpy()
    
    n_q = noisy_b[idx, 1, ant, :].cpu().numpy()
    c_q = clean_b[idx, 1, ant, :].cpu().numpy()
    g_q = gen_b[idx, 1, ant, :].cpu().numpy()
    
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))
    
    # I-Component
    axes[0].set_title(f"I-Component (Antenna {ant+1})")
    axes[0].plot(n_i, '--', color='gray', label='Noisy Input', alpha=0.6)
    axes[0].plot(c_i, '-', color='blue', label='Clean GT')
    axes[0].plot(g_i, '-.', color='red', label='Generated')
    axes[0].legend()
    axes[0].grid(True)
    
    # Q-Component
    axes[1].set_title(f"Q-Component (Antenna {ant+1})")
    axes[1].plot(n_q, '--', color='gray', label='Noisy Input', alpha=0.6)
    axes[1].plot(c_q, '-', color='blue', label='Clean GT')
    axes[1].plot(g_q, '-.', color='red', label='Generated')
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.show()

# ==========================================
# Main Execution Block
# ==========================================
if __name__ == "__main__":
    cur_dir = os.path.dirname(os.path.abspath(__file__))

    noisy_dir = os.path.join(cur_dir, "../data/")
    clean_dir = os.path.join(cur_dir, "../gt/")
    
    
    # 2. 데이터셋 및 로더 준비
    dataset = PairedIQDataset(noisy_dir, clean_dir)
    
    if len(dataset) > 0:
        dataloader = DataLoader(dataset, batch_size=CONFIG['batch_size'], shuffle=True)
        
        # 3. 모델 초기화
        G = UNetGenerator().to(CONFIG['device'])
        D = PatchDiscriminator().to(CONFIG['device'])
        
        # 4. 학습
        G_trained = train_model(dataloader, G, D, CONFIG['device'])
        
        # 5. 평가 및 시각화
        evaluate_and_visualize(G_trained, dataloader, CONFIG['device'])
    else:
        print("데이터셋이 비어있습니다. 경로를 확인해주세요.")

    # (선택) 임시 폴더 삭제
    # shutil.rmtree("data")

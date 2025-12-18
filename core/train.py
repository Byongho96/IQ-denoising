import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import numpy as np
import matplotlib.pyplot as plt
import glob
import os
from tqdm import tqdm  # 진행률 표시바 (pip install tqdm)

# 기존에 작성한 모듈들 임포트
from config import Config
from dataset import PhaseDataset
from model import ComplexDenoisingNet
from loss import PhysicsInformedLoss

def visualize_prediction(model, val_loader, device, save_path="result_plot.png"):
    """
    학습된 모델의 예측 결과를 시각화하여 저장합니다.
    (Noisy Input vs GT vs Denoised Output)
    """
    model.eval()
    
    # 배치 하나만 가져오기
    noisy_input, gt_iq, gt_aoa = next(iter(val_loader))
    noisy_input = noisy_input.to(device)
    
    with torch.no_grad():
        output = model(noisy_input)
        
    # CPU로 이동 및 첫 번째 샘플 추출
    noisy = noisy_input[0].cpu().numpy() # (2, 111)
    gt = gt_iq[0].cpu().numpy()          # (2, 111)
    pred = output[0].cpu().numpy()       # (2, 111)
    
    # 시각화 (I/Q 채널 분리)
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    
    # X축: 샘플 인덱스
    x = np.arange(Config.SEQ_LEN)
    
    # 1. Real Part (In-Phase)
    axes[0].set_title(f"Real Part (I) - GT AoA: {gt_aoa[0].item():.2f} deg")
    axes[0].plot(x, noisy[0], label='Noisy Input', color='lightgray', alpha=0.7)
    axes[0].plot(x, gt[0], label='Ideal GT', color='green', linestyle='--', linewidth=2)
    axes[0].plot(x, pred[0], label='Denoised Output', color='red', alpha=0.8)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # 2. Imaginary Part (Quadrature)
    axes[1].set_title("Imaginary Part (Q)")
    axes[1].plot(x, noisy[1], label='Noisy Input', color='lightgray', alpha=0.7)
    axes[1].plot(x, gt[1], label='Ideal GT', color='green', linestyle='--', linewidth=2)
    axes[1].plot(x, pred[1], label='Denoised Output', color='red', alpha=0.8)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"[Visualization] Result saved to {save_path}")
    plt.close()

def train_model():
    # --------------------------------------------------------------------------
    # 1. 설정 및 준비
    # --------------------------------------------------------------------------
    # 데이터 경로 설정 (여기를 실제 경로로 수정하세요)
    import os
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(cur_dir, '../data/')
    MODEL_SAVE_PATH = 'best_model.pth'
    
    # 하이퍼파라미터 (Config에서 가져오거나 덮어쓰기)
    BATCH_SIZE = Config.BATCH_SIZE
    LR = Config.LEARNING_RATE
    EPOCHS = Config.EPOCHS
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print(f"Device: {DEVICE}")
    print(f"Data Directory: {DATA_DIR}")

    # --------------------------------------------------------------------------
    # 2. 데이터셋 로드 및 분할
    # --------------------------------------------------------------------------
    # glob으로 모든 csv 파일 찾기
    csv_files = glob.glob(os.path.join(DATA_DIR, '*.csv'))
    
    if not csv_files:
        raise FileNotFoundError(f"No .csv files found in {DATA_DIR}")
        
    print(f"Found {len(csv_files)} CSV files.")
    
    # 데이터셋 인스턴스 생성
    full_dataset = PhaseDataset(csv_files)
    total_samples = len(full_dataset)
    print(f"Total Samples: {total_samples}") # 예상: 12600
    
    # Train/Val 분할 (9:1)
    train_size = int(0.9 * total_samples)
    val_size = total_samples - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4)
    
    print(f"Train: {len(train_dataset)} | Val: {len(val_dataset)}")

    # --------------------------------------------------------------------------
    # 3. 모델, Loss, Optimizer 초기화
    # --------------------------------------------------------------------------
    model = ComplexDenoisingNet(seq_len=Config.SEQ_LEN).to(DEVICE)
    criterion = PhysicsInformedLoss(lambda_cov=Config.LAMBDA_PHYSICS).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    
    # Learning Rate Scheduler (선택사항: Loss가 안 줄어들면 LR 감소)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    # --------------------------------------------------------------------------
    # 4. 학습 루프
    # --------------------------------------------------------------------------
    best_val_loss = float('inf')
    
    for epoch in range(EPOCHS):
        # [Training Phase]
        model.train()
        train_loss = 0.0
        
        # tqdm으로 진행바 표시
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        
        for noisy_input, gt_iq, _ in loop:
            noisy_input, gt_iq = noisy_input.to(DEVICE), gt_iq.to(DEVICE)
            
            optimizer.zero_grad()
            
            # Forward
            output = model(noisy_input)
            
            # Loss Calculation
            loss = criterion(output, gt_iq)
            
            # Backward
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            loop.set_postfix(loss=loss.item())
            
        avg_train_loss = train_loss / len(train_loader)
        
        # [Validation Phase]
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for noisy_input, gt_iq, _ in val_loader:
                noisy_input, gt_iq = noisy_input.to(DEVICE), gt_iq.to(DEVICE)
                output = model(noisy_input)
                loss = criterion(output, gt_iq)
                val_loss += loss.item()
                
        avg_val_loss = val_loss / len(val_loader)
        
        # 스케줄러 업데이트
        scheduler.step(avg_val_loss)
        
        print(f"Epoch [{epoch+1}/{EPOCHS}] Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        
        # 최고 성능 모델 저장
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print(f"--> Best Model Saved (Loss: {best_val_loss:.4f})")
            
            # 성능이 갱신될 때마다 시각화 결과도 업데이트
            visualize_prediction(model, val_loader, DEVICE, save_path=f"epoch_{epoch+1}_result.png")

    print("Training Finished.")

if __name__ == '__main__':
    train_model()
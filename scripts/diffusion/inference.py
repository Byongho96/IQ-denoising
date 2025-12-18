import torch
import pandas as pd
import numpy as np
import os
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from config import Config

# 앞서 작성한 Diffusion 모델과 유틸리티 임포트
from model import DiffusionUNet1D
from diffusion import Diffusion

def preprocess_phase_to_iq_tensor(phase_data, seq_length):
    """
    Phase(Degree) -> Model Input(IQ Tensor)
    (기존과 동일)
    """
    # Padding / Truncating
    current_len = phase_data.shape[1]
    
    if current_len > seq_length:
        phase_data = phase_data[:, :seq_length]
    elif current_len < seq_length:
        pad_width = seq_length - current_len
        phase_data = np.pad(phase_data, ((0, 0), (0, pad_width)), mode='edge')

    # Phase -> IQ Conversion
    # z = exp(j * theta)
    iq_data = np.exp(1j * np.deg2rad(phase_data)) 

    # Tensor Conversion (Batch, 2, Seq_Len)
    real = torch.from_numpy(iq_data.real).float()
    imag = torch.from_numpy(iq_data.imag).float()
    
    tensor_out = torch.stack([real, imag], dim=1)
    return tensor_out

def restore_and_save_csv(input_csv_path, output_csv_path, model_path, batch_size=64):
    # Diffusion은 메모리와 연산량이 많으므로 batch_size를 조절(256 -> 64 등)하는 것이 좋습니다.
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[Info] Device: {device}")
    
    # 1. Load Diffusion Model
    # Input Channels: 4 (2 Latent + 2 Condition)
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
        
    model = DiffusionUNet1D(in_channels=4, out_channels=2).to(device)
    
    # Load Weights
    checkpoint = torch.load(model_path, map_location=device)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
        
    model.eval()
    print(f"[Info] Diffusion Model loaded from {model_path}")

    # 2. Initialize Diffusion Scheduler
    # 학습 때 사용한 noise_steps와 동일해야 합니다 (예: 1000)
    diffusion = Diffusion(noise_steps=1000, device=device)

    # 3. Load CSV
    if not os.path.exists(input_csv_path):
        raise FileNotFoundError(f"Input CSV not found: {input_csv_path}")
    
    df = pd.read_csv(input_csv_path, header=None)
    print(f"[Info] Loaded CSV: {input_csv_path} (Rows: {len(df)})")
    
    # Split Meta and Phase Data
    meta_df = df.iloc[:, :2] # Timestamp, Beacon ID
    raw_phase_data = df.iloc[:, 2:].values.astype(np.float32)
    
    # Preprocessing
    print("[Info] Preprocessing Phase to IQ Tensor...")
    input_tensor = preprocess_phase_to_iq_tensor(raw_phase_data, Config.SEQ_LENGTH)
    
    dataset = TensorDataset(input_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    # Inference Loop
    restored_phase_list = []
    
    print("[Info] Starting Diffusion Sampling (This may take a while)...")
    
    # Diffusion Sampling은 시간이 오래 걸리므로 tqdm으로 진행 상황 표시
    # Outer loop: Batches
    for batch in tqdm(loader, desc="Batch Processing"):
        batch_x = batch[0].to(device) # Condition (Noisy Input): (Batch, 2, 111)
        
        # Diffusion Reverse Sampling
        # diffusion.sample 내부에서 1000 step 루프를 돌며 노이즈를 제거합니다.
        # condition으로 batch_x(noisy input)를 넣어줍니다.
        denoised_iq = diffusion.sample(model, condition=batch_x, n_samples=batch_x.shape[0])
        
        # Result to Numpy
        outputs_np = denoised_iq.cpu().numpy() # (Batch, 2, 111)
        
        # I (Real): Channel 0, Q (Imag): Channel 1
        real = outputs_np[:, 0, :]
        imag = outputs_np[:, 1, :]
        
        # Calculate Phase using arctan2 (Result: -180 ~ 180 degrees)
        phase_rad = np.arctan2(imag, real)
        phase_deg = np.rad2deg(phase_rad) # (Batch, 111)
        
        restored_phase_list.append(phase_deg)
            
    # Concatenate all batches
    restored_phase_data = np.concatenate(restored_phase_list, axis=0) # (Total_Rows, 111)
    
    # Restore to DataFrame
    denoised_df = pd.DataFrame(restored_phase_data)
    final_df = pd.concat([meta_df.reset_index(drop=True), denoised_df.reset_index(drop=True)], axis=1)
    
    # Save to CSV
    os.makedirs(os.path.dirname(output_csv_path), exist_ok=True)
    final_df.to_csv(output_csv_path, index=False, header=False)
    print(f"[Success] Denoised Phase data saved to: {output_csv_path}")

if __name__ == "__main__":
    cur_dir = os.path.dirname(os.path.abspath(__file__))

    # 경로 수정: Diffusion 모델 체크포인트
    MODEL_PATH = os.path.join(cur_dir, '../../checkpoints/best_diffusion_model.pth') 
    TARGET_CSV = os.path.join(cur_dir, '../../data/mapSmall_x0y1.csv') 
    OUTPUT_CSV = os.path.join(cur_dir, '../../denoised/mapSmall_x0y1_diffusion.csv')
    
    try:
        restore_and_save_csv(TARGET_CSV, OUTPUT_CSV, MODEL_PATH, batch_size=32)
    except Exception as e:
        print(f"[Error] {e}")
        import traceback
        traceback.print_exc()
import torch
import pandas as pd
import numpy as np
import os
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm
from config import Config
from model import ComplexUNet1D

def preprocess_phase_to_iq_tensor(phase_data, seq_length):
    """
    Phase(Degree) -> Model Input(IQ Tensor)
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

def restore_and_save_csv(input_csv_path, output_csv_path, model_path, batch_size=256):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[Info] Device: {device}")
    
    # Load Model
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
        
    model = ComplexUNet1D(in_channels=2, out_channels=2).to(device)
    
    checkpoint = torch.load(model_path, map_location=device)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
        
    model.eval()
    print(f"[Info] Model loaded from {model_path}")

    # Load CSV
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
    
    print("[Info] Starting Inference...")
    with torch.no_grad():
        for batch in tqdm(loader, desc="Denoising"):
            batch_x = batch[0].to(device) # (Batch, 2, 111)
            
            # Inference (Denoised IQ)
            outputs = model(batch_x) # (Batch, 2, 111)
            outputs_np = outputs.cpu().numpy() # (Batch, 2, 111)
            
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

    MODEL_PATH = os.path.join(cur_dir, '../../checkpoints/best_model.pth') 
    TARGET_CSV = os.path.join(cur_dir, '../../data/mapSmall_x0y1.csv') 
    OUTPUT_CSV = os.path.join(cur_dir, '../../denoised/mapSmall_x0y1_unet.csv')
    
    try:
        restore_and_save_csv(TARGET_CSV, OUTPUT_CSV, MODEL_PATH)
    except Exception as e:
        print(f"[Error] {e}")
        import traceback
        traceback.print_exc()
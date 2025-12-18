import torch
import pandas as pd
import numpy as np
import os
from tqdm import tqdm

# 기존 모듈 임포트
from config import Config
from model import ComplexDenoisingNet

def restore_and_save_csv(input_csv_path, output_csv_path, model_path):
    """
    특정 CSV 파일을 읽어 노이즈를 제거한 뒤 새로운 CSV로 저장하는 함수
    """
    # --------------------------------------------------------------------------
    # 1. 설정 및 모델 로드
    # --------------------------------------------------------------------------
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[Info] Device: {device}")
    
    # 모델 초기화
    model = ComplexDenoisingNet(seq_len=Config.SEQ_LEN).to(device)
    
    # 학습된 가중치 로드
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
        
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint)
    model.eval() # 평가 모드 (Dropout 해제 등)
    print(f"[Info] Model loaded from {model_path}")

    # --------------------------------------------------------------------------
    # 2. CSV 파일 로드
    # --------------------------------------------------------------------------
    if not os.path.exists(input_csv_path):
        raise FileNotFoundError(f"Input CSV not found: {input_csv_path}")
        
    df = pd.read_csv(input_csv_path)
    print(f"[Info] Loaded CSV: {input_csv_path} (Rows: {len(df)})")
    
    # 결과 데이터를 담을 리스트 (원본 DataFrame 복사)
    restored_df = df.copy()
    
    # --------------------------------------------------------------------------
    # 3. 복원(Inference) 루프
    # --------------------------------------------------------------------------
    print("[Info] Starting restoration...")
    
    # 배치 처리를 하면 더 빠르지만, 코드 간결성을 위해 행 단위 처리 (tqdm으로 진행상황 확인)
    denoised_phases = []
    
    with torch.no_grad():
        for index, row in tqdm(df.iterrows(), total=len(df)):
            # A. 데이터 추출 (2번 컬럼부터 끝까지가 Phase Samples)
            raw_phase_deg = row.iloc[2:].values.astype(float)
            
            if len(raw_phase_deg) != Config.SEQ_LEN:
                # 길이가 안 맞으면 원본 그대로 두거나 스킵 (여기선 원본 유지)
                denoised_phases.append(raw_phase_deg)
                continue
            
            # B. 전처리: Degree -> Radian -> IQ -> Tensor
            raw_phase_rad = np.deg2rad(raw_phase_deg)
            input_iq = np.exp(1j * raw_phase_rad)
            
            # (Batch, 2, 111) 형태로 변환
            input_tensor = torch.stack([
                torch.tensor(input_iq.real), 
                torch.tensor(input_iq.imag)
            ], dim=0).float().unsqueeze(0).to(device)
            
            # C. 모델 추론 (Inference)
            output = model(input_tensor) # (1, 2, 111)
            
            # D. 후처리: Tensor -> IQ -> Phase(Rad) -> Degree
            # CPU로 이동
            pred_iq_tensor = output.squeeze(0).cpu().numpy() # (2, 111)
            pred_iq_complex = pred_iq_tensor[0] + 1j * pred_iq_tensor[1]
            
            # IQ에서 위상 추출 (np.angle은 -pi ~ pi 반환)
            restored_phase_rad = np.angle(pred_iq_complex)
            restored_phase_deg = np.rad2deg(restored_phase_rad)
            
            denoised_phases.append(restored_phase_deg)

    # --------------------------------------------------------------------------
    # 4. 결과 저장
    # --------------------------------------------------------------------------
    # 복원된 위상 데이터로 DataFrame 업데이트
    # 기존 Timestamp, Beacon ID 컬럼은 유지하고 2번째 컬럼부터 덮어쓰기
    phase_columns = df.columns[2:] # 위상 데이터 컬럼명들
    
    # 리스트를 DataFrame으로 변환하여 병합
    denoised_df_part = pd.DataFrame(denoised_phases, columns=phase_columns)
    
    # 원본 앞부분(메타데이터) + 복원된 뒷부분(위상)
    restored_df.iloc[:, 2:] = denoised_df_part
    
    # CSV 저장
    restored_df.to_csv(output_csv_path, index=False)
    print(f"[Success] Restored CSV saved to: {output_csv_path}")

# ==============================================================================
# 실행 설정
# ==============================================================================
if __name__ == "__main__":
    # 1. 모델 경로 (학습된 파일)
    import os
    cur_dir = os.path.dirname(os.path.abspath(__file__))

    MODEL_PATH = os.path.join(cur_dir, '../../best_model.pth') 
    
    # 2. 복원할 대상 CSV 파일
    TARGET_CSV = os.path.join(cur_dir, '../data/mapSmall_x0y1.csv') 
    
    # 3. 저장할 파일명
    OUTPUT_CSV = os.path.join(cur_dir, '../denoised/mapSmall_x0y1.csv')
    
    try:
        restore_and_save_csv(TARGET_CSV, OUTPUT_CSV, MODEL_PATH)
    except Exception as e:
        print(f"[Error] {e}")
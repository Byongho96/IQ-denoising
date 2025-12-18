import pandas as pd
import numpy as np
import os
import re
from tqdm import tqdm

# 사용자 정의 모듈 임포트
from config import Config
from utils_physics import generate_ideal_iq

def generate_gt_csv(input_csv_path, output_csv_path):
    """
    입력 CSV 파일을 읽어 PhaseDataset과 동일한 로직으로 GT를 생성한 뒤,
    위상 데이터를 GT 값으로 교체하여 저장하는 함수.
    """
    # -------------------------------------------------------------------------
    # 1. 파일명에서 Rx 좌표 파싱 (Dataset 로직 동일)
    # -------------------------------------------------------------------------
    filename = os.path.basename(input_csv_path)
    match = re.search(r'x(\d+)y(\d+)', filename)
    
    if not match:
        print(f"[Error] 파일명에서 좌표를 찾을 수 없습니다: {filename}")
        return

    rx_x = float(match.group(1))
    # [주의] 사용자 코드의 Y축 반전 로직 유지
    rx_y = 4 - float(match.group(2)) 
    
    print(f"[Info] Processing {filename}")
    print(f"       Rx Position: ({rx_x}, {rx_y})")

    # -------------------------------------------------------------------------
    # 2. CSV 로드
    # -------------------------------------------------------------------------
    try:
        df = pd.read_csv(input_csv_path)
    except Exception as e:
        print(f"[Error] CSV 로드 실패: {e}")
        return

    # 결과 저장을 위한 리스트
    gt_rows = []
    
    # Tx 위치 정보
    tx_positions = Config.TX_POSITIONS

    # -------------------------------------------------------------------------
    # 3. 행 단위 처리 (GT 생성 및 정렬)
    # -------------------------------------------------------------------------
    print("[Info] Generating GT data...")
    
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        # 원본 데이터 복사 (Timestamp, Beacon ID 등 보존)
        new_row = row.copy()
        
        # A. Beacon ID 및 Tx 좌표 확인
        try:
            beacon_id = int(row.iloc[1])
        except ValueError:
            gt_rows.append(new_row) # 에러 행은 그대로 유지
            continue
            
        if beacon_id not in tx_positions:
            gt_rows.append(new_row) # 정의되지 않은 비콘은 그대로 유지
            continue
            
        tx_x, tx_y = tx_positions[beacon_id]
        
        # B. GT AoA 계산 (표준 수학 좌표계: East=0, CCW)
        dx = tx_x - rx_x
        dy = tx_y - rx_y
        theta_rad = np.arctan2(dy, dx)
        gt_aoa_deg = np.rad2deg(theta_rad)
        
        # C. Input Data 전처리 (Alignment 용)
        raw_phase_deg = row.iloc[2:].values
        if len(raw_phase_deg) != Config.SEQ_LEN:
            gt_rows.append(new_row)
            continue
            
        try:
            raw_phase_deg = raw_phase_deg.astype(float)
        except ValueError:
            gt_rows.append(new_row)
            continue
            
        input_iq = np.exp(1j * np.deg2rad(raw_phase_deg))
        
        # D. GT IQ 생성 (utils_physics)
        gt_iq = generate_ideal_iq(gt_aoa_deg)
        
        # =====================================================================
        # [핵심] Phase Alignment (3-Sample Average)
        # Dataset의 __getitem__ 로직과 100% 동일해야 함
        # =====================================================================
        n_anchor = Config.SAMPLES_PER_ANT
        input_anchor = input_iq[:n_anchor]
        gt_anchor = gt_iq[:n_anchor]
        
        # Correlation으로 평균 위상차 계산
        correlation = np.sum(input_anchor * np.conj(gt_anchor))
        phase_diff = np.angle(correlation)
        
        # GT 회전 (Alignment)
        gt_iq_aligned = gt_iq * np.exp(1j * phase_diff)
        
        # =====================================================================
        
        # E. 다시 Degree로 변환하여 저장 (CSV 포맷 유지)
        # np.angle은 -pi ~ pi를 반환하므로 degree로 변환
        gt_phase_deg = np.rad2deg(np.angle(gt_iq_aligned))
        
        # F. DataFrame Row 업데이트 (2번 컬럼부터 끝까지 교체)
        new_row.iloc[2:] = gt_phase_deg
        gt_rows.append(new_row)

    # -------------------------------------------------------------------------
    # 4. 결과 저장
    # -------------------------------------------------------------------------
    gt_df = pd.DataFrame(gt_rows)
    gt_df.to_csv(output_csv_path, index=False)
    print(f"[Success] GT CSV Generated: {output_csv_path}")
    
    # -------------------------------------------------------------------------
    # 5. 간단한 검증 (시각화)
    # -------------------------------------------------------------------------
    try:
        import matplotlib.pyplot as plt
        
        # 첫 번째 유효한 샘플 시각화
        valid_idx = 0
        original_phases = df.iloc[valid_idx, 2:].values.astype(float)
        gt_phases = gt_df.iloc[valid_idx, 2:].values.astype(float)
        
        plt.figure(figsize=(12, 6))
        plt.plot(original_phases, label='Noisy Input (Raw)', alpha=0.5)
        plt.plot(gt_phases, label='Generated GT (Aligned)', linewidth=2, linestyle='--')
        plt.title(f"GT Generation Check - Row {valid_idx}")
        plt.xlabel("Sample Index")
        plt.ylabel("Phase (Degree)")
        plt.legend()
        plt.grid(True)
        
        img_save_path = output_csv_path.replace('.csv', '.png')
        plt.savefig(img_save_path)
        print(f"[Visual] Plot saved to: {img_save_path}")
        
    except ImportError:
        print("[Info] matplotlib가 없어 시각화는 건너뜁니다.")

if __name__ == "__main__":
    # 테스트할 파일 경로 설정
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    INPUT_FILE = os.path.join(cur_dir, "../data/mapSmall_x0y1.csv")       # 원본 파일 경로
    OUTPUT_FILE = os.path.join(cur_dir, "../gt/mapSmall_x0y1.csv")   # 생성될 GT 파일 경로
    
    if os.path.exists(INPUT_FILE):
        generate_gt_csv(INPUT_FILE, OUTPUT_FILE)
    else:
        print(f"파일을 찾을 수 없습니다: {INPUT_FILE}")
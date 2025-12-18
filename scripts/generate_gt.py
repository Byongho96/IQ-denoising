import pandas as pd
import numpy as np
import os
import math

# ==========================================
# 1. 설정 및 상수 정의 (Configuration)
# ==========================================
class Config:
    C = 3e8                 # 빛의 속도
    FREQ = 2.4e9 + 250e3    # BLE 5.1 중심 주파수
    LAMBDA = C / FREQ       # 파장
    
    N_ANTENNAS = 8          # 안테나 수
    D_SPACING = 0.0456      # 인접 안테나 간격 (m)
    # UCA 반지름 계산
    RADIUS = D_SPACING / (2 * np.sin(np.pi / N_ANTENNAS))
    
    SAMPLES_PER_SLOT = 3    # 슬롯 당 샘플 수
    TOTAL_SAMPLES = 111     # 패킷 당 총 위상 샘플 수

# ==========================================
# 2. Beacon ID 별 GT AoA 매핑 (Dictionary)
# ==========================================
# 이전 대화 내용을 바탕으로 설정했습니다. 필요에 따라 값을 수정하세요.
BEACON_GT_AOA = {
    1: -116.56,   # Beacon 1
    2: 90,     # Beacon 2
    4: -90,   # Beacon 4 (or -180)
    5: 116.56   # Beacon 5
}


# ==========================================
# 3. 합성 데이터 생성 함수 (Core Logic)
# ==========================================
def generate_synthetic_packet(timestamp, anchor_id, gt_aoa_deg, snr_db):
    """
    특정 AoA와 SNR을 기반으로 가우시안 노이즈가 섞인 위상 샘플을 생성합니다.
    """
    # 1. 안테나 배치 각도 (반시계 방향 가정)
    element_angles = np.arange(0, 2*np.pi, 2*np.pi/Config.N_ANTENNAS)
    
    # 2. 이상적 위상차 계산
    theta_rad = np.deg2rad(gt_aoa_deg)
    phi_rad = np.deg2rad(90) # Elevation 수평 가정
    
    k = 2 * np.pi / Config.LAMBDA
    ideal_phases = k * Config.RADIUS * np.sin(phi_rad) * np.cos(theta_rad - element_angles)
    
    # 3. 111개 샘플 시퀀스 생성 (A1->A8 순환, 슬롯당 3개)
    phase_sequence = []
    current_sample_count = 0
    antenna_idx = 0
    
    while current_sample_count < Config.TOTAL_SAMPLES:
        current_phase = ideal_phases[antenna_idx % Config.N_ANTENNAS]
        for _ in range(Config.SAMPLES_PER_SLOT):
            if current_sample_count >= Config.TOTAL_SAMPLES:
                break
            phase_sequence.append(current_phase)
            current_sample_count += 1
        antenna_idx += 1

    phase_sequence = np.array(phase_sequence)

    # 4. IQ 변환 및 노이즈 추가 (AWGN)
    signal_iq = np.exp(1j * phase_sequence)
    
    # SNR 계산 및 노이즈 생성
    p_signal = 1.0
    p_noise = p_signal / (10 ** (snr_db / 10.0))
    noise_std = np.sqrt(p_noise / 2)
    
    noise = noise_std * (np.random.randn(len(signal_iq)) + 1j * np.random.randn(len(signal_iq)))
    noisy_iq = signal_iq + noise
    
    # 5. 위상 복원 (Phase Extraction)
    noisy_phases_rad = np.angle(noisy_iq)
    noisy_phases_deg = np.rad2deg(noisy_phases_rad)
    
    # 정수형 변환 (CSV 저장용)
    noisy_phases_int = np.round(noisy_phases_deg).astype(int)
    
    # [Timestamp, AnchorID, ...Samples...]
    return [timestamp, anchor_id] + noisy_phases_int.tolist()

# ==========================================
# 4. CSV 파일 처리 및 저장 함수
# ==========================================
def process_csv_and_generate_data(input_csv_path, output_csv_path, snr_db=10):
    """
    입력 CSV를 읽어 Beacon ID에 맞는 GT AoA로 합성 데이터를 생성하여 저장합니다.
    """
    print(f"Reading from: {input_csv_path}")
    
    # CSV 읽기 (헤더가 없다고 가정, 있으면 header=0)
    try:
        df = pd.read_csv(input_csv_path, header=None)
    except FileNotFoundError:
        print("파일을 찾을 수 없습니다.")
        return

    generated_rows = []

    print(f"Processing {len(df)} rows with SNR={snr_db}dB...")

    # DataFrame 순회
    for index, row in df.iterrows():
        # 데이터 파싱
        timestamp = row[0]
        beacon_id = int(row[1])
        
        # Dictionary에서 GT AoA 조회
        if beacon_id in BEACON_GT_AOA:
            gt_aoa = BEACON_GT_AOA[beacon_id]
            
            # 합성 데이터 생성 함수 호출
            new_data_row = generate_synthetic_packet(timestamp, beacon_id, gt_aoa, snr_db)
            generated_rows.append(new_data_row)
        else:
            # 정의되지 않은 Beacon ID는 건너뛰거나 로그 출력
            # print(f"Warning: Unknown Beacon ID {beacon_id} at row {index}")
            pass

    # 결과 저장
    # 컬럼명 생성: Timestamp, Beacon_ID, theta_0, theta_1, ...
    column_names = ['timestamp', 'beacon_id'] + [f'theta_{i}' for i in range(Config.TOTAL_SAMPLES)]
    
    result_df = pd.DataFrame(generated_rows, columns=column_names)
    result_df.to_csv(output_csv_path, index=False, header=False)
    
    print(f"Successfully saved {len(result_df)} rows to: {output_csv_path}")

# ==========================================
# 5. 실행부
# ==========================================
if __name__ == "__main__":
    # 예시 파일 경로
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    # 노이즈 레벨 설정 (dB)
    target_snr = 100.0 

    
    cur_dir = os.path.dirname(os.path.abspath(__file__))
    for i in range(0, 5):  # 예시로 5개 파일
        for j in range(0, 5):
            if i == 0 and j == 0:
                continue  # (0,0) 위치는 데이터 없음
            if i == 4 and j == 4:
                continue  # (4,4) 위치는 데이터 없음
            if i == 0 and j == 4:
                continue  # (0,4) 위치는 데이터 없음
            if i == 4 and j == 0:
                continue  # (4,0) 위치는 데이터 없음
            x = i
            y = 4 - j
            BEACON_GT_AOA = {
                1: -180 + math.atan2(y - 0, 4 - x) * 180 / np.pi,  # Beacon 1
                2: math.atan2(4 - y, x - 0) * 180 / np.pi,     # Beacon 2
                4: -math.atan2(y - 0, x - 0) * 180 / np.pi,   # Beacon 4 (or -180)
                5: 180 - math.atan2(4 - y, 4 - x) * 180 / np.pi  # Beacon 5
            }
            noisy_path = os.path.join(cur_dir, f'../data/mapSmall_x{i}y{j}.csv')
            gt_path = os.path.join(cur_dir, f'../gt/mapSmall_x{i}y{j}.csv')
            process_csv_and_generate_data(noisy_path, gt_path, snr_db=target_snr)
                
    
    # 1. (테스트용) 더미 입력 파일 생성
    # 실제 파일이 있다면 이 부분은 주석 처리하세요.
    # dummy_data = [
    #     [0.092501, 2] + [0]*111,
    #     [0.248520, 5] + [0]*111,
    #     [0.451230, 4] + [0]*111,
    #     [0.612340, 1] + [0]*111
    # ]
    # pd.DataFrame(dummy_data).to_csv(input_file, header=None, index=False)
    # print(f"Created dummy input file: {input_file}")

    # 2. 메인 로직 실행
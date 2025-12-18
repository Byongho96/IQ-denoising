import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# import seaborn as sns
import os
# ==========================================
# 1. 모의 데이터 생성 (실제 데이터가 없을 경우 사용)
# ==========================================
def create_mock_dataset(num_packets=100):
    """
    논문에 묘사된 형식의 가짜 데이터를 생성합니다.
    - 111개 샘플 (3개씩 안테나 스위칭)
    - 8개 안테나 순환
    """
    data = []
    
    # 가상의 도래각 (AoA) 설정 (예: 45도)
    true_aoa_deg = 45
    
    # 안테나 배열 설정 (UCA, 8 elements)
    num_antennas = 8
    radius = 0.0456 / (2 * np.sin(np.pi/num_antennas)) # 대략적 반지름
    wavelength = 0.125
    
    for _ in range(num_packets):
        row = [0.0, 1] # Timestamp, BeaconID
        
        # 111개 샘플 생성
        packet_phases = []
        for i in range(111):
            # 현재 샘플이 몇 번째 안테나인지 계산
            # 3샘플마다 스위칭, 8개 안테나 순환
            switch_idx = i // 3
            ant_idx = switch_idx % 8 
            
            # 이론적 위상 계산 (Steering Vector)
            angle_pos = 2 * np.pi * ant_idx / num_antennas
            # 간단한 UCA 위상 공식 (Elevation 90도 가정)
            phase_val = (2 * np.pi * radius / wavelength) * \
                        np.cos(np.deg2rad(true_aoa_deg) - angle_pos)
            
            # 노이즈 추가 (Gaussian Noise)
            noise = np.random.normal(0, 0.5) # 라디안 단위 노이즈
            phase_deg = np.rad2deg(phase_val + noise)
            
            # -180 ~ 180 범위로 래핑
            phase_deg = ((phase_deg + 180) % 360) - 180
            packet_phases.append(int(phase_deg))
            
        row.extend(packet_phases)
        data.append(row)
    
    # 컬럼 이름 생성
    cols = ['Timestamp', 'BeaconID'] + [f'sample_{i}' for i in range(111)]
    return pd.DataFrame(data, columns=cols)

# ==========================================
# 2. 데이터 로드 및 전처리 (핵심 로직)
# ==========================================

# [사용자 설정] 실제 csv 파일이 있다면 경로를 입력하세요.
current_dir = os.path.dirname(os.path.abspath(__file__))
csv_file_path = os.path.join(current_dir, '../data/mapSmall_x2y2.csv')
    
df = pd.read_csv(csv_file_path)
# df = create_mock_dataset() # 테스트용 생성

print("데이터 로드 완료. 형태:", df.shape)

# 위상 데이터만 추출 (앞의 2개 컬럼 제외)
phase_data_deg = df.iloc[:, 2:].values

# (1) Degree -> Radian 변환
phase_data_rad = np.deg2rad(phase_data_deg)

# (2) IQ 데이터 변환 (I = cos, Q = sin)
I_data = np.cos(phase_data_rad)
Q_data = np.sin(phase_data_rad)

# ==========================================
# 3. 시각화: 노이즈 분포 및 IQ Constellation
# ==========================================
plt.figure(figsize=(15, 6))

# --- Plot 1: 단일 패킷의 시계열 위상 변화 ---
plt.subplot(1, 2, 1)
sample_packet_idx = 0
plt.plot(phase_data_deg[sample_packet_idx], '.-', label='Measured Phase')
plt.title(f"Raw Phase Sequence (Packet {sample_packet_idx})")
plt.xlabel("Sample Index (0~110)")
plt.ylabel("Phase (Degree)")
plt.grid(True, alpha=0.3)
plt.legend()

# --- Plot 2: IQ Constellation (Unit Circle) ---
plt.subplot(1, 2, 2)

# 안테나 별로 색상을 다르게 하여 IQ 평면에 점을 찍습니다.
# 111개 샘플이 각각 어떤 안테나에 해당하는지 인덱싱
sample_indices = np.arange(111)
antenna_indices = (sample_indices // 3) % 8  # 0~7 사이 값

# 컬러맵 설정
colors = plt.cm.tab10(np.linspace(0, 1, 10))

for ant_id in range(8):
    # 현재 안테나에 해당하는 열(column) 인덱스 마스크
    mask = (antenna_indices == ant_id)
    
    # 모든 패킷, 해당 안테나의 I, Q 값 추출
    i_vals = I_data[:, mask].flatten()
    q_vals = Q_data[:, mask].flatten()
    
    plt.scatter(i_vals, q_vals, s=5, alpha=0.5, label=f'Ant {ant_id+1}', color=colors[ant_id])

# 단위 원 그리기 (이상적인 신호의 궤적)
circle = plt.Circle((0, 0), 1, color='black', fill=False, linestyle='--', label='Unit Circle')
plt.gca().add_patch(circle)

plt.title("IQ Constellation per Antenna (All Packets)")
plt.xlabel("In-Phase (I)")
plt.ylabel("Quadrature (Q)")
plt.axis('equal') # 원이 찌그러지지 않게 비율 고정
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
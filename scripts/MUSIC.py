import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.signal import find_peaks

# ==========================================
# 1. 설정 및 상수 정의 (Configuration)
# ==========================================
class Config:
    C = 3e8                 
    FREQ = 2.4e9 + 250e3   
    LAMBDA = C / FREQ 
    
    N_SOURCES = 1
    N_ANTENNAS = 8       

    D_SPACING = 0.0456      
    RADIUS = D_SPACING / (2 * np.sin(np.pi / N_ANTENNAS))
    
    SAMPLES_PER_SLOT = 3
    
    ANGLE_RES = 1           
    SCAN_ANGLES = np.arange(-180, 180, ANGLE_RES)


def load_and_process_data(filepath):
    """
    CSV 파일을 읽고 MUSIC에 넣을 수 있는 형태(IQ 복소수)로 변환합니다.
    """
    df = pd.read_csv(filepath)
    
    phase_cols = df.columns[2:]     #  Timestamp, Beacon ID  | theta_A1, theta_A2, ..., theta_A8
    phase_data = df[phase_cols].values # (N_packets, 111)

    # Degree -> Radian 변환
    phase_rad = np.deg2rad(phase_data)

    # Phase -> IQ (복소수) 변환 (진폭은 1로 가정)
    iq_data_raw = np.exp(1j * phase_rad)

    processed_packets = []
    
    for i in range(len(df)):
        packet_iq = iq_data_raw[i, :] # (111,)

        samples_per_full_rotation = Config.N_ANTENNAS * Config.SAMPLES_PER_SLOT # 8 * 3 = 24
        n_full_rotations = len(packet_iq) // samples_per_full_rotation # 111 // 24 = 4

        valid_len = n_full_rotations * samples_per_full_rotation    # 4 * 24 = 96
        valid_data = packet_iq[:valid_len]
        
        # 형태 변환: (4바퀴, 8안테나, 3샘플)
        reshaped = valid_data.reshape(n_full_rotations, Config.N_ANTENNAS, Config.SAMPLES_PER_SLOT)
        
        # 축 2(샘플 축)를 기준으로 3개 평균을 구함 -> (4바퀴, 8안테나)
        slot_avg = np.mean(reshaped, axis=2)
        
        # 4. MUSIC 입력 형태 (Antennas x Snapshots)로 변환
        # (4, 8) -> Transpose -> (8, 4)
        X = slot_avg.T 
        
        processed_packets.append({
            'timestamp': df.iloc[i, 0],
            'beacon_id': df.iloc[i, 1],
            'X': X  # Shape: (8, 4)
        })

        
    return processed_packets
# ==========================================
# 3. UCA 스티어링 벡터 생성 (Steering Vector)
# ==========================================
def get_uca_steering_vector(theta_deg):
    """
    Uniform Circular Array(UCA)의 조향 벡터를 계산합니다.
    theta: 도래각 (Degree)
    """
    phi_rad = np.deg2rad(90) # Elevation은 90도(수평)로 가정
    theta_rad = np.deg2rad(theta_deg)

    element_angles = np.arange(0, 2*np.pi, 2*np.pi/Config.N_ANTENNAS)   # A0 = 0 rad, A1 = π/4 rad, ..., A7 = 7π/4 rad

    # UCA 위상 지연 계산 (Phase delay)
    # psi = k * R * sin(elevation) * cos(theta - element_angle)
    k = 2 * np.pi / Config.LAMBDA # phase constant
    psi = k * Config.RADIUS * np.sin(phi_rad) * np.cos(theta_rad - element_angles)
    
    # Steering Vector: a(theta) = exp(j * psi)
    a = np.exp(1j * psi)
    return a.reshape(-1, 1) # (8, 1)

# ==========================================
# 4. MUSIC 알고리즘 코어 (MUSIC Algorithm)
# ==========================================
def run_music_algorithm(X, num_sources=1):
    """
    MUSIC 알고리즘을 수행하여 Spectrum을 반환합니다.
    X: 수신 신호 행렬 (Antennas x Snapshots)
    """
    N = X.shape[0] # 안테나 수
    M = X.shape[1] # 스냅샷 수

    # Step 1: 공분산 행렬 (Covariance Matrix) 계산
    # Rxx = E[X * X^H]
    Rxx = (X @ X.conj().T) / M

    # Step 2: 고유값 분해 (Eigen Decomposition)
    eig_vals, eig_vecs = np.linalg.eig(Rxx)
    
    # 고유값 크기순 정렬 (내림차순)
    idx = eig_vals.argsort()[::-1]
    eig_vals = eig_vals[idx]
    eig_vecs = eig_vecs[:, idx]

    # Step 3: 노이즈 부분공간 (Noise Subspace) 추출
    # 신호 개수(L)만큼 큰 고유값을 제외한 나머지(N-L)가 노이즈 부분공간
    En = eig_vecs[:, num_sources:]

    # Step 4: MUSIC 스펙트럼 계산 (Spatial Spectrum)
    # P(theta) = 1 / (a(theta)^H * En * En^H * a(theta))
    spectrum = []
    
    # 미리 노이즈 부분공간 투영 행렬 계산 (연산 최적화)
    En_EnH = En @ En.conj().T

    for theta in Config.SCAN_ANGLES:
        a = get_uca_steering_vector(theta) # (8, 1)
        
        # 분모 계산: a^H * (En * En^H) * a
        denom = a.conj().T @ En_EnH @ a
        pspectrum = 1 / abs(denom.item())
        spectrum.append(pspectrum)

    return np.array(spectrum)

def estimate_aoa(spectrum):
    """ 스펙트럼에서 피크(최대값)의 각도를 찾습니다. """
    # 정규화 (선택사항)
    spectrum_log = 10 * np.log10(spectrum / np.max(spectrum))
    
    peaks, properties = find_peaks(spectrum_log, height=-10) # 상위 피크 탐색
    
    if len(peaks) > 0:
        # 가장 높은 피크 하나만 반환 (단일 경로 가정)
        best_peak_idx = peaks[np.argmax(properties['peak_heights'])]
        return Config.SCAN_ANGLES[best_peak_idx]
    else:
        return None

# ==========================================
# 5. 시각화 (Visualizer)
# ==========================================
def visualize_results(results_df):
    """
    시간에 따른 Beacon별 AoA 추정 결과를 산점도로 시각화합니다.
    """
    plt.figure(figsize=(12, 6))
    
    # 비콘 ID 별로 색상 구분
    beacons = results_df['beacon_id'].unique()
    
    for bid in beacons:
        subset = results_df[results_df['beacon_id'] == bid]
        plt.scatter(subset['timestamp'], subset['aoa'], label=f'Beacon {bid}', alpha=0.6, s=15)

    plt.title(f'AoA Estimation over Time (MUSIC Algorithm) - 8 Element UCA')
    plt.xlabel('Timestamp (s)')
    plt.ylabel('Estimated Angle of Arrival (Degree)')
    plt.ylim(-180, 180)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend()
    plt.tight_layout()
    plt.show()

# ==========================================
# 6. 메인 실행 (Main)
# ==========================================
def main():
    # 예시 파일 경로 (실제 파일 경로로 변경 필요)
    # CSV 파일이 있는 경로를 입력하세요.
    current_dir = os.path.dirname(os.path.abspath(__file__))
    csv_file_path = os.path.join(current_dir, '../denoised/mapSmall_x0y1.csv')
    
    print(f"1. Loading data from {csv_file_path}...")
    try:
        packets = load_and_process_data(csv_file_path)
    except FileNotFoundError:
        print("Error: CSV 파일을 찾을 수 없습니다. 경로를 확인해주세요.")
        return

    print(f"2. Processing {len(packets)} packets with MUSIC algorithm...")
    
    results = []
    
    for packet in packets:
        # 패킷별로 MUSIC 수행
        spectrum = run_music_algorithm(packet['X'], num_sources=Config.N_SOURCES)
        estimated_angle = estimate_aoa(spectrum)
        
        if estimated_angle is not None:
            results.append({
                'timestamp': packet['timestamp'],
                'beacon_id': packet['beacon_id'],
                'aoa': estimated_angle
            })

    # 결과를 DataFrame으로 변환
    results_df = pd.DataFrame(results)

    # Print mean AoA for each beacon
    if not results_df.empty:
        mean_aoa = results_df.groupby('beacon_id')['aoa'].mean()
        print("Mean AoA estimates for each Beacon ID:")
        print(mean_aoa)
    else:
        print("No valid AoA results to compute mean.")
    
    print("3. Visualizing results...")
    if not results_df.empty:
        visualize_results(results_df)
        print("Done.")
    else:
        print("No valid AoA results found.")

if __name__ == "__main__":
    main()
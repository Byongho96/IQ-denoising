import numpy as np
from config import Config

def get_antenna_positions():
    angles_rad = []
    
    step = 360.0 / Config.NUM_ANTENNAS
    
    for i in range(Config.NUM_ANTENNAS):
        deg = 180 + (i * step * -1) # West=180도, CW=-1
        angles_rad.append(np.deg2rad(deg))
        
    angles_rad = np.array(angles_rad)
    
    x = Config.RADIUS * np.cos(angles_rad)
    y = Config.RADIUS * np.sin(angles_rad)
    
    return x, y

def generate_ideal_iq(aoa_degree):
    """
    Args:
        aoa_degree: 신호가 오는 각도 (Standard Math: East=0, CCW)
                    (dataset.py에서 atan2로 계산된 값 그대로 사용)
    """
    # 1. 입사 각도 (Radian)
    aoa_rad = np.deg2rad(aoa_degree)
    
    # 2. 입사 벡터 (Wave Vector)
    # Source -> Rx 방향 단위 벡터
    u_x = np.cos(aoa_rad)
    u_y = np.sin(aoa_rad)
    
    # 3. 안테나 위치 가져오기 (A1=West, CW)
    ant_x, ant_y = get_antenna_positions()
    
    # 4. 위상 지연 계산 (Plane Wave Equation)
    k = 2 * np.pi / Config.LAMBDA
    
    phases = []
    for i in range(Config.NUM_ANTENNAS):
        # 경로차 (Path Difference) = r dot u
        path_diff = ant_x[i] * u_x + ant_y[i] * u_y
        
        # Phase = -k * path_diff
        # (파동이 진행하는 방향에 있는 안테나가 위상이 늦음)
        phase = -k * path_diff
        phases.append(phase)
        
    # 5. 스위칭 패턴 적용 (111 samples)
    full_sequence_phase = []
    cycle_idx = 0
    curr_len = 0
    
    while curr_len < Config.SEQ_LEN:
        ant_idx = Config.ANTENNA_PATTERN[cycle_idx % 8]
        
        for _ in range(Config.SAMPLES_PER_ANT):
            if curr_len < Config.SEQ_LEN:
                full_sequence_phase.append(phases[ant_idx])
                curr_len += 1
        cycle_idx += 1
        
    # 6. IQ 변환
    iq_complex = np.exp(1j * np.array(full_sequence_phase))
    
    return iq_complex
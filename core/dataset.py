import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
import re
import os
from config import Config
from utils_physics import generate_ideal_iq

class PhaseDataset(Dataset):

    def __init__(self, file_paths):
        self.data = []
        # Config에서 송신기(Tx) 위치 정보 가져오기 (Key: Beacon ID, Value: (x, y))
        self.tx_positions = Config.TX_POSITIONS
        
        print(f"[Dataset] Initializing with {len(file_paths)} files...")

        for file_path in file_paths:
            # 예시: "mapSmall_x0y1.csv" -> Rx 좌표 (0, 1) 추출
            filename = os.path.basename(file_path)
            match = re.search(r'x(\d+)y(\d+)', filename)
            
            if not match:
                print(f"Warning: '{filename}'에서 좌표를 찾을 수 없어 건너뜁니다.")
                continue
                
            rx_x = float(match.group(1))
            rx_y = 4 - float(match.group(2))    # y축 반전 보정 (파일명 기준 y=0이 상단)
            
            # CSV 파일 읽기
            try:
                df = pd.read_csv(file_path)
            except Exception as e:
                print(f"Error reading {file_path}: {e}")
                continue

            # -----------------------------------------------------------
            # 2. 데이터 행(Row)별 파싱 및 GT 계산
            # -----------------------------------------------------------
            for _, row in df.iterrows():
                try:
                    # Beacon ID 확인 (컬럼 1: 'Beacon ID' 또는 'Board ID')
                    beacon_id = int(row.iloc[1])
                except ValueError:
                    continue # ID가 숫자가 아니면 스킵
                
                # 정의되지 않은 비콘 ID는 무시
                if beacon_id not in self.tx_positions:
                    continue
                
                # 송신기(Tx) 좌표 가져오기
                tx_x, tx_y = self.tx_positions[beacon_id]
                
                # =======================================================
                # [핵심] GT AoA (Angle of Arrival) 계산
                # 기준: 표준 수학 좌표계 (East = 0도, Counter-Clockwise = +)
                # =======================================================
                dx = tx_x - rx_x
                dy = tx_y - rx_y
                
                # arctan2(y, x): (-pi ~ pi) 라디안 값 반환
                # (1, 0) -> 0도, (0, 1) -> 90도
                theta_rad = np.arctan2(dy, dx)
                gt_aoa_deg = np.rad2deg(theta_rad) # Degree로 변환

                # =======================================================
                # [데이터 추출] Phase Samples
                # =======================================================
                # 2번 컬럼부터 끝까지가 위상 샘플 데이터
                samples = row.iloc[2:].values
                
                # 데이터 길이 검증 (설정된 시퀀스 길이와 맞는지)
                if len(samples) != Config.SEQ_LEN:
                    continue
                
                # 문자열 등이 섞여 있을 경우를 대비해 float 변환
                try:
                    samples = samples.astype(float)
                except ValueError:
                    continue

                # 메모리에 저장 (Dictionary 형태)
                self.data.append({
                    'samples': samples,     # 측정된 Raw Phase (Degree)
                    'gt_aoa': float(gt_aoa_deg) # 계산된 정답 각도 (Standard Math)
                })
        
        print(f"[Dataset] Loaded {len(self.data)} valid samples.")

    def __len__(self):
        """전체 데이터 샘플 수 반환"""
        return len(self.data)

    def __getitem__(self, idx):
        """
        인덱스(idx)에 해당하는 데이터 샘플을 반환
        
        Returns:
            input_tensor (Tensor): (2, 111) Noisy Input (Real, Imag)
            gt_tensor (Tensor): (2, 111) Clean GT (Real, Imag) - Aligned
            gt_aoa (Tensor): (1,) 정답 각도
        """
        # -----------------------------------------------------------
        # 1. Noisy Input 처리 (측정 데이터)
        # -----------------------------------------------------------
        raw_phase_deg = self.data[idx]['samples']
        
        # Degree -> Radian 변환
        raw_phase_rad = np.deg2rad(raw_phase_deg)
        
        # Phase -> IQ (Complex Number) 변환: e^(j * theta)
        # 진폭(Amplitude) 정보가 없으므로 크기는 1로 가정
        input_iq = np.exp(1j * raw_phase_rad)
        
        # -----------------------------------------------------------
        # 2. GT IQ 생성 (이상적인 신호)
        # -----------------------------------------------------------
        gt_aoa = self.data[idx]['gt_aoa']
        
        # utils_physics 함수를 호출하여 잡음 없는 이상적인 IQ 시퀀스 생성
        # 주의: generate_ideal_iq 함수도 'East, CCW' 각도를 받도록 설정되어 있어야 함
        gt_iq = generate_ideal_iq(gt_aoa)
        
        # -----------------------------------------------------------
        # 3. Global Phase Alignment (위상 정렬) - 중요!
        # -----------------------------------------------------------
        # 문제: 측정 데이터는 패킷마다 랜덤한 초기 위상(Offset)을 가짐
        # 해결: Input의 첫 번째 샘플 위상과 GT의 첫 번째 안테나 위상을 일치시킴

        # 안테나 1개의 샘플 수만큼 가져옴 (Config.SAMPLES_PER_ANT = 3)
        n_anchor = Config.SAMPLES_PER_ANT 
        
        input_anchor = input_iq[:n_anchor] # (3,)
        gt_anchor = gt_iq[:n_anchor]       # (3,)
        
        # 복소수 내적 (Input * conj(GT))의 합을 구하면,
        # 벡터적으로 평균적인 위상 회전각을 얻을 수 있음 (Wrap 문제 해결)
        correlation = np.sum(input_anchor * np.conj(gt_anchor))
        
        # 내적 벡터의 각도가 곧 평균 위상차
        phase_diff = np.angle(correlation)
        
        # GT 신호 전체를 회전시켜 Input과 동기화 (Align)
        gt_iq_aligned = gt_iq * np.exp(1j * phase_diff)
        
        # -----------------------------------------------------------
        # 4. PyTorch Tensor 변환
        # -----------------------------------------------------------
        # Complex numpy array -> Float Tensor (Channel First: 2 x Length)
        # Real part -> Channel 0, Imag part -> Channel 1
        
        input_tensor = torch.stack([
            torch.tensor(input_iq.real), 
            torch.tensor(input_iq.imag)
        ], dim=0).float()
        
        gt_tensor = torch.stack([
            torch.tensor(gt_iq_aligned.real), 
            torch.tensor(gt_iq_aligned.imag)
        ], dim=0).float()
        
        return input_tensor, gt_tensor, torch.tensor(gt_aoa).float()
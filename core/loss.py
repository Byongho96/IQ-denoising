import torch
import torch.nn as nn

class PhysicsInformedLoss(nn.Module):
    """
    AoA 추정(MUSIC) 성능 최적화를 위한 복합 손실 함수
    
    구성:
    1. Shape Loss (Cosine Similarity): 파형의 유사도 학습 (Phase Alignment 오차에 강건함)
    2. Covariance Loss (Physics): MUSIC 알고리즘의 핵심인 공간 공분산 행렬(R) 일치 유도
    """
    def __init__(self, lambda_cov=0.5):
        super(PhysicsInformedLoss, self).__init__()
        self.lambda_cov = lambda_cov

    def forward(self, pred_iq, gt_iq):
        """
        Args:
            pred_iq: (Batch, 2, 111) - 모델 예측값 (Real, Imag)
            gt_iq:   (Batch, 2, 111) - 정답값 (Real, Imag)
        """
        # 1. 복소수 차원으로 변환 (Batch, 111) Complex
        pred_c = torch.complex(pred_iq[:, 0, :], pred_iq[:, 1, :])
        gt_c = torch.complex(gt_iq[:, 0, :], gt_iq[:, 1, :])

        # ==================================================================
        # [Loss 1] Shape Loss: Complex Cosine Similarity
        # ==================================================================
        # 목적: 신호의 '모양'과 '상대적 위상 흐름'을 맞춤.
        # 특징: 데이터셋에서 위상 정렬을 했더라도 남아있는 미세한 Global Phase 오차를 무시하고 
        #       순수하게 신호가 얼마나 닮았는지만 평가함.
        
        # 내적 (Hermitian Inner Product): sum(pred * conj(gt))
        # dim=1 (Time axis) 기준으로 내적
        dot_product = torch.sum(pred_c * torch.conj(gt_c), dim=1)
        
        # 각 벡터의 크기(Norm) 계산
        pred_norm = torch.norm(pred_c, dim=1)
        gt_norm = torch.norm(gt_c, dim=1)
        
        # 코사인 유사도: |<x, y>| / (||x|| * ||y||)
        # 절대값(abs)을 취함으로써 위상이 통째로 회전해 있어도 모양만 같으면 1.0이 됨
        cosine_sim = torch.abs(dot_product) / (pred_norm * gt_norm + 1e-8)
        
        # Loss는 (1 - 유사도). 0에 가까울수록 좋음.
        loss_shape = 1.0 - cosine_sim.mean()

        # ==================================================================
        # [Loss 2] Physics Loss: Spatial Covariance Matrix Distance
        # ==================================================================
        # 목적: MUSIC 알고리즘의 입력이 되는 R 행렬을 직접 최적화.
        #       Time Domain의 노이즈가 제거되고, Spatial Domain의 특징이 보존되도록 함.
        
        # 공분산 행렬 계산 (Batch, 8, 8)
        R_pred = self._compute_covariance_matrix(pred_c)
        R_gt = self._compute_covariance_matrix(gt_c)
        
        # Frobenius Norm (행렬 간의 거리) 계산
        # 두 행렬의 차이를 제곱해서 평균 냄
        diff = R_pred - R_gt
        loss_cov = torch.norm(diff, p='fro', dim=(1,2)).mean()

        # ==================================================================
        # Final Loss
        # ==================================================================
        # Covariance Loss의 스케일이 작을 수 있으므로 가중치(lambda)로 조절
        total_loss = loss_shape + (self.lambda_cov * loss_cov)
        
        return total_loss

    def _compute_covariance_matrix(self, seq_complex):
        """
        1D 시계열 데이터(111 samples)를 8x8 공간 공분산 행렬로 변환
        Args:
            seq_complex: (Batch, 111)
        Returns:
            R: (Batch, 8, 8) Covariance Matrix
        """
        batch_size = seq_complex.shape[0]
        
        # 데이터 구조: [A1, A1, A1, A2, A2, A2, ..., A8, A8, A8] 반복
        # 한 바퀴(Rotation) = 8개 안테나 * 3샘플 = 24샘플
        # 111개 샘플 중 온전한 4바퀴(96샘플)만 사용하여 R을 추정 (나머지는 버림)
        
        n_rotations = 4
        samples_per_ant = 3
        samples_per_rot = 8 * samples_per_ant # 24
        limit = n_rotations * samples_per_rot # 96
        
        # 1. 온전한 회전 데이터만 자르기 (Batch, 96)
        seq_cut = seq_complex[:, :limit]
        
        # 2. 차원 재배열 (Batch, Rotations, Antennas, Samples_per_ant)
        # (Batch, 4, 8, 3)
        x_reshaped = seq_cut.reshape(batch_size, n_rotations, 8, samples_per_ant)
        
        # 3. Snapshot 차원 생성
        # 공분산은 "안테나 간의 관계"이므로 (Antennas x Time_Snapshots) 형태로 만듦
        # Time_Snapshots = Rotations * Samples_per_ant = 4 * 3 = 12
        # (Batch, 8, 12) 형태로 변환 (안테나를 행으로)
        x_spatial = x_reshaped.permute(0, 2, 1, 3).reshape(batch_size, 8, -1)
        
        # 4. 공분산 행렬 계산: R = X * X^H
        # x_spatial: (B, 8, 12)
        # x_conj_T:  (B, 12, 8)
        x_conj_T = torch.conj(x_spatial.transpose(1, 2))
        
        R = torch.bmm(x_spatial, x_conj_T) # 결과: (B, 8, 8)
        
        # 샘플 수(12)로 나누어 정규화 (Unbiased Estimator)
        R = R / x_spatial.shape[2]
        
        return R
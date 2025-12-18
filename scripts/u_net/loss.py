import torch
import torch.nn as nn
from config import Config

class PhysicsInformedLoss(nn.Module):
    def __init__(self, lambda_mse=0.5, lambda_cosine=0.3, lambda_cov=0.2, n_antennas=8):
        super().__init__()
        self.mse = nn.MSELoss()

        self.lambda_mse = lambda_mse
        self.lambda_cosine = lambda_cosine
        self.lambda_cov = lambda_cov

        self.n_antennas = n_antennas

    def forward(self, pred, target):
        """
        pred, target: (Batch, 2, Seq_Len)
        """
        # ---------------------------------------------------------------------------
        # (1) MSE Loss
        # : The standard Mean Squared Error Loss between predicted and target signals.
        # : 1/N * sum((pred_I - target_I)^2 + (pred_Q - target_Q)^2)
        # ---------------------------------------------------------------------------
        loss_mse = self.mse(pred, target)
        
        # ---------------------------------------------------------------------------
        # (2) Cosine Similarity Loss
        # : Measures the cosine product between the predicted and target complex vectors.
        # : This encourages the model to preserve the phase relationships in the signals even if phase shifts occur.
        # : If the vectors are perfectly aligned, cosine similarity = 1 -> loss = 0
        # ---------------------------------------------------------------------------
        # Convert to complex tensors
        pred_c = torch.complex(pred[:, 0, :], pred[:, 1, :])
        target_c = torch.complex(target[:, 0, :], target[:, 1, :])

        # Calculate Cosine Similarity
        dot_product = torch.sum(pred_c * torch.conj(target_c), dim=1)
        norm_pred = torch.norm(pred_c, dim=1)
        norm_target = torch.norm(target_c, dim=1)
        # cosine_sim = torch.abs(dot_product) / (norm_pred * norm_target + 1e-8) # Epsilon to avoid div by zero
        cosine_sim = dot_product.real / (norm_pred * norm_target + 1e-8) # Epsilon to avoid div by zero

        loss_cosine = 1.0 - torch.mean(cosine_sim)

        # ---------------------------------------------------------------------------
        # (3) Covariance Matrix Loss (specialized for MUSIC)
        # : Computes the covariance matrices of the predicted and target signals.
        # : Encourages the model to produce outputs that maintain the spatial correlation structure necessary for accurate
        # ---------------------------------------------------------------------------
        # (Batch, Seq_Len) -> (Batch, 37)
        pred_avg = pred_c[:, :Config.SEQ_LENGTH].reshape(-1, 37, 3).mean(dim=2)
        target_avg = target_c[:, :Config.SEQ_LENGTH].reshape(-1, 37, 3).mean(dim=2)
                
        # Select first 32 samples to form complete cycles
        # (Batch, 4, 8) -> (Batch, 8, 4) [Antennas x Snapshots]
        n_cycles = 37 // Config.N_ELEMENTS # 4
        pred_snap = pred_avg[:, :32].reshape(-1, n_cycles, Config.N_ELEMENTS).permute(0, 2, 1)
        target_snap = target_avg[:, :32].reshape(-1, n_cycles, Config.N_ELEMENTS).permute(0, 2, 1)

        # Calculate Covariance Matrices
        cov_pred = torch.bmm(pred_snap, pred_snap.conj().transpose(1, 2)) / n_cycles
        cov_target = torch.bmm(target_snap, target_snap.conj().transpose(1, 2)) / n_cycles

        # Covariance Loss (Frobenius Norm)
        loss_cov = torch.mean(torch.norm(cov_pred - cov_target, p='fro', dim=(1,2)))

        # ------------------
        #     Total Loss
        # ------------------
        total_loss = (self.lambda_mse * loss_mse) + \
                     (self.lambda_cosine * loss_cosine) + \
                     (self.lambda_cov * loss_cov)
                     
        return total_loss